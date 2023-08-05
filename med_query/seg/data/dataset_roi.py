# Copyright (c) DAMO Health
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import torch
from easydict import EasyDict
from torch.utils.data import Dataset
from typing import Tuple

from med_query.utils.common import axis_angle_to_rotation_matirx
from med_query.utils.frame import Frame3d, world_box
from med_query.utils.itk import mask_bbox, normalize_sitk_im
from med_query.utils.resample import resample_base
from ..builder import DATASETS


@DATASETS.register_module()
class RoiDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        cfg = EasyDict(kwargs)
        self.dataset_name = cfg.dataset_name
        self.phase = cfg.phase
        filelist_csv = cfg[f"{self.dataset_name}_csv"]
        filelist_df = pd.read_csv(filelist_csv)
        self.im_list = list(filelist_df.filename)
        self.im_dir = cfg.im_dir
        self.mask_dir = cfg.mask_dir

        self.default_value = cfg.default_value

        self.spacing = np.array(cfg.spacing, dtype=np.double)
        assert self.spacing.size == 3, "spacing should be a vector with 3 elements"

        self.crop_size = np.array(cfg.crop_size, dtype=np.int32)
        assert self.crop_size.size == 3, "crop_size should be a vector with 3 elements"

        self.aug_params = cfg.aug_params

        assert cfg.interpolation in (
            "linear",
            "nearest",
        ), "interpolation should be one of 'linear' or 'nearest'!"
        self.interpolation = cfg.interpolation
        self.min_value = cfg.normalization_params.min_value
        self.max_value = cfg.normalization_params.max_value
        self.clip = cfg.normalization_params.clip

        self.samples = cfg.samples

    def __len__(self) -> int:
        return len(self.im_list)

    def __repr__(self):
        return f"{self.dataset_name} for roi extractor"

    def global_sample(self) -> np.ndarray:
        """
        random sample a position in the image

        :return: a position in world coordinate
        """
        im_size_mm = self.max_world - self.min_world
        crop_size_mm = self.crop_size * self.spacing

        sp = np.array(self.min_world, dtype=np.double)
        for i in range(3):
            if im_size_mm[i] > crop_size_mm[i]:
                sp[i] = self.min_world[i] + np.random.uniform(0, im_size_mm[i] - crop_size_mm[i])
        center = sp + crop_size_mm / 2
        return center

    def mask_sample(self, min_corner: np.ndarray, max_corner: np.ndarray) -> np.ndarray:
        """
        random sample a position from the mask area

        :param min_corner: min_corner of the foreground area
        :param max_corner: max corner of the foreground area
        :return: a position in world coordinate
        """
        crop_size_mm = self.crop_size * self.spacing
        center = np.array([0, 0, 0])
        for i in range(3):
            center[i] = np.random.uniform(min_corner[i], max_corner[i])
        center = np.maximum(center, self.min_world + crop_size_mm / 2)
        center = np.minimum(center, self.max_world - crop_size_mm / 2)

        return center

    def get_crops(
        self, image: sitk.Image, mask: sitk.Image, world_center: np.ndarray
    ) -> Tuple[sitk.Image, sitk.Image]:
        crop_spacing = self.spacing.copy()
        # each column of direction represents an axis
        direction = np.eye(3, dtype=np.float32)
        new_direction = direction.copy()

        if (
            self.aug_params.on
            and self.phase == "train"
            and np.random.uniform() < self.aug_params.prob
        ):
            scale_min, scale_max = self.aug_params.scale
            crop_spacing *= scale_min + (scale_max - scale_min) * np.random.uniform(0, 1, 3)

            translation_min, translation_max = self.aug_params.translation
            translation_mm = translation_min + (
                translation_max - translation_min
            ) * np.random.uniform(0, 1, 3)
            world_center += translation_mm

            rotation_min, rotation_max = self.aug_params.rotation
            angle = rotation_min + (rotation_max - rotation_min) * np.random.uniform()

            rotation_axis = self.aug_params.rotation_axis
            if np.linalg.norm(rotation_axis) < 1e-3:
                rotation_axis = np.random.uniform(-1, 1, 3)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            rotation_matrix = axis_angle_to_rotation_matirx(rotation_axis, angle)
            new_direction = np.dot(rotation_matrix, direction)

        frame = Frame3d()
        frame.origin = world_center
        frame.spacing = crop_spacing
        frame.direction = new_direction

        crop_origin = frame.voxel_to_world(0 - self.crop_size // 2)
        frame.origin = crop_origin

        im_crop = resample_base(
            image,
            frame.origin,
            frame.direction,
            frame.spacing,
            self.crop_size,
            interpolator=self.interpolation,
            pad_value=self.default_value,
        )

        mask_crop = resample_base(
            mask,
            frame.origin,
            frame.direction,
            frame.spacing,
            self.crop_size,
            interpolator="nearest",
            pad_value=0,
        )

        im_crop = sitk.Cast(im_crop, sitk.sitkFloat32)
        im_crop = normalize_sitk_im(im_crop, self.min_value, self.max_value, self.clip, True)
        mask_crop = sitk.Clamp(mask_crop, lowerBound=0, upperBound=1)

        return im_crop, mask_crop

    def __getitem__(self, index):
        case_name = self.im_list[index]

        if "FLARE" in case_name:
            image_path = os.path.join(self.im_dir, f"{case_name}_0000.nii.gz")
            mask_path = os.path.join(self.mask_dir, f"{case_name}.nii.gz")
        elif "RibFrac" in case_name:
            image_path = os.path.join(self.im_dir, f"{case_name}-image.nii.gz")
            mask_path = os.path.join(self.mask_dir, f"{case_name}-ribmask_labelled.nii.gz")
        else:  # CTSpine1K
            image_path = os.path.join(self.im_dir, f"{case_name}.nii.gz")
            mask_path = os.path.join(self.mask_dir, f"{case_name}_seg.nii.gz")

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # sample a crop center, need to compensate for class imbalance in a batch
        world_centers = []

        # for entire image
        self.min_world, self.max_world = world_box(image)
        for i in range(self.samples[0]):
            world_centers.append(self.global_sample())

        # for mask area
        min_corner, max_corner = mask_bbox(mask)
        for i in range(self.samples[1]):
            if min_corner is None:
                world_centers.append(self.global_sample())
            else:
                world_centers.append(self.mask_sample(min_corner, max_corner))

        # currently there are several centers
        # sample a crop from image and normalize it with given parameters
        im_crop_tensors, mask_crop_tensors = [], []
        for idx, world_center in enumerate(world_centers):
            im_crop, mask_crop = self.get_crops(image, mask, world_center)
            im_crop_tensor = torch.from_numpy(sitk.GetArrayFromImage(im_crop)).unsqueeze(0)
            mask_crop_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask_crop)).unsqueeze(0)
            im_crop_tensors.append(im_crop_tensor.unsqueeze(0))
            mask_crop_tensors.append(mask_crop_tensor.unsqueeze(0))

        im_tensor = torch.cat(im_crop_tensors, dim=0)
        mask_tensor = torch.cat(mask_crop_tensors, dim=0).long()
        data = {"pid": case_name, "img": im_tensor, "msk": mask_tensor}
        return data
