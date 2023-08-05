# Copyright (c) DAMO Health
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import torch
from easydict import EasyDict
from torch.utils.data import Dataset

from med_query.utils.common import axis_angle_to_rotation_matirx, convert_df_to_dicts
from med_query.utils.itk import normalize_sitk_im
from med_query.utils.resample import crop_roi_with_center
from ..builder import DATASETS


@DATASETS.register_module()
class SingleOrganDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super(SingleOrganDataset, self).__init__()
        cfg = EasyDict(kwargs)
        self.dataset_name = cfg.dataset_name
        self.phase = cfg.phase
        filelist_csv = cfg[f"{self.dataset_name}_csv"]
        filelist_df = pd.read_csv(filelist_csv)

        self.im_list = list(filelist_df.filename)
        self.im_dir = cfg.im_dir
        self.mask_dir = cfg.mask_dir

        cases_info_df = pd.read_csv(cfg.gt_csv)
        cases_info_df = cases_info_df[~cases_info_df.filename.str.contains("crop")]
        cases_info = convert_df_to_dicts(cases_info_df)
        self.cases_info = dict()
        for line in cases_info:
            case_name = line["filename"]
            if case_name not in self.cases_info:
                self.cases_info[case_name] = []
            self.cases_info[case_name].append(line)

        self.debug_dir = cfg.debug_dir

        self.interpolation = cfg.interpolation
        self.default_value = cfg.default_value
        self.num_classes = cfg.num_classes

        self.normalization_params = cfg.normalization_params

        self.min_value = None
        self.max_value = None
        self.clip = True
        if self.normalization_params is not None:
            self.min_value = self.normalization_params.min_value
            self.max_value = self.normalization_params.max_value
            self.clip = self.normalization_params.clip

        self.is_debugging = cfg.is_debugging
        if self.is_debugging:
            if not os.path.isdir(self.debug_dir):
                os.makedirs(self.debug_dir)

        self.organ_cluster = cfg.organ_cluster
        self.set_cluster(cfg.cluster)

        self.expand = cfg.expand
        self.aug_params = cfg.aug_params

    def set_cluster(self, cluster: str):
        assert cluster in self.organ_cluster.keys(), f"unexpected cluster value: {cluster}"
        self.cluster = cluster
        self.crop_size = np.array(self.organ_cluster[cluster]["crop_size"])
        self.cluster_labels = self.organ_cluster[cluster]["label"]

    def sample(self):
        index = np.random.randint(len(self))
        return self[index]

    def get_crops(self, case_name, image, mask):
        bboxes = self.cases_info.get(case_name)
        chosen = [b for b in bboxes if b["label"] in self.cluster_labels]
        if self.cluster in ["0", "1", "6", "8", "10"] and len(chosen) < len(self.cluster_labels):
            raise ValueError("Not enough organs to pack.")

        im_tensor, mask_tensor = [], []
        for box in chosen:
            label = box.get("label")
            world_center = box.get("center_w")
            x_axis, y_axis, z_axis = (
                box.get("x_axis_local"),
                box.get("y_axis_local"),
                box.get("z_axis_local"),
            )
            expand_for_box = [x for x in self.expand]
            size = (
                np.array([box.get("width"), box.get("height"), box.get("depth")]) * expand_for_box
            )
            if (
                self.aug_params.on
                and self.phase == "train"
                and np.random.uniform() < self.aug_params.prob
            ):
                scale_min, scale_max = self.aug_params.scale
                scale_random = scale_min + (scale_max - scale_min) * np.random.uniform()
                size *= scale_random

                t_min, t_max = self.aug_params.translation
                translation_mm = t_min + (t_max - t_min) * np.random.uniform(0, 1, 3)
                world_center += translation_mm

                rotation_min, rotation_max = self.aug_params.rotation_angle
                angle = rotation_min + (rotation_max - rotation_min) * np.random.uniform()
                rotation_axis = self.aug_params.rotation_axis
                if np.linalg.norm(rotation_axis) < 1e-3:
                    rotation_axis = np.random.uniform(-1, 1, 3)
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_matrix = axis_angle_to_rotation_matirx(rotation_axis, angle)
                x_axis = np.dot(rotation_matrix, x_axis)
                y_axis = np.dot(rotation_matrix, y_axis)
                z_axis = np.dot(rotation_matrix, z_axis)

            spacing = size / self.crop_size
            im_roi = crop_roi_with_center(
                image,
                world_center,
                spacing,
                x_axis,
                y_axis,
                z_axis,
                self.crop_size,
                self.interpolation,
                self.default_value,
            )
            im_roi = sitk.Cast(im_roi, sitk.sitkFloat32)
            im_roi = normalize_sitk_im(im_roi, self.min_value, self.max_value, self.clip, True)
            mask_roi = crop_roi_with_center(
                mask, world_center, spacing, x_axis, y_axis, z_axis, self.crop_size, pad_value=0
            )
            im_roi_tensor = torch.from_numpy(sitk.GetArrayFromImage(im_roi))
            mask_roi_tensor = torch.from_numpy(sitk.GetArrayFromImage(mask_roi))

            if label == 2 or label == 13:
                index = mask_roi_tensor == label
                mask_roi_tensor[mask_roi_tensor > 0] = 0
                mask_roi_tensor[index] = 1
            else:
                indices = []
                for i in self.cluster_labels:
                    indices.append(mask_roi_tensor == i)
                mask_roi_tensor[mask_roi_tensor > 0] = 0
                for i in range(len(self.cluster_labels)):
                    mask_roi_tensor[indices[i]] = i + 1

            im_tensor.append(im_roi_tensor.unsqueeze(0).unsqueeze(0))
            mask_tensor.append(mask_roi_tensor.unsqueeze(0).unsqueeze(0))
        im_tensor = torch.cat(im_tensor, dim=0)
        mask_tensor = torch.cat(mask_tensor, dim=0).long()
        return im_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.im_list)

    def __repr__(self):
        return f"{self.dataset_name} for single organ seg"

    def __getitem__(self, index):
        case_name = self.im_list[index]
        image_path = os.path.join(self.im_dir, f"{case_name}_0000.nii.gz")
        mask_path = os.path.join(self.mask_dir, f"{case_name}.nii.gz")
        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        try:
            im_tensor, mask_tensor = self.get_crops(case_name, image, mask)
            data = {"pid": case_name, "img": im_tensor, "msk": mask_tensor}
            return data
        except Exception:
            return self.sample()
