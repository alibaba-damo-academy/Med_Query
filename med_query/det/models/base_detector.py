# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from med_query.utils.frame import Frame3d
from med_query.utils.itk import mask_bbox
from med_query.utils.resample import resample_base
from ..builder import DETECTORS


@DETECTORS.register_module()
class BaseDetector(nn.Module):
    def __init__(self, **kwargs):
        super(BaseDetector, self).__init__()
        self.spacing = np.array(kwargs.get("dataset").get("spacing"))
        self.max_bbox_size = np.array(kwargs.get("dataset").get("max_bbox_size"))
        self.aug_params = kwargs.get("dataset").get("aug_params")
        self.max_stride = kwargs.get("dataset").get("max_stride")

    def train_forward(self, **kwargs):
        raise NotImplementedError

    def test_forward(self, **kwargs):
        raise NotImplementedError

    def prepare_data(self, **kwargs):
        image = kwargs.get("image")
        roi_center = kwargs.get("roi_center")
        if roi_center is None:
            min_box, max_box = mask_bbox(image)
            roi_center = (min_box + max_box) / 2

        # each column of direction represents an axis
        direction = np.eye(3, dtype=np.float32)
        rot_mat_global = direction

        frame = Frame3d()
        frame.origin = roi_center
        frame.spacing = self.spacing
        frame.direction = direction

        t, s, rad = 0, 1, 0
        if self.aug_params.on:
            t = self.aug_params.translation[1]
            s = self.aug_params.scale[0]
            rad = np.deg2rad(self.aug_params.rotation_angle[1])
        crop_size = (self.max_bbox_size + t) / (self.spacing * s) / np.cos(rad)
        crop_size = (crop_size // self.max_stride + 1) * self.max_stride
        crop_size = crop_size.astype(np.int32)

        crop_origin = frame.voxel_to_world(0 - crop_size // 2)
        frame.origin = crop_origin

        img_crop = resample_base(
            image,
            frame.origin,
            frame.direction,
            frame.spacing,
            crop_size,
            interpolator="linear",
            pad_value=0,  # image has been normalized
        )

        img_tensor = torch.from_numpy(sitk.GetArrayFromImage(img_crop)).permute(2, 1, 0)[None][None]
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        return img_tensor, img_crop, rot_mat_global
