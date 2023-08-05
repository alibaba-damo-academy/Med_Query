# Copyright (c) DAMO Health

import numpy as np
import SimpleITK as sitk
import torch

from .frame import Frame3d, set_frame


class FramedTensor(object):
    def __init__(
        self, tensor: torch.Tensor = None, frame: Frame3d = None,
    ):
        self.tensor = tensor
        self.frame = frame
        self.device = torch.device("cpu")
        self.dtype_ = None
        if self.tensor is not None:
            self.device = self.tensor.device
            self.dtype_ = self.tensor.dtype

    @staticmethod
    def from_itk_image(itk_image: sitk.Image):
        ft = FramedTensor()
        ft.frame = Frame3d(itk_image)
        im_np = sitk.GetArrayFromImage(itk_image)
        if im_np.dtype == np.uint16:
            im_np = im_np.astype(np.int32)
        ft.tensor = torch.from_numpy(im_np)
        return ft

    def to_itk_image(self):
        image = sitk.GetImageFromArray(self.tensor.cpu().numpy())
        set_frame(image, self.frame)
        return image

    def cpu(self):
        if self.device != torch.device("cpu"):
            self.tensor = self.tensor.cpu()
            self.device = self.tensor.device
        return self

    def cuda(self):
        if self.device == torch.device("cpu"):
            self.tensor = self.tensor.cuda()
            self.device = self.tensor.device
        return self

    def to(self, device: torch.device):
        if self.device != device:
            self.device = device
            self.tensor = self.tensor.to(device)
        return self

    def type(self, dtype):
        new_framed_tensor = FramedTensor()
        new_framed_tensor.tensor = self.tensor.type(dtype)
        new_framed_tensor.device = self.device
        new_framed_tensor.frame = self.frame
        return new_framed_tensor

    @property
    def dtype(self):
        return self.dtype_

    def world_box(self):
        start = [0, 0, 0]
        end = self.tensor.size()[::-1]
        coords = list(zip(start, end))
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    voxel_coord = [coords[0][i], coords[1][j], coords[2][k]]
                    corners.append(self.frame.voxel_to_world(voxel_coord))
        corners = np.vstack(corners)
        min_corner = corners.min(axis=0)
        max_corner = corners.max(axis=0)
        return min_corner, max_corner

    def __repr__(self):
        return f"{self.tensor.__repr__()} with frame {self.frame} on {self.device.__repr__()}"
