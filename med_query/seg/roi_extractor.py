# Copyright (c) DAMO Health
import json
import numpy as np
import os
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from pathlib import Path

from med_query.utils.common import get_network
from med_query.utils.frame import set_frame_as_ref, voxel_to_world
from med_query.utils.resample import resample_base, resample_itkimage_torai


class RoiExtractor(object):
    def __init__(self, model_path, mode):
        super(RoiExtractor, self).__init__()
        assert mode in ["rib", "spine", "organ"], f"unsupported mode {mode}!"
        self.mode = mode
        hyper_params_json = (
            Path(os.path.abspath(__file__)).parent.parent / "scripts" / f"hyper_params_{mode}.json"
        )
        with open(hyper_params_json, "r") as f:
            hyper_params = json.load(f)
        assert "center_offset" in hyper_params.keys(), "center offset not calculated!"
        self.center_offset = np.array(hyper_params.get("center_offset"), dtype=np.float32)

        assert os.path.isfile(model_path), "model file not found!"
        if model_path.endswith(".pt"):
            extra_files = {"max_stride": "", "cfg": ""}
            net = torch.jit.load(model_path, _extra_files=extra_files, map_location="cpu")
            self.max_stride = int(extra_files["max_stride"])
            cfg = edict(json.loads(extra_files["cfg"]))
        else:
            ckpt = torch.load(model_path, map_location="cpu")
            cfg = ckpt["cfg"]

            net, self.max_stride = get_network(
                cfg.network.net_name,
                cfg.network.in_channels,
                cfg.network.out_channels,
                cfg.network.dimensions,
                cfg.network.normalization,
            )
            net.load_state_dict(ckpt["model_state_dict"])

        self.min_value = cfg.dataset.normalization_params.min_value
        self.max_value = cfg.dataset.normalization_params.max_value
        self.spacing = cfg.dataset.spacing
        self.cfg = cfg
        if torch.cuda.is_available():
            net = net.cuda()
        self.net = net.eval()
        self.seg_prob_th = 0.7

    def extract_case(self, image: sitk.Image, return_mask: bool = False):
        assert (
            image.GetPixelIDValue() == sitk.sitkFloat32
        ), f"wrong input pixel type {image.GetPixelIDValue()}"
        iso_image = resample_itkimage_torai(image, self.spacing, "linear", 0)

        im_np = sitk.GetArrayFromImage(iso_image)
        pad_nums = [0, 0, 0]
        orig_shape = im_np.shape
        need_pad = False
        for dim in range(3):
            remain_num = orig_shape[dim] % self.max_stride
            if remain_num != 0:
                need_pad = True
                pad_nums[dim] = self.max_stride - remain_num
        if need_pad:
            im_np = np.pad(
                im_np,
                [(0, pad_nums[0]), (0, pad_nums[1]), (0, pad_nums[2])],
                "constant",
                constant_values=0,
            )
        im_tensor = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            im_tensor = im_tensor.cuda()
        
        predictions = self.net(im_tensor)

        predictions = F.softmax(predictions, dim=1).squeeze(0)
        if need_pad:
            predictions = predictions[:, : orig_shape[0], : orig_shape[1], : orig_shape[2]]
        predictions = predictions.cpu()
        mask_np = predictions.argmax(dim=0).numpy().astype(np.uint8)
        probmap = predictions.max(dim=0)[0].detach().numpy()
        mask_np[probmap < self.seg_prob_th] = 0

        if return_mask:
            mask = sitk.GetImageFromArray(mask_np)
            set_frame_as_ref(mask, iso_image)
            mask = resample_base(
                mask,
                image.GetOrigin(),
                image.GetDirection(),
                image.GetSpacing(),
                image.GetSize(),
                pad_value=0,
            )
            return mask
        else:
            zs, ys, xs = np.where(mask_np > 0)
            weight_center = voxel_to_world(iso_image, [xs.mean(), ys.mean(), zs.mean()])
            roi_center = weight_center
            return roi_center - self.center_offset
