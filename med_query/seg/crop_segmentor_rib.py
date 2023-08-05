# Copyright (c) DAMO Health

import json
import numpy as np
import os
import SimpleITK as sitk
import skimage.measure as measure
import time
import torch
from easydict import EasyDict as edict

from med_query.utils.common import get_network
from med_query.utils.frame import Frame3d, set_frame_as_ref
from med_query.utils.framed_tensor import FramedTensor
from med_query.utils.resample import crop_roi_with_center, merge_mask_by_prob_patches


class CropSegmentorRib(object):
    def __init__(self, model_path, verbose=True, run_profile=False, rank=0):
        super(CropSegmentorRib, self).__init__()
        self.verbose = verbose
        self.run_profile = run_profile
        self.run_remove_scc = True
        load_begin = time.time()
        assert os.path.isfile(model_path), "model file not found!"
        if model_path.endswith(".pt"):
            extra_files = {"max_stride": "", "cfg": ""}
            net = torch.jit.load(model_path, _extra_files=extra_files)
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
                cfg.network.base,
                cfg.network.downsample,
            )
            net.load_state_dict(ckpt["model_state_dict"])

        self.dimensions = cfg.network.get("dimensions", 3)
        if torch.cuda.is_available():
            net = net.cuda()
        self.net = net.eval()
        self.cfg = cfg
        self.crop_size = np.array(cfg.dataset.crop_size, dtype=np.int32)
        self.interpolation = cfg.dataset.interpolation
        self.expand = cfg.dataset.expand
        self.min_value = cfg.dataset.normalization_params.min_value
        self.max_value = cfg.dataset.normalization_params.max_value
        self.seg_prob_th = 0.55
        self.seg_size_th = 300

        if verbose:
            print(f"loaded model in {time.time() - load_begin:.1f} seconds")

    def get_crops(self, image: sitk.Image, bboxes):
        labels, im_crops, crop_frames = [], [], []
        for box in bboxes:
            label = box.get("label")
            # to be compatible with DETR outputs
            if label == 0:
                continue
            labels.append(label)
            world_center = box.get("center_w")
            x_axis, y_axis, z_axis = (
                box.get("x_axis_local"),
                box.get("y_axis_local"),
                box.get("z_axis_local"),
            )
            expand_for_box = [x for x in self.expand]
            expand_for_box[2] += ((label - 1) % 12) / 22
            size = (
                np.array([box.get("width"), box.get("height"), box.get("depth")]) * expand_for_box
            )
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
                0,
            )
            im_crops.append(im_roi)
            crop_frames.append(Frame3d(im_roi))

        return labels, im_crops, crop_frames

    def seg_crop(self, im_crop_tensor):
        with torch.no_grad():
            predictions = self.net(im_crop_tensor)
            if len(predictions.size()) > 5:
                predictions = torch.unbind(predictions, dim=1)[0]
            predictions = torch.softmax(predictions, dim=1).squeeze(0)

        probmap_crop_tensor = predictions[1, :, :, :].detach().clone()
        probmap_crop_tensor[probmap_crop_tensor < self.seg_prob_th] = 0
        mask_crop_tensor = probmap_crop_tensor.new_zeros(
            probmap_crop_tensor.size(), dtype=torch.uint8
        )
        mask_crop_tensor[probmap_crop_tensor >= self.seg_prob_th] = 1
        if self.run_remove_scc:
            mask_crop_np = mask_crop_tensor.cpu().numpy() > 0
            label_mask_np, num_cc = measure.label(mask_crop_np, return_num=True)
            if num_cc > 1:
                label_mask_tensor = torch.from_numpy(label_mask_np)
                if torch.cuda.is_available():
                    label_mask_tensor = label_mask_tensor.cuda()
                regions = measure.regionprops(label_mask_np)
                regions.sort(key=lambda x: x.area, reverse=True)
                seg_size_th = regions[0].area / 6
                for region in regions:
                    if region.area > seg_size_th:
                        label_mask_tensor[label_mask_tensor == region.label] = 0
                mask_crop_tensor[label_mask_tensor > 0] = 0
                probmap_crop_tensor[label_mask_tensor > 0] = 0

        return mask_crop_tensor, probmap_crop_tensor

    def seg_crops(self, im_crops):
        mask_crops, probmap_crops = [], []
        for im_crop in im_crops:
            im_crop_np = sitk.GetArrayFromImage(im_crop)
            im_crop_tensor = torch.from_numpy(im_crop_np).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                im_crop_tensor = im_crop_tensor.cuda()
            mask_crop_tensor, probmap_crop_tensor = self.seg_crop(im_crop_tensor)

            mask_crops.append(mask_crop_tensor)
            probmap_crops.append(probmap_crop_tensor)

        return mask_crops, probmap_crops

    def merge_crops(
        self,
        case_name,
        image,
        labels,
        mask_crops,
        probmap_crops,
        crop_frames,
        work_with_float16=True,
    ):
        working_dtype = torch.float32
        if work_with_float16:
            working_dtype = torch.float16

        framed_mask_crops = [
            FramedTensor(mask_crop, crop_frame)
            for mask_crop, crop_frame in zip(mask_crops, crop_frames)
        ]
        framed_probmap_crops = [
            FramedTensor(probmap_crop.type(working_dtype), crop_frame)
            for probmap_crop, crop_frame in zip(probmap_crops, crop_frames)
        ]
        mask_ft, _, _ = merge_mask_by_prob_patches(
            Frame3d(image),
            image.GetSize(),
            framed_mask_crops,
            framed_probmap_crops,
            labels,
            use_gpu=True,
            work_with_float16=work_with_float16,
        )

        if self.run_remove_scc:
            tmp_mask_np = mask_ft.tensor.cpu().numpy()
            tmp_label_np, num_cc = measure.label(tmp_mask_np, return_num=True)
            if num_cc > len(labels):
                regions = measure.regionprops(tmp_label_np)
                tmp_label_tensor = torch.from_numpy(tmp_label_np)
                if torch.cuda.is_available():
                    tmp_label_tensor = tmp_label_tensor.cuda()
                for r in regions:
                    if r.area > self.seg_size_th:
                        tmp_label_tensor[tmp_label_tensor == r.label] = 0
                mask_ft.tensor[tmp_label_tensor != 0] = 0

        mask_np = mask_ft.tensor.cpu().numpy()
        mask = sitk.GetImageFromArray(mask_np)
        set_frame_as_ref(mask, image)

        return dict(case_name=case_name, mask=mask)

    def seg_case(self, case_name, image, bboxes, **kwargs):
        seg_case_begin = time.time()
        labels, im_crops, crop_frames = self.get_crops(image, bboxes)
        get_crops_end = time.time()
        if self.verbose:
            print(f"get crops time: {get_crops_end - seg_case_begin: .2f} seconds")

        mask_crops, probmap_crops = self.seg_crops(im_crops)
        seg_crops_end = time.time()
        if self.verbose:
            print(f"seg crops time: {seg_crops_end - get_crops_end: .2f} seconds")

        seg_result = self.merge_crops(
            case_name, image, labels, mask_crops, probmap_crops, crop_frames
        )

        merge_crops_end = time.time()
        if self.verbose:
            print(f"merge crops time: {merge_crops_end - seg_crops_end: .2f} seconds")
            print(f"seg case time: {merge_crops_end - seg_case_begin: .2f} seconds\n")

        return seg_result
