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
from med_query.utils.io_utils import load_module_from_file
from med_query.utils.resample import crop_roi_with_center, merge_mask_by_prob_patches


class CropSegmentorOrgan(object):
    def __init__(self, model_path, verbose=False, run_profile=False, rank=0):
        super(CropSegmentorOrgan, self).__init__()
        self.verbose = verbose
        self.run_profile = run_profile
        self.run_remove_scc = True
        load_begin = time.time()
        assert os.path.isfile(model_path) or os.path.isdir(model_path), "model file not found!"

        if os.path.isdir(model_path):
            if rank == 0:
                print("loading a cluster of models...")

            self.net = {}
            model_dirs = os.listdir(model_path)
            model_dirs = sorted(model_dirs, key=lambda x: int(x.split(".")[0].split("_")[-1]))

            if os.path.isdir(os.path.join(model_path, model_dirs[0])):
                for model_dir in model_dirs:
                    model_cluster_name = model_dir.split("_")[-1]
                    model_ckpt = os.path.join(model_path, model_dir, "checkpoints")
                    ckpts = os.listdir(model_ckpt)
                    ckpts = sorted(ckpts, key=lambda x: int(x.split("_")[-1]), reverse=True)
                    latest_model_path = os.path.join(model_ckpt, ckpts[0], "params.pth")

                    config_file = os.path.join(os.path.dirname(latest_model_path), "config.py")
                    assert os.path.isfile(config_file), "file config.py not found!"
                    cfg_module = load_module_from_file(config_file)
                    cfg = cfg_module.cfg
                    self.organ_dict = cfg.dataset.organ_dict
                    self.organ_cluster = cfg.dataset.organ_cluster

                    epoch = int(ckpts[0].split("_")[-1])
                    ckpt = torch.load(latest_model_path, map_location="cpu")

                    out_channels = len(self.organ_cluster[model_cluster_name]["label"]) + 1
                    out_channels = (
                        2
                        if model_cluster_name == "0" or model_cluster_name == "1"
                        else out_channels
                    )
                    net, self.max_stride = get_network(
                        cfg.network.net_name,
                        cfg.network.in_channels,
                        out_channels,
                        cfg.network.dimensions,
                        cfg.network.normalization,
                        cfg.network.base,
                        cfg.network.downsample,
                    )
                    net.load_state_dict(ckpt["model_state_dict"])
                    if torch.cuda.is_available():
                        net = net.cuda()
                    self.net[model_cluster_name] = net.eval()
                    labels = self.organ_cluster[model_cluster_name]["label"]
                    organs = [self.organ_dict[str(i)]["organ"] for i in labels]
                    if rank == 0:
                        model_cluster_name = (
                            "X" if model_cluster_name == "10" else model_cluster_name
                        )
                        print(
                            f"segmentor_{model_cluster_name} at epoch {epoch} for {organs} \
                                was loaded"
                        )
            else:  # models in .pt file
                for model in model_dirs:
                    model_single = os.path.join(model_path, model)
                    model_cluster_name = model.split(".")[0].split("_")[-1]
                    extra_files = {"max_stride": "", "cfg": ""}
                    net = torch.jit.load(
                        model_single,
                        _extra_files=extra_files,
                        map_location="cpu",
                    )
                    if torch.cuda.is_available():
                        net = net.cuda()
                    self.net[model_cluster_name] = net.eval()
                    cfg = edict(json.loads(extra_files["cfg"]))
                    self.organ_dict = cfg.dataset.organ_dict
                    self.organ_cluster = cfg.dataset.organ_cluster
                    labels = self.organ_cluster[model_cluster_name]["label"]
                    organs = [self.organ_dict[str(i)]["organ"] for i in labels]

        elif model_path.endswith(".pt"):
            extra_files = {"max_stride": "", "cfg": ""}
            net = torch.jit.load(model_path, _extra_files=extra_files)
            self.max_stride = int(extra_files["max_stride"])
            cfg = edict(json.loads(extra_files["cfg"]))
            self.organ_dict = cfg.dataset.organ_dict
            if torch.cuda.is_available():
                net = net.cuda()
            self.net = net.eval()
        else:
            ckpt = torch.load(model_path, map_location="cpu")
            cfg = ckpt["cfg"]
            self.organ_dict = cfg.dataset.organ_dict
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
            if torch.cuda.is_available():
                net = net.cuda()
            self.net = net.eval()

        self.cfg = cfg
        self.dimensions = cfg.network.get("dimensions", 3)
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
            # to be compatible with background class outputs
            if label == 0:
                continue

            world_center = box.get("center_w")
            x_axis, y_axis, z_axis = (
                box.get("x_axis_local"),
                box.get("y_axis_local"),
                box.get("z_axis_local"),
            )
            if label == 3 and box.get("depth") > 110:
                expand_for_box = [1.2, 1.2, 1.0]
            else:
                expand_for_box = self.expand
            size = (
                np.array([box.get("width"), box.get("height"), box.get("depth")]) * expand_for_box
            )
            crop_size = np.array(self.organ_dict[str(label)]["crop_size"], dtype=np.int32)
            spacing = size / crop_size
            im_roi = crop_roi_with_center(
                image,
                world_center,
                spacing,
                x_axis,
                y_axis,
                z_axis,
                crop_size,
                self.interpolation,
                0,
            )
            labels.append(label)
            im_crops.append(im_roi)
            crop_frames.append(Frame3d(im_roi))

            if label in [2, 13]:
                expand_for_box = [1.5, 1.5, 1.5]
                size = np.array([box.get("width"), box.get("height"), box.get("depth")])
                if np.product(np.array([70, 70, 104]) / size) > 2:  # compare to mean shape
                    expand_for_box = [2.5, 2.5, 2.5]
                size = size * expand_for_box

                spacing = size / crop_size
                im_roi = crop_roi_with_center(
                    image,
                    world_center,
                    spacing,
                    x_axis,
                    y_axis,
                    z_axis,
                    crop_size,
                    self.interpolation,
                    0,
                )
                labels.append(label)
                im_crops.append(im_roi)
                crop_frames.append(Frame3d(im_roi))

            if label in [4, 5, 7, 9]:
                start = [-10, -10, -10]
                end = [10, 10, 10]

                coords = list(zip(start, end))
                biases = []
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            corner_coord = np.array([coords[0][i], coords[1][j], coords[2][k]])
                            biases.append(corner_coord)

                spacing = size / crop_size

                for i in range(len(biases)):
                    world_center_bias = world_center + biases[i]
                    im_roi = crop_roi_with_center(
                        image,
                        world_center_bias,
                        spacing,
                        x_axis,
                        y_axis,
                        z_axis,
                        crop_size,
                        self.interpolation,
                        0,
                    )
                    labels.append(label)
                    im_crops.append(im_roi)
                    crop_frames.append(Frame3d(im_roi))

        return labels, im_crops, crop_frames

    def seg_crop(self, im_crop_tensor, label, net):
        with torch.no_grad():
            predictions = net(im_crop_tensor)
            if len(predictions.size()) > 5:
                predictions = torch.unbind(predictions, dim=1)[0]
            predictions = torch.softmax(predictions, dim=1).squeeze(0)

        # unpack probabilities
        assert len(label) == predictions.size()[0] - 1
        mask_crop_tensor_list, probmap_crop_tensor_list = [], []
        for i in range(1, predictions.size()[0]):
            seg_prob_th = self.seg_prob_th
            probmap_crop_tensor = predictions[i, :, :, :].detach().clone()
            probmap_crop_tensor[probmap_crop_tensor < seg_prob_th] = 0
            mask_crop_tensor = probmap_crop_tensor.new_zeros(
                probmap_crop_tensor.size(), dtype=torch.uint8
            )
            mask_crop_tensor[probmap_crop_tensor >= seg_prob_th] = 1
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
            mask_crop_tensor_list.append(mask_crop_tensor)
            probmap_crop_tensor_list.append(probmap_crop_tensor)

        return mask_crop_tensor_list, probmap_crop_tensor_list

    def seg_crops(self, labels, crop_frames, im_crops):
        labels_ret, crop_frames_ret, mask_crops, probmap_crops = [], [], [], []
        for label, crop_frame, im_crop in zip(labels, crop_frames, im_crops):
            im_crop_np = sitk.GetArrayFromImage(im_crop)
            im_crop_tensor = torch.from_numpy(im_crop_np).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                im_crop_tensor = im_crop_tensor.cuda()

            cluster = self.organ_dict[str(label)]["cluster"]
            if cluster not in self.net.keys():
                net = self.net["0"]
                labels_ret.append(label)
                crop_frames_ret.append(crop_frame)
                label = [label]
            else:
                net = self.net[cluster]
                if cluster not in ["2", "3", "5"]:
                    labels_ret.append(label)
                    crop_frames_ret.append(crop_frame)
                    label = [label]
                else:  # multi-channel model
                    label = self.organ_cluster[cluster]["label"]
                    labels_ret.extend(label)
                    crop_frames_ret.extend([crop_frame] * len(self.organ_cluster[cluster]["label"]))
            mask_crop_tensor, probmap_crop_tensor = self.seg_crop(im_crop_tensor, label, net)

            mask_crops.extend(mask_crop_tensor)
            probmap_crops.extend(probmap_crop_tensor)

        return labels_ret, crop_frames_ret, mask_crops, probmap_crops

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
        if len(mask_crops) == 0 or len(probmap_crops) == 0:
            print(f"No cropped frames in {case_name}")
            mask_np = np.zeros(image.GetSize()[::-1], dtype=np.uint8)
            mask = sitk.GetImageFromArray(mask_np)
            set_frame_as_ref(mask, image)

            return dict(case_name=case_name, mask=mask)

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

        try:
            mask_ft, _, _ = merge_mask_by_prob_patches(
                Frame3d(image),
                image.GetSize(),
                framed_mask_crops,
                framed_probmap_crops,
                labels,
                use_gpu=True,
                work_with_float16=work_with_float16,
            )
        except Exception as e:
            raise ValueError(f"Error on {case_name}: {e}")

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

        # labels and crop_frames will be updated
        labels, crop_frames, mask_crops, probmap_crops = self.seg_crops(
            labels, crop_frames, im_crops
        )
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
