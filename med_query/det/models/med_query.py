# Copyright (c) DAMO Health

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional

from med_query.utils.dist_utils import allreduce
from med_query.utils.frame import voxel_to_world
from med_query.utils.io_utils import load_model_single
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base_detector import BaseDetector


class MedQueryCore(nn.Module):
    def __init__(self, **kwargs):
        super(MedQueryCore, self).__init__()

        self.backbone = build_backbone(kwargs.get("backbone"))
        neck = kwargs.get("neck")
        if neck is not None:
            self.neck = build_neck(neck)
        self.bbox_head = build_head(kwargs.get("bbox_head"))
        self.init_weight()

    def init_weight(self, pretrained: Optional[str] = None) -> None:
        """
        Initialize the weights in detector.

        :param pretrained: path to pre-trained weights.
        :return:
        """
        self.backbone.init_weights(pretrained=pretrained)
        if hasattr(self, "neck"):
            self.neck.init_weights()
        self.bbox_head.init_weights()

    def forward(
        self,
        img: torch.Tensor,
        img_metas: List[Dict[str, torch.Tensor]],
        msk: Optional[torch.Tensor] = None,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_angles: Optional[List[torch.Tensor]] = None,
        query_index: Optional[List] = None,
    ) -> torch.Tensor:
        """
        :param img: Input images of shape [NCWHD]. Typically these should be mean centered and std
                 scaled.
        :param img_metas: A list of image info dict where each dict has: 'img_shape', 'scale_factor'
                 'flip', etc.
        :param msk: Input masks of shape [NCWHD].
        :param gt_bboxes: Each item are the truth boxes for each image in [cx, cy, cz, w, h, d]
                format.
        :param gt_labels: Class indices corresponding to each box.
        :param gt_angles: Euler angles (alpha, beta, gamma) to each box.
        :param query_index: List of indexes to query.
        :return:
        """
        x = self.backbone(img)
        if hasattr(self, "neck"):
            x = self.neck(x)
        cls_score, bbox_pred, angle_pred, msk_pred = self.bbox_head.forward(
            x, img_metas, query_index
        )
        detections = self.bbox_head.get_bboxes(
            cls_score, bbox_pred, angle_pred, img_metas, query_index
        )

        if gt_bboxes is not None and query_index is None:
            loss_dict = self.bbox_head.loss(
                cls_score,
                bbox_pred,
                angle_pred,
                msk_pred,
                gt_bboxes,
                gt_labels,
                gt_angles,
                msk,
                img_metas,
            )
            return detections, loss_dict
        else:
            return detections, img.new_ones([1])


@DETECTORS.register_module()
class MedQuery(BaseDetector):
    def __init__(self, **kwargs):
        super(MedQuery, self).__init__(**kwargs)
        self.epoch_count = 0
        self.min_det_thresh = kwargs.get("min_det_thresh")
        self.with_seg = kwargs.get("bbox_head").get("with_seg", False)
        self.num_decoder_layers = kwargs.get("bbox_head")["transformer"]["decoder"]["num_layers"]
        snapshot = kwargs.get("snapshot")
        self.use_jit = False
        if snapshot is not None and snapshot.endswith(".pt"):
            self.model = torch.jit.load(snapshot)
            if torch.cuda.is_available():
                self.model.cuda()
            self.use_jit = True
        else:
            self.model = MedQueryCore(**kwargs)
            if torch.cuda.is_available():
                self.model.cuda()
            if snapshot is not None:
                self.model, self.epoch_count = load_model_single(self.model, snapshot)
        if kwargs.get("phase") == "train":
            self.model = DDP(self.model, device_ids=[kwargs.get("local_rank")])

    def train_forward(self, batch, phase="train", **kwargs):
        """
        train method (also used for validation monitoring). wrapper around forward pass of network.
        prepares input data for processing, computes losses, and stores outputs in a dictionary.

        :param batch: [img, bboxes, labels, angles, meta_data]
        :param kwargs:
        :return: results_dict: dictionary with keys:
                'boxes': list over batch elements. each batch element is a list of boxes. each box
                   is a dictionary:
                        [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
        """
        img = batch["img"]
        msk = batch["msk"]
        gt_bboxes = batch["bboxes"]
        gt_labels = batch["labels"]
        gt_angles = batch["angles"]
        meta_data = batch["meta_data"]

        query_index = kwargs.get("query_index", None)
        (detections, loss_dict) = self.model.forward(
            img, meta_data, msk, gt_bboxes, gt_labels, gt_angles, query_index
        )
        res_dict = self.get_results(detections, meta_data)
        if phase == "valid" and query_index is not None:
            return res_dict

        loss_cls = loss_dict["loss_cls"]
        loss_box = loss_dict["loss_box"]
        loss_ang = loss_dict["loss_ang"]

        for i in range(self.num_decoder_layers - 1):
            loss_cls += loss_dict[f"d{i}.loss_cls"]
            loss_box += loss_dict[f"d{i}.loss_box"]
            loss_ang += loss_dict[f"d{i}.loss_ang"]

        loss = loss_cls + loss_box + loss_ang

        loss_seg = loss_cls.new_tensor(0)
        if self.with_seg:
            loss_seg = loss_dict["loss_seg"]
            loss += loss_seg

        if phase == "train":
            reduce_list = [loss_cls, loss_box, loss_ang]
            if self.with_seg:
                reduce_list.append(loss_seg)
            for i in reduce_list:
                allreduce(i)

        res_dict["loss"] = loss
        res_dict["loss_cls"] = loss_cls
        res_dict["loss_box"] = loss_box
        res_dict["loss_ang"] = loss_ang
        res_dict["logger_string"] = (
            f"loss_nonreduce: {loss.item():.5f}, class: {loss_cls.item():.5f}, "
            f"bbox: {loss_box.item():.5f}, angle: {loss_ang.item():.5f}"
        )
        if self.with_seg:
            res_dict["logger_string"] += f" segm: {loss_seg.item():.5f}"

        return res_dict

    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth
        information. prepares input data for processing and stores outputs in a dictionary.

        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is
                a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
        """
        roi_center = kwargs.get("roi_center", None)
        query_index = kwargs.get("query_index", None)
        meta_data = batch["meta_data"]

        img_tensor, img_itk, rot_mat_global = self.prepare_data(
            image=meta_data[0]["img_res0"], roi_center=roi_center
        )
        meta_data[0]["img_itk"] = img_itk
        meta_data[0]["img_shape"] = img_itk.GetSize()
        meta_data[0]["rot_mat_global"] = rot_mat_global
        meta_data[0]["scale_factor"] = 1

        # to be compatible with torchsrcipt which only accept two args and not support query_index.
        if self.use_jit:
            meta_data_jit = [
                {
                    "img_shape": torch.tensor(m.get("img_shape")).cuda(),
                    "scale_factor": torch.tensor([1] * 6).cuda(),
                }
                for m in meta_data
            ]
            detections, _ = self.model.forward(img_tensor, meta_data_jit)
        else:
            detections, _ = self.model.forward(img_tensor, meta_data, query_index=query_index)
        res_dict = self.get_results(detections, meta_data)

        return res_dict

    def get_results(self, detections, meta_data):
        """
        Restore batch dimension of merged detections, unmolds detections, create and fill results
        dict.

        :param detections: (n_final_detections, (cx, cy, cz, w, h, d), (alpha, beta, gamma),
                pred_class_id, pred_score))
        :return: results_dict: dictionary with keys:
            'boxes': list over batch element, each batch element is a pandas.DataFrame. containing:
                'filename', 'label', 'world_x', 'world_y', 'world_z', 'dx1', 'dx2', 'dx3', 'dy1',
                'dy2', 'dy3', 'dz1', 'dz2', 'dz3', 'width', 'height', 'depth', 'box_score',
                'box_type'.
        """

        detections = [det.cpu().data.numpy() for det in detections]

        box_results_list = [[] for _ in range(len(meta_data))]

        for ix in range(len(meta_data)):
            if 0 not in detections[ix].shape:
                pid = meta_data[ix]["pid"]
                img_itk = meta_data[ix]["img_itk"]
                rot_mat_global = meta_data[ix]["rot_mat_global"]
                spacing = img_itk.GetSpacing()

                bbox_preds = detections[ix][:, :6]
                angle_preds = detections[ix][:, 6:9]
                class_ids = detections[ix][:, 9].astype(np.int32)
                scores = detections[ix][:, 10]

                if 0 not in bbox_preds.shape:
                    for ix2, score in enumerate(scores):
                        label = class_ids[ix2]
                        if score >= self.min_det_thresh and label > 0:
                            angle = angle_preds[ix2]
                            rot_mat = R.from_euler("zyx", angle, degrees=False).as_matrix()
                            rot_mat = np.linalg.inv(rot_mat_global) @ rot_mat

                            bp = bbox_preds[ix2]
                            center_v = bp[:3]
                            center_w = voxel_to_world(img_itk, center_v)

                            width = bp[3] * spacing[0]
                            height = bp[4] * spacing[1]
                            depth = bp[5] * spacing[2]
                            rot_mat = rot_mat.transpose().flatten()

                            box_results_list[ix].append(
                                {
                                    "filename": pid,
                                    "label": label,
                                    "world_x": center_w[0],
                                    "world_y": center_w[1],
                                    "world_z": center_w[2],
                                    "dx1": rot_mat[0],
                                    "dx2": rot_mat[1],
                                    "dx3": rot_mat[2],
                                    "dy1": rot_mat[3],
                                    "dy2": rot_mat[4],
                                    "dy3": rot_mat[5],
                                    "dz1": rot_mat[6],
                                    "dz2": rot_mat[7],
                                    "dz3": rot_mat[8],
                                    "width": width,
                                    "height": height,
                                    "depth": depth,
                                    "box_score": score,
                                    "box_type": "det",
                                }
                            )
                if len(box_results_list[ix]) > 0:
                    box_results_list[ix] = pd.DataFrame(box_results_list[ix])
                else:
                    box_results_list[ix] = pd.DataFrame(
                        columns=[
                            "filename",
                            "label",
                            "world_x",
                            "world_y",
                            "world_z",
                            "dx1",
                            "dx2",
                            "dx3",
                            "dy1",
                            "dy2",
                            "dy3",
                            "dz1",
                            "dz2",
                            "dz3",
                            "width",
                            "height",
                            "depth",
                            "box_score",
                            "box_type",
                        ]
                    )

        results_dict = {"boxes": box_results_list}

        return results_dict
