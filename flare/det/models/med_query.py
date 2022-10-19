# Copyright (c) DAMO Health

import numpy as np
import pandas as pd
import torch
from flare.det.builder import DETECTORS
from flare.utils.frame import voxel_to_world
from scipy.spatial.transform import Rotation as R

from .base_detector import BaseDetector


@DETECTORS.register_module()
class MedQuery(BaseDetector):
    def __init__(self, **kwargs):
        super(MedQuery, self).__init__(**kwargs)
        self.epoch_count = 0
        self.min_det_thresh = kwargs.get("min_det_thresh")
        self.num_decoder_layers = kwargs.get("bbox_head")["transformer"]["decoder"]["num_layers"]
        snapshot = kwargs.get("snapshot")
        self.use_jit = False
        if snapshot is not None and snapshot.endswith(".pt"):
            self.model = torch.jit.load(snapshot, map_location="cpu").cuda()
            self.use_jit = True

    def test_forward(self, batch, **kwargs):
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

        # to be compatible with torchsrcipt which only accept two args and not support query_index:
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

        :param detections: (n_final_detections, (x, y, z, w, h, d),
            (alpha, beta, gamma), pred_class_id, pred_score)
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
