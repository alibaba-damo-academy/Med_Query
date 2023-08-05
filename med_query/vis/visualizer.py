# Copyright (c) DAMO Health

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional, Tuple, Union
from vedo import Box, Point, Video, Volume, show

from med_query.utils.common import mask_np_bbox
from med_query.utils.frame import world_to_voxel
from med_query.utils.resample import resample_itkimage_torai
from med_query.vis.common import get_itksnap_color_dict


class Visualizer3D(object):
    def __init__(
        self,
        image_cmap: str = "RdBu_r",
        mask_cmap: str = "jet",
        image_alpha: Tuple = (0.0, 0.0, 0.0, 0.0, 0.7, 0.8, 1.0),
        mask_alpha: Tuple = (0.0, 1.0),
        spacing=2,
    ):
        self.image_cmap = image_cmap
        self.mask_cmap = mask_cmap
        self.image_alpha = image_alpha
        self.mask_alpha = mask_alpha
        self.spacing = spacing

    def parse_df(
        self,
        ref_itk: sitk.Image = None,
        df: pd.DataFrame = pd.DataFrame({}),
        center_color: str = "r",
        cmap: Dict = {},
        show_box_not_roi: bool = False,
        label_visible: Optional[Union[List, int]] = None,
    ):
        """
        :param ref_itk: reference ITK Image
        :param df: pandas data frame
        :param center_color: color for center point
        :param cmap: color map
        :param show_box_not_roi: flag for showing box with surfaces or just roi
        :param label_visible: specific labels in mask to display
        :return: centers & boxes actors in List format
        """
        assert set(
            [
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
            ]
        ).issubset(df.columns)

        if label_visible is not None and type(label_visible) is int:
            label_visible = [label_visible]

        labels = df["label"].values
        center_x = df["world_x"].values
        center_y = df["world_y"].values
        center_z = df["world_z"].values

        centers = []
        boxes = []
        for i in range(len(labels)):
            label = labels[i]
            if label_visible is not None and label not in label_visible:
                continue
            center_w = [center_x[i], center_y[i], center_z[i]]
            center_v = world_to_voxel(ref_itk, center_w)
            centers.append(Point(pos=center_v, c=center_color))
            shape = np.array(df.loc[i, ["width", "height", "depth"]].tolist()) / self.spacing
            direction = (
                np.array(
                    df.loc[
                        i, ["dx1", "dx2", "dx3", "dy1", "dy2", "dy3", "dz1", "dz2", "dz3"]
                    ].tolist()
                )
                .reshape(3, 3)
                .transpose()
            )

            T = np.eye(4)
            T[:3, :3] = direction
            T[:3, 3] = center_v

            box = Box(
                pos=(0, 0, 0), length=shape[0], width=shape[1], height=shape[2], alpha=0.5,
            ).applyTransform(T, reset=True)
            if show_box_not_roi:
                boxes.append(box.c(cmap[label]))
            else:
                boxes.append(box.boundaries().c(cmap[label]))

        return centers, boxes

    def compute_mean_angle(
        self, df: pd.DataFrame = pd.DataFrame({}), degrees=True, save_diff=False
    ):
        """
        Get mean rotation euler angle for left ribs and right ribs, respectively.
        :param df: pandas DataFrame containing rib rotation matrices.
        :param degrees: whether to represent angles in terms of degrees, otherwise rad.
        :param save_diff: whether to save angle difference for each bbox.
        :return: left mean angle and right mean angle.
        """
        assert set(
            ["filename", "label", "dx1", "dx2", "dx3", "dy1", "dy2", "dy3", "dz1", "dz2", "dz3"]
        ).issubset(df.columns)

        ids = pd.unique(df["filename"])
        right_angles = []
        left_angles = []
        for pid in ids:
            df_case = df[df["filename"] == pid].reset_index()
            for i in range(len(df_case)):
                direction = (
                    np.array(
                        df_case.loc[
                            i, ["dx1", "dx2", "dx3", "dy1", "dy2", "dy3", "dz1", "dz2", "dz3"]
                        ].tolist()
                    )
                    .reshape(3, 3)
                    .transpose()
                )
                rot_mat = R.from_matrix(direction)
                rib_label = df_case.loc[i, "label"]
                euler_angles = rot_mat.as_euler("zyx", degrees=degrees)
                if rib_label <= 12:
                    right_angles.append(euler_angles)
                elif rib_label > 12:
                    left_angles.append(euler_angles)

        right_angles = np.vstack(right_angles)

        right_mean_angles = right_angles.mean(axis=0)
        right_angles_delta = right_angles - right_mean_angles

        right_angles_delta_mean = right_angles_delta.mean(axis=0)
        right_angles_delta_std = right_angles_delta.std(axis=0)

        left_angles = np.vstack(left_angles)
        left_mean_angles = left_angles.mean(axis=0)
        left_angles_delta = left_angles - left_mean_angles

        left_angles_delta_mean = left_angles_delta.mean(axis=0)
        left_angles_delta_std = left_angles_delta.std(axis=0)

        right_mean_rot_mat = R.from_euler("zyx", right_mean_angles, degrees=degrees)
        left_mean_rot_mat = R.from_euler("zyx", left_mean_angles, degrees=degrees)

        T = np.eye(4)
        T[:3, :3] = right_mean_rot_mat.as_matrix()
        right_mean_box = Box(
            pos=(0, 0, 0), length=50, width=30, height=10, alpha=0.5
        ).applyTransform(T)

        T = np.eye(4)
        T[:3, :3] = left_mean_rot_mat.as_matrix()
        T[:3, 3] = [50, 0, 0]
        left_mean_box = Box(
            pos=(0, 0, 0), length=50, width=30, height=10, alpha=0.5
        ).applyTransform(T)

        res_dict = {
            "right_mean_angles": right_mean_angles,
            "right_angles_delta_mean": right_angles_delta_mean,
            "right_angles_delta_std": right_angles_delta_std,
            "left_mean_angles": left_mean_angles,
            "left_angles_delta_mean": left_angles_delta_mean,
            "left_angles_delta_std": left_angles_delta_std,
            "right_mean_box_visual": right_mean_box,
            "left_mean_box_visual": left_mean_box,
        }

        if save_diff:
            # rearrange angle delta in patient order
            right_idx = 0
            left_idx = 0
            angles_delta_seq = []
            for pid in ids:
                df_case = df[df["filename"] == pid].reset_index()
                for i in range(len(df_case)):
                    rib_label = df_case.loc[i, "label"]
                    if rib_label <= 12:
                        angles_delta = right_angles_delta[right_idx]
                        right_idx += 1
                    elif rib_label > 12:
                        angles_delta = left_angles_delta[left_idx]
                        left_idx += 1
                    angles_delta_seq.append(angles_delta)
            angles_delta_seq = np.vstack(angles_delta_seq)

            res_dict["alpha"] = angles_delta_seq[:, 0]
            res_dict["beta"] = angles_delta_seq[:, 1]
            res_dict["gamma"] = angles_delta_seq[:, 2]

        return res_dict

    def show(
        self,
        title: str = "vedo",
        image_path: str = None,
        mask_path: str = None,
        gt_df: pd.DataFrame = pd.DataFrame({}),
        pred_df: pd.DataFrame = pd.DataFrame({}),
        show_box_not_roi: bool = False,
        label_visible: Optional[Union[List, int]] = None,
        others_to_show: Optional[List] = None,
        save_video: bool = False,
    ):
        """
        :param title: window title
        :param image_path: image_path
        :param mask_path: mask_path
        :param gt_df: pandas data frame storing ground truth information
        :param pred_df: pandas data frame storing prediction information
        :param show_box_not_roi: flag for showing box with surfaces or just roi
        :param label_visible: specific labels in mask to display
        :param others_to_show: show additional markers, like arrow, line etc.
        :param save_video: whether to save a video
        :return: None
        """
        actors = []
        image_itk = None
        mask_itk = None
        shape = None
        if label_visible is not None and type(label_visible) is int:
            label_visible = [label_visible]
        if image_path is not None:
            image_itk = sitk.ReadImage(image_path)
            spacing = image_itk.GetSpacing()
            if not np.all(np.abs(np.array(spacing) - self.spacing) < 0.01):
                image_itk = resample_itkimage_torai(image_itk, spacing=[self.spacing] * 3)

            image = Volume(
                sitk.GetArrayFromImage(image_itk), c=self.image_cmap, alpha=self.image_alpha, mode=0
            )
            actors.append(image)
            shape = np.array(image_itk.GetSize())
        if mask_path is not None:
            mask_itk = sitk.ReadImage(mask_path)
            spacing = mask_itk.GetSpacing()
            if not np.all(np.abs(np.array(spacing) - self.spacing) < 0.01):
                mask_itk = resample_itkimage_torai(
                    mask_itk, spacing=[self.spacing] * 3, pad_value=0
                )

            mask_vol = sitk.GetArrayFromImage(mask_itk)
            cmap = get_itksnap_color_dict()
            for i in range(50, 0, -1):
                if label_visible is not None and i not in label_visible:
                    continue
                minb, maxb = mask_np_bbox(mask_vol, mask_value=i)
                if minb is None or maxb is None:
                    continue
                mask_vol_local = mask_vol[
                    minb[0] : maxb[0] + 1, minb[1] : maxb[1] + 1, minb[2] : maxb[2] + 1
                ].copy()
                mask_vol_local[mask_vol_local != i] = 0
                cmap_local = {0: cmap[0], i: cmap[i]}
                mask = Volume(
                    mask_vol_local.transpose(),
                    origin=minb[::-1],  # origin is in [x, y, z] order
                    c=list(cmap_local.items()),
                    alpha=self.mask_alpha,
                    mode=1,
                    mapper="smart",
                )
                actors.append(mask)
            shape = np.array(mask_itk.GetSize())

        if not gt_df.empty:
            assert image_itk is not None or mask_itk is not None
            ref_itk = image_itk if image_itk is not None else mask_itk
            gt_cmap = get_itksnap_color_dict()
            gt_centers, gt_boxes = self.parse_df(
                ref_itk, gt_df, "r", gt_cmap, show_box_not_roi, label_visible
            )
            actors.extend([*gt_centers, *gt_boxes])
        if not pred_df.empty:
            assert image_itk is not None or mask_itk is not None
            ref_itk = image_itk if image_itk is not None else mask_itk
            pred_cmap = get_itksnap_color_dict()
            pred_centers, pred_boxes = self.parse_df(
                ref_itk, pred_df, "b", pred_cmap, show_box_not_roi, label_visible
            )
            actors.extend([*pred_centers, *pred_boxes])

        if others_to_show is not None:
            actors.extend(others_to_show)

        ms = shape.max() + 20

        if save_video:
            vd = Video(name="3D_detection_visualization.gif")
            plt = show(
                actors[0:24],
                title=title,
                viewup="z",
                axes=dict(
                    xrange=(-50, ms),
                    yrange=(-50, ms),
                    zrange=(-20, ms),
                    c="white",
                    numberOfDivisions=10,
                    yzGrid=False,
                ),
                zoom=False,
                bg="black",
                interactive=False,
            )
            plt.show(actors[0:24])
            vd.add_frame()
            vd.pause(1)
            for i in range(24, len(actors) + 1):
                plt.show(*actors[:i])
                vd.add_frame()
            vd.pause(1)
            for i in range(100):
                for actor in actors:
                    actor.rotate(angle=3.6, axis=(0, 0, 1), point=shape // 2)
                plt.show(*actors)
                vd.add_frame()
            vd.close()

        else:
            show(
                *actors,
                title=title,
                viewup="z",
                axes=dict(c="white", numberOfDivisions=10, yzGrid=False,),
                bg="black"
            ).close()
