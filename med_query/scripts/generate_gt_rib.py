# Copyright (c) DAMO Health
import argparse
import glob
import json
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import time
import tqdm
from multiprocessing.pool import Pool
from pathlib import Path
from skimage.morphology import binary_closing, binary_opening, skeletonize_3d

from med_query.utils.common import mask_np_bbox
from med_query.utils.frame import voxel_to_world
from med_query.utils.itk import mask_bbox
from med_query.utils.resample import resample_base, resample_itkimage_torai


class GtGenerator(object):
    def __init__(self, **kwargs):
        super(GtGenerator, self).__init__()
        self.filelist = list(pd.read_csv(kwargs.get("filelist_csv")).filename)
        self.mask_dir = kwargs.get("mask_dir")
        self.output_dir = kwargs.get("output_dir")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.debug = kwargs.get("debug_flag", False)
        self.num_process = kwargs.get("num_process")
        self.use_skeleton = kwargs.get("use_skeleton", False)
        if self.debug:
            self.num_process = 1
            self.debug_dir = os.path.join(self.output_dir, "debug")
            if not os.path.isdir(self.debug_dir):
                os.makedirs(self.debug_dir)
        self.spacing = [kwargs.get("spacing")] * 3
        self.min_voxel_thresh = kwargs.get("min_voxel_thresh", 100)

        hyper_params_json = Path(__file__).parent / "hyper_params_rib.json"
        with open(hyper_params_json, "r") as f:
            hyper_params = json.load(f)
        if not hyper_params.get("max_bbox_size", None):
            hyper_params["max_bbox_size"] = self.get_max_bbox_size()
            with open(hyper_params_json, "w") as f:
                json.dump(hyper_params, f, indent=2)

    def get_max_bbox_size(self):
        sizes = []
        for case in tqdm.tqdm(self.filelist):
            mask_path = glob.glob(os.path.join(self.mask_dir, case) + "*")[0]
            mask = sitk.ReadImage(mask_path)
            mask_rai = resample_itkimage_torai(mask, self.spacing, pad_value=0)
            min_corner, max_corner = mask_bbox(mask_rai)
            bbox_size = max_corner - min_corner
            sizes.append(bbox_size)
        sizes = np.vstack(sizes)
        max_size = sizes.max(axis=0)
        print(max_size)
        return max_size.tolist()

    def run(self):
        self.start_time = time.time()
        rows = []
        if self.num_process == 1:
            for case_idx, case in enumerate(self.filelist):
                rows.extend(self.process(case_idx, case))
        else:
            p = Pool(self.num_process)
            res = []
            for case_idx, case in enumerate(self.filelist):
                res.append(p.apply_async(self.process, [case_idx, case]))
            p.close()
            p.join()
            for r in res:
                rows.extend(r.get())

        cases_info_df = pd.DataFrame(
            rows,
            columns=[
                "filename",
                "mask_center_x",
                "mask_center_y",
                "mask_center_z",
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
            ],
        )

        return cases_info_df

    def process(self, case_idx, case):
        print(f"processing {case_idx}: {case}")
        mask_path = glob.glob(os.path.join(self.mask_dir, case) + "*")[0]
        mask = sitk.ReadImage(mask_path)
        mask_rai = resample_itkimage_torai(mask, self.spacing, pad_value=0)
        case_info = self.get_case_info(case, mask_rai)
        case_rows = []
        for box in case_info["bboxes"]:
            row = [
                case_info["filename"],
                *case_info["mask_center_w"],
                box["label"],
                *box["center_w"],
                *box["x_axis_local"],
                *box["y_axis_local"],
                *box["z_axis_local"],
                box["width"],
                box["height"],
                box["depth"],
            ]
            case_rows.append(row)
        return case_rows

    def get_case_info(self, case, mask_rai):
        min_corner, max_corner = mask_bbox(mask_rai)
        mask_center_w = (min_corner + max_corner) / 2

        mask_np = sitk.GetArrayFromImage(mask_rai).astype(np.uint8)
        labels = np.unique(mask_np).tolist()
        labels.remove(0)
        labels.sort()
        case_info = {
            "filename": case,
            "mask_center_w": mask_center_w.round(1).tolist(),
            "bboxes": [],
        }

        for label_idx, label in enumerate(labels):
            sp, ep = mask_np_bbox(mask_np, mask_value=label)
            mask_np_crop = mask_np[sp[0] : ep[0] + 1, sp[1] : ep[1] + 1, sp[2] : ep[2] + 1].copy()
            mask_np_crop[mask_np_crop != label] = 0
            mask_np_crop[mask_np_crop > 0] = 1
            if mask_np_crop.sum() < self.min_voxel_thresh:
                continue
            if self.use_skeleton:
                # 使用中心线来算PCA
                mask_np_crop = binary_opening(mask_np_crop)
                mask_np_crop = binary_closing(mask_np_crop).astype(np.uint8)
                skeleton_np_crop = skeletonize_3d(mask_np_crop)
                # z,y,x to x,y,z
                coords_skeleton = np.flipud(np.array(np.where(skeleton_np_crop == 1)))
                if len(coords_skeleton) < 3 or len(coords_skeleton[0]) <= 3:
                    coords_skeleton = np.flipud(np.array(np.where(mask_np_crop == 1)))
            else:
                coords_skeleton = np.flipud(np.array(np.where(mask_np_crop == 1)))

            cov_array = np.cov(coords_skeleton)
            w, v = np.linalg.eig(cov_array)
            ind = np.argsort(w)
            min_axis = v[:, ind[0]]
            max_axis = v[:, ind[-1]]
            x_axis_local = max_axis
            z_axis_local = min_axis
            # keep x_axis_local in the direction of right to left
            if label < 13 and x_axis_local[2] < 0:
                x_axis_local *= -1
            if label > 12 and x_axis_local[2] > 0:
                x_axis_local *= -1
            # keep z_axis_local in the direction of bottom to up
            if z_axis_local[2] < 0:
                z_axis_local *= -1
            y_axis_local = np.cross(z_axis_local, x_axis_local)
            axes = np.vstack([x_axis_local, y_axis_local, z_axis_local])
            # `direction` is the rotation expression in SimpleITK
            direction = axes.transpose()
            rot_mat = axes

            coords = np.flipud(np.array(np.where(mask_np == label)))
            coords_rot = rot_mat @ coords

            center_v = (coords_rot.max(1) + coords_rot.min(1)) / 2
            center_w = voxel_to_world(mask_rai, np.linalg.inv(rot_mat) @ center_v).round(1)
            width, height, depth = (
                ((coords_rot.max(1) + 1 - coords_rot.min(1)) * self.spacing).round(1).tolist()
            )

            bbox_info = {
                "label": label,
                "center_w": center_w.tolist(),
                "x_axis_local": x_axis_local.tolist(),
                "y_axis_local": y_axis_local.tolist(),
                "z_axis_local": z_axis_local.tolist(),
                "width": width,
                "height": height,
                "depth": depth,
            }
            case_info["bboxes"].append(bbox_info)

            if self.debug:
                origin = voxel_to_world(mask_rai, np.linalg.inv(rot_mat) @ coords_rot.min(1))
                cropped_mask = resample_base(
                    mask_rai,
                    origin,
                    direction.flatten().tolist(),
                    self.spacing,
                    (coords_rot.max(1) + 1 - coords_rot.min(1)).astype(np.int32),
                    pad_value=0,
                )
                sitk.WriteImage(cropped_mask, os.path.join(self.debug_dir, f"{label}.mhd"))

        return case_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_csv", type=str, default="/path/to/filelist.csv")
    parser.add_argument(
        "-m",
        "--mask_dir",
        type=str,
        help="dir for multi-label rib mask",
        default="/path_to/labelled_mask/",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="output dir", default="/path/to/output_ground_truth",
    )
    parser.add_argument("-s", "--spacing", type=float, default=2, help="isotropic spacing")
    parser.add_argument(
        "-t",
        "--min_voxel_thresh",
        type=int,
        help="minimum voxel threshold @ given spacing",
        default=100,
    )
    parser.add_argument("-n", "--num_process", type=int, default=1, help="number of processes")
    parser.add_argument("-d", "--debug_flag", action="store_true", default=False)
    parser.add_argument(
        "--use_skeleton",
        action="store_true",
        default=False,
        help="whether to use rib skeleton to get the bbox_info",
    )

    args = parser.parse_args()
    worker = GtGenerator(**args.__dict__)
    cases_info_df = worker.run()
    cases_info_df.to_csv(os.path.join(args.output_dir, "cases_info.csv"), index=False)


if __name__ == "__main__":
    main()
