# Copyright (c) DAMO Health

import glob
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import torch
from easydict import EasyDict
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from med_query.utils.common import axis_angle_to_rotation_matirx
from med_query.utils.frame import Frame3d, world_box, world_to_voxel
from med_query.utils.itk import normalize_sitk_im
from med_query.utils.resample import resample_base
from ..builder import DATASETS


@DATASETS.register_module()
class RibDetSegDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        super(RibDetSegDataset, self).__init__()
        
        cfg = EasyDict(kwargs)
        dataset_name = cfg.dataset_name
        phase = cfg.phase
        assert dataset_name in ["trainset", "validset", "testset"]
        assert phase in ["train", "valid"]
        if phase == "train":
            assert dataset_name == "trainset"
        self.dataset_name = dataset_name
        self.phase = phase

        filelist_csv = cfg[f"{self.dataset_name}_csv"]
        filelist_df = pd.read_csv(filelist_csv)
        self.image_list = list(filelist_df.filename)
        self.bbox_df = pd.read_csv(cfg.gt_path)

        self.image_dir = cfg.image_dir
        self.mask_dir = cfg.mask_dir
        self.debug_dir = cfg.debug_dir

        self.interpolation = cfg.interpolation
        self.default_value = cfg.default_value

        self.normalization_params = cfg.normalization_params
        self.aug_params = cfg.aug_params

        if self.aug_params.use_crop:
            temp_list = self.image_list.copy()
            for file in temp_list:
                self.image_list.extend([f"{file}-crop-{i+1}" for i in range(3)])
        total_list = self.bbox_df.filename.unique()
        self.image_list = [i for i in self.image_list if i in total_list]

        self.bbox_info = self.get_bbox_info()

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

        self.spacing = np.array(cfg.spacing, dtype=np.float32)

        # in the unit of voxel
        self.max_stride = cfg.max_stride

        # in the unit of mm
        if cfg.use_fixed_bbox_size:
            self.fixed_bbox_size = np.array(cfg.fixed_bbox_size)
            self.crop_size = self.fixed_bbox_size / self.spacing
        else:
            self.max_bbox_size = np.array(cfg.max_bbox_size)
            t, s, rad = 0, 1, 0
            if self.aug_params.on:
                t = self.aug_params.translation[1]
                s = self.aug_params.scale[0]
                rad = np.deg2rad(self.aug_params.rotation_angle[1])
            self.crop_size = (self.max_bbox_size + t) / (self.spacing * s) / np.cos(rad)
            self.crop_size = (self.crop_size // self.max_stride + 1) * self.max_stride
            self.crop_size = self.crop_size.astype(np.int32)

    def __len__(self) -> int:
        return len(self.image_list)

    def __repr__(self):
        return f"{self.dataset_name} for rib_det_seg_experimentation"

    def get_bbox_info(self):
        filenames = set(self.image_list)
        assert set(
            [
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
            ]
        ).issubset(self.bbox_df.columns)

        def fun(filename):
            tmp_df = self.bbox_df[self.bbox_df.filename == filename].reset_index()
            case_info = dict()
            case_info["mask_center_w"] = np.array(
                tmp_df.loc[0, ["mask_center_x", "mask_center_y", "mask_center_z"]]
            ).astype(np.float32)
            boxes = []
            for idx in range(len(tmp_df)):
                label = tmp_df.loc[idx, "label"]
                center = np.array(tmp_df.loc[idx, ["world_x", "world_y", "world_z"]]).astype(
                    np.float32
                )
                direction = (
                    np.array(
                        tmp_df.loc[
                            idx, ["dx1", "dx2", "dx3", "dy1", "dy2", "dy3", "dz1", "dz2", "dz3"]
                        ]
                    )
                    .astype(np.float32)
                    .reshape(3, 3)
                    .transpose()
                )
                whd = np.array(tmp_df.loc[idx, ["width", "height", "depth"]]).astype(np.float32)
                boxes.append([label, center, direction, whd])
            case_info["boxes"] = boxes
            return case_info

        res = Parallel(n_jobs=8)(delayed(fun)(filename) for filename in filenames)
        bbox_info = dict()
        for i, k in enumerate(filenames):
            bbox_info[k] = res[i]

        return bbox_info

    def gen_gt_tensors(self, filename, image, mask):
        case_info = self.bbox_info[filename]

        crop_spacing = self.spacing.copy()
        # each column of direction represents an axis
        direction = np.eye(3, dtype=np.float32)
        new_direction = direction.copy()
        rot_mat_global = np.eye(3, dtype=np.float32)
        world_center = case_info["mask_center_w"]
        min_corner, max_corner = world_box(image)

        if (
            self.aug_params.on
            and self.phase == "train"
            and np.random.uniform() < self.aug_params.prob
        ):
            scale_min, scale_max = self.aug_params.scale
            crop_spacing *= scale_min + (scale_max - scale_min) * np.random.uniform()
            t_min, t_max = self.aug_params.translation
            translation_mm = t_min + (t_max - t_min) * np.random.uniform(0, 1, 3)
            world_center += translation_mm
            rotation_min, rotation_max = self.aug_params.rotation_angle
            angle = rotation_min + (rotation_max - rotation_min) * np.random.uniform()

            rotation_axis = self.aug_params.rotation_axis
            if np.linalg.norm(rotation_axis) < 1e-3:
                rotation_axis = np.random.uniform(-1, 1, 3)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            rot_mat_global = axis_angle_to_rotation_matirx(rotation_axis, angle)
            new_direction = np.dot(rot_mat_global, direction)

        frame = Frame3d()
        frame.origin = world_center
        frame.spacing = crop_spacing
        frame.direction = new_direction

        crop_origin = frame.voxel_to_world(0 - self.crop_size // 2)
        frame.origin = crop_origin

        img_crop = resample_base(
            image,
            frame.origin,
            frame.direction,
            frame.spacing,
            self.crop_size,
            interpolator=self.interpolation,
            pad_value=self.default_value,
        )
        img_crop = sitk.Cast(img_crop, sitk.sitkFloat32)
        img_crop = normalize_sitk_im(img_crop, self.min_value, self.max_value, self.clip, True)
        # keep img_tensor in (x, y, z) order
        img_tensor = (
            torch.from_numpy(sitk.GetArrayFromImage(img_crop)).permute(2, 1, 0).unsqueeze(0)
        )

        msk_crop = resample_base(
            mask,
            frame.origin,
            frame.direction,
            frame.spacing,
            self.crop_size,
            interpolator="nearest",
            pad_value=0,
        )
        msk_crop = sitk.Cast(msk_crop, sitk.sitkInt8)
        # keep msk_tensor in (x, y, z) order
        msk_tensor = (
            torch.from_numpy(sitk.GetArrayFromImage(msk_crop)).permute(2, 1, 0).unsqueeze(0)
        )

        erase_ribs = []
        if (
            self.aug_params.on
            and self.phase == "train"
            and np.random.uniform() < self.aug_params.erase_prob
        ):
            erase_ribs = [12, 24]
            for rib in erase_ribs:
                mask_value = np.random.randint(10, 50)
                img_tensor[msk_tensor == rib] = (mask_value - self.min_value) / (
                    self.max_value - self.min_value
                )
            if self.is_debugging:
                from med_base.data.frame import set_frame_as_ref

                img = sitk.GetImageFromArray(
                    img_tensor.squeeze(0).permute(2, 1, 0).numpy()
                    * (self.max_value - self.min_value)
                    + self.min_value
                )
                set_frame_as_ref(img, img_crop)
                sitk.WriteImage(
                    img, os.path.join(self.debug_dir, f"{filename}_masked.nii.gz"), True
                )

        data = EasyDict()
        data.img = img_tensor
        data.msk = msk_tensor
        data.meta_data = EasyDict()
        data.meta_data.pid = filename
        data.meta_data.img_itk = img_crop
        data.meta_data.img_shape = img_crop.GetSize()
        data.meta_data.min_corner = min_corner
        data.meta_data.max_corner = max_corner
        data.meta_data.rot_mat_global = rot_mat_global
        data.meta_data.scale_factor = 1

        bboxes = []
        labels = []
        rotats = []
        angles = []
        for box in case_info["boxes"]:
            # center_w is in [x, y, z] order
            label, center_w, direction, whd = box
            if label in erase_ribs:
                continue
            # center_v is in [x, y, z] order
            center_v = world_to_voxel(img_crop, center_w)

            rot_mat_local = rot_mat_global @ direction

            rot_mat_obj = R.from_matrix(rot_mat_local)
            euler_angles = rot_mat_obj.as_euler("zyx", degrees=False)

            # the inversed rot_mat_local is only for FCOS
            rot_mat_local = np.linalg.inv(rot_mat_local)
            rotats.append(rot_mat_local)
            labels.append(label)
            angles.append(euler_angles)
            # consider spacing
            whd_v = whd / crop_spacing
            # bbox in [cx, cy, cz, w, h, d]
            bbox = [center_v[0], center_v[1], center_v[2], whd_v[0], whd_v[1], whd_v[2]]
            bboxes.append(bbox)

        data.bboxes = torch.Tensor(bboxes)
        data.labels = torch.Tensor(labels).long()
        data.rotats = torch.Tensor(rotats)
        data.angles = torch.Tensor(angles)

        return data

    def __getitem__(self, index):
        case_name = self.image_list[index]
        image_path = os.path.join(self.image_dir, f"{case_name}-image.nii.gz")
        mask_path = os.path.join(self.mask_dir, f"{case_name}-ribmask_labelled.nii.gz")

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_path, sitk.sitkInt16)

        data = self.gen_gt_tensors(case_name, image, mask)

        return data

    def collate_fn(self, batch):
        return {
            "img": torch.stack([item["img"] for item in batch]),
            "msk": torch.stack([item["msk"] for item in batch]),
            "bboxes": [item["bboxes"] for item in batch],
            "labels": [item["labels"] for item in batch],
            "rotats": [item["rotats"] for item in batch],
            "angles": [item["angles"] for item in batch],
            "meta_data": [item["meta_data"] for item in batch],
        }


class RibDetSegDatasetTest(Dataset):
    def __init__(self, cfg, image_dir: str = None) -> None:
        super(RibDetSegDatasetTest, self).__init__()

        assert os.path.exists(image_dir), f"No such directory '{image_dir}'"
        self.image_dir = image_dir

        self.image_list = glob.glob(os.path.join(self.image_dir, "*.nii.gz"))

        self.normalization_params = cfg.normalization_params

        self.min_value = None
        self.max_value = None
        self.clip = True
        if self.normalization_params is not None:
            self.min_value = self.normalization_params.min_value
            self.max_value = self.normalization_params.max_value
            self.clip = self.normalization_params.clip

    def __len__(self) -> int:
        return len(self.image_list)

    def __repr__(self):
        return "testset for rib_det_seg_experimentation"

    def __getitem__(self, index):
        image_path = self.image_list[index]
        pid = os.path.basename(image_path)
        if image_path.endswith("-image.nii.gz"):
            pid = os.path.basename(image_path).replace("-image.nii.gz", "")
        elif image_path.endswith(".nii.gz"):
            pid = os.path.basename(image_path).replace(".nii.gz", "")

        image = sitk.ReadImage(image_path, sitk.sitkInt16)
        image = sitk.Cast(image, sitk.sitkFloat32)
        image = normalize_sitk_im(image, self.min_value, self.max_value, self.clip, True)

        meta_data = {"pid": pid, "img_res0": image}

        data = {"meta_data": meta_data}

        return data

    def collate_fn(self, batch):
        return {"meta_data": [item["meta_data"] for item in batch]}


if __name__ == "__main__":
    import time
    from torch.utils.data import DataLoader

    from med_query.utils.data_utils import DataPrefetcher
    from med_query.utils.io_utils import load_module_from_file

    cfg_module = load_module_from_file(
        os.path.join(os.path.dirname(__file__), "../configs/rib_config.py")
    )
    cfg = cfg_module.Configs()

    ds = RibDetSegDataset(cfg.dataset, dataset_name="trainset", phase="train")
    print("dataset len: ", len(ds))
    loader = DataLoader(
        ds, batch_size=8, num_workers=2, shuffle=False, pin_memory=False, collate_fn=ds.collate_fn,
    )

    for e in range(1):
        print(f"epoch: {e+1}")
        prefetcher = DataPrefetcher(cfg.dataset, loader)
        cnt = 0
        start = time.time()
        t0 = time.time()
        for i, data in enumerate(prefetcher):
            # data = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
            # data["meta_data"] = batch["meta_data"]
            time.sleep(1)
            cnt += 1
            t1 = time.time()
            print(data["meta_data"][0]["pid"], data["img"].shape, f"time {t1-t0:.2f} sec.")
            # print("bboxes: ", data["bboxes"])
            # print("labels: ", data["labels"])
            t0 = t1
            if cnt == 10:
                break
        end = time.time()
        print(f"total time {end-start:.2f} sec.")
