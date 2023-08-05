# Copyright (c) DAMO Health
import argparse
import itertools
import numpy as np
import os.path as osp
import pandas as pd
import random
import SimpleITK as sitk
from joblib import Parallel, delayed
from skimage import morphology
from tqdm import tqdm

from med_query.scripts.generate_gt_flare import GtGenerator
from med_query.utils.frame import set_frame_as_ref, world_to_voxel
from med_query.utils.io_utils import make_dir
from med_query.utils.itk import crop, mask_bbox
from med_query.utils.resample import resample_itkimage_torai


class Preprocessor(object):
    """
    steps:
    1. generate_gt
    2. crop & generate_gt
    3. resample
    """

    def __init__(self, **kwargs):
        super(Preprocessor, self).__init__()
        self.filelist = list(pd.read_csv(kwargs.get("filelist_csv")).filename)
        self.image_dir = kwargs.get("image_dir")
        self.mask_dir = kwargs.get("mask_dir")
        self.output_dir = kwargs.get("output_dir")
        make_dir(self.output_dir)
        self.debug = kwargs.get("debug_flag", False)
        self.crop = kwargs.get("crop", False)
        self.num_process = kwargs.get("num_process")
        self.use_skeleton = kwargs.get("use_skeleton", False)
        if self.debug:
            self.num_process = 1
            self.debug_dir = osp.join(self.output_dir, "debug")
            make_dir(self.debug_dir)
        self.spacing = kwargs.get("spacing", 2)

        self.gt_generator = GtGenerator(**kwargs)

        if self.crop:
            random.seed(42)
            self.margin = 10  # crop_size in z-axis is larger than 2 * self.margin
            self.num_crop_for_each_sample = 1
            self.cropped_image_dir = osp.join(self.output_dir, "images_cropped")
            self.cropped_mask_dir = osp.join(self.output_dir, "masks_cropped")
            make_dir(self.cropped_image_dir)
            make_dir(self.cropped_mask_dir)

    def resample(self, filelist, image_dir, mask_dir, spacing):
        save_image_dir = osp.join(self.output_dir, f"images_{spacing}mm")
        save_mask_dir = osp.join(self.output_dir, f"masks_{spacing}mm")
        make_dir(save_image_dir)
        make_dir(save_mask_dir)

        def fun(file):
            img_path = osp.join(image_dir, f"{file}_0000.nii.gz")
            msk_path = osp.join(mask_dir, f"{file}.nii.gz")
            img_itk = sitk.ReadImage(img_path)
            msk_itk = sitk.ReadImage(msk_path)
            img_itk_resampled = resample_itkimage_torai(
                img_itk, [spacing] * 3, interpolator="linear", pad_value=-1024
            )
            msk_itk_resampled = resample_itkimage_torai(
                msk_itk, [spacing] * 3, interpolator="nearest", pad_value=0
            )
            sitk.WriteImage(
                img_itk_resampled, osp.join(save_image_dir, osp.basename(img_path)), True
            )
            sitk.WriteImage(
                msk_itk_resampled, osp.join(save_mask_dir, osp.basename(msk_path)), True
            )

        Parallel(n_jobs=self.num_process)(
            delayed(fun)(file)
            for file in tqdm(
                filelist, desc=f"resample {osp.basename(image_dir)} @ spacing {spacing}"
            )
        )

    def dilate_mask(self, filelist, mask_dir, labels):
        save_mask_dir = osp.join(self.output_dir, "masks_2mm_dilated")
        make_dir(save_mask_dir)

        def fun(file):
            msk_path = osp.join(mask_dir, f"{file}.nii.gz")
            msk_itk = sitk.ReadImage(msk_path)
            msk_np = sitk.GetArrayFromImage(msk_itk)
            for rib in labels:
                rib_np = np.zeros_like(msk_np)
                rib_np[msk_np == rib] = 1
                rib_np_dilated = morphology.binary_dilation(rib_np, morphology.ball(radius=2))
                msk_np[rib_np_dilated == 1] = rib

            msk_itk_dilated = sitk.GetImageFromArray(msk_np)
            set_frame_as_ref(msk_itk_dilated, msk_itk)

            sitk.WriteImage(msk_itk_dilated, osp.join(save_mask_dir, osp.basename(msk_path)), True)

        Parallel(n_jobs=self.num_process)(
            delayed(fun)(file)
            for file in tqdm(filelist, desc=f"dilate {osp.basename(mask_dir)} @ organs {labels}")
        )

    def crop_image_and_mask(self, cropped_image_dir, cropped_mask_dir):
        def fun(file):
            img_path = osp.join(self.image_dir, f"{file}_0000.nii.gz")
            msk_path = osp.join(self.mask_dir, f"{file}.nii.gz")
            img_itk = sitk.ReadImage(img_path)
            msk_itk = sitk.ReadImage(msk_path)
            img_itk = resample_itkimage_torai(
                img_itk, [1] * 3, interpolator="linear", pad_value=-1024
            )
            msk_itk = resample_itkimage_torai(msk_itk, [1] * 3, interpolator="nearest", pad_value=0)

            min_box, max_box = mask_bbox(msk_itk)
            min_box_v = world_to_voxel(msk_itk, min_box)
            max_box_v = world_to_voxel(msk_itk, max_box)
            size = img_itk.GetSize()
            cropped_list = []

            zs = max(min_box_v[2] - 10, 0)
            ze = min(max_box_v[2] + 10, size[2])
            assert ze - zs >= 2 * self.margin, f"FOV of {file} is too small, z range: {ze-zs}"
            for i in range(self.num_crop_for_each_sample):
                center = random.randint(zs + self.margin, ze - self.margin)
                z0 = random.randint(zs, center - self.margin)
                z1 = random.randint(center + self.margin, ze)

                img_cropped = crop(img_itk, [0, 0, z0], [size[0], size[1], z1])
                msk_cropped = crop(msk_itk, [0, 0, z0], [size[0], size[1], z1])
                save_img_path = osp.join(cropped_image_dir, f"{file}-crop-{i+1}_0000.nii.gz")
                save_msk_path = osp.join(cropped_mask_dir, f"{file}-crop-{i+1}.nii.gz")
                sitk.WriteImage(img_cropped, save_img_path)
                sitk.WriteImage(msk_cropped, save_msk_path)
                cropped_list.append(f"{file}-crop-{i+1}")

            return cropped_list

        res = Parallel(n_jobs=self.num_process)(
            delayed(fun)(file) for file in tqdm(self.filelist, desc="crop process")
        )
        res = list(itertools.chain.from_iterable(res))
        cropped_df = pd.DataFrame({"filename": res})
        cropped_df_path = osp.join(self.output_dir, "filelist_cropped.csv")
        cropped_df.to_csv(cropped_df_path, index=False)

        return cropped_df_path

    def run(self):
        print("STAGE I: Generate GT Info for Raw Images...")
        info_df = self.gt_generator.run()
        print("STAGE I: Generate GT Info for Raw Images... DONE!")
        cropped_info_df = None
        if self.crop:
            print("STAGE II: Crop Raw Images for Augmentation...")
            cropped_df_path = self.crop_image_and_mask(
                self.cropped_image_dir, self.cropped_mask_dir
            )
            worker_for_crop = GtGenerator(
                filelist_csv=cropped_df_path,
                mask_dir=self.cropped_mask_dir,
                output_dir=self.output_dir,
                spacing=self.spacing,
                num_process=self.num_process,
                use_skeleton=self.use_skeleton,
            )
            cropped_info_df = worker_for_crop.run()
            print("STAGE II: Crop Raw Images for Augmentation...DONE!")
        else:
            print("Crop operation is not required, skip to STAGE III!")

        total_info_df = info_df
        if cropped_info_df is not None:
            total_info_df = pd.concat([info_df, cropped_info_df])
        total_info_df.to_csv(osp.join(self.output_dir, "cases_info.csv"), index=False)

        print("STAGE III: Resample Images...")
        self.resample(self.filelist, self.image_dir, self.mask_dir, spacing=2)
        if cropped_info_df is not None:
            self.resample(
                cropped_info_df.filename.unique(),
                self.cropped_image_dir,
                self.cropped_mask_dir,
                spacing=2,
            )
        print("STAGE III: Resample Images...DONE!")

        # print("STAGE IV: Dilate Images...")
        # mask_dir = osp.join(self.output_dir, "masks_2mm")
        # if osp.exists(osp.join(self.output_dir, "cases_info.csv")):
        #     total_info_df = pd.read_csv(osp.join(self.output_dir, "cases_info.csv"))
        # self.dilate_mask(total_info_df.filename.unique(), mask_dir, labels=[12, 24])
        # print("STAGE IV: Dilate Images...DONE!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filelist_csv", type=str, default="/path_to_exp/filelist.csv",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        help="dir for rib images with original resolution",
        default="/path_to_exp/images",
    )
    parser.add_argument(
        "-m",
        "--mask_dir",
        type=str,
        help="dir for multi-label rib mask",
        default="/path_to_exp/masks_labelled/",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="output dir", default="/path_to_exp/",
    )
    parser.add_argument("-s", "--spacing", type=float, default=2, help="isotropic spacing")
    parser.add_argument("-n", "--num_process", type=int, default=1, help="number of processes")
    parser.add_argument("-d", "--debug_flag", action="store_true", default=False)
    parser.add_argument(
        "--crop",
        action="store_true",
        default=False,
        help="wether to save cropped data as augmentation",
    )
    parser.add_argument(
        "--use_skeleton",
        action="store_true",
        default=False,
        help="whether to use rib skeleton to get the bbox_info",
    )
    parser.add_argument(
        "--use_pca",
        action="store_true",
        default=False,
        help="whether to use rib skeleton to get the bbox_info",
    )

    args = parser.parse_args()
    worker = Preprocessor(**args.__dict__)
    worker.run()


if __name__ == "__main__":
    main()
