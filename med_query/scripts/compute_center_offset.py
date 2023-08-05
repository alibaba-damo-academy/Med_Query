# Copyright (c) DAMO Health
import argparse
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from med_query.utils.frame import voxel_to_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainset_csv", type=str, help="path for trainset filelist")
    parser.add_argument("-m", "--mask_dir", type=str, help="path for mask dir")
    parser.add_argument("-i", "--info_csv", type=str, help="path for cases_info.csv")
    parser.add_argument("-o", "--output_csv", type=str, help="path to save output")
    args = parser.parse_args()

    filelist_df = pd.read_csv(args.trainset_csv)
    cases_info_df = pd.read_csv(args.info_csv)
    cases_info_df = cases_info_df[~cases_info_df.filename.str.contains("crop")]

    mask_center_dict = {}
    for idx, row in cases_info_df.iterrows():
        if row.filename not in mask_center_dict.keys():
            mask_center_dict.update({row.filename: row.iloc[1:4].__array__(np.float32)})

    diffs = []
    for case_name in tqdm(filelist_df.filename):
        if "FLARE" in case_name:
            mask_path = os.path.join(args.mask_dir, f"{case_name}.nii.gz")
        elif "RibFrac" in case_name:
            mask_path = os.path.join(args.mask_dir, f"{case_name}-ribmask_labelled.nii.gz")
        else:  # CTSpine1K
            mask_path = os.path.join(args.mask_dir, f"{case_name}_seg.nii.gz")

        mask = sitk.ReadImage(mask_path)
        mask_np = sitk.GetArrayFromImage(mask)
        vz, vy, vx = np.where(mask_np > 0)
        v = [vx.mean(), vy.mean(), vz.mean()]
        w = voxel_to_world(mask, v)
        diff = w - mask_center_dict.get(case_name)
        diffs.append([case_name, *diff])

    df = pd.DataFrame(diffs, columns=["filename", "diff_x", "diff_y", "diff_z"])
    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
