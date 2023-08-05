# Copyright (c) DAMO Health

import argparse
import os.path as osp
import pandas as pd

from med_query.vis.visualizer import Visualizer3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--case_id", type=str, default="RibFrac1")
    parser.add_argument(
        "-i", "--image_dir", type=str, default="/path/to/images/",
    )
    parser.add_argument(
        "-m", "--mask_dir", type=str, default="/path/to/masks/",
    )
    parser.add_argument(
        "--info_csv", type=str, default="/path/to/cases_info.csv",
    )
    parser.add_argument(
        "-l", "--label_visible", type=str, default=None, help="specific labels in mask to display",
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        default=False,
        help="whether to show the original image in the overlay",
    )
    parser.add_argument(
        "--save_video", action="store_true", default=False, help="whether to save a video",
    )

    args = parser.parse_args()
    label_visible = None
    if args.label_visible is not None:
        label_visible = [int(i) for i in args.label_visible.split(",")]
    df = pd.read_csv(args.info_csv, dtype={"filename": str})
    df_case = df[df["filename"] == args.case_id].reset_index()
    if len(df_case) == 0:
        raise KeyError(f"{args.case_id} does not exist in {args.info_csv}!")
    print(f"Visualize {args.case_id} ...")

    vis = Visualizer3D()
    image_path = None
    if args.show_image:
        if "FLARE" in args.case_id:
            image_path = osp.join(args.image_dir, f"{args.case_id}_0000.nii.gz")
        elif "RibFrac" in args.case_id:
            image_path = osp.join(args.image_dir, f"{args.case_id}-image.nii.gz")
        else:
            image_path = osp.join(args.image_dir, f"{args.case_id}.nii.gz")
    # extract mask_path
    if "FLARE" in args.case_id:
        mask_path = osp.join(args.mask_dir, f"{args.case_id}.nii.gz")
    elif "RibFrac" in args.case_id:
        mask_path = osp.join(args.mask_dir, f"{args.case_id}-ribmask_labelled.nii.gz")
    else:  # CTSpine1K
        mask_path = osp.join(args.mask_dir, f"{args.case_id}_seg.nii.gz")

    vis.show(
        title=args.case_id,
        image_path=image_path,
        mask_path=mask_path,
        gt_df=df_case,
        label_visible=label_visible,
        others_to_show=[],
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
