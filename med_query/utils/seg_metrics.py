import argparse
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from surface_distance.metrics import (
    compute_average_surface_distance,
    compute_robust_hausdorff,
    compute_surface_distances,
)
from tqdm import tqdm

from med_query.utils.common import longest_common_postfix_str
from med_query.utils.itk import cal_dsc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_csv", type=str, help="filelist csv file")
    parser.add_argument("-r", "--result_dir", type=str, help="result dir")
    parser.add_argument("-l", "--label_dir", type=str, help="label dir")
    parser.add_argument(
        "-n", "--num_process", type=int, default=1, help="num process for paralleled processing"
    )
    parser.add_argument(
        "-s",
        "--simple_mode",
        action="store_true",
        default=False,
        help="simple mode only calculate dice",
    )
    parser.add_argument(
        "-p", "--hausdorff_percent", default=95, type=float, help="hausdorff percent"
    )
    args = parser.parse_args()
    filelist_df = pd.read_csv(args.filelist_csv)
    filelist = filelist_df.filename
    lcpo_r = longest_common_postfix_str([f for f in os.listdir(args.result_dir) if ".nii" in f])
    lcpo_l = longest_common_postfix_str([f for f in os.listdir(args.label_dir) if ".nii" in f])

    def process(case):
        result_mask = sitk.ReadImage(os.path.join(args.result_dir, f"{case}{lcpo_r}"))
        label_mask = sitk.ReadImage(os.path.join(args.label_dir, f"{case}{lcpo_l}"))

        spacing_mm_np = np.array(label_mask.GetSpacing()[::-1])
        result_mask_np = sitk.GetArrayFromImage(result_mask)
        label_mask_np = sitk.GetArrayFromImage(label_mask)
        case_dice = cal_dsc(result_mask_np, label_mask_np, 25)
        results = [case, *case_dice, np.nanmean(case_dice).item(), np.nan]

        if not args.simple_mode:
            case_hd, case_avg_sd = [], []
            for label in range(1, 25):
                gt_idx = label_mask_np == label
                pred_idx = result_mask_np == label
                sd_idx = compute_surface_distances(gt_idx, pred_idx, spacing_mm_np)
                hd_idx = compute_robust_hausdorff(sd_idx, args.hausdorff_percent)
                hd_idx = np.nan if hd_idx == np.inf else hd_idx
                avg_sd_idx = compute_average_surface_distance(sd_idx)
                avg_sd_idx = [np.nan if f == np.inf else f for f in avg_sd_idx]

                case_hd.append(hd_idx)
                case_avg_sd.append(sum(avg_sd_idx) / 2)
            results.extend([*case_hd, np.nanmean(case_hd), np.nan])
            results.extend([*case_avg_sd, np.nanmean(case_avg_sd)])

        return results

    res = Parallel(n_jobs=args.num_process)(
        delayed(process)(case) for case in tqdm(filelist, desc="evaluation process")
    )

    columns = ["filename"]
    columns.extend(f"R{i}_dice" for i in range(1, 13))
    columns.extend(f"L{i}_dice" for i in range(1, 13))
    columns.extend(["case_level_avg_dice", "sep_1"])
    if not args.simple_mode:
        columns.extend(f"R{i}_hausdorff" for i in range(1, 13))
        columns.extend(f"L{i}_hausdorff" for i in range(1, 13))
        columns.extend(["case_level_avg_hd", "sep_2"])
        columns.extend(f"R{i}_asd" for i in range(1, 13))
        columns.extend(f"L{i}_asd" for i in range(1, 13))
        columns.append("case_level_avg_asd")
    metrics_df = pd.DataFrame(res, columns=columns)
    # convert np.inf to np.nan
    metrics_df = metrics_df.replace(np.inf, np.nan, inplace=False)
    rib_level_avg = metrics_df.mean(axis=0, skipna=True, numeric_only=True)
    rib_level_avg["filename"] = "rib_level_avg"
    metrics_df = pd.concat([metrics_df, rib_level_avg.to_frame().T])
    output_csv = os.path.join(
        args.result_dir,
        f'{os.path.basename(args.filelist_csv).split(".")[0]}' f"_seg_metrics_24.xlsx",
    )
    metrics_df.to_excel(output_csv, index=False, float_format="%.3f")


if __name__ == "__main__":
    main()
