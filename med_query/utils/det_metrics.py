# Copyright (c) DAMO Health

import os
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_csv", type=str, help="filelist csv file")
    parser.add_argument("-r", "--result_csv", type=str, help="result csv")
    parser.add_argument("-l", "--label_csv", type=str, help="label csv")
    parser.add_argument(
        "-pt", "--position_thresh", type=float, default=20.0, help="center distance threshold"
    )
    parser.add_argument(
        "-st",
        "--scale_thresh",
        type=float,
        default=20.0,
        help="average scale difference threshold in the unit of mm",
    )
    parser.add_argument(
        "-rt",
        "--rotate_thresh",
        type=float,
        default=10.0,
        help="average rotation threshold in the unit of degree",
    )

    parser.add_argument("-n", "--num_process", type=int, default=1, help="number of processes")
    args = parser.parse_args()

    filelist_df = pd.read_csv(args.filelist_csv)
    result_df = pd.read_csv(args.result_csv)
    label_df = pd.read_csv(args.label_csv)
    max_ribs = 24
    p_thresh = args.position_thresh
    s_thresh = args.scale_thresh
    r_thresh = args.rotate_thresh

    def fun(case):
        case_result = result_df[result_df.filename == case]
        case_label = label_df[label_df.filename == case]
        hit_rate, avg_center_diff, avg_whd_diff, avg_angle_diff, res_dict = process(
            case_result, case_label, p_thresh, s_thresh, r_thresh, max_ribs
        )
        res_dict["filename"] = [case] * max_ribs
        res_df = pd.DataFrame(res_dict)
        return hit_rate, avg_center_diff, avg_whd_diff, avg_angle_diff, res_df

    res = Parallel(n_jobs=args.num_process)(
        delayed(fun)(case) for case in tqdm(filelist_df.filename, desc="Detection evaluation")
    )
    hit_rates = [i[0] for i in res]
    avg_center_diffs = [i[1] for i in res]
    avg_whd_diffs = [i[2] for i in res]
    avg_angle_diffs = [i[3] for i in res]
    res_list = [i[4] for i in res]

    avg_hit_rate = np.mean(hit_rates)
    std_hit_rate = np.std(hit_rates)
    avg_center_diff_p = np.nanmean(np.stack(avg_center_diffs))
    std_center_diff_p = np.nanstd(np.stack(avg_center_diffs))
    avg_whd_diff_p = np.nanmean(np.stack(avg_whd_diffs), axis=0)
    avg_angle_diff_p = np.nanmean(np.stack(avg_angle_diffs), axis=0)

    total_df = pd.concat(res_list)
    columns = [
        "filename",
        "label",
        "hit",
        "center_diff",
        "width_diff",
        "height_diff",
        "depth_diff",
        "alpha_diff",
        "beta_diff",
        "gamma_diff",
    ]
    output_file = os.path.join(os.path.dirname(args.result_csv), "test_det_metrics.xlsx")
    total_df.to_excel(output_file, index=False, columns=columns, float_format="%.2f")
    print("==================================Detections results==================================")
    print(
        f"Average hit rate in patient level: {avg_hit_rate:.3f} (std: {std_hit_rate:.3f}) "
        f"@ (p_thresh={p_thresh}mm, "
        f"s_thresh={s_thresh}mm, r_thresh={r_thresh}\N{DEGREE SIGN})"
    )
    print(
        f"Average center difference for hit ribs in patient level: {avg_center_diff_p:.3f}mm. "
        f"(std: {std_center_diff_p:.3f})."
    )
    print(
        f"Average scale difference for hit ribs w.r.t width, height, depth: "
        f"{avg_whd_diff_p[0]:.3f}mm, {avg_whd_diff_p[1]:.3f}mm, {avg_whd_diff_p[2]:.3f}mm."
    )
    print(
        f"Average angle difference for hit ribs w.r.t alpha, beta, gamma: "
        f"{avg_angle_diff_p[0]:.3f}\N{DEGREE SIGN}, {avg_angle_diff_p[1]:.3f}\N{DEGREE SIGN}, "
        f"{avg_angle_diff_p[2]:.3f}\N{DEGREE SIGN}."
    )
    print(f"Output csv has been saved to {output_file}")


def compute_distance_mat(pred_array, gt_array, max_ribs):
    pred_mat = np.expand_dims(pred_array, 1).repeat(max_ribs, 1)
    gt_mat = np.expand_dims(gt_array, 1).repeat(max_ribs, 1).transpose(1, 0, 2)
    dist_mat = np.sqrt(np.sum(np.square(pred_mat - gt_mat), axis=2))

    return dist_mat


def compute_angle_diff(pred_direction, gt_direction, intersects):
    angle_diff = np.full((len(intersects), 3), np.nan)
    for i, inter in enumerate(intersects):
        rot_mat = pred_direction[inter - 1].reshape(3, 3).transpose()
        rot_mat_obj = R.from_matrix(rot_mat)
        pred_angles = rot_mat_obj.as_euler("zyx", degrees=True)

        rot_mat = gt_direction[inter - 1].reshape(3, 3).transpose()
        rot_mat_obj = R.from_matrix(rot_mat)
        gt_angles = rot_mat_obj.as_euler("zyx", degrees=True)
        angle_diff[i] = np.abs(pred_angles - gt_angles)

    return angle_diff


def process(pred, gt, p_thresh, s_thresh, r_thresh, max_ribs):
    intersection_columns = pred.columns.intersection(gt.columns)
    pred = pred[intersection_columns].sort_values(by="label")
    pred = pred[pred.label != 0]
    gt = gt[intersection_columns]

    hit_list = np.full(max_ribs, np.nan)
    hit_list[gt.label.values - 1] = 0

    intersects = np.intersect1d(pred.label.values, gt.label.values)
    if len(intersects) == 0:
        gt_label = gt.label.values
        label_column = np.full(max_ribs, np.nan)
        label_column[gt_label - 1] = gt_label
        res_dict = dict()
        res_dict["label"] = label_column
        res_dict["hit"] = hit_list
        res_dict["center_diff"] = np.full(max_ribs, np.nan)

        res_dict["width_diff"] = np.full(max_ribs, np.nan)
        res_dict["height_diff"] = np.full(max_ribs, np.nan)
        res_dict["depth_diff"] = np.full(max_ribs, np.nan)

        res_dict["alpha_diff"] = np.full(max_ribs, np.nan)
        res_dict["beta_diff"] = np.full(max_ribs, np.nan)
        res_dict["gamma_diff"] = np.full(max_ribs, np.nan)

        return 0.0, np.nan, np.full(3, np.nan), np.full(3, np.nan), res_dict
    else:
        pred_label = pred.label.values
        gt_label = gt.label.values

        pred_array = np.full((max_ribs, 15), np.nan)
        gt_array = np.full((max_ribs, 15), np.nan)

        pred_all_key = np.array(pred.iloc[:, 2:]).astype(np.float32)
        gt_all_key = np.array(gt.iloc[:, 2:]).astype(np.float32)

        pred_array[pred_label - 1, :] = pred_all_key
        gt_array[gt_label - 1, :] = gt_all_key

        pred_centroids = pred_array[:, :3]
        gt_centroids = gt_array[:, :3]
        pred_directions = pred_array[:, 3:12]
        gt_directions = gt_array[:, 3:12]
        pred_whd = pred_array[:, 12:]
        gt_whd = gt_array[:, 12:]

        dist_mat = compute_distance_mat(pred_centroids, gt_centroids, max_ribs)
        dist_mat_inter = dist_mat[intersects - 1, :][:, intersects - 1]

        # 1. check if pred label is closest to gt label
        mask = np.ones_like(dist_mat_inter, dtype=bool)
        mask[:, np.argmin(dist_mat_inter, axis=0)] = False
        dist_mat_inter[mask] = np.nan

        # 2. check if the center distance between closest pred and gt is less than preset threshold
        dist_diag = np.full(max_ribs, np.nan)
        dist_diag[intersects - 1] = np.copy(np.diagonal(dist_mat_inter))
        dist_diag_cp = np.copy(dist_diag)
        dist_diag[dist_diag >= p_thresh] = np.nan

        # 3. check average whd difference is less than preset threshold
        whd_diff = np.full((len(pred_whd), 3), np.nan)
        whd_diff[intersects - 1] = np.abs(pred_whd - gt_whd)[intersects - 1]
        whd_diff_mean = np.mean(whd_diff, axis=1)
        whd_diff_mean[whd_diff_mean >= s_thresh] = np.nan

        # 4. check average angle difference is less than preset threshold
        angle_diff_inter = compute_angle_diff(pred_directions, gt_directions, intersects)
        angle_diff = np.full((max_ribs, 3), np.nan)
        angle_diff[intersects - 1] = np.copy(angle_diff_inter)
        angle_diff_mean = np.mean(angle_diff, axis=1)
        angle_diff_mean[angle_diff_mean >= r_thresh] = np.nan

        # 5. compute hit rate
        hit_list[~np.isnan(dist_diag) & ~np.isnan(whd_diff_mean) & ~np.isnan(angle_diff_mean)] = 1
        hits = np.count_nonzero(hit_list == 1)
        hit_rate = hits / len(gt_label)

        avg_center_diff = np.mean(dist_diag[hit_list == 1])
        avg_whd_diff = np.mean(whd_diff[hit_list == 1], axis=0)
        avg_angle_diff = np.mean(angle_diff[hit_list == 1], axis=0)

        label_column = np.full(max_ribs, np.nan)
        label_column[gt_label - 1] = gt_label
        res_dict = dict()
        res_dict["label"] = label_column
        res_dict["hit"] = hit_list
        res_dict["center_diff"] = dist_diag_cp

        res_dict["width_diff"] = whd_diff[:, 0]
        res_dict["height_diff"] = whd_diff[:, 1]
        res_dict["depth_diff"] = whd_diff[:, 2]

        res_dict["alpha_diff"] = angle_diff[:, 0]
        res_dict["beta_diff"] = angle_diff[:, 1]
        res_dict["gamma_diff"] = angle_diff[:, 2]

        return hit_rate, avg_center_diff, avg_whd_diff, avg_angle_diff, res_dict


if __name__ == "__main__":
    main()
