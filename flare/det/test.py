# Copyright (c) DAMO Health

import argparse
import datetime
import itertools
import json
import logging
import math
import os
import pandas as pd
import SimpleITK as sitk
import time
import torch
import torch.multiprocessing as mp
import traceback
from easydict import EasyDict as edict
from flare.det.builder import build_detector
from flare.det.data.dataset_flare import FLAREDatasetTest
from flare.seg.crop_segmentor import CropSegmentor
from flare.seg.roi_extractor import RoiExtractor
from flare.utils.common import convert_df_to_dicts
from flare.utils.io_utils import make_dir
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename="mytest.log", level=logging.DEBUG, format=LOG_FORMAT)


def parse_args():
    parser = argparse.ArgumentParser(description="Med_Query Project")
    parser.add_argument(
        "-n",
        "--num_process",
        type=int,
        default=None,
        dest="num_process",
        help="number of processes to spawn",
    )
    parser.add_argument("--snapshot", default=None, type=str, help="snapshot")
    parser.add_argument("--snapshot_seg", default=None, type=str, help="snapshot for seg")
    parser.add_argument("--snapshot_roi", default=None, type=str, help="snapshot for roi extractor")
    parser.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        help="if specified, print more debugging information",
    )
    parser.add_argument(
        "--image_dir", type=str, dest="image_dir", default=None, help="directory of testing images"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/path/to/results", help="ouput_dir",
    )
    parser.add_argument(
        "-q",
        "--query_index",
        type=str,
        dest="query_index",
        default=None,
        help="indices for which to query",
    )
    parser.add_argument(
        "-s",
        "--save_det_csv",
        action="store_true",
        dest="save_det_csv",
        help="if specified, save detection results to csv",
    )
    args = parser.parse_args()
    return args


def init_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # set this key to enable timeout wait.
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def merge_cfg():
    args = parse_args()
    if args.query_index is not None:
        args.query_index = [int(i) for i in args.query_index.split(",")]

    assert args.snapshot is not None
    args.snapshots = args.snapshot.split(",")

    if args.output_dir is not None:
        setattr(args, "RESULT_SAVE_PATH", args.output_dir)
        make_dir(args.RESULT_SAVE_PATH)

    return args


def test_single_patient(cfg, detectors, roi_extractor, segmentor, data):
    with torch.no_grad():
        pid = data["meta_data"][0]["pid"]
        try:
            t0 = time.time()
            roi_center = None
            if roi_extractor is not None:
                roi_center = roi_extractor.extract_case(data["meta_data"][0]["img_res0"])
            t1 = time.time()

            torch.cuda.empty_cache()
            res_list = []
            for det in detectors:
                res_dict = det.test_forward(
                    data, roi_center=roi_center, query_index=cfg.query_index
                )
                res_list.append(res_dict)
            t2 = time.time()

            if len(res_list) == 1:
                box_df = res_list[0]["boxes"][0]
                box_df.loc[box_df["label"] == 9, "label"] = 13
                box_df.loc[box_df["label"] == 8, "label"] = 11
                box_df.loc[box_df["label"] == 7, "label"] = 9
                box_df.loc[box_df["label"] == 6, "label"] = 7

            elif len(res_list) == 2:
                box_df1 = res_list[0]["boxes"][0]
                box_df2 = res_list[1]["boxes"][0]
                # Note that detector1 should be the merged version
                box_df1.loc[box_df1["label"] == 9, "label"] = 13
                box_df1.loc[box_df1["label"] == 8, "label"] = 11
                box_df1.loc[box_df1["label"] == 7, "label"] = 9
                box_df1.loc[box_df1["label"] == 6, "label"] = 7

                # 6,10,8,12 have been merged
                box_df1 = box_df1[box_df1["label"].isin([1, 3, 4, 5, 7, 11])]
                box_df2 = box_df2[box_df2["label"].isin([2, 9, 13])]
                box_df = pd.concat([box_df1, box_df2])

            else:
                raise ValueError("Unexpected ensemble composition")

            torch.cuda.empty_cache()
            if segmentor is not None:
                bboxes = convert_df_to_dicts(box_df)
                seg_result = segmentor.seg_case(pid, data["meta_data"][0]["img_res0"], bboxes)
                t3 = time.time()

                torch.cuda.empty_cache()
                mask_save_path = os.path.join(cfg.save_mask_dir, f"{pid}.nii.gz")
                if "_0000" in mask_save_path:
                    mask_save_path = mask_save_path.replace("_0000", "")
                sitk.WriteImage(
                    seg_result["mask"], mask_save_path, useCompression=True,
                )
                t4 = time.time()
                print(
                    f"Testing {pid}: roi_extraction: {t1-t0:.2f}s, detection: {t2-t1:.2f}s, "
                    f"segmentation: {t3-t2:.2f}s, saving: {t4-t3:.2f}s"
                )
                return box_df, t3 - t0
            else:
                print(f"Testing {pid}: roi_extraction: {t1-t0:.2f}s, detection: {t2-t1:.2f}s")
                return box_df, t2 - t1

        except Exception:
            err_tb = "".join(traceback.format_exc(limit=10))
            logging.error(f"Error occurred on {pid}\n {err_tb}")
            print(err_tb)
            return pd.DataFrame([]), 0


def test(q, rank, world_size, cfg):
    init_process(rank, world_size)

    cfgs = []
    for i, snap in enumerate(cfg.snapshots):
        if snap.endswith(".pt"):
            extra_files = {"max_stride": "", "cfg": ""}
            torch.jit.load(snap, _extra_files=extra_files)
            cfg_ = edict(json.loads(extra_files["cfg"]))
        else:
            raise ValueError("Unexpected snapshot format!")
        # cfg_.print_configs()
        cfg.snapshot = snap
        # merge args with configs
        for (k, v) in cfg.__dict__.items():
            setattr(cfg_, k, v)
        cfgs.append(cfg_)

    # define testing dataset
    test_ds = FLAREDatasetTest(cfgs[0].dataset, image_dir=cfgs[0].image_dir)
    test_sampler = DistributedSampler(test_ds, shuffle=False)
    test_size = len(test_ds)
    # create a testing dataloader
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=1,
        sampler=test_sampler,
        pin_memory=True,
        collate_fn=test_ds.collate_fn,
    )

    print(f"This is pid: {os.getpid()}, will test {len(test_loader)} " f"images in this process.")

    detectors = []
    for cfg_ in cfgs:
        model = build_detector(cfg_.__dict__)
        cfg_.epoch_count = model.epoch_count
        model.eval()
        detectors.append(model)

    roi_extractor = None
    if cfg.snapshot_roi is not None:
        roi_extractor = RoiExtractor(cfg.snapshot_roi)
    segmentor = None
    if cfg.snapshot_seg is not None:
        segmentor = CropSegmentor(cfg.snapshot_seg, rank=rank)

    if rank == 0:
        print("Detector has been loaded.")
        if roi_extractor is not None:
            print("Roi_Extractor has been loaded.")
        if segmentor is not None:
            print("Segmentor has been loaded.")

    start = time.time()

    process_res_list = []
    inf_costs = []
    for data in test_loader:
        with autocast(enabled=True):
            res, inf_cost = test_single_patient(cfg, detectors, roi_extractor, segmentor, data)
        process_res_list.append(res)
        inf_costs.append(inf_cost)

    if rank == 0:
        process_res_list.append(test_size)
    print(
        f"Testing completed in pid: {os.getpid()}, cost {time.time()-start:.3f} sec, "
        f"inference avg time: {sum(inf_costs)/len(test_loader):.3f} sec."
    )
    q.put(process_res_list)


def main():
    start = datetime.datetime.now()
    mp.set_start_method("spawn")

    cfg = merge_cfg()

    num_processes = cfg.num_process if cfg.num_process is not None else torch.cuda.device_count()
    processes = []
    if cfg.snapshot_seg is not None:
        cfg.save_mask_dir = cfg.RESULT_SAVE_PATH
        make_dir(cfg.save_mask_dir)

    for i in range(num_processes):
        q = mp.Queue()
        p = mp.Process(target=test, args=(q, i, num_processes, cfg,),)
        p.start()
        processes.append([q, p])
    process_res_list = []
    for q, p in processes:
        process_res_list.append(q.get())
        p.join()

    test_size = process_res_list[0][-1]
    process_res_list[0].pop()
    print(f"Has tested {test_size} images in total.")

    if cfg.save_det_csv:
        patient_res_list = list(itertools.chain.from_iterable(process_res_list))
        # rearrange the result list
        patient_res_list_neat = []
        num_patient_per_process = math.ceil(test_size / num_processes)
        for i in range(num_patient_per_process):
            for j in range(num_processes):
                patient_res_list_neat.append(patient_res_list[i + j * num_patient_per_process])
        # duplicates might be appear due to DistributedSampler
        patient_res_list = patient_res_list_neat[:test_size]

        df = pd.concat(patient_res_list)

        save_csv_path = os.path.join(cfg.RESULT_SAVE_PATH, "OrganDetSeg_TestRes.csv")
        if os.path.exists(save_csv_path):
            os.remove(save_csv_path)
        df.to_csv(save_csv_path, index=False)
    end = datetime.datetime.now()
    print(f"Parallel Testing completed, time elapsed {str(end - start).split('.')[0]}.")


if __name__ == "__main__":
    main()
