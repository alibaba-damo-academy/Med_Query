# Copyright (c) DAMO Health

import argparse
import datetime
import itertools
import math
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.multiprocessing as mp
import traceback
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from med_query.det.builder import build_dataset, build_detector
from med_query.utils.data_utils import DataPrefetcher
from med_query.utils.io_utils import load_module_from_file, make_dir
from med_query.utils.training_utils import setup_logger

logger = setup_logger("valid.log", "infer_log", True)


def parse_args():
    parser = argparse.ArgumentParser(description="MED_Query Platform")
    parser.add_argument(
        "-c", "--cfg", type=str, default="./configs/rib_config.py", dest="cfg", help="config file"
    )
    parser.add_argument(
        "-n",
        "--num_process",
        type=int,
        default=None,
        dest="num_process",
        help="number of processes to spawn, None means using all gpus",
    )
    parser.add_argument("--snapshot_det", default=None, type=str, help="snapshot for det")
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="validset",
        dest="dataset_name",
        help="dataset part to validate, trainset | validset | testset",
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
        "-o",
        "--output_csv_path",
        type=str,
        dest="output_csv_path",
        default=None,
        help="path for saving results",
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
    cfg_module = load_module_from_file(args.cfg)
    cfg = cfg_module.Configs()

    if args.query_index is not None:
        args.query_index = [int(i) for i in args.query_index.split(",")]
    else:
        num_queries = cfg.num_classes + 1
        args.query_index = np.arange(num_queries).tolist()

    # merge args with configs
    for k, v in args.__dict__.items():
        setattr(cfg, k, v)

    # cfg.print_configs()
    make_dir(cfg.RESULT_SAVE_PATH)

    return cfg


def valid_single_patient(cfg, model, data):
    with torch.no_grad():
        pid = data["meta_data"][0]["pid"]
        try:
            t0 = time.time()
            res_dict = model.train_forward(data, phase="valid", query_index=cfg.query_index)
            t1 = time.time()
            logger.info(f"Validate {pid}: cost {t1-t0:.2f} sec.")

            return res_dict["boxes"][0], t1 - t0

        except Exception:
            err_tb = "".join(traceback.format_exc(limit=10))
            logger.error(f"Error occurred on {pid}\n {err_tb}")
            return pd.DataFrame([]), 0


def valid(
    q,
    rank,
    world_size,
    cfg,
):
    init_process(rank, world_size)

    # define validation dataset
    cfg.dataset.update({"dataset_name": cfg.dataset_name, "phase": "valid"})
    val_ds = build_dataset(cfg.dataset)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    val_size = len(val_ds)
    # create a validation dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=1,
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=val_ds.collate_fn,
    )
    val_prefetcher = DataPrefetcher(val_loader)
    logger.info(
        f"This is pid: {os.getpid()}, will valid {len(val_loader)} " f"images in this process."
    )

    model = build_detector(cfg.__dict__)
    cfg.epoch_count = model.epoch_count
    model.eval()
    if rank == 0:
        logger.info(
            f"Model {cfg.type} has {sum(x.numel() for x in model.parameters())/1e6:.1f}M "
            f"parameters in total."
        )

    start = time.time()

    process_res_list = []
    inf_costs = []
    for v_i, data in enumerate(val_prefetcher):
        res, inf_cost = valid_single_patient(cfg, model, data)
        process_res_list.append(res)
        inf_costs.append(inf_cost)

    if rank == 0:
        process_res_list.append(val_size)
    logger.info(
        f"Validation completed in pid: {os.getpid()}, cost {time.time()-start:.3f} sec, "
        f"inference avg time: {sum(inf_costs)/len(val_prefetcher):.3f} sec."
    )
    q.put(process_res_list)


def main():
    start = datetime.datetime.now()
    mp.set_start_method("spawn")

    cfg = merge_cfg()

    num_processes = cfg.num_process if cfg.num_process is not None else torch.cuda.device_count()
    processes = []

    for i in range(num_processes):
        q = mp.Queue()
        p = mp.Process(
            target=valid,
            args=(
                q,
                i,
                num_processes,
                cfg,
            ),
        )
        p.start()
        processes.append([q, p])
    process_res_list = []
    for q, p in processes:
        process_res_list.append(q.get())
        p.join()

    val_size = process_res_list[0][-1]
    process_res_list[0].pop()
    logger.info(f"Has validated {val_size} images in total.")

    patient_res_list = list(itertools.chain.from_iterable(process_res_list))
    # rearrange the result list
    patient_res_list_neat = []
    num_patient_per_process = math.ceil(val_size / num_processes)
    for i in range(num_patient_per_process):
        for j in range(num_processes):
            patient_res_list_neat.append(patient_res_list[i + j * num_patient_per_process])
    # duplicates might be appear due to DistributedSampler
    patient_res_list = patient_res_list_neat[:val_size]

    df = pd.concat(patient_res_list)

    save_csv_path = os.path.join(cfg.RESULT_SAVE_PATH, "OrganDetSeg_ValRes.csv")
    if cfg.output_csv_path is not None:
        save_csv_path = cfg.output_csv_path

    if os.path.exists(save_csv_path):
        os.remove(save_csv_path)
    df.to_csv(save_csv_path, index=False)
    end = datetime.datetime.now()
    logger.info(f"Parallel validation completed, time elapsed {str(end - start).split('.')[0]}.")


if __name__ == "__main__":
    main()
