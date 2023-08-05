# Copyright (c) DAMO Health

import argparse
import copy
import datetime
import os
import shutil
import time
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from med_query.det.builder import build_dataset, build_detector
from med_query.utils.common import get_scheduler
from med_query.utils.data_utils import DataPrefetcher
from med_query.utils.io_utils import (
    load_module_from_file,
    load_optimizer,
    make_dir,
    save_model,
    save_optimizer,
)
from med_query.utils.training_utils import setup_logger_and_tensorboard


def init_process(local_rank):
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def parse_args():
    parser = argparse.ArgumentParser(description="MED_Query Platform")
    parser.add_argument(
        "-c", "--cfg", type=str, default="./configs/fcos_config.py", dest="cfg", help="config file"
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        default="default_tag",
        dest="tag",
        help="distinct from other trials",
    )
    parser.add_argument(
        "--save_plot", action="store_true", dest="save_plot", help="if specified, save image plots."
    )
    parser.add_argument(
        "--test_num", type=int, dest="test_num", default=4, help="number of data for testing"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        dest="local_rank",
        default=0,
        help="local rank in distributed training",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        dest="world_size",
        default=8,
        help="world size in distributed training",
    )

    args = parser.parse_args()
    return args


def merge_cfg():
    args = parse_args()
    
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    init_process(args.local_rank)

    cfg_module = load_module_from_file(args.cfg)
    cfg = cfg_module.Configs()

    # merge args with configs
    for (k, v) in args.__dict__.items():
        setattr(cfg, k, v)

    if args.local_rank == 0:
        cfg.print_configs()

    make_dir(cfg.MODEL_SAVE_PATH)
    make_dir(os.path.join(cfg.MODEL_SAVE_PATH, cfg.tag))
    # copy config.py
    config_file_to_save = os.path.join(cfg.MODEL_SAVE_PATH, cfg.tag, "config.py")
    shutil.copy(args.cfg, config_file_to_save)

    return cfg


def train(cfg):
    # define training dataset
    cfg.dataset.update({"dataset_name": "trainset", "phase": "train"})
    train_ds = build_dataset(cfg.dataset)
    tr_sampler = DistributedSampler(train_ds)
    # create a training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.nThreads,
        # shuffle=True,
        sampler=tr_sampler,
        pin_memory=True,
        collate_fn=train_ds.collate_fn,
    )

    # define validation dataset
    cfg.dataset.update({"dataset_name": "validset", "phase": "valid"})
    cfg_to_save = copy.deepcopy(cfg)
    val_ds = build_dataset(cfg.dataset)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    # create a validation data loader
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=cfg.nThreads,
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=val_ds.collate_fn,
    )

    save_model_dir = os.path.join(cfg.MODEL_SAVE_PATH, cfg.tag)
    save_plot_dir = os.path.join(cfg.RESULT_SAVE_PATH, cfg.tag)

    if cfg.local_rank == 0:
        make_dir(save_model_dir)
        make_dir(save_plot_dir)
        logger, writer = setup_logger_and_tensorboard(save_model_dir, "med-query")

        dataloader_len = len(train_loader)
        print(
            f"Training images: {len(train_ds)}  |  "
            f"Training iterations per epoch: {dataloader_len}"
        )
        print(
            f"Validation images: {len(val_ds)}  | Validation actually: "
            f"{min(cfg.test_num, len(val_loader)) * dist.get_world_size()}"
        )

    model = build_detector(cfg.__dict__)
    cfg.epoch_count = model.epoch_count

    if cfg.local_rank == 0:
        print(
            f"Model {cfg.type} have {sum(x.numel() for x in model.parameters())/1e6:.1f}M "
            f"parameters in total."
        )
        print("Start Training...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )
    if cfg.epoch_count > 0:
        optimizer_path = cfg.snapshot_opt
        assert os.path.isfile(optimizer_path), "optimizer ckpt not found!"
        optimizer = load_optimizer(optimizer, optimizer_path, cfg.local_rank)

    scheduler = get_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg.use_amp, init_scale=1024.0)

    total_steps = 0
    for epoch in range(cfg.epoch_count + 1, cfg.max_epochs + 1):
        model.train()
        tr_sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_loss = 0
        epoch_ang_loss = 0

        train_prefetcher = DataPrefetcher(train_loader)
        val_prefetcher = DataPrefetcher(val_loader)
        t0 = time.time()
        for epoch_i, data in enumerate(train_prefetcher):
            total_steps += 1
            with autocast(enabled=cfg.use_amp):
                res_dict = model.train_forward(data, epoch=epoch)

                # trick to avoid stuck
                loss = res_dict["loss"] + 0 * sum([x.sum() for x in model.parameters()])

            if cfg.use_amp:
                # with torch.autograd.detect_anomaly():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
            optimizer.zero_grad()
            t1 = time.time()

            if cfg.local_rank == 0:
                loss_cls = res_dict["loss_cls"]
                loss_box = res_dict["loss_box"]
                loss_ang = res_dict["loss_ang"]
                epoch_loss += loss
                epoch_cls_loss += loss_cls
                epoch_box_loss += loss_box
                epoch_ang_loss += loss_ang

                print(f"total step: {total_steps} timer: {t1 - t0:.4f} sec.")
                print(f"epoch {epoch}/{cfg.max_epochs}, " f"step {epoch_i + 1}/{len(train_loader)}")
                logger.info(res_dict["logger_string"])
                print("\r")
            t0 = t1

        if cfg.local_rank == 0:
            epoch_str = f"-----Epoch {epoch}-----"
            logger.info(epoch_str)
            print("\r")

            lr = optimizer.param_groups[0]["lr"]
            print(f"Learning Rate = {lr:.7f}")
            print("\r")

            writer.add_scalar("lr", lr, epoch)
            writer.add_scalar("epoch_train_loss", epoch_loss / dataloader_len, epoch)
            writer.add_scalar("epoch_cls_loss", epoch_cls_loss / dataloader_len, epoch)
            writer.add_scalar("epoch_box_loss", epoch_box_loss / dataloader_len, epoch)
            writer.add_scalar("epoch_ang_loss", epoch_ang_loss / dataloader_len, epoch)
            writer.flush()

        if epoch % cfg.save_epoch_freq == 0 and cfg.local_rank == 0:
            save_model(
                model.model, epoch, save_model_dir, cfg_to_save.__dict__, tag=cfg.tag, save_best=False,
            )
            save_optimizer(
                optimizer, os.path.join(save_model_dir, f"optimizer_at_epoch_{epoch:0>4d}.pth")
            )
            print(f"Saved model at epoch {epoch}")
            print("\r")
            # save a single csv_file for visualization
            pid = data["meta_data"][0]["pid"]
            csv_path = os.path.join(cfg.RESULT_SAVE_PATH, cfg.tag, f"train_{pid}_epoch{epoch}.csv")
            res_dict["boxes"][0].to_csv(csv_path, index=False)

        if epoch % cfg.valid_epoch_freq == 0:
            model.eval()

            with torch.no_grad():
                valid_epoch_loss = 0
                valid_epoch_cls_loss = 0
                valid_epoch_box_loss = 0
                valid_epoch_ang_loss = 0
                for v_i, data in enumerate(val_prefetcher):
                    if v_i < cfg.test_num:
                        t0 = time.time()
                        pid = data["meta_data"][0]["pid"]
                        res_dict = model.train_forward(data, phase="valid", epoch=epoch)

                        loss = res_dict["loss"]
                        loss_cls = res_dict["loss_cls"]
                        loss_box = res_dict["loss_box"]
                        loss_ang = res_dict["loss_ang"]
                        valid_epoch_loss += loss
                        valid_epoch_cls_loss += loss_cls
                        valid_epoch_box_loss += loss_box
                        valid_epoch_ang_loss += loss_ang

                        t1 = time.time()
                        if cfg.local_rank == 0:
                            print(
                                f"---Validation--- {pid} : {res_dict['logger_string']} , "
                                f"inference cost {t1 - t0:.2f} sec."
                            )
                            # save a single csv_file for visualization
                            csv_path = os.path.join(save_plot_dir, f"valid_{pid}_epoch{epoch}.csv")
                            res_dict["boxes"][0].to_csv(csv_path, index=False)

                    else:
                        break
                if cfg.local_rank == 0:
                    num = min(cfg.test_num, len(val_prefetcher))
                    writer.add_scalar("valid_epoch_loss", valid_epoch_loss / num, epoch)
                    writer.add_scalar("valid_epoch_cls_loss", valid_epoch_cls_loss / num, epoch)
                    writer.add_scalar("valid_epoch_box_loss", valid_epoch_box_loss / num, epoch)
                    writer.add_scalar("valid_epoch_ang_loss", valid_epoch_ang_loss / num, epoch)
                    writer.flush()

        scheduler.step()


def main():
    cfg = merge_cfg()
    if cfg.local_rank == 0:
        start = datetime.datetime.now()
    train(cfg)
    if cfg.local_rank == 0:
        end = datetime.datetime.now()
        print(f"Train completed, time elapsed {str(end - start).split('.')[0]}.")


if __name__ == "__main__":
    main()
