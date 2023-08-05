# Copyright (c) DAMO Health

import argparse
import datetime
import numpy as np
import os
import shutil
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceLoss, TverskyLoss
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from med_query.det.loss.focal_loss import FocalLoss
from med_query.seg.builder import build_dataset
from med_query.utils.common import get_network, get_scheduler
from med_query.utils.io_utils import load_module_from_file
from med_query.utils.queue import Queue
from med_query.utils.training_utils import (
    compute_voxel_metrics,
    load_optimizer,
    reset_random_states,
    reshape_nd_tensors_to_2d,
    resume_training,
    save_checkpoint,
    setup_logger_and_tensorboard,
)


def train(cfg):
    global train_sampler
    trail_name = f"{cfg.task}-{cfg.network.net_name}-{cfg.loss.loss_name}-{cfg.tag}"
    save_dir = os.path.join(cfg.training_params.save_dir, trail_name)

    # clean up existing folder if not resume training
    if cfg.local_rank == 0:
        if cfg.training_params.resume_epoch < 1 and os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # random state
    reset_random_states(cfg.training_params.random_seed)

    # enable logging
    if cfg.local_rank == 0:
        logger, writer = setup_logger_and_tensorboard(save_dir, trail_name)

    # enable cudnn benchmark
    if torch.cuda.is_available():
        cudnn.benchmark = True
    else:
        if cfg.local_rank == 0:
            logger.info("CUDA is not available")

    # create dataset and dataloader
    cfg.dataset.update(
        {"dataset_name": "trainset", "phase": "train", "cluster": cfg.dataset.cluster}
    )
    train_dataset = build_dataset(cfg.dataset)
    cfg.dataset.update(
        {"dataset_name": "validset", "phase": "valid", "cluster": cfg.dataset.cluster}
    )
    valid_dataset = build_dataset(cfg.dataset)

    if cfg.dist_train:
        train_sampler = DistributedSampler(train_dataset)
        train_queue = Queue(
            train_dataset,
            max_length=8,
            samples_per_volume=1,
            sampler=None,
            num_workers=8,
            shuffle_patches=False,
            dist_train=True,
        )
        train_loader = DataLoader(
            train_queue,
            batch_size=cfg.training_params.batch_size,
            num_workers=0,
            # num_workers=cfg.training_params.num_threads,
            pin_memory=True,
            shuffle=False,
            # sampler=train_sampler,
            drop_last=False,
        )
        valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=cfg.training_params.num_threads,
            pin_memory=True,
            sampler=valid_sampler,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.training_params.batch_size,
            num_workers=cfg.training_params.num_threads,
            pin_memory=True,
            shuffle=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=cfg.training_params.num_threads,
            pin_memory=True,
            shuffle=False,
        )

    if cfg.local_rank == 0:
        print(
            f"Training images: {len(train_dataset)}  |  "
            f"Training iterations per epoch: {len(train_loader)}"
        )
        print(
            f"Validation images: {len(valid_dataset)}  | Validation actually: "
            f"{cfg.test_num * dist.get_world_size()}"
        )

    # network
    net_name = cfg.network.net_name
    in_channels = cfg.network.in_channels
    out_channels = cfg.network.out_channels
    dimensions = cfg.network.dimensions
    normalization = cfg.network.normalization
    base = cfg.network.base
    downsample = cfg.network.downsample
    net, max_stride = get_network(
        net_name, in_channels, out_channels, dimensions, normalization, base, downsample
    )
    if torch.cuda.is_available():
        net = net.cuda()

    # load checkpoint if resume epoch > 0
    resume_epoch = cfg.training_params.resume_epoch
    max_epochs = cfg.training_params.max_epochs
    if max_epochs <= resume_epoch:
        raise ValueError(
            f"resume_epoch: {resume_epoch} should be smaller than max_epochs: {max_epochs}!"
        )
    checkpoints_dir, epoch_start, batch_start = resume_training(
        save_dir, net, resume_epoch, cfg.local_rank
    )
    last_save_epoch = epoch_start
    batch_index = batch_start

    if cfg.dist_train:
        net = DDP(net, [cfg.local_rank], find_unused_parameters=False)

    # create optimizer
    optimizer = optim.Adam(net.parameters(), lr=cfg.training_params.lr)

    if resume_epoch > 0:
        optimizer = load_optimizer(checkpoints_dir, optimizer, resume_epoch, cfg.local_rank)

    scheduler = get_scheduler(optimizer, cfg)
    scaler = GradScaler(enabled=cfg.training_params.use_amp, init_scale=4096.0)

    # define loss function
    if cfg.loss.loss_name.upper() == "FOCAL":
        alpha = cfg.loss.focal_alpha
        gamma = cfg.loss.focal_gamma
        loss_func = FocalLoss(class_num=out_channels, alpha=alpha, gamma=gamma, size_average=True)
    elif cfg.loss.loss_name.upper() == "DICE":
        loss_func = DiceLoss(include_background=True, softmax=True, to_onehot_y=True)
    elif cfg.loss.loss_name.upper() == "MIX":
        alpha = cfg.loss.focal_alpha
        gamma = cfg.loss.focal_gamma
        fl = FocalLoss(class_num=out_channels, alpha=alpha, gamma=gamma, size_average=True)
        dl = DiceLoss(include_background=True, softmax=True, to_onehot_y=True)
        a, b = cfg.loss.balance
        loss_func = lambda x, y: a * fl(x, y) + b * dl(x, y)
    elif cfg.loss.loss_name.upper() == "TVERSKY":
        loss_func = TverskyLoss(
            include_background=True, softmax=True, to_onehot_y=True, alpha=0.4, beta=0.6
        )
    elif cfg.loss.loss_name.upper() == "MSE":
        loss_func = nn.MSELoss(reduction="sum")
    else:
        raise NotImplementedError(f"unspported loss function: {cfg.loss.loss_name}")

    for epoch_index in range(epoch_start + 1, max_epochs + 1):
        net.train()
        dataloader_length = len(train_loader)
        epoch_loss_sum, epoch_acc = 0, 0
        epoch_sens, epoch_spec = [], 0
        begin_time = time.time()

        train_iter = train_loader
        valid_iter = valid_loader
        for data in train_iter:
            img_tensor = data["img"].cuda()
            msk_tensor = data["msk"].cuda()
            img_tensor = img_tensor.view(-1, *img_tensor.size()[-4:])
            msk_tensor = msk_tensor.view(-1, *msk_tensor.size()[-4:])

            # clear previous gradients
            optimizer.zero_grad()
            # forward
            with autocast(enabled=cfg.training_params.use_amp):
                predictions = net(img_tensor)

            if len(predictions.size()) - len(msk_tensor.size()) == 1:  # deep supervision
                predictions = torch.unbind(predictions, dim=1)
            else:
                predictions = [predictions]

            # compute loss
            train_loss = 0
            for i, pred in enumerate(predictions):
                pred, target = reshape_nd_tensors_to_2d(pred, msk_tensor)
                selected_sample_indices = torch.nonzero((target > -1).int(), as_tuple=False)
                pred = torch.index_select(pred, 0, selected_sample_indices[:, 0])
                target = torch.index_select(target, 0, selected_sample_indices[:, 0])
                train_loss += 0.5 ** i * loss_func(pred, target)

            if cfg.training_params.use_amp:
                scaler.scale(train_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training_params.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training_params.clip_grad)
                optimizer.step()

            epoch_loss_sum += train_loss.item()

            if cfg.local_rank == 0:
                end_time = time.time()
                batch_time = end_time - begin_time
                begin_time = end_time
                sample_time = batch_time / cfg.training_params.batch_size

                # compute error
                pred, target = reshape_nd_tensors_to_2d(predictions[0], msk_tensor)
                selected_sample_indices = torch.nonzero((target > -1).int(), as_tuple=False)
                pred = torch.index_select(pred, 0, selected_sample_indices[:, 0])
                target = torch.index_select(target, 0, selected_sample_indices[:, 0])
                total_acc, sens, spec = compute_voxel_metrics(pred, target)

                # log
                msg = (
                    f"epoch: {epoch_index}, batch: {batch_index}, loss: {train_loss.item(): .4f}, "
                    f"acc: {total_acc: .4f}, sens: {sens: .4f}, "
                    f"spec: {spec: .4f}, time: {sample_time: .4f} s/volume"
                )

                logger.info(msg)
                writer.add_scalar("train_loss", train_loss.item(), batch_index)
                writer.add_scalar("train_acc", total_acc, batch_index)
                writer.add_scalar("train_sens", sens, batch_index)
                writer.add_scalar("train_spec", spec, batch_index)
                writer.flush()

                epoch_acc += total_acc
                epoch_sens.append(sens)
                epoch_spec += spec

            del predictions
            batch_index += 1

        # save
        if (
            (cfg.local_rank == 0)
            and (epoch_index % cfg.training_params.save_epochs == 0)
            and (last_save_epoch != epoch_index)
        ):
            checkpoint_dir = os.path.join(checkpoints_dir, f"ckpt_{epoch_index:04d}")
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # copy config.py
            config_file_to_save = os.path.join(checkpoint_dir, "config.py")
            shutil.copy(cfg.config_file, config_file_to_save)

            # save checkpoint
            save_checkpoint(
                checkpoint_dir,
                net,
                cfg,
                optimizer,
                in_channels,
                epoch_index,
                batch_index,
                net_name,
                max_stride,
                save_optimizer=True,
            )
            last_save_epoch = epoch_index

        # tensorboard log per epoch
        if cfg.local_rank == 0:
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_index)
            writer.add_scalar("epoch_loss", epoch_loss_sum / dataloader_length, epoch_index)
            writer.add_scalar("epoch_acc", epoch_acc / dataloader_length, epoch_index)
            writer.add_scalar("epoch_sens", np.nanmean(np.array(epoch_sens)), epoch_index)
            writer.add_scalar("epoch_spec", epoch_spec / dataloader_length, epoch_index)
            writer.flush()

        if epoch_index % cfg.training_params.valid_epoch_freq == 0:
            net.eval()
            with torch.no_grad():
                valid_epoch_loss = 0
                valid_epoch_acc = 0
                valid_epoch_sens = []
                valid_epoch_spec = 0
                for v_i, data in enumerate(valid_iter):
                    if v_i < cfg.test_num:
                        t0 = time.time()
                        pid = data["pid"][0]
                        img_tensor = data["img"].cuda()
                        msk_tensor = data["msk"].cuda()

                        img_tensor = img_tensor.view(-1, *img_tensor.size()[-4:])
                        msk_tensor = msk_tensor.view(-1, *msk_tensor.size()[-4:])

                        predictions = net(img_tensor)

                        if (
                            len(predictions.size()) - len(msk_tensor.size()) == 1
                        ):  # deep supervision
                            predictions = torch.unbind(predictions, dim=1)
                        else:
                            predictions = [predictions]

                        valid_loss = 0
                        for i, pred in enumerate(predictions):
                            pred, target = reshape_nd_tensors_to_2d(pred, msk_tensor)
                            selected_sample_indices = torch.nonzero(
                                (target > -1).int(), as_tuple=False
                            )
                            pred = torch.index_select(pred, 0, selected_sample_indices[:, 0])
                            target = torch.index_select(target, 0, selected_sample_indices[:, 0])
                            valid_loss += 0.5 ** i * loss_func(pred, target)

                        valid_epoch_loss += valid_loss
                        pred, target = reshape_nd_tensors_to_2d(predictions[0], msk_tensor)
                        selected_sample_indices = torch.nonzero((target > -1).int(), as_tuple=False)
                        pred = torch.index_select(pred, 0, selected_sample_indices[:, 0])
                        target = torch.index_select(target, 0, selected_sample_indices[:, 0])
                        total_acc, sens, spec = compute_voxel_metrics(pred, target)

                        del predictions

                        valid_epoch_acc += total_acc
                        valid_epoch_sens.append(sens)
                        valid_epoch_spec += spec

                        t1 = time.time()
                        if cfg.local_rank == 0:
                            print(
                                f"---Validation-{v_i+1}--- pid: {pid} loss: {valid_loss:.2f}, "
                                f"acc: {total_acc: .2f}, sens: {sens: .2f}, "
                                f"spec: {spec: .2f}, "
                                f"time: {t1 - t0:.2f} sec."
                            )

                    else:
                        break
                if cfg.local_rank == 0:
                    num = min(cfg.test_num, len(valid_iter))
                    writer.add_scalar("valid_epoch_loss", valid_epoch_loss / num, epoch_index)
                    writer.add_scalar("valid_epoch_acc", valid_epoch_acc / num, epoch_index)
                    writer.add_scalar(
                        "valid_epoch_sens", np.nanmean(np.array(valid_epoch_sens)), epoch_index,
                    )
                    writer.add_scalar(
                        "valid_epoch_spec", valid_epoch_spec / num, epoch_index,
                    )
                    writer.flush()
        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, default="seg_config.py", help="path to the config file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        dest="local_rank",
        default=0,
        help="local rank in distributed training",
    )
    parser.add_argument(
        "-t", "--tag", type=str, default="default_tag", help="distinct from other trials"
    )
    parser.add_argument(
        "--test_num", type=int, dest="test_num", default=10, help="number of data for testing"
    )

    args = parser.parse_args()
    cfg_module = load_module_from_file(args.config)
    cfg = cfg_module.cfg
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    cfg.config_file = args.config
    cfg.tag = args.tag
    cfg.test_num = args.test_num
    cfg.dist_train = not cfg.dataset.is_debugging

    if cfg.dist_train:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(cfg.local_rank)

    if cfg.local_rank == 0:
        start = datetime.datetime.now()
    train(cfg)
    if cfg.local_rank == 0:
        end = datetime.datetime.now()
        print(f"Train completed, time elapsed {str(end - start).split('.')[0]}.")


if __name__ == "__main__":
    main()
