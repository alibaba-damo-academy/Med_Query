# Copyright (c) DAMO Health

import logging
import numpy as np
import os
import random
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

from med_query.utils.io_utils import load_module_from_file


def load_optimizer(checkpoints_dir, optimizer, checkpoint_index=-1, local_rank=0):
    """
    load the optimizer from disk to resume training

    :param checkpoints_dir: the directory where checkpoints sub-directories are stored
    :param optimizer: the initialized optimizer object
    :param checkpoint_index: a number to indict which epoch should be loaded
    :param local_rank: information will be printed when local_rank is 0
    :return: the optimizer with state_dict loaded
    """
    checkpoint_list = [
        f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if len(checkpoint_list) == 0:
        if local_rank == 0:
            print(f"No checkpoints found in {checkpoints_dir}")
        return optimizer

    if checkpoint_index == -1:
        optimizer_file = os.path.join(checkpoints_dir, checkpoint_list[-1], "optimizer.pth")
    else:
        optimizer_file = os.path.join(
            checkpoints_dir, f"ckpt_{checkpoint_index:04d}", "optimizer.pth"
        )
    if not os.path.isfile(optimizer_file):
        if local_rank == 0:
            print(f"Optimizer file {optimizer_file} can not be found in the directory.")
    else:
        optimizer_ckpt = torch.load(optimizer_file, map_location="cpu")
        optimizer.load_state_dict(optimizer_ckpt["optimizer_state_dict"])
        if local_rank == 0:
            print(f"Loaded the optimizer file {optimizer_file}")
    return optimizer


def load_checkpoint(checkpoints_dir, net, checkpoint_index=-1, local_rank=0):
    """
    load the model checkpoint from disk to resume training

    :param checkpoints_dir: the directory where checkpoints sub-directories are stored
    :param net: the initialized network model
    :param checkpoint_index: a number to indict which epoch should be loaded
    :param local_rank: information will be printed when local_rank is 0
    :return: epoch_index and batch_index of the loaded checkpoint, will be (0, 0) if nothing loaded
    """
    checkpoint_list = [
        f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))
    ]
    if len(checkpoint_list) < 1:
        if local_rank == 0:
            print("No checkpoint files found, start training from scratch.")
        return 0, 0

    checkpoint_list.sort()
    if checkpoint_index == -1:
        checkpoint_file = os.path.join(checkpoints_dir, checkpoint_list[-1], "params.pth")
    else:
        checkpoint_file = os.path.join(
            checkpoints_dir, f"ckpt_{checkpoint_index:04d}", "params.pth"
        )

    if not os.path.isfile(checkpoint_file):
        if local_rank == 0:
            print(f"Checkpoint file {checkpoint_file} can not be found in the directory.")
        return 0, 0

    ckpt = torch.load(checkpoint_file, map_location="cpu")
    model_state_dict = ckpt["model_state_dict"]
    res = net.load_state_dict(model_state_dict)
    if res.__str__() != "<All keys matched successfully>":
        raise KeyError("load model failed")
    if local_rank == 0:
        print(f"Loaded the checkpoint file {checkpoint_file}")

    return ckpt["epoch_index"], ckpt["batch_index"] + 1


def save_checkpoint(
    checkpoint_dir,
    net,
    cfg,
    optimizer,
    in_channels,
    epoch_index,
    batch_index,
    net_name="default_net",
    max_stride=None,
    save_optimizer=True,
):
    """
    Save the network model and optimizer as checkpoint into the checkpoint directory

    :param checkpoint_dir: the root directory to save the checkpoints
    :param net: the network module to save
    :param cfg: network configuration
    :param optimizer: the optimizer to save
    :param net_name: the network's name
    :param in_channels: the number of input channels
    :param epoch_index: index of the epoch
    :param batch_index: index of the batch
    :param max_stride: for networks like VNet, to indicate it's down-sample ratio, defaults to None
    :param save_optimizer: whether to save the optimizer object, defaults to True
    :return: None
    """
    if "module" in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to("cpu")
    ckpt = {
        "epoch_index": epoch_index,
        "batch_index": batch_index,
        "model_state_dict": state_dict,
        "net_name": net_name,
        "in_channels": in_channels,
        "cfg": cfg,
    }
    if max_stride is not None:
        ckpt.update({"max_stride": max_stride})

    checkpoint_file = os.path.join(checkpoint_dir, "params.pth")
    print(f"saving checkpoint file {checkpoint_file}")
    torch.save(ckpt, checkpoint_file)

    if save_optimizer:
        optimizer_ckpt = {"optimizer_state_dict": optimizer.state_dict()}
        optimizer_file = os.path.join(checkpoint_dir, "optimizer.pth")
        torch.save(optimizer_ckpt, optimizer_file)


def resume_training(save_dir, net, resume_epoch=0, local_rank=0):
    """
    resume training, load checkpoint if resume epoch > 0, otherwise training from scratch

    :param save_dir: directory where checkpoints are stored, will be created if not exists
    :param net: the initialized network module
    :param resume_epoch: which epoch to resume from
    :param max_epochs: the max epoch
    :param local_rank: information will be printed when local_rank is 0
    :return:
    """
    checkpoints_dir = os.path.join(save_dir, "checkpoints")
    if local_rank == 0:
        if not os.path.isdir(checkpoints_dir):
            os.makedirs(checkpoints_dir)

    if resume_epoch > 0:
        epoch_start, batch_start = load_checkpoint(checkpoints_dir, net, resume_epoch, local_rank)
    else:
        epoch_start, batch_start = 0, 0
    net.train()

    return checkpoints_dir, epoch_start, batch_start


def reshape_nd_tensors_to_2d(predictions, mask_tensor):
    """
    reshape predictions for the convenience of loss calculation

    :param predictions:  the prediction tensor
    :param mask_tensor:  the ground mask tensor
    :return: the reshaped tensors
    """
    if predictions.dim() == 4:
        b, c, h, w = predictions.size()
        predictions = predictions.permute(0, 2, 3, 1).contiguous()
        target = mask_tensor.permute(0, 2, 3, 1).contiguous()
    elif predictions.dim() == 5:
        b, c, d, h, w = predictions.size()
        predictions = predictions.permute(0, 2, 3, 4, 1).contiguous()
        target = mask_tensor.permute(0, 2, 3, 4, 1).contiguous()
    else:
        raise ValueError(f"Invalid inputs with dimensions {predictions.dim()}")

    predictions = predictions.view(-1, c)
    target = target.view(-1, 1)

    return predictions, target


def compute_voxel_metrics(predictions, target):
    """
    compute accuracy, sensitivity, specificity

    :param predictions: 2d tensor of predicted multi-channel logits, N * num_classes
    :param target: 2d tensor of ground-truth mask, N * 1
    :return: error, positive error, negative error
    """
    assert predictions.dim() == 2, f"the dim of predictions should be 2, not {predictions.dim()}"
    if predictions.size(0) == 0:
        return 0.0
    _, predicted_labels = torch.max(predictions.detach(), dim=1)
    predicted_labels = predicted_labels.cpu().int().flatten()
    groundtruth_labels = target.detach().cpu().int().flatten()

    total_acc = 1.0 * (predicted_labels == groundtruth_labels).sum() / predicted_labels.size(0)

    pred_for_positive = predicted_labels[groundtruth_labels > 0]
    gt_for_positive = groundtruth_labels[groundtruth_labels > 0]
    sensitivity = 1.0 * (pred_for_positive == gt_for_positive).sum() / gt_for_positive.size(0)

    pred_for_negative = predicted_labels[groundtruth_labels == 0]
    gt_for_negative = groundtruth_labels[groundtruth_labels == 0]
    specificity = 1.0 * (pred_for_negative == gt_for_negative).sum() / gt_for_negative.size(0)

    return total_acc.item(), sensitivity.item(), specificity.item()


def setup_logger(
    log_file: str = None,
    name: str = "train_log",
    get_cmd_handler=True,
    level=logging.INFO,
    formatter=None,
) -> logging.Logger:
    """
    setup a logger that can put message to log file and command line simultaneously

    :param log_file: path to the log file
    :param name: name for the logger
    :param get_cmd_handler: whether to get a cmd_handler at the same time
    :return: a logger object
    """
    if log_file is None and not get_cmd_handler:
        raise ValueError("a path for log_file should be passed when get_cmd_handler=False!")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if formatter is None:
        formatter = logging.Formatter(fmt="%(name)s-%(levelname)s-%(asctime)s-%(message)s")

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if get_cmd_handler:
        cmd_handler = logging.StreamHandler(sys.stdout)
        cmd_handler.setFormatter(formatter)
        logger.addHandler(cmd_handler)

    logger.propagate = False

    return logger


def setup_logger_and_tensorboard(save_dir, logger_name):
    """
    setup a logger and tensorboard writer for training process

    :param save_dir: the directory where checkpoints are stored, log files will also be stored in it
    :param logger_name:  name for the logger
    :return: a logger and a summary_writter
    """
    log_dir = os.path.join(save_dir, "logging")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "train_log.txt")
    logger = setup_logger(log_file, name=logger_name)

    # tensorboard
    tensorboard_log_dir = os.path.join(log_dir, "tensorboard")
    if not os.path.isdir(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    summary_writter = SummaryWriter(tensorboard_log_dir)

    return logger, summary_writter


def init_process(local_rank):
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def torch_version_is_higher(other: str = "1.9.1"):
    """
    compare system torch version with given version, return True if version installed is higher

    :param other: version to compare with
    :return: True if version installed is higher, else False
    """
    torch_version = torch.__version__.split("+")[0].split(".")[:2]
    other = other.split(".")
    assert (
        len(other) > 1
    ), 'version to be compared should be a string with at least two digital separated by "."'

    higher = False
    for i in range(2):
        if int(torch_version[i]) > int(other[i]):
            higher = True
            break

    return higher


def merge_cfg(args):
    cfg_module = load_module_from_file(args.cfg)
    cfg = cfg_module.Configs()
    cfg.cfg_file = args.cfg

    # merge args with configs
    for k, v in args.__dict__.items():
        setattr(cfg, k, v)

    if torch_version_is_higher("1.9.1"):
        cfg.local_rank = int(os.environ["LOCAL_RANK"])

    if args.no_dist_flag:
        cfg.is_dist = False
    else:
        cfg.is_dist = True
        init_process(cfg.local_rank)

    if cfg.local_rank == 0:
        cfg.print_configs()

    return cfg


def reset_random_states(seed: int = 42) -> None:
    """
    control the randomness of all modules for debugging purpose

    :param seed: the random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
