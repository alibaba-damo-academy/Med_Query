# Copyright (c) DAMO Health

import importlib
import os
import os.path as osp
import sys
import time
import torch
import warnings
from random import random


def make_dir(*args, retry_count: int = 3) -> str:
    """
    the one-liner directory creator
    """
    path = osp.join(*[arg.strip(" ") for arg in args])
    if path in ("", ".", "/"):
        # invalid dir name
        return path
    path = osp.expanduser(path)
    while not osp.isdir(path) and retry_count > 0:
        retry_count -= 1
        try:
            os.makedirs(path)
        except Exception:
            pass
        # add a random sleep to avoid race between threads
        time.sleep(random() * 0.001)

    if not osp.isdir(path):
        warnings.warn(f"failed to create {path}")
    return path


def load_module_from_file(pyfile: str):
    """
    load module from .py file

    :param pyfile: path to the module file
    :return: module
    """

    dirname = os.path.dirname(pyfile)
    basename = os.path.basename(pyfile)
    module_name, _ = os.path.splitext(basename)

    need_reload = module_name in sys.modules

    # to avoid duplicate module name with existing modules, add the specified path first
    os.sys.path.insert(0, dirname)
    module = importlib.import_module(module_name)
    if need_reload:
        importlib.reload(module)
    os.sys.path.pop(0)

    return module


def save_optimizer(optimizer, save_path):
    optimizer_ckpt = {"optimizer_state_dict": optimizer.state_dict()}
    torch.save(optimizer_ckpt, save_path)


def load_optimizer(optimizer, load_path, local_rank=0):
    """
    load the optimizer from disk to resume training

    :param optimizer: the initialized optimizer object
    :param load_path: the path to load
    :param local_rank: information will be printed when local_rank is 0
    :return: the optimizer with state_dict loaded
    """

    if not os.path.isfile(load_path):
        if local_rank == 0:
            print(f"Optimizer file {load_path} can not be found in the directory.")
    else:
        optimizer_ckpt = torch.load(load_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_ckpt["optimizer_state_dict"])
        if local_rank == 0:
            print(f"Loaded the optimizer file {load_path}")
    return optimizer


def partial_weight_update(model, pretrained_state):
    model_state = model.state_dict()
    state_dict = {k: v for k, v in pretrained_state.items() if k in model_state.keys()}
    model_state.update(state_dict)
    model.load_state_dict(model_state)
    return model


def load_model(net, load_path, gpu_ids=None):
    """
    loading model function

    :param net: defined network
    :param load_path: the path for loading
    :param gpu_ids: gpu to use
    :return: network
    """
    checkpoint = torch.load(load_path)
    partial_weight_update(net, checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if gpu_ids is None:  # use all gpus in default
            net = torch.nn.DataParallel(net).cuda()
        else:
            net = torch.nn.DataParallel(net, device_ids=list(range(len(gpu_ids)))).cuda()
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        net = net.cuda()
    return net, int(epoch)


def load_model_single(net, load_path):
    """
    loading model and do not wrap it with DataParallel

    :param net: defined network
    :param load_dir: the path for loading
    :return: network
    """
    checkpoint = torch.load(load_path)
    partial_weight_update(net, checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    return net, int(epoch)


def save_model(net, epoch, save_dir, cfg, tag=None, save_best=False):
    """
    saving model function

    :param net:  defined network
    :param epoch: current saved epoch
    :param save_dir: the path for saving
    :param cfg: model configuration
    :param tag: tag to distinguish this model
    :param save_best: whether the model to save is the best among previous epochs
    :return: None
    """
    make_dir(save_dir)
    if "module" in dir(net):
        state_dict = net.module.state_dict()
    else:
        state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    if save_best:
        torch.save(
            {"epoch": epoch, "state_dict": state_dict, "cfg": cfg},
            os.path.join(save_dir, f"{tag}_best_metric_model.pth"),
        )
    else:
        torch.save(
            {"epoch": epoch, "state_dict": state_dict, "cfg": cfg},
            os.path.join(save_dir, "model_at_epoch_%04d.pth" % (epoch)),
        )
