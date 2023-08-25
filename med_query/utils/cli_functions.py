# Copyright (c) DAMO Health
import argparse
import os
import subprocess

import med_query
from med_query.utils.training_utils import torch_version_is_higher


def stop_distributed_training():
    """
    use this to stop distributed training

    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tag", type=str, help="training trail to be killed, distinct from other trials"
    )
    args = parser.parse_args()
    tag = args.tag
    if tag is None:
        print("[warning] A tag is needed to kill the right processes, using default tag 'train.py'")
        tag = "train.py"
    sh_path = os.path.join(med_query.__path__[0], "utils", "stop_distributed_training.sh")
    cmd = f"sh {sh_path} {tag}"
    subprocess.run(cmd, shell=True)


def start_distributed_training(**kwargs):
    """
    An utility function to start distributed training, one can import this function and
    wrap it into his own command-line distributed_training function

    :param kwargs: key arguments including: train_file, config_file, gpu_ids,
                   tag[Optional], num_threads[Optional], master_port[Optional]
    :return: None

    Example Usage:
        from med_base.utils.cli_functions import start_distributed_training
        def vdet_train_dist():
            parser = argparse.ArgumentParser()
            parser.add_argument("-c", "--config_file", type=str, help="path to the config file")
            parser.add_argument("-g", "--gpu_ids", type=str, default="0,1,2,3", help="gpus to use")
            parser.add_argument("-t", "--tag", type=str, default="default_tag",
                                help="distinct from other trials")
            parser.add_argument('-n', "--num_threads", type=int, default=2)
            parser.add_argument("-p", "--master_port", type=int, default=29500, help="master port")

            args = parser.parse_args()
            args.train_file = os.path.join(ct_spine_labelling.__path__[0], 'landmark', 'train.py')
            start_distributed_training(args)
        put vdet_train_dist into project' setup.py and then use it in command-line:
        vdet_train_dist -c xxx/config.py -g 1,2,3,4 -t version_1
        note that train.py should at least accept "-c", "-t", "-d", which stand for "config_path",
        "tag", "whether to use distributed training", respectively.
        and "--local_rank" is also needed in it if torch version lower than 1.10
    """

    sh_path = os.path.join(med_query.__path__[0], "utils", "start_distributed_training.sh")
    train_file = kwargs.get("train_file")
    dist_mode = "launch"
    if torch_version_is_higher("1.9.1"):
        dist_mode = "torchrun"
    config_file = kwargs.get("config_file")
    gpu_ids = kwargs.get("gpu_ids")
    tag = kwargs.get("tag", "dist_train")
    if not all([train_file, config_file, gpu_ids, tag]):
        raise KeyError(f"{kwargs}")
    num_threads = kwargs.get("num_threads", 2)
    master_port = kwargs.get("master_port", 29500)
    cmd = (
        f"sh {sh_path} {train_file} {dist_mode} {config_file} "
        f"{gpu_ids} {tag} {num_threads} {master_port}"
    )
    subprocess.run(cmd, shell=True)


def med_query_det_train_dist():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="path to the config file")
    parser.add_argument("-g", "--gpu_ids", type=str, default="0,1,2,3", help="gpus to use")
    parser.add_argument(
        "-t", "--tag", type=str, default="default_tag", help="distinct from other trials"
    )
    parser.add_argument("-n", "--num_threads", type=int, default=2)
    parser.add_argument("-p", "--master_port", type=int, default=29500, help="master port")

    args = parser.parse_args()
    args.train_file = os.path.join(os.path.dirname(__file__), "../det", "train.py")
    kwargs = {k: v for k, v in args._get_kwargs()}
    start_distributed_training(**kwargs)


def med_query_seg_train_dist():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="path to the config file")
    parser.add_argument("-g", "--gpu_ids", type=str, default="0,1,2,3", help="gpus to use")
    parser.add_argument(
        "-t", "--tag", type=str, default="default_tag", help="distinct from other trials"
    )
    parser.add_argument("-n", "--num_threads", type=int, default=2)
    parser.add_argument("-p", "--master_port", type=int, default=29500, help="master port")

    args = parser.parse_args()
    args.train_file = os.path.join(os.path.dirname(__file__), "../seg", "train.py")
    kwargs = {k: v for k, v in args._get_kwargs()}
    start_distributed_training(**kwargs)
