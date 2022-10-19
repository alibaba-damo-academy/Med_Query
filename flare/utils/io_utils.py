# Copyright (c) DAMO Health

import importlib
import os
import os.path as osp
import sys
import time
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
