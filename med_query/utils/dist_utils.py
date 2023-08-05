# Copyright (c) DAMO Health

import torch.distributed as dist


def allreduce(tensor, op=dist.ReduceOp.SUM):
    """
    reduce param for monitoring.

    :param tensor: torch.Tensor.
    :param op: reduce operation, Defaults to dist.ReduceOp.SUM.
    """
    world_size = dist.get_world_size()
    dist.all_reduce(tensor.div_(world_size), op)
