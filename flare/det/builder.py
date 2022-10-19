# Copyright (c) DAMO Health

from mmcv.utils import Registry

DETECTORS = Registry("detectors")


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)
