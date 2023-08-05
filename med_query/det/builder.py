# Copyright (c) DAMO Health

from mmcv.utils import Registry, build_from_cfg

BACKBONES = Registry("backbones")
NECKS = Registry("necks")
LOSSES = Registry("losses")
ROI_EXTRACTORS = Registry("roi_extractors")
HEADS = Registry("heads")
DETECTORS = Registry("detectors")
SHARED_HEADS = Registry("shared_heads")
BBOX_ASSIGNERS = Registry("bbox_assigners")

TRANSFORMER = Registry("transformer")
TRANSFORMER_LAYER_SEQUENCE = Registry("transformer_layer_sequence")
POSITIONAL_ENCODING = Registry("positional_encoding")
MATCH_COST = Registry("match_cost")

DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build dataset."""
    return DATASETS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)


def build_assigner(cfg):
    """Build bbox_assigners."""
    return BBOX_ASSIGNERS.build(cfg)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


def build_transformer(cfg, default_args=None):
    """Builder for transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for positional encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_match_cost(cfg, default_args=None):
    """Builder of match cost."""
    return build_from_cfg(cfg, MATCH_COST, default_args)
