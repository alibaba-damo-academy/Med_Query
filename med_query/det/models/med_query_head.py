# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) DAMO Health

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv3d, Linear
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import force_fp32
from mmdet.core import build_sampler, multi_apply, reduce_mean
from monai.losses import DiceLoss

from ..builder import (
    HEADS,
    build_assigner,
    build_loss,
    build_positional_encoding,
    build_transformer,
)
from .seg_head import MaskHeadSmallConv, MHAttentionMap


@HEADS.register_module()
class MedQueryHead(nn.Module):
    """Implements the MedQuery transformer head based on DETRhead.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        transformer=None,
        sync_cls_avg_factor=False,
        use_relative_distance_constraint=True,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
        loss_cls=dict(
            type="CrossEntropyLoss",
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_angle=dict(type="L1Loss", loss_weight=5.0),
        with_positional_encoding=True,
        with_seg=False,
        train_cfg=dict(
            assigner=dict(
                type="HungarianAssigner",
                cls_cost=dict(type="ClassificationCost", weight=1.0),
                box_cost=dict(type="BBoxL1Cost", weight=5.0),
                ang_cost=dict(type="BBoxL1Cost", weight=5.0),
            )
        ),
        test_cfg=dict(max_per_img=100),
        init_cfg=None,
        **kwargs,
    ):
        super(MedQueryHead, self).__init__()
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.use_relative_distance_constraint = use_relative_distance_constraint
        class_weight = loss_cls.get("class_weight", None)
        if class_weight is not None:
            assert isinstance(class_weight, float), (
                "Expected " "class_weight to have type float. Found " f"{type(class_weight)}."
            )
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get("bg_cls_weight", class_weight)
            assert isinstance(bg_cls_weight, float), (
                "Expected " "bg_cls_weight to have type float. Found " f"{type(bg_cls_weight)}."
            )
            class_weight = [class_weight] * (num_classes + 1)
            # set background class as the first indice
            # Note: This is different from mmdet fashion
            class_weight[0] = bg_cls_weight

            loss_cls.update({"class_weight": class_weight})
            if "bg_cls_weight" in loss_cls:
                loss_cls.pop("bg_cls_weight")
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert "assigner" in train_cfg, "assigner should be provided " "when train_cfg is set."
            assigner = train_cfg["assigner"]
            if assigner["type"] == "HungarianAssigner3D":
                assert loss_cls["loss_weight"] == assigner["cls_cost"]["weight"], (
                    "The classification weight for loss and matcher should be" "exactly the same."
                )
                assert loss_bbox["loss_weight"] == assigner["box_cost"]["weight"], (
                    "The regression L1 weight for loss and matcher " "should be exactly the same."
                )
                assert loss_angle["loss_weight"] == assigner["ang_cost"]["weight"], (
                    "The regression L1 weight for loss and matcher " "should be exactly the same."
                )
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_angle = build_loss(loss_angle)
        self.with_positional_encoding = with_positional_encoding
        self.with_seg = with_seg

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=False))
        if self.with_positional_encoding:
            self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)

        self.embed_dims = self.transformer.embed_dims
        self.num_heads = transformer["encoder"]["transformerlayers"]["attn_cfgs"][0]["num_heads"]

        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 3 == self.embed_dims, (
            "embed_dims should"
            f" be exactly 3 times of num_feats. Found {self.embed_dims}"
            f" and {num_feats}."
        )

        if self.with_seg:
            self.bbox_attention = MHAttentionMap(
                self.embed_dims, self.embed_dims, self.num_heads, dropout=0.0
            )
            self.mask_head = MaskHeadSmallConv(
                self.embed_dims + self.num_heads, [256, 128, 64], self.embed_dims
            )
            self.loss_seg = DiceLoss(include_background=True, sigmoid=True, to_onehot_y=False)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        self.input_proj = Conv3d(self.in_channels, self.embed_dims, kernel_size=1)
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)

        # Linear FeedForwardNetwork
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            ffn_drop=0.0,
            add_identity=False,
        )

        self.fc_box = Linear(self.embed_dims, 6)
        self.fc_ang = Linear(self.embed_dims, 3)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def forward(self, feats, img_metas, query_index=None):
        """Forward function."""
        return self.forward_single(feats, img_metas, query_index)

    def forward_single(self, feats, img_metas, query_index=None):
        """"Forward function for a single feature level, but with all features as input for \
            skip connection

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 5D-tensor.
            img_metas (list[dict]): List of image information.
            query_index (list): Indices for which to query
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, cz, w, h, d).
                Shape [nb_dec, bs, num_query, 6].
            all_angle_preds (Tensor): outputs from the angle regression head (alpha, beta, gamma).
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        x = feats[-1]
        batch_size = x.size(0)
        masks = x.new_zeros((batch_size, *x.shape[-3:])).to(torch.bool)

        x = self.input_proj(x)

        # position encoding
        if self.with_positional_encoding:
            pos_embed = self.positional_encoding(masks, type="spatial")  # [bs, embed_dim, w, h, d]
        else:
            pos_embed = None

        if query_index is not None:
            if self.loss_cls.use_sigmoid:
                query_index = x.new_tensor([q - 1 for q in query_index]).long()
            else:
                query_index = x.new_tensor(query_index).long()
        else:
            query_index = torch.arange(self.num_query).to(x.device)

        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        outs_dec, memory = self.transformer(
            x, masks, self.query_embedding.weight[query_index], pos_embed
        )

        all_cls_scores = self.fc_cls(outs_dec)
        reg_feats = self.reg_ffn(outs_dec)
        all_bbox_preds = self.fc_box(reg_feats).sigmoid()
        all_angle_preds = self.fc_ang(reg_feats)

        if self.with_seg:
            bbox_mask = self.bbox_attention(outs_dec[-1], memory, mask=masks)
            seg_masks = self.mask_head(x, bbox_mask, [feats[2], feats[1], feats[0]])
            # [bs * qs, 1, W/2, H/2, D/2] -> [bs, qs, W/2, H/2, D/2]
            mask_preds = seg_masks.view(
                batch_size,
                self.num_query,
                seg_masks.shape[-3],
                seg_masks.shape[-2],
                seg_masks.shape[-1],
            )

            return all_cls_scores, all_bbox_preds, all_angle_preds, mask_preds
        else:
            return all_cls_scores, all_bbox_preds, all_angle_preds, None

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds", "all_angle_preds", "mask_preds"))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        all_angl_preds,
        mask_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_angles_list,
        gt_masks,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """ "Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores (Tensor): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs for each feature level. Each is a 6D-tensor with
                normalized coordinate format (cx, cy, cz, w, h, d) and shape
                [nb_dec, bs, num_query, 6].
            all_angl_preds (Tensor): Outputs from the angle regression head (alpha, beta, gamma).
            mask_preds (Tensor): [Optional] segmentation logits.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 6) in [cx, cy, cz, w, h, d] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_angles_list (list[Tensor]): Ground truth angles for each image with shape
                (num_gts, 3) in [alpha, beta, gamma] format.
            gt_masks (Tensor): Ground truth segmentation masks.
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, "Only supports for gt_bboxes_ignore setting to None."

        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_angles_list = [gt_angles_list for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_box, losses_ang = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_angl_preds,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_angles_list,
            img_metas_list,
            all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_box"] = losses_box[-1]
        loss_dict["loss_ang"] = losses_ang[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_box_i, loss_ang_i in zip(
            losses_cls[:-1], losses_box[:-1], losses_ang[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_box"] = loss_box_i
            loss_dict[f"d{num_dec_layer}.loss_ang"] = loss_ang_i
            num_dec_layer += 1

        if self.with_seg:
            loss_seg = []
            for i, gt_labels in enumerate(gt_labels_list):
                msk = mask_preds[i]
                gt = gt_masks[i]
                for label in gt_labels:
                    gt_bin = gt == label
                    msk_ch = msk[label]
                    loss_seg_ = self.loss_seg(msk_ch.unsqueeze(0).unsqueeze(0), gt_bin.unsqueeze(0))
                    loss_seg.append(loss_seg_)

            loss_dict["loss_seg"] = torch.mean(torch.stack(loss_seg))

        return loss_dict

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        angl_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_angles_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, cz, w, h, d) and
                shape [bs, num_query, 6].
            angl_preds (Tensor): Outputs from the angle regression head (alpha, beta, gamma), with
                shape [bs, num_query, 3].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 6) in [cx, cy, cz, w, h, d] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_angles_list (list[Tensor]): Ground truth angles for each image with shape
                (num_gts, 3) in [alpha, beta, gamma] format.
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        # split into batch element
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        angl_preds_list = [angl_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(
            cls_scores_list,
            bbox_preds_list,
            angl_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_angles_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            angl_targets_list,
            angl_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        angl_targets = torch.cat(angl_targets_list, 0)
        angl_weights = torch.cat(angl_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # DETR regress the relative position of boxes (cxcyczwhd) in the image.
        # regression L1 loss on box
        bbox_preds = bbox_preds.reshape(-1, 6)
        loss_bbox = self.loss_bbox(bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss on box relative distance
        bbox_weights = bbox_weights.clone().reshape(-1, self.num_query, 6)
        bbox_preds = bbox_preds.reshape(-1, self.num_query, 6)
        bbox_targets = bbox_targets.reshape(-1, self.num_query, 6)

        if self.use_relative_distance_constraint:
            bbox_preds_delta = (
                bbox_preds[:, 1 : self.num_query, :3] - bbox_preds[:, 0 : self.num_query - 1, :3]
            )
            bbox_targets_delta = (
                bbox_targets[:, 1 : self.num_query, :3]
                - bbox_targets[:, 0 : self.num_query - 1, :3]
            )

            # condition: query weight with current weight = 1 & previous weight = 0 will be set to 0
            bbox_weights = bbox_weights[:, 1 : self.num_query, :3]
            bbox_weights_delta = (
                bbox_weights[:, 1 : self.num_query - 1, :]
                - bbox_weights[:, 0 : self.num_query - 2, :]
            )
            index = list(torch.where(bbox_weights_delta > 0))
            index[1] = index[1] + 1
            bbox_weights[tuple(index)] = 0

            bbox_preds_delta = bbox_preds_delta.reshape(-1, 3)
            bbox_targets_delta = bbox_targets_delta.reshape(-1, 3)
            bbox_weights = bbox_weights.reshape(-1, 3)

            loss_redist = self.loss_bbox(bbox_preds_delta, bbox_targets_delta, bbox_weights)
            loss_bbox += loss_redist

        # regression L1 loss on angle
        angl_preds = angl_preds.reshape(-1, 3)
        loss_angle = self.loss_angle(
            angl_preds, angl_targets, angl_weights, avg_factor=num_total_pos
        )

        return loss_cls, loss_bbox, loss_angle

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        angl_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_angles_list,
        img_metas,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, cz, w, h, d) and shape [num_query, 6].
            angl_preds_list (list[Tensor]): Outputs from the angle regression head
                (alpha, beta, gamma), with shape [bs, num_query, 3].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 6) in [cx, cy, cz, w, h, d] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_angles_list (list[Tensor]): Ground truth angles for each image with shape
                (num_gts, 3) in [alpha, beta, gamma] format.
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            angl_targets_list,
            angl_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            angl_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_angles_list,
            img_metas,
            gt_bboxes_ignore_list,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            angl_targets_list,
            angl_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        cls_score,
        bbox_pred,
        angl_pred,
        gt_bboxes,
        gt_labels,
        gt_angles,
        img_meta,
        gt_bboxes_ignore=None,
    ):
        """ "Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, cz, w, h, d) and
                shape [num_query, 6].
            angl_pred (Tensor): Outputs from the angle regression head
                (alpha, beta, gamma), with shape [num_query, 3].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 6) in [cx, cy, cz, w, h, d] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_angles (Tensor): Ground truth angles for each image with shape
                (num_gts, 3) in [alpha, beta, gamma] format.
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(
            bbox_pred=bbox_pred,
            cls_pred=cls_score,
            angle_pred=angl_pred,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_angles=gt_angles,
            img_meta=img_meta,
            gt_bboxes_ignore=gt_bboxes_ignore,
        )
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if self.loss_cls.use_sigmoid:
            labels[pos_inds] = gt_labels.new_ones(len(pos_inds))
            # hard code
            labels = labels.unsqueeze(1)
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_w, img_h, img_d = img_meta["img_shape"]

        # DETR regress the relative position of boxes (cxcyczwhd) in the image.
        factor = bbox_pred.new_tensor([img_w, img_h, img_d, img_w, img_h, img_d]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        bbox_targets[pos_inds] = pos_gt_bboxes_normalized

        # angle targets
        angl_targets = torch.zeros_like(angl_pred)
        angl_weights = torch.zeros_like(angl_pred)
        angl_weights[pos_inds] = 1.0
        angl_targets[pos_inds] = gt_angles[sampling_result.pos_assigned_gt_inds] / (2 * math.pi)

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            angl_targets,
            angl_weights,
            pos_inds,
            neg_inds,
        )

    @force_fp32(apply_to=("all_cls_scores", "all_bbox_preds", "all_angl_preds"))
    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        all_angl_preds,
        img_metas,
        query_index=None,
        rescale=False,
    ):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs for each feature level. Each is a 6D-tensor with
                normalized coordinate format (cx, cy, cz, w, h, d) and shape
                [nb_dec, bs, num_query, 6].
            all_angl_preds (Tensor): Outputs from the angle regression head
                (alpha, beta, gamma), with shape [num_query, 3].
            img_metas (list[dict]): Meta information of each image.
            query_index (list): Indices for which to query
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            det_bboxes: Predicted bboxes with shape [num_query, 11], \
                    where the first 6 columns are bounding box positions \
                    (cx, cy, cz, w, h, d), the 7th-9th columns are bounding box orientations \
                    (alpha, beta, gamma), the 10-th column are predicted labels, and the last \
                    column are scores between 0 and 1.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]
        angl_preds = all_angl_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            angl_pred = angl_preds[img_id]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            proposals = self._get_bboxes_single(
                cls_score, bbox_pred, angl_pred, img_shape, scale_factor, query_index, rescale,
            )
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(
        self,
        cls_score,
        bbox_pred,
        angl_pred,
        img_shape,
        scale_factor,
        query_index=None,
        rescale=False,
        sort=False,
    ):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, cz, w, h, d) and
                shape [num_query, 6].
            angl_pred (Tensor): Outputs from the angle regression head
                (alpha, beta, gamma), with shape [num_query, 3].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            query_index (list): Indices for which to query
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.
            sort (str): whether to sort results by score

        Returns:
            Tensor: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 11], \
                    where the first 6 columns are bounding box positions \
                    (cx, cy, cz, w, h, d), the 7th-9th columns are bounding box orientations \
                    (alpha, beta, gamma), the 10-th column are predicted labels, and the last \
                    column are scores between 0 and 1.
        """
        cls_score = cls_score.clone().detach()
        bbox_pred = bbox_pred.clone().detach()
        angl_pred = angl_pred.clone().detach()
        assert len(cls_score) == len(bbox_pred)

        max_per_img = self.test_cfg.get("max_per_img", self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            scores = cls_score.sigmoid().squeeze()
            if query_index is not None:
                indices = cls_score.new_tensor(query_index, dtype=torch.long)
                det_labels = torch.where(
                    scores > 0.5, indices, scores.new_tensor(0, dtype=torch.long)
                )
            else:
                indices = torch.arange(scores.shape[0]).to(scores.device)
                det_labels = torch.where(
                    scores > 0.5, indices + 1, scores.new_tensor(0, dtype=torch.long)
                )
            if sort:
                scores, indices = scores.topk(max_per_img)
                det_labels = det_labels[indices]
                bbox_pred = bbox_pred[indices]
                angl_pred = angl_pred[indices]
        else:
            cls_logits = F.softmax(cls_score, dim=-1)[..., :]
            scores, det_labels = cls_logits.max(-1)
            if query_index is not None:
                indices = cls_score.new_tensor(query_index, dtype=torch.long)
            else:
                indices = torch.arange(scores.shape[0]).to(scores.device)

            det_labels = torch.where((det_labels == 0) & (scores < 0.8), indices, det_labels)
            scores = torch.gather(cls_logits, dim=1, index=det_labels.unsqueeze(1)).squeeze()

            if sort:
                scores, bbox_index = scores.topk(max_per_img)
                bbox_pred = bbox_pred[bbox_index]
                det_labels = det_labels[bbox_index]
                angl_pred = angl_pred[bbox_index]

        det_bboxes = bbox_pred
        det_bboxes[:, 0::3] = det_bboxes[:, 0::3] * img_shape[0]
        det_bboxes[:, 1::3] = det_bboxes[:, 1::3] * img_shape[1]
        det_bboxes[:, 2::3] = det_bboxes[:, 2::3] * img_shape[2]

        det_bboxes[:, 0::3].clamp_(min=0, max=img_shape[0])
        det_bboxes[:, 1::3].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 2::3].clamp_(min=0, max=img_shape[2])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)

        angl_pred *= 2 * math.pi
        detections = torch.cat(
            (det_bboxes, angl_pred, det_labels.unsqueeze(1), scores.unsqueeze(1)), -1
        )

        return detections
