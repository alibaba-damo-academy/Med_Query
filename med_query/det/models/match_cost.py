# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) DAMO Health

import torch
import torch.nn.functional as F

from ..builder import MATCH_COST


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import \
         ... ClassificationCost
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class BBoxL1Cost3D:
    """BBoxL1Cost3D.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'cxcyczwhd' for DETR

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost3D()
         >>> bbox_pred = torch.rand(1, 6)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4, 3, 5], [1, 2, 3, 4, 5, 6]])
         >>> self(bbox_pred, gt_bboxes)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1.0, box_format="cxcyczwhd"):
        self.weight = weight
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, cz, w, h, d), which are all in range [0, 1]. Shape
                [num_query, 6].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (cx, cy, cz, w, h, d). Shape [num_gt, 6].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class CosineSimilarityCost:
    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, pred, target):
        pred = pred.unsqueeze(1).repeat(1, target.shape[0], 1)
        target = target.unsqueeze(0)
        cosine_cost = 1 - F.cosine_similarity(pred, target, dim=2)
        return cosine_cost * self.weight


@MATCH_COST.register_module()
class IndexCost:
    def __init__(self, num_classes, weight=1.0):
        self.weight = weight

        idx = torch.arange(num_classes)
        cost_mat = torch.zeros((num_classes, num_classes))

        for i in range(num_classes):
            cost_mat[i, i:num_classes] = idx[: num_classes - i]

        self.idx_cost = cost_mat + cost_mat.T

    def __call__(self, targets):
        self.idx_cost = self.idx_cost.to(targets.device)
        idx_cost = self.idx_cost[:, targets] * self.weight

        return idx_cost.to(targets.device)
