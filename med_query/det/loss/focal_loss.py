# Copyright (c) DAMO Health

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union


class FocalLoss(nn.Module):
    """
    Reimplementation of the Focal Loss described in:
        - "Focal Loss for Dense Object Detection", T. Lin et al., ICCV 2017
    """

    def __init__(
        self,
        class_num,
        alpha: Union[List, Tuple, None] = None,
        gamma: float = 2.0,
        size_average=True,
    ) -> None:
        """
        :param class_num: number of classes, including background class
        :param alpha: a list or tuple containing the weights for each class,
                      weights for all classes would be the same if alpha is not specified
        :param gamma: value of the exponent gamma in the definition of the Focal loss.
        :param size_average: whether to return averaged mini-batch loss

        Example:
            >>> import torch
            >>> import torch.nn.functional as F
            >>> from med_base.loss.focal_loss import FocalLoss
            >>> pred = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.float32)
            >>> grnd = torch.tensor([[0], [1], [0]], dtype=torch.int64)
            >>> fl = FocalLoss(class_num=2)
            >>> res = fl(pred, grnd)
            >>> print(res)
        """
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1) / class_num
        else:
            assert len(alpha) == class_num
            self.alpha = torch.FloatTensor(alpha)
            self.alpha.unsqueeze_(1)
            self.alpha = self.alpha / self.alpha.sum()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.one_hot_codes = torch.eye(self.class_num)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        assume the input y_pred has one of the following shapes:
        1. [sample, class_num]
        2. [batch, class_num, dim_y, dim_x]
        3. [batch, class_num, dim_z, dim_y, dim_x]
        assume the target y_true has one of the following shapes,
        which corresponds to the shapes of the input:
        1. [sample, 1] or [sample, ]
        2. [batch, 1, dim_y, dim_x]
        3. [batch, 1, dim_z, dim_y, dim_x]

        :param y_pred: prediction tensor, should be logits without normalization.
        :param y_true: ground truth tensor.
        :return: mean or sum of losses of batch elements.
        """

        assert y_pred.dim() == 2 or y_pred.dim() == 4 or y_pred.dim() == 5
        assert y_pred.dim() == y_true.dim(), "the dims of y_pred and y_true should be the same"
        if y_pred.dim() == 4:
            y_pred = y_pred.permute(0, 2, 3, 1).contiguous()
            y_pred = y_pred.view(-1, self.class_num)
            y_true = y_true.permute(0, 2, 3, 1).contiguous()
            y_true = y_true.view(-1, 1)
        elif y_pred.dim() == 5:
            y_pred = y_pred.permute(0, 2, 3, 4, 1).contiguous()
            y_pred = y_pred.view(-1, self.class_num)
            y_true = y_true.permute(0, 2, 3, 4, 1).contiguous()
            y_true = y_true.view(-1, 1)

        self.alpha = self.alpha.to(y_pred.device)
        self.one_hot_codes = self.one_hot_codes.to(y_pred.device)

        y_true = y_true.long().view(-1)

        mask = self.one_hot_codes[y_true]
        mask.requires_grad_(False)

        alpha = self.alpha[y_true]
        alpha.requires_grad_(False)

        y_pred = F.softmax(y_pred, dim=1)
        probs = (y_pred * mask).sum(1).view(-1, 1)
        probs = torch.maximum(probs.new_tensor(1e-3), probs)
        probs = torch.minimum(probs.new_tensor(1 - 1e-3), probs)
        log_probs = probs.log()

        if self.gamma > 0:
            batch_loss = -alpha * torch.pow((1 - probs), self.gamma) * log_probs
        else:
            batch_loss = -alpha * log_probs

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
