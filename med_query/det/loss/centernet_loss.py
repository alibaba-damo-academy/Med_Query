# Copyright (c) DAMO Health

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuaternionProd(nn.Module):
    def __init__(self):
        super(QuaternionProd, self).__init__()

    def forward(self, q1, q2):
        """
        quaternion product forward
        Args:
            q1:  torch.float_tensor, shape [N, 4]
            q2:  torch.float_tensor, shape [N, 4]

        Returns: torch.float_tensor, shape [N, 4]

        """
        q1 = q1.view(-1, 4)
        q2 = q2.view(-1, 4)
        q1 = q1 / torch.linalg.norm(q1, dim=1, keepdim=True)
        q2 = q2 / torch.linalg.norm(q2, dim=1, keepdim=True)
        a, b, c, d = q1.split(1, dim=1)
        e, f, g, h = q2.split(1, dim=1)
        s = a * e - b * f - c * g - d * h
        v1 = b * e + a * f + c * h - d * g
        v2 = c * e + a * g + d * f - b * h
        v3 = d * e + a * h + b * g - c * f
        return torch.cat([s, v1, v2, v3], dim=1)


def modified_focal_loss(pred, gt):
    """
    focal loss copied from CenterNet, modified version focal loss
    change log: numeric stable version implementation
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-7 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, mask, target, reduction="sum"):
    loss = F.l1_loss(pred * mask, target * mask, reduction=reduction)
    loss = loss / (mask.sum() + 1e-4)
    return loss


def quaternion2direction_loss(pred, target, needed_index, axis_idx):
    """

    Args:
        pred:  pred tensor of quaternion
        mask:  the mask
        target: target x1_tensor

    Returns: quaternion loss

    """
    v0 = torch.tensor([0, 0, 0, 0], dtype=torch.float, device=pred.device)
    v0[axis_idx + 1] = 1
    qp = QuaternionProd()

    channel = pred.size(1)
    pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, channel)
    target = target.permute(0, 2, 3, 4, 1).contiguous().view(-1, channel)

    pred = torch.index_select(pred, 0, needed_index[:, 0])
    target = torch.index_select(target, 0, needed_index[:, 0])

    pred_conj = pred.clone()
    pred_conj[:, 1:] *= -1

    v1 = qp(qp(pred, v0), pred_conj)
    loss = 1 * needed_index.size(0) - F.cosine_similarity(v1, target, dim=1).sum()

    return loss, v1


def quaternion_loss(pred, batch_input):
    x1_tensor, z1_tensor = batch_input["x1_tensor"], batch_input["z1_tensor"]
    channel = x1_tensor.size(1)

    tmp_tensor = x1_tensor.clone().permute(0, 2, 3, 4, 1).contiguous().view(-1, channel)
    needed_index = torch.nonzero(tmp_tensor[:, 1].ne(0), as_tuple=False)

    direction_loss_x, x1 = quaternion2direction_loss(pred[2], x1_tensor, needed_index, 0)
    direction_loss_z, z1 = quaternion2direction_loss(pred[3], z1_tensor, needed_index, 2)
    dot_prod_loss = F.cosine_similarity(x1, z1, dim=1).abs().sum()
    y1 = torch.cross(z1[:, 1:], x1[:, 1:], dim=1)
    y1_tensor = batch_input["y1_tensor"]

    y1_tensor = y1_tensor.permute(0, 2, 3, 4, 1).contiguous().view(-1, channel)
    target_y = torch.index_select(y1_tensor, 0, needed_index[:, 0])[:, 1:]
    direction_loss_y = 1 * needed_index.size(0) - F.cosine_similarity(y1, target_y, dim=1).sum()

    a, b, c, d = 1, 1, 0.5, 0.5
    q_loss = a * direction_loss_x + b * direction_loss_z + c * direction_loss_y + d * dot_prod_loss
    return q_loss
