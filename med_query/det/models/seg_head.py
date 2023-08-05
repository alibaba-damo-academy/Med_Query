import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional


def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [
            dim,
            context_dim // 2,
            context_dim // 4,
            context_dim // 8,
            context_dim // 16,
            context_dim // 64,
        ]
        self.lay1 = torch.nn.Conv3d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(4, dim)
        self.lay2 = torch.nn.Conv3d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(4, inter_dims[1])
        self.lay3 = torch.nn.Conv3d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(4, inter_dims[2])
        self.lay4 = torch.nn.Conv3d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(4, inter_dims[3])
        self.lay5 = torch.nn.Conv3d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(4, inter_dims[4])
        self.out_lay = torch.nn.Conv3d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv3d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv3d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv3d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # [bs, c, w, h, d], [bs, qs, n_heads, w, h, d] -> [bs * qs, c + n_heads, w, h, d]
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(
            x, size=cur_fpn.shape[-3:], mode="trilinear", align_corners=True
        )
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(
            x, size=cur_fpn.shape[-3:], mode="trilinear", align_corners=True
        )
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(
            x, size=cur_fpn.shape[-3:], mode="trilinear", align_corners=True
        )
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = self.out_lay(x)

        # [bs * qs, 1, W/2, H/2, D/2]
        return x


class MHAttentionMap(nn.Module):
    """This is a 3D attention module, which only returns the attention softmax"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv3d(
            k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), self.k_linear.bias
        )
        # q shape: [bs, num_query, hidden_dim] -> [bs, num_query, hs, hidden_dim/hs]
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        # k shape: [bs, hidden_dim, w, h, d] -> [bs, hs, hidden_dims/hs, w, h, d]
        kh = k.view(
            k.shape[0],
            self.num_heads,
            self.hidden_dim // self.num_heads,
            k.shape[-3],
            k.shape[-2],
            k.shape[-1],
        )
        weights = torch.einsum("bqnc, bncwhd->bqnwhd", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))

        # weights = torch.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = torch.sigmoid(weights)
        weights = self.dropout(weights)
        # return query spatial weights on multi heads
        return weights
