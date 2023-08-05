import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_

from ..builder import TRANSFORMER, TRANSFORMER_LAYER_SEQUENCE, build_transformer_layer_sequence


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type="LN"), **kwargs):
        super(DetrTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = (
                build_norm_layer(post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
            )
        else:
            assert not self.pre_norm, (
                f"Use prenorm in " f"{self.__class__.__name__}," f"Please specify post_norm_cfg"
            )
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(DetrTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, post_norm_cfg=dict(type="LN"), return_intermediate=False, **kwargs):
        transformerlayers = kwargs.get("transformerlayers", None)
        num_layers = kwargs.get("num_layers", None)
        init_cfg = kwargs.get("init_cfg", None)
        super(DetrTransformerDecoder, self).__init__(transformerlayers, num_layers, init_cfg)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)


@TRANSFORMER.register_module()
class Transformer3D(BaseModule):
    """Implements the DETR transformer in 3D space.

    Following the official mmdet DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(Transformer3D, self).__init__(init_cfg=init_cfg)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.dim() > 1:
                xavier_init(m, distribution="uniform")
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed):
        """Forward function for `Transformer3D`.

        Args:
            x (Tensor): Input query with shape [bs, c, w, h, d] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, w, h, d].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, w, h, d].
        """
        bs, c, w, h, d = x.shape

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        x = x.view(bs, c, -1).permute(2, 0, 1)  # [bs, c, w, h, d] -> [w*h*d, bs, c]

        if pos_embed is not None:
            pos_embed = pos_embed.reshape(bs, c, -1).permute(2, 0, 1)
        # [num_query, dim] -> [num_query, bs, dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # query_embed = query_embed.permute(2, 0, 1)
        mask = mask.view(bs, -1)  # [bs, w, h, d] -> [bs, w*h*d]

        memory = self.encoder(
            query=x, key=None, value=None, query_pos=pos_embed, query_key_padding_mask=mask
        )

        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0).reshape(bs, c, w, h, d)

        return out_dec, memory


@TRANSFORMER.register_module()
class MultiScaleTransformer3D(Transformer3D):
    """Implements the MultiScaleDETR transformer.
    """

    def __init__(self, num_feature_levels=4, **kwargs):
        super(MultiScaleTransformer3D, self).__init__(**kwargs)
        self.num_feature_levels = num_feature_levels
        self.init_layers()

    def init_layers(self):
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

    def init_weights(self):
        super().init_weights()
        normal_(self.level_embeds)

    def forward(self, mlvl_feats, mlvl_masks, query_embed, mlvl_pos_embeds):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from different level, each with shape
                [bs, c, w, h, d] where c = embed_dims.
            mlvl_masks (list(Tensor)): The key_padding_mask from different level for encoder
                and decoder, each with shape [bs, h, w, d].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding of feats from different level,
                with shape [bs, c, w, h, d].

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, num_points]. where num_points = w1*h1*d1 + ... + wn*hn*dn
        """
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, w, h, d = feat.shape
            spatial_shapes.append((w, h, d))
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            if pos_embed is not None:
                pos_embed = pos_embed.flatten(2).transpose(1, 2)  # [bs, n_points, c]
                lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = self.level_embeds[lvl].view(1, 1, -1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)

        # format into [w*h*d, bs, embed_dims]
        feat_flatten = feat_flatten.permute(1, 0, 2)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)

        # [num_query, dim] -> [num_query, bs, dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # query_embed = query_embed.permute(2, 0, 1)

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
        )

        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=lvl_pos_embed_flatten,
            query_pos=query_embed,
            key_padding_mask=mask_flatten,
        )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.permute(1, 2, 0)

        return out_dec, memory
