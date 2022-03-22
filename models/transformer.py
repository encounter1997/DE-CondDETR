# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

from util import box_ops
from util.misc import inverse_sigmoid

from detectron2.structures import Boxes
from detectron2.modeling.poolers import ROIPooler
from .ms_poolers import MSROIPooler


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 pooler_resolution=4, num_feature_levels=3,
                 ms_roi=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model,
                                          pooler_resolution=pooler_resolution,
                                          num_feature_levels=num_feature_levels,
                                          ms_roi=ms_roi)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, meta_info=None, ms_feats=None):
        # flatten NxCxHxW to HWxNxC
        meta_info['src_size'] = src.shape
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs, references, outputs_coord = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                     pos=pos_embed, query_pos=query_embed,
                                                     meta_info=meta_info, ms_feats=ms_feats)
        return hs, references, outputs_coord


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, d_model=256,
                 pooler_resolution=4, num_feature_levels=3, ms_roi=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        for layer_id in range(num_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None
        # Build RoI.
        self.ms_roi = ms_roi
        func_init_pooler = self._init_ms_box_pooler if self.ms_roi else self._init_box_pooler
        box_pooler = func_init_pooler(
            pooler_resolution=pooler_resolution,
            num_feature_levels=num_feature_levels,
        )
        self.box_pooler = box_pooler
        # hack implementation of iterative bbox refine, like cascade rcnn
        self.bbox_embed = None
        self.box_refine = False

    @staticmethod
    def _init_box_pooler(num_feature_levels=1, pooler_resolution=4, sampling_ratio=2, pooler_type="ROIAlignV2"):
        """
        Modified from Sparse RCNN project
        NOTE currently only consider single scale input
        """
        assert num_feature_levels in [1, 3, 4]
        all_stride = [8, 16, 32, 64]
        input_stride = [32] if num_feature_levels == 1 else all_stride[:num_feature_levels]
        pooler_scales = [1. / input_s for input_s in input_stride]
        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    @staticmethod
    def _init_ms_box_pooler(
            num_feature_levels=3,
            pooler_resolution=4,
            sampling_ratio=2,
            pooler_type="ROIAlignV2",
    ):
        """
        Customized MSROIPooler that crops roi feature from all feat levels
        for each bbox.
        It takes x: List[torch.Tensor], box_lists: List[Boxes] as inputs,
        and outputs List[torch.Tensor], each of shape (M, C, pooler_resolution_list[l], pooler_resolution_list[l])
        """
        assert num_feature_levels in [1, 3, 4]
        all_stride = [8, 16, 32, 64]
        input_strides = [32] if num_feature_levels == 1 else all_stride[:num_feature_levels]
        pooler_scales = [1. / s for s in input_strides]
        pooler_resolution_list = [pooler_resolution for _ in range(len(input_strides))] \
            if type(pooler_resolution) == int else pooler_resolution
        assert len(pooler_resolution_list) == len(input_strides)

        box_pooler = MSROIPooler(
            output_size=pooler_resolution_list,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def update_memory_with_roi(self, memory, pos, coord_norm, meta_info, ms_feats=None):
        """
        memory: (seq_len, bs, dim)
                output of transformer encoder, used as value in cross attn
        memory_key: (seq_len, bs, dim)
                memory + pos, used as key in cross attn
        """
        # adjust boxes to xyxy format, and input image size
        boxes = box_ops.box_cxcywh_to_xyxy(coord_norm)
        boxes = boxes * meta_info['size'].repeat(1, 2).unsqueeze(1)  # (bs, 100, 4)
        # warp as Boxes list (directly copied from Sparse RCNN)
        N, nr_boxes = boxes.shape[:2]
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(boxes[b]))
        # extract roi and reshape, note concate feat and pos and pool to increase parallel and decrease latency
        # todo: fix the extensive amount of views, permutes, and repeats !
        bs, c, h, w = meta_info['src_size']  # reshape memory
        memory_pos = torch.cat([memory, pos], dim=-1)  # (seq_len, bs, dim)
        memory_pos_2d = memory_pos.permute(1, 2, 0).view(bs, 2*c, h, w)

        if ms_feats is not None:
            assert len(ms_feats) in [3, 4], "currently only support 3 or 4 level multi-scale"
            assert ms_feats[2].shape == memory_pos_2d.shape
            ms_feats[2] = memory_pos_2d
            feat2pool = ms_feats
        else:
            feat2pool = [memory_pos_2d]
        output = self.box_pooler(feat2pool, proposal_boxes)  # box_pooler takes list as inputs
        if self.ms_roi:
            assert type(output) == list
            output_list = []
            for output_ in output:
                output_ = output_.flatten(2).view(bs, nr_boxes, 2 * c, -1)
                output_ = output_.permute(3, 1, 0, 2)  # (seq_len, nr_boxes, bs, c)
                output_list.append(output_)
            output = torch.cat(output_list, dim=0)
        else:
            output = output.flatten(2).view(bs, nr_boxes, 2*c, -1)
            output = output.permute(3, 1, 0, 2)  # (seq_len, nr_boxes, bs, c) roi crop order: (bs * nr_boxes)

        output, output_pos = torch.split(output, c, dim=-1)

        return output, output_pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                meta_info=None,  # dict containing: 'size', 'src_size'
                ms_feats=None,  # for roi align from multi-scale feat
                ):
        _, B, _ = memory.shape
        memory_full = memory
        pos_full = pos

        output = tgt

        intermediate = []
        intermediate_coords = []
        reference_points_before_sigmoid = self.ref_point_head(query_pos)    # [num_queries, batch_size, 2]
        reference_points = reference_points_before_sigmoid.sigmoid()  # seq_len first

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2]  # [num_queries, batch_size, 2]

            # For the first decoder layer, we do not apply transformation over p_s
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation  # seq_len first
            output = layer(output, memory, tgt_mask=tgt_mask,  # None
                           memory_mask=memory_mask,  # None
                           tgt_key_padding_mask=tgt_key_padding_mask,  # None
                           memory_key_padding_mask=memory_key_padding_mask,  # (bs, Nk)
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # prediction on normed feat, while cross attn and roi on original feat
            output_norm = self.norm(output)

            tmp = self.bbox_embed[layer_id](output_norm)  # seq_len first
            # note when use bbox_refine, coord_norm equals to the updated references (Same as DeformableDETR)
            if not self.box_refine:
                # get normalized coordinates prediction (range 0-1)
                # transpose happens in Transformer in DETR, while happens here in CondDETR
                tmp[..., :2] += reference_points_before_sigmoid
                coord_norm = tmp.sigmoid().transpose(0, 1)
            else:
                # hack implementation for iterative bounding box refinement
                # reference points are kept in seq_len first shape
                if layer_id == 0:
                    tmp[..., :2] += reference_points_before_sigmoid  # seq_len first
                    new_reference_points = tmp.sigmoid()  # reference points are kept in seq_len first shape
                else:
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()
                coord_norm = new_reference_points.transpose(0, 1)  # batch first

            if self.return_intermediate:
                intermediate.append(output_norm)
                intermediate_coords.append(coord_norm)

            # update output feature here by roi align  (Nk_new, B*Nq, dim)
            roi_bbox = coord_norm.detach()  # follow SparseRCNN
            # all tensors that should be cropped: memory, pos
            memory, pos = self.update_memory_with_roi(memory_full, pos_full, roi_bbox, meta_info, ms_feats=ms_feats)
            # make memory_key_padding_mask, which never contain non-image area
            memory_key_padding_mask = torch.zeros(
                (B, memory.shape[0]),
                dtype=memory_key_padding_mask.dtype, device=memory_key_padding_mask.device
            )

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output)

        # note reference_points is still seq_len first, but it is not used in our impl.
        if self.return_intermediate:
            # return [torch.stack(intermediate).transpose(1, 2), reference_points]
            return [torch.stack(intermediate).transpose(1, 2), reference_points, torch.stack(intermediate_coords)]

        return output_norm.unsqueeze(0), reference_points, coord_norm


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        num_queries, bs, n_model = q_content.shape
        hw = k_content.shape[0]  # may be four dimensional

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw = k_content.shape[0]  # may be four dimensional

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)

        # # proposals as batches
        # reshape q, k, v and memory_key_padding_mask. Make sure bs, nr_boxes order is matched
        Nq, Nk, dim = num_queries, hw, n_model  # in cond_detr, the dim of q and k is two times of v (result of concat)
        if len(k.shape) == 3:
            v = v.unsqueeze(1).repeat(1, Nq, 1, 1)
            k = k.unsqueeze(1).repeat(1, Nq, 1, 1)
            k_pos = k_pos.unsqueeze(1).repeat(1, Nq, 1, 1)
        # cond_detr ops
        k = k.view(hw, Nq, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, Nq, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=4).view(hw, Nq, bs, n_model * 2)
        # p2b
        q = q.reshape(1, Nq*bs, dim*2)
        k = k.reshape(Nk, Nq*bs, dim*2)  # (Nk, Nq, bs, c*2)  ==> (Nk, Nq*bs, c*2)
        v = v.reshape(Nk, Nq*bs, dim)
        memory_key_padding_mask = memory_key_padding_mask.unsqueeze(0).repeat(Nq, 1, 1).view(Nq*bs, Nk)

        tgt2 = self.cross_attn(query=q, key=k, value=v,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # b2p
        tgt2 = tgt2.view(Nq, bs, dim)
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        if self.normalize_before:
            raise NotImplementedError
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pooler_resolution=args.pool_res,
        num_feature_levels=args.num_feature_levels,
        ms_roi=args.ms_roi,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")