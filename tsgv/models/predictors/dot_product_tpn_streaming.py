import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed

from ..builder import PREDICTORS
from .base_streaming import BaseStreamingPredictor
from ..losses import BCELossWithLogits
from ...core.evaluation import average_precision


class MLPHead(nn.Module):
    def __init__(self, hidden_dim, num_mlp_layers, dropout=0.2, video_only=False):
        super().__init__()
        self.video_only = video_only

        self.video_mlp = self.build_mlps(
            num_mlp_layers, hidden_dim, dropout)
        if not video_only:
            self.sentence_mlp = self.build_mlps(
                num_mlp_layers, hidden_dim, dropout)

    @staticmethod
    def build_mlps(num_mlp_layers, hidden_dim, dropout):
        mlps = []
        for _ in range(num_mlp_layers-1):
            mlps.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()])
        mlps.extend([
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3*hidden_dim)])
        # mlps.append(nn.Linear(hidden_dim, 3*hidden_dim))
        return nn.Sequential(*mlps)

    def forward(self, video_features, sentence_features):
        video_features = self.video_mlp(video_features)
        # [B, T, 3h]
        if not self.video_only:
            sentence_features = self.sentence_mlp(sentence_features)
            # [B, 3h]
            return video_features, sentence_features
        return video_features


@PREDICTORS.register_module()
class DotProductPredictorTPN(BaseStreamingPredictor):
    """Span based predictor in temporal video grounding methods.

    Args:
        hidden_dim (int): Feature dimension of hidden multi-modal space.
        sp_type ('MLP', 'Tied-LSTM', 'Conditioned-LSTM'): Type of span-based
            predictor.
            Default: 'MLP'.
        num_layers (int): Number of predictor layers. Default: 1.
        tau (int): Annealing temperature. Default: 32.
        dropout (float): Dropout ratio. Default: 0.2.
        NS_strategy ('local'|'dist'): Negative sampling strategy.
        FNE_strategy (None): False negative elimination strategy.
    """
    def __init__(self, 
                 hidden_dim, 
                 num_layers=1,
                 tau=16.0,
                 dropout=0.2, 
                 gamma=.0,
                 span_weight=[0.5, 0.5, 1.0],
                 tpn_type=None,
                 tpn_distillation_weight=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.dropout = dropout
        self.gamma = gamma
        self.register_buffer('span_weight', torch.tensor(span_weight))
        self.tpn_type = tpn_type
        self.tpn_distillation_weight = tpn_distillation_weight
        
        self.head = MLPHead(hidden_dim, num_layers, dropout)
        if tpn_type is not None:
            assert tpn_type in ['KD', 'twinnet']
            self.head_inv = MLPHead(hidden_dim, num_layers, dropout)
            if tpn_type == 'twinnet':
                self.kd_affine = nn.Linear(hidden_dim, hidden_dim)
                self.twinnet_weight = tpn_distillation_weight
                self.tpn_distillation_weight = .0
        self.bce_loss = BCELossWithLogits()

    def span_predictor(self, memories, sentence_features):
        B, T = memories.shape[:2]
        sentence_features = sentence_features.reshape(B, 3, self.hidden_dim)
        sentence_features = F.normalize(sentence_features, 2, -1)
        if len(memories.shape) == 3: # Normal
            memories = memories.reshape(B, T, 3, self.hidden_dim)
            memories = F.normalize(memories, 2, -1)
            logits = self.tau * torch.einsum('BTCh,BCh->BTC', memories, sentence_features)
        elif len(memories.shape) == 4: # Multi-view
            num_view = memories.shape[2]
            memories = memories.reshape(B, T, num_view, 3, self.hidden_dim)
            memories = F.normalize(memories, 2, -1)
            logits = self.tau * torch.einsum('BTkCh,BCh->BTkC', memories, sentence_features)
            logits = logits.max(2).values
        # [B, T, 3]
        return logits

    def forward(self, short_memories, short_memory_masks, sentence_features, **kwargs):
        """Define the computation performed at every call."""
        aux_info = {}
        # Short prediction
        short_memories_head, sentence_features_head = self.head(short_memories, sentence_features)
        span_logits = self.span_predictor(short_memories_head, sentence_features_head)
        # [b, T, 3]
        if self.training and self.tpn_type is not None:
            short_memories_tpn = kwargs.get('short_memories_tpn')
            short_memories_head_inv, sentence_features_head_inv = self.head_inv(
                short_memories_tpn, sentence_features)
            span_logits_inv = self.span_predictor(
                short_memories_head_inv, sentence_features_head_inv)
            # [b, T, 3]
            aux_info['span_logits_inv'] = span_logits_inv
            aux_info['short_memories_head'] = short_memories_head
            aux_info['short_memories_head_inv'] = short_memories_head_inv
        
        return span_logits, None, aux_info

    def span_loss(self, short_logits, span_labels, memory_masks, weigth=1.0):
        # [b, T, 3] [b, T, 3] [b, T]
        span_labels = span_labels.float()
        memory_masks = memory_masks.unsqueeze(-1)
        loss = self.bce_loss(short_logits, span_labels, reduction='none')
        # [b, T, 3]
        if self.gamma != 0:
            p = torch.abs(torch.sigmoid(short_logits) - span_labels).detach()
            p = torch.pow(p, self.gamma)
            loss = p * loss
        loss = torch.masked_select(loss, ~memory_masks).reshape(-1, 3)
        # [valid_sample_cnt, 3]
        loss = weigth * self.span_weight * loss.mean(0)
        # [3]
        loss_dict = dict(
            loss_s=loss[0],
            loss_e=loss[1],
            loss_se=loss[2])
        return loss_dict

    def loss(self, short_logits, short_memory_masks,
             start_label, end_label, semantic_label, 
             video_name=None, anno_framestamp=None, memory_framestamp=None, **kwargs):
        # Preprocessing
        span_labels = torch.stack([start_label, end_label, semantic_label], dim=-1)
        # [b, T, 3]

        # Init loss
        loss_dict = {}
        for k, v in kwargs.items():
            if 'loss' in k:
                loss_dict[k] = v
        
        # Short memory span Loss
        span_loss = self.span_loss(
            short_logits, span_labels, short_memory_masks, 1-self.tpn_distillation_weight)
        loss_dict.update(span_loss)

        # TPN Loss
        if self.tpn_type:
            # Inverted loss
            short_logits_inv = kwargs.get('span_logits_inv')
            span_loss_inv = self.span_loss(
                short_logits_inv, span_labels, short_memory_masks)
            for k, v in span_loss_inv.items():
                loss_dict[f'{k}_inv'] = v
            # KD loss
            if self.tpn_type == 'KD':
                short_score_inv = torch.sigmoid(short_logits_inv).detach()
                span_loss_kd = self.span_loss(
                    short_logits, short_score_inv, short_memory_masks, self.tpn_distillation_weight)
                loss_kd = sum(list(span_loss_kd.values()))
            elif self.tpn_type == 'twinnet':
                short_memories_head = kwargs.get('short_memories_head')
                short_memories_head_inv = kwargs.get('short_memories_head_inv')
                b, T = short_memories_head.shape[:2]
                distill_loss = ( self.kd_affine(short_memories_head.reshape(b, T, 3, -1)) - 
                    short_memories_head_inv.reshape(b, T, 3, -1).detach().clone() )
                distill_loss = torch.masked_select(distill_loss**2, ~short_memory_masks[:,:,None,None])
                loss_kd = self.twinnet_weight * distill_loss.mean()
            loss_dict['loss_kd'] = loss_kd
        
        # Average precision
        semantic_pred = torch.sigmoid(short_logits[:,-1,2]).detach().cpu().numpy()
        semantic_label = (semantic_label[:,-1] > 0.3).long()
        semantic_label = semantic_label.detach().cpu().numpy().astype(np.int32)
        AP, cAP = average_precision(semantic_pred, semantic_label)
        loss_dict['cAP'] = torch.tensor(cAP, device=short_logits.device)

        return loss_dict