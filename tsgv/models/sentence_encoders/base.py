import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder
from ..builder import SENTENCE_ENCODERS
from ..common import get_activation_nn


@SENTENCE_ENCODERS.register_module()
class BaseSentenceEncoder(nn.Module):
    """Base class for sentence encoder in temporal sentence grounding methods.
    
    All sentence encoder methods should subclass it. All subclass
    should overwrite: Methods:``_encoder``.

    Args:
        sent_dim (int): Feature dimension of sentence.
        hidden_dim (int): Feature dimension of hidden multi-modal space.
        pool_strategy ('Mean' | None): Pooling strategy for sentence features.
            Default None.
        linear_before (bool): Whether to use linear transformation before ``_encoder()``.
            Default True.
        norm_after (bool): Whether to add ``LayerNorm`` after ``_encoder()``.
            Default True.
    """
    def __init__(self, 
                 sent_dim, 
                 hidden_dim, 
                 pool_strategy=None, 
                 linear_before=True,
                 linear_options=dict(dropout_rate=0.0,
                                     norm_before=True,
                                     activation='gelu'),
                 norm_after=False):
        super().__init__()
        self.sent_dim = sent_dim
        self.hidden_dim = hidden_dim
        self.pool_strategy = pool_strategy
        self.linear_before = linear_before
        self.norm_after = norm_after

        if pool_strategy and pool_strategy not in ['Mean', 'Last']:
            raise NotImplementedError
        if linear_before:
            dropout_rate = linear_options.get('dropout_rate', 0.0)
            norm_before = linear_options.get('norm_before', True)
            activation = linear_options.get('activation', 'gelu')
            Norm = nn.LayerNorm(hidden_dim) if norm_before else nn.Identity()
            self.before_encoder = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(sent_dim, hidden_dim),
                Norm,
                get_activation_nn(activation)
            )
        else:
            self.before_encoder = nn.Identity()
        if norm_after:
            self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sentence_features, sentence_length, sentence_masks=None, **kwargs):
        """Define the computation performed at every call.
        
        Args:
            sentence_features (torch.Tensor): The input sentence features.
            sentence_length (torch.LongTensor): The length of input sentence features.
        """
        if sentence_masks is None:
            sentence_masks = self.setence_mask(sentence_length, sentence_features.size(1))
            # [B, maxL]
        if self.linear_before:
            sentence_features = self.before_encoder(sentence_features)
            sentence_features = torch.masked_fill(
                sentence_features, sentence_masks.unsqueeze(-1), .0)
            # [B, maxL, hidden_dim]
        sentence_features = self._encoder(sentence_features=sentence_features,
                                          sentence_masks=sentence_masks, 
                                          sentence_length=sentence_length, 
                                          **kwargs)
        # [B, maxL, hidden_dim]
        sentence_features = self.feature_pooling(
            sentence_features, sentence_masks, sentence_length)
        if self.norm_after:
            sentence_features = self.final_norm(sentence_features)
        # [B, *, hidden_dim]
        return sentence_features

    @staticmethod
    def setence_mask(sentence_length, maxlen, dtype=torch.bool):
        """Generate sentence mask according to sentence length.
            sentence_length [B, 1]
        """
        row_vector = torch.arange(0, maxlen, 1, device=sentence_length.device)
        # [maxL]
        mask = row_vector > sentence_length
        # [B, maxL]
        mask.type(dtype)
        return mask

    def feature_pooling(self, sentence_features, sentence_masks, sentence_length):
        """Pool the sentence sequence feature sequence to a vector, or keep the sequence, 
        according to ``pool_strategy``.
        """
        sentence_features = sentence_features.masked_fill(sentence_masks[:,:,None], .0)
        if self.pool_strategy == 'Mean':
            sentence_features = sentence_features.sum(1) / sentence_length
            # [B, hidden_dim]
        elif self.pool_strategy == 'Last':
            batch_indices = range(sentence_features.size(0))
            length_indices = sentence_length.squeeze(-1).long() - 1
            # [B]
            sentence_features = sentence_features[batch_indices, length_indices]
            # [B, hidden_dim]
        return sentence_features

    def _encoder(self, sentence_features, sentence_masks, sentence_length, **kwargs):
        """Define the sentence encoder."""
        return sentence_features