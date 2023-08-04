import math

import numpy as np
import torch
import torch.nn as nn

from ..builder import SENTENCE_ENCODERS
from .base import BaseSentenceEncoder


@SENTENCE_ENCODERS.register_module()
class RNNSentenceEncoder(BaseSentenceEncoder):
    """Recurrent neural network sentence encoder for TSGV methods.

    Args:
        sent_dim (int): Feature dimension of sentence.
        hidden_dim (int): Feature dimension of hidden multi-modal space.
        num_layers (int): Number of RNN layers.
        rnn_type (str): Type of recurrent neural network.
            Default: GRU
        dropout (floar): Dropout ratio of RNN. Default 0.2.
        bidirectional (bool): Whether to use bidirectional RNN.
            Default: True.
        pool_strategy (str | None): Pooling strategy for sentence features.
            Default: None.
        linear_before (bool): Whether to use linear transformation before ``_encoder()``.
            Default True.
        norm_after (bool): Whether to add ``LayerNorm`` after ``_encoder()``.
            Default True.
    """
    def __init__(self, 
                 sent_dim, 
                 hidden_dim, 
                 num_layers,
                 rnn_type='GRU',
                 dropout=0.2,
                 bidirectional=True,
                 bidirection_fusion='Mean',
                 pool_strategy=None,
                 linear_before=True,
                 norm_after=True):
        super().__init__(sent_dim=sent_dim, 
                         hidden_dim=hidden_dim, 
                         pool_strategy=pool_strategy, 
                         linear_before=linear_before, 
                         norm_after=norm_after)
        self.bidirectional = bidirectional
        self.bidirection_fusion = bidirection_fusion if bidirectional else 'Mean'
        self.rnn_indim = hidden_dim if self.linear_before else sent_dim
        if bidirectional and bidirection_fusion == 'Concat':
            hidden_dim = hidden_dim >> 1

        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.rnn_indim, 
                               hidden_dim, 
                               num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(self.rnn_indim, 
                              hidden_dim, 
                              num_layers,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidirectional)
        else:
            raise NotImplementedError

    def _encoder(self, sentence_features, sentence_length, **kwargs):
        """Define the sentence encoder."""
        B, maxL = sentence_features.shape[0:2]
        lengths = sentence_length.squeeze(-1).cpu()
        # [B]
        sent_emb_packed = nn.utils.rnn.pack_padded_sequence(
            sentence_features, lengths, batch_first=True, enforce_sorted=False)
        sent_emb_packed, _ = self.rnn(sent_emb_packed)
        sent_emb, _ = nn.utils.rnn.pad_packed_sequence(
            sent_emb_packed, batch_first=True, total_length=maxL)
        sent_emb = sent_emb.contiguous()
        # [B, maxL, D*hidden_dim]
        if self.bidirectional and self.bidirection_fusion == 'Mean':
            sent_emb = sent_emb.reshape(B, maxL, 2, self.hidden_dim)
            # [B, maxL, 2, hidden_dim]
            sent_emb = sent_emb.mean(2)
        # [B, maxL, hidden_dim]
        return sent_emb