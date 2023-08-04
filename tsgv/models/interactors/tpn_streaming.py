import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import INTERACTORS
from .base_streaming import BaseStreamingInteractor
from ..common import FixedPositionalEncoding, get_activation_fn


class DoNothingCompressor(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        pass

    def forward(self, input, query):
        return input


class LSTRCompressor(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.tokens = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.lstr_encoder = nn.TransformerDecoderLayer(d_model=hidden_dim, 
                                                       nhead=num_heads, 
                                                       dim_feedforward=hidden_dim<<1, 
                                                       dropout=dropout, 
                                                       activation="gelu", 
                                                       batch_first=True)
        
        nn.init.xavier_normal_(self.tokens)

    def forward(self, input, query):
        B = input.shape[0]
        feat = self.lstr_encoder(self.tokens.repeat(B, 1, 1), input)
        return feat


class TokenLearner(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout)
        )
        self.se_norm = nn.LayerNorm(hidden_dim)

        self.trn = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

    def forward(self, input, query):
        # [B, T, h]
        selected = self.input_mlp(input)
        # [B, T, n]
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, -1)
        # [B, n, T]
        feat = torch.einsum('...nt,...td->...nd', selected, input)
        feat = self.se_norm(feat)
        # [B, n, h]
        feat = self.trn(feat)
        return feat


class SCDMTokenLearner(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.build_tokenlearner(hidden_dim, num_tokens, dropout)
        self.build_scdm(hidden_dim, dropout)

    def build_tokenlearner(self, hidden_dim, num_tokens, dropout):
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout)
        )

    def build_scdm(self, hidden_dim, dropout):
        self.fc_feat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.w = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        self.fc_gamma = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_beta = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def tokenlearner(self, input):
        # [B, T, h]
        selected = self.input_mlp(input)
        # [B, T, n]
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, -1)
        # [B, n, T]
        feat_selected = torch.einsum('...nt,...td->...nd', selected, input)
        # [B, n, h]
        return feat_selected

    def scdm(self, feat, query):
        # [B, n, h] [B, L, h]
        attn = self.fc_feat(feat.unsqueeze(2)) + self.fc_query(query.unsqueeze(1))
        # [B, n, L, h]
        attn = self.w(attn).squeeze(-1)
        attn = F.softmax(attn, dim=-1)
        # [B, n, L]
        c = torch.bmm(attn, query)
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        # [B, n, h]
        feat = self.norm(feat + (gamma * feat + beta))
        return feat

    def forward(self, input, query):
        feat = self.tokenlearner(input)
        feat = self.scdm(feat, query)
        return feat


class CQATokenLearner(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.build_tokenlearner(hidden_dim, num_tokens, dropout)
        self.build_cqa(hidden_dim, dropout)

    def build_tokenlearner(self, hidden_dim, num_tokens, dropout):
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def build_cqa(self, hidden_dim, dropout):
        self.cqa_dropout = nn.Dropout(dropout)
        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ffn3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.ffn4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fusion = nn.LayerNorm(hidden_dim)

    def tokenlearner(self, input):
        # [B, T, h]
        selected = self.input_mlp(input)
        # [B, T, n]
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, -1)
        # [B, n, T]
        feat_selected = torch.einsum('...nt,...td->...nd', selected, input)
        # [B, n, h]
        return self.norm(feat_selected)

    def cqa(self, feat, query):
        # [B, n, h] [B, L, h]
        sim = torch.bmm(
            self.cqa_dropout(feat), 
            self.cqa_dropout(query.transpose(1, 2)))
        sim_r = torch.softmax(sim, dim=2)
        sim_c = torch.softmax(sim, dim=1)
        # [B, n, L]
        A = torch.bmm(sim_r, query)
        B = torch.einsum('bNl,bnl,bnh->bNh', sim_r, sim_c, feat)
        # [B, n, h]
        cat1 = self.ffn1(feat)
        cat2 = self.ffn2(A)
        cat3 = self.ffn3(feat*A)
        cat4 = self.ffn4(feat*B)
        feat = self.fusion(cat1+cat2+cat3+cat4)
        return feat

    def forward(self, input, query):
        feat = self.tokenlearner(input)
        feat = self.cqa(feat, query)
        return feat


class MultimodalTokenLearner(nn.Module):
    def __init__(self, hidden_dim, num_tokens, num_heads, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim

        self.build_tokenlearner(hidden_dim, num_tokens, dropout)

    def build_tokenlearner(self, hidden_dim, num_tokens, dropout):
        # Vision tokenlearner
        self.input_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens),
            nn.Dropout(dropout)
        )
        self.video_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim<<1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        # Linguistic tokenlearner
        self.fc_feat = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_query = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.w = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, num_tokens, bias=False)
        )
        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim<<1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim<<1, 1),
            nn.Tanh(),
            nn.GELU(),
            nn.Dropout(0.4)
        )

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

    def tokenlearner(self, input, query):
        # [B, T, h]
        selected = self.input_mlp(input)
        # [B, T, n]
        selected = selected.transpose(1, 2)
        selected = F.softmax(selected, -1)
        # [B, n, T] 
        feat_selected = torch.einsum('...nt,...td->...nd', selected, input)
        # [B, n, h]

        query = query.mean(1, keepdim=True)
        # [B, T, h] [B, 1, h]
        attn = self.fc_feat(input) + self.fc_query(query)
        # [B, T, h]
        attn = self.w(attn).transpose(1, 2)
        attn = F.softmax(attn, dim=-1)
        # [B, n, T]
        feat_query = torch.einsum('...nt,...td->...nd', attn, input)
        # [B, n, h]

        feat = self.norm(
            self.video_gate(input.mean(1, keepdim=True)) * feat_selected + 
            self.query_gate(query) * feat_query)
        
        return feat

    def forward(self, input, query):
        feat = self.tokenlearner(input, query)
        return feat


def _build_memory_compressor(memory_compressor, 
                             hidden_dim, 
                             num_compressor_tokens, 
                             num_heads, 
                             dropout):
    compressors = nn.ModuleList()
    if type(memory_compressor) == str:
        for num_token in num_compressor_tokens:
            compressors.append(
                globals()[memory_compressor](
                    hidden_dim=hidden_dim, num_tokens=num_token, 
                    num_heads=num_heads, dropout=dropout))
    else:
        raise NotImplementedError
    return compressors


class TemporalPincerDecoderLayer(nn.Module):
    def __init__(self, forward_dec, d_model, nhead, dim_feedforward = 2048, dropout = 0.1,
                 activation = F.relu,
                 layer_norm_eps = 1e-5, batch_first = False, norm_first = False,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TemporalPincerDecoderLayer, self).__init__()
        self.forward_dec = forward_dec
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                        **factory_kwargs)
        # Implementation of Feedforward model
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, short_memories_forward, short_memories_inverted, 
                long_memories, future_memories, causal_mask):
        if self.forward_dec:
            x = short_memories_forward + short_memories_inverted.flip(1)
        else:
            x = short_memories_inverted.flip(1)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ca_block(self.norm2(x), long_memories, future_memories)
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ca_block(x, long_memories, future_memories))

        return x

    # multihead self-attention block
    def _sa_block(self, x):
        x = self.self_attn(x, x, x,
                             need_weights=False)[0]
        return self.dropout1(x)

    # multihead cross-attention block
    def _ca_block(self, x, mem1, mem2):
        if self.forward:
            mem = torch.concat([mem1, mem2], dim=1)
        else:
            mem = mem2
        x = self.cross_attn(x, mem, mem,
                            need_weights=False)[0]
        return self.dropout2(x)


class TemporalPincerDecoder(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 num_decoder_layers,
                 num_heads,
                 feedforward_dim,
                 dropout,
                 forward_dec=True):
        super().__init__()
        self.forward_dec = forward_dec
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True)
        if forward_dec:
            self.forward_decoder = nn.TransformerDecoder(
                decoder_layer,
                num_decoder_layers-1)
        self.inverted_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_decoder_layers-1)
        self.tpn_decoder = TemporalPincerDecoderLayer(
            forward_dec,
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True)

    def forward(self, short_memories, long_memories, future_memories, 
                causal_mask, pos_encoding_fn):
        short_memories_forward = pos_encoding_fn(short_memories)
        if self.forward_dec:
            short_memories_forward = self.forward_decoder(
                short_memories_forward, 
                long_memories,
                tgt_mask=causal_mask)
        short_memories_inverted = pos_encoding_fn(short_memories.flip(1))
        short_memories_inverted = self.inverted_decoder(
            short_memories_inverted, 
            future_memories,
            tgt_mask=causal_mask)
        short_memories_tpn = self.tpn_decoder(
            short_memories_forward, short_memories_inverted,
            long_memories, future_memories, causal_mask)
        return short_memories_tpn


@INTERACTORS.register_module()
class TPNInteractor(BaseStreamingInteractor):
    """LSTR video encoder for TSGV methods.
    Please refer to ``Long Short-Term Transformer for Online Action Detection``.

    Args:
        hidden_dim (int): Feature dimension of hidden multi-modal space.
        feedforward_dim (int): Feedforward dimension in transformers.
        short_memory_length (int): Length of short memory feature sequence.
        long_memory_length (int): Length of long memory feature sequence.
        future_memory_length (int): Length of anticipated future memory sequence.
        num_compressor_tokens (list[int]): List of length of encoder tokens, i.e.,
            the ``n_i`` int the pape.
            Default: [16, 32, 32].
        num_decoder_layers (int): Number of decoder layers. 
            Default: 2.
        num_heads (int): Number of attention heads. Default: 8.
        dropout (floar): Dropout ratio. Default: 0.2.
        use_future (bool): Whether to process future memory.
            Default False.
        norm_after (bool): Whether to apply ``LayerNorm`` after the _encoder.
            Default True.
    """
    def __init__(self, 
                 hidden_dim, 
                 feedforward_dim,
                 short_memory_length,
                 long_memory_length,
                 future_memory_length=0,
                 memory_compressor='MultimodalTokenLearner',
                 num_compressor_tokens=[16, 32, 32],
                 num_decoder_layers=2,
                 num_heads=8,
                 dropout=0.2,
                 tpn_forward=True,
                 sent_norm_before=True,
                 future_usage=None):
        super().__init__()
        # ``norm_after=False`` because norm is already done by transformer.
        self.future_memory_length = future_memory_length
        self.tpn_forward = tpn_forward
        self.future_usage = future_usage
        self.sent_norm_before = sent_norm_before
        # Pre Normalization
        if sent_norm_before:
            self.sent_norm = nn.LayerNorm(hidden_dim)
        # Build encoder
        max_length = max(long_memory_length, short_memory_length+future_memory_length)
        self.pos_encoding = FixedPositionalEncoding(hidden_dim, 
                                                    batch_first=True, 
                                                    dropout=dropout, 
                                                    max_length=max_length)
        self.compressors = _build_memory_compressor(memory_compressor, 
                                                    hidden_dim, 
                                                    num_compressor_tokens, 
                                                    num_heads, 
                                                    dropout)
        # Build decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=feedforward_dim, 
            dropout=dropout, 
            activation="gelu", 
            batch_first=True)
        self.ordinary_decoder = nn.TransformerDecoder(decoder_layer,
                                                      num_decoder_layers)
        if future_usage is None:
            assert future_memory_length == 0
        else:
            assert future_memory_length > 0
            if future_usage == 'FutureAnticipation':
                self.future_token = nn.Parameter(torch.zeros(1, future_memory_length, hidden_dim))
            elif future_usage == 'TemporalPincer':
                self.tpn_decoder = TemporalPincerDecoder(hidden_dim,
                                                         num_decoder_layers,
                                                         num_heads,
                                                         feedforward_dim,
                                                         dropout,
                                                         self.tpn_forward)
            else:
                raise NotImplementedError

        self.init_weights()

    def init_weights(self):
        if self.future_usage == 'FutureAnticipation':
            nn.init.xavier_normal_(self.future_token)

    @ staticmethod
    def generate_causal_mask(M_s, device):
        mask = (torch.triu(torch.ones(M_s, M_s)) == 1).transpose(0, 1)
        mask = ~mask.bool()
        return mask.to(device)

    def forward(self, sentence_features, sentence_length,
                short_memories, short_memory_masks, 
                long_memories, long_memory_masks,
                future_memories, future_memory_masks,
                **kwargs):
        """Define the sentence encoder."""
        B, M_s = short_memories.shape[:2]
        M_l = long_memories.shape[1]
        M_f = self.future_memory_length
        aux_info = {}

        # Normalization
        if self.sent_norm_before:
            # [B, h]
            sentence_features = self.sent_norm(sentence_features)
        
        # Process long-term memory
        long_memories = self.pos_encoding(long_memories)
        # [B, M_l, h] 
        for encoder in self.compressors:
            long_memories = encoder(long_memories, sentence_features)
        # [B, n, h]

        # Process short-term memory
        causal_mask = self.generate_causal_mask(M_s, short_memories.device)
        if self.future_usage == 'FutureAnticipation':
            causal_mask = self.generate_causal_mask(M_s+M_f, short_memories.device)
            future_token = self.future_token.expand(B, -1, -1)
            short_memories = torch.cat(
                [short_memories, future_token],
                dim=1)
            # [B, M_s+M_f, h]
        short_memories = self.pos_encoding(short_memories)
        # [B, M_s+M_f, h]
        short_memories = self.ordinary_decoder(short_memories,
                                               long_memories,
                                               tgt_mask=causal_mask)
        # [B, M_s+M_f, h]
        if self.training:
            if self.future_usage == 'FutureAnticipation':
                short_memories, future_memories = short_memories[:,:M_s], short_memories[:,M_s:]
                # [B, M_s, h], [B, M_f, h]
            elif self.future_usage == 'TemporalPincer':
                future_memories = self.pos_encoding(future_memories.flip(1))
                for encoder in self.compressors:
                    future_memories = encoder(future_memories, sentence_features)
                # [B, n, h]
                short_memories_tpn = self.tpn_decoder(
                    short_memories, long_memories, future_memories, 
                    causal_mask, self.pos_encoding)
                aux_info['short_memories_tpn'] = short_memories_tpn
        
        # Sentence post processor
        if len(sentence_features.shape) == 3:
            sentence_features = sentence_features.mean(1)
        return short_memories, future_memories, sentence_features, aux_info
