import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder
from ..builder import VIDEO_ENCODERS
from ..common import get_activation_nn


@VIDEO_ENCODERS.register_module()
class BaseVideoEncoder(nn.Module):
    """Base class for video encoder in temporal video grounding methods.
    
    If ``linear_before`` is ``True``, the video features will be feed to 
    a linear transform layer. If ``norm_after`` is ``True``, the output 
    features of ``_encoder()`` will be feed to short/long/tuture ``LayerNorm`` 
    layers.

    All video encoder methods should subclass it. All subclass
    should overwrite: Methods:``_encoder()``.

    Args:
        video_dim (int): Feature dimension of video.
        hidden_dim (int): Feature dimension of hidden multi-modal space.
        linear_before (bool): Whether to apply linear transform before _encoder.
            Default True.
        norm_after (bool): Whether to apply ``LayerNorm`` after the _encoder.
            Default True.
    """
    def __init__(self, 
                 video_dim, 
                 hidden_dim, 
                 linear_before=True,
                 linear_options=dict(dropout_rate=0.0,
                                     norm_before=True,
                                     activation='gelu'),
                 norm_after=False):
        super().__init__()
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.linear_before = linear_before
        self.norm_after = norm_after

        if linear_before:
            dropout_rate = linear_options.get('dropout_rate', 0.0)
            norm_before = linear_options.get('norm_before', True)
            activation = linear_options.get('activation', 'gelu')
            Norm = nn.LayerNorm(hidden_dim) if norm_before else nn.Identity()
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(video_dim, hidden_dim),
                Norm,
                get_activation_nn(activation)
            )
        if norm_after:
            self.norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def zero_padding_memories(memories, memory_masks):
        return torch.masked_fill(memories, memory_masks.unsqueeze(-1), .0)

    def before_encoder(self, video_features, video_masks):
        if self.linear_before:
            video_features = self.fc(video_features)
        if video_masks is not None:
            video_features = self.zero_padding_memories(
                video_features, video_masks)
        return video_features

    def after_encoder(self, video_features):
        if self.norm_after:
            video_features = self.norm(video_features)
        return video_features

    def forward(self, video_features, video_masks, video_length, **kwargs):
        """Define the computation performed at every call.

        The input future_memories should be either the real future
            frame sequence or ``None``.
        
        However, the outout future_memories could be the real future
            frame sequence, anticipated future frame sequence, or ``None``.``.
        
        Args:
            video_features (torch.Tensor): The input video features.
            video_masks (torch.Tensor): The mask of input video features.
        """
        video_features = self.before_encoder(video_features, video_masks)
        video_features = self._encoder(
            video_features=video_features, video_masks=video_masks, 
            video_length=video_length, **kwargs)
        video_features = self.after_encoder(video_features)
        return video_features

    @abstractmethod
    def _encoder(self, video_features, video_masks, video_length, **kwargs):
        """Define the video encoder."""
        assert self.norm_after is False
        return video_features