import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder
from ..builder import VIDEO_ENCODERS


@VIDEO_ENCODERS.register_module()
class BaseStreamingVideoEncoder(nn.Module):
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
        use_long (bool): Whether to process long-term memory.
            Default True.
        use_future (bool): Whether to process future memory.
            Default False.
        linear_before (bool): Whether to apply linear transform before _encoder.
            Default True.
        norm_after (bool): Whether to apply ``LayerNorm`` after the _encoder.
            Default True.
    """
    def __init__(self, 
                 video_dim, 
                 hidden_dim, 
                 use_long=True,
                 use_future=False,
                 linear_before=True,
                 norm_after=False):
        super().__init__()
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.use_long = use_long
        self.use_future = use_future
        self.linear_before = linear_before
        self.norm_after = norm_after

        if linear_before:
            self.fc = nn.Sequential(
                nn.Linear(video_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        if norm_after:
            self.short_norm = nn.LayerNorm(hidden_dim)
            if self.use_long:
                self.long_norm = nn.LayerNorm(hidden_dim)
            if self.use_future:
                self.future_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def zero_padding_memories(memories, memory_masks):
        return torch.masked_fill(memories, memory_masks.unsqueeze(-1), .0)

    def before_encoder(self, short_memories, short_memory_masks, 
                       long_memories, long_memory_masks,
                       future_memories, future_memory_masks):
        if self.linear_before:
            short_memories = self.fc(short_memories)
            if self.use_long:
                long_memories = self.fc(long_memories)
            if self.use_future and future_memories is not None:
                # If use_future==True, the future_memories could also be None.
                # In such circurmustance, the video encoder anticipates the future.
                future_memories = self.fc(future_memories)
        short_memories = self.zero_padding_memories(short_memories, 
                                                    short_memory_masks)
        if self.use_long:
            long_memories = self.zero_padding_memories(long_memories, 
                                                       long_memory_masks)
        if self.use_future and future_memories is not None:
            future_memories = self.zero_padding_memories(future_memories, 
                                                         future_memory_masks)
        return short_memories, long_memories, future_memories

    def after_encoder(self, short_memories, long_memories, future_memories):
        if self.norm_after:
            short_memories = self.short_norm(short_memories)
            if self.use_long:
                long_memories = self.long_norm(long_memories)
            if self.use_future:
                future_memories = self.future_norm(future_memories)
        return short_memories, long_memories, future_memories

    def forward(self, short_memories, short_memory_masks, 
                long_memories, long_memory_masks,
                future_memories, future_memory_masks,
                **kwargs):
        """Define the computation performed at every call.

        The input future_memories should be either the real future
            frame sequence or ``None``.
        
        However, the outout future_memories could be the real future
            frame sequence, anticipated future frame sequence, or ``None``.``.
        
        Args:
            short_memories (torch.Tensor): The input video features.
            short_memory_masks (torch.Tensor): The mask of input video features.
        """
        short_memories, long_memories, future_memories = \
            self.before_encoder(short_memories, short_memory_masks, 
                                long_memories, long_memory_masks,
                                future_memories, future_memory_masks)
        short_memories, long_memories, future_memories, *aux_info = \
            self._encoder(short_memories=short_memories, 
                          short_memory_masks=short_memory_masks, 
                          long_memories=long_memories, 
                          long_memory_masks=long_memory_masks,
                          future_memories=future_memories, 
                          future_memory_masks=future_memory_masks, 
                          **kwargs)
        short_memories, long_memories, future_memories = \
            self.after_encoder(short_memories, long_memories, future_memories)
        if len(aux_info):
            return short_memories, long_memories, future_memories, aux_info[0]
        else:
            return short_memories, long_memories, future_memories

    @abstractmethod
    def _encoder(self, short_memories, short_memory_masks, 
                 long_memories, long_memory_masks,
                 future_memories, future_memory_masks, **kwargs):
        """Define the video encoder."""
        assert self.norm_after is False
        return short_memories, long_memories, future_memories