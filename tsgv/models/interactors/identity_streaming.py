import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import INTERACTORS
from .base_streaming import BaseStreamingInteractor


@INTERACTORS.register_module()
class IdentityInteractor(BaseStreamingInteractor):
    """Identity multi-modal interactor in temporal sentence grounding methods.

    Args:
        
    """
    def __init__(self):
        super().__init__()

    def forward(self, sentence_features, sentence_length,
                short_memories, short_memory_masks, 
                long_memories, long_memory_masks,
                future_memories, future_memory_masks,
                **kwargs):
        """Define the computation performed at every call.
        
        Args:
            sentence_features (torch.Tensor): The encoded sentence features.
            sentence_len (torch.LongTensor): The length of input sentence features.
            short_memories (torch.Tensor): The encoded video features.
            short_memory_masks (torch.Tensor): The mask of input video features.
        """
        return short_memories, future_memories, sentence_features
