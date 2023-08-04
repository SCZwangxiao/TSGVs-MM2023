import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder


class BaseStreamingPredictor(nn.Module):
    """Base class for predictor in temporal video grounding methods.

    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, short_memories, short_memory_masks,
                future_memories, future_memory_masks, **kwargs):
        """Define the computation performed at every call."""