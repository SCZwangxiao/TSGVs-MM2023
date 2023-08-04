import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder


class BasePredictor(nn.Module):
    """Base class for predictor in temporal video grounding methods.

    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, sentence_features, sentence_length,
                video_features, video_masks, **kwargs):
        """Define the computation performed at every call."""

    @abstractmethod
    def loss(self, pred_logits, video_masks, label, **kwargs):
        """Define the computation performed at every call."""