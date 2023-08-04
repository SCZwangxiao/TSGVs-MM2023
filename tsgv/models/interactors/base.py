import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder


class BaseInteractor(nn.Module, metaclass=ABCMeta):
    """Base class for multi-modal interactor in temporal sentence grounding methods.
    
    All sentence encoder methods should subclass it. All subclass
    should overwrite: Methods:``forward``.

    """

    @abstractmethod
    def forward(self, sentence_features, sentence_length,
                video_features, video_masks,
                **kwargs):
        """Define the computation performed at every call.
        
        Args:
            sentence_features (torch.Tensor): The encoded sentence features.
            sentence_len (torch.LongTensor): The length of input sentence features.
            video_features (torch.Tensor): The encoded video features.
            video_masks (torch.Tensor): The mask of input video features.
        """