# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

from .. import builder
from ..builder import LOCALIZERS


class BaseLocalizer(nn.Module, metaclass=ABCMeta):
    """Base class for temporal sentence grounding in videos.
    All localizers should subclass it.
    All subclass should overwrite:
    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.
    Args:
        sentence_encoder (dict): Module for sentence encoding.
        video_encoder (dict): Module for video encoding.
        interactor (dict): Module for multi-modal interection.
        predictor (dict): Moulde for localize moments.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Defines the computation performed at training."""

    @abstractmethod
    def forward_test(self, *args, **kwargs):
        """Defines the computation performed at testing."""

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, *args, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(*args, **kwargs)
        return self.forward_test(*args, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        aux_info = {}
        for item in self.train_aux_info:
            assert item in data_batch
            aux_info[item] = data_batch.pop(item)
        
        losses = self.forward(**data_batch, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        aux_info = {}
        for item in self.test_aux_info:
            assert item in data_batch
            aux_info[item] = data_batch.pop(item)
        results = self.forward(return_loss=False, **data_batch, **aux_info)

        outputs = dict(results=results)

        return outputs


@LOCALIZERS.register_module()
class BaseTSGV(BaseLocalizer):
    """Base class for temporal sentence grounding methods.
    
    All temporal sentence grounding methods should subclass it. 
    All subclass should overwrite: 

    - Methods:``forward_train``, supporting to forward when training. 
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        sentence_encoder (dict): Sentence encoding.
        video_encoder (dict): Video encoding.
        interactor (dict): Multi-modal interaction. Default: None.
        predictor (dict): Prediction.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """
    def __init__(self,
                 sentence_encoder,
                 video_encoder,
                 interactor,
                 predictor,
                 train_aux_info=None,
                 test_aux_info=None):
        super().__init__()
        self.sentence_encoder = builder.build_sentence_encoder(sentence_encoder)
        self.video_encoder = builder.build_video_encoder(video_encoder)
        self.interactor = builder.build_interactor(interactor)
        self.predictor = builder.build_predictor(predictor)

        self.train_aux_info = []
        self.test_aux_info = []
        if train_aux_info is not None:
            self.train_aux_info = train_aux_info
        if test_aux_info is not None:
            self.test_aux_info = test_aux_info

    @staticmethod
    def merge_aux_info(aux_info, kwargs):
        if len(aux_info):
            aux_info = aux_info[0]
            assert type(aux_info) == dict
        else:
            aux_info = {}
        kwargs.update(aux_info)
        return kwargs

    def forward_train(self, sentence_features, sentence_masks, sentence_length, 
                      video_features, video_masks, video_length,
                      label, **train_aux_info):
        """Defines the computation performed at training.

        Note that the output future_memories of ``video_encoder()`` should be 
            the real future frame sequence, an anticipated one (OadTR, LSTR),
            or ``None``.
            However, the output future_memories of ``interactor()`` must be either
            an anticipated future frame sequence or ``None``.
        """
        # Sentence Encoding
        sentence_features = self.sentence_encoder(sentence_features=sentence_features, 
                                                  sentence_masks=sentence_masks,
                                                  sentence_length=sentence_length, 
                                                  **train_aux_info)
        # Video Encoding
        video_features = self.video_encoder(video_features=video_features,
                                            video_masks=video_masks, 
                                            video_length=video_length,
                                            **train_aux_info)
        # Multi-modal Interaction
        video_features, *aux_info = \
            self.interactor(sentence_features=sentence_features, 
                            sentence_masks=sentence_masks,
                            sentence_length=sentence_length,
                            video_features=video_features, 
                            video_masks=video_masks, 
                            video_length=video_length,
                            **train_aux_info)
        train_aux_info = self.merge_aux_info(aux_info, train_aux_info)
        # Prediction
        pred_logits, *aux_info = \
            self.predictor(sentence_features=sentence_features, 
                           sentence_masks=sentence_masks,
                           sentence_length=sentence_length,
                           video_features=video_features, 
                           video_masks=video_masks,
                           video_length=video_length,
                           **train_aux_info)
        train_aux_info = self.merge_aux_info(aux_info, train_aux_info)
        # Loss
        losses = self.predictor.loss(pred_logits=pred_logits, 
                                     label=label,
                                     video_masks=video_masks,
                                     video_length=video_length,
                                     **train_aux_info)
        return losses

    def forward_test(self, sentence_features, sentence_masks, sentence_length, 
                     video_features, video_masks, video_length, **test_aux_info):
        """Defines the computation performed at testing."""
        # Sentence Encoding
        sentence_features = self.sentence_encoder(sentence_features=sentence_features, 
                                                  sentence_masks=sentence_masks,
                                                  sentence_length=sentence_length, 
                                                  **test_aux_info)
        # Video Encoding
        video_features = self.video_encoder(video_features=video_features,
                                            video_masks=video_masks, 
                                            video_length=video_length,
                                            **test_aux_info)
        # Multi-modal Interaction
        video_features, *aux_info = \
            self.interactor(sentence_features=sentence_features,
                            sentence_masks=sentence_masks, 
                            sentence_length=sentence_length,
                            video_features=video_features, 
                            video_masks=video_masks, 
                            video_length=video_length,
                            **test_aux_info)
        test_aux_info = self.merge_aux_info(aux_info, test_aux_info)
        # Prediction
        pred_scores, *aux_info = \
            self.predictor(sentence_features=sentence_features, 
                           sentence_masks=sentence_masks,
                           sentence_length=sentence_length,
                           video_features=video_features, 
                           video_masks=video_masks,
                           video_length=video_length,
                           **test_aux_info)
        test_aux_info = self.merge_aux_info(aux_info, test_aux_info)
        return pred_scores.cpu().numpy()

    def forward(self, sentence_features, video_features, 
                sentence_masks=None, sentence_length=None, 
                video_masks=None, video_length=None,
                label=None, return_loss=True, **aux_info):
        """Define the computation performed at every call.
        
        Args:
            sentence_features (torch.Tensor): Shape `(B, L, h_s)`. Containing 
                the zero-padded sentence feature.
            sentence_length (torch.Tensor): Shape `(B, 1)`. Containing the 
                length of each sentence.
            video_features (torch.Tensor):  Shape `(B, T, h)`. Containing 
                the zero-padded feature of short memory.
            video_masks (torch.Tensor): Shape `(B, T)`. Containing 
                the mask of invalid feature frames. Value ``True`` for invalid frames.
            label (torch.Tensor): Shape differs.
        """
        if return_loss:
            return self.forward_train(sentence_features, sentence_masks, sentence_length, 
                                      video_features, video_masks, video_length,
                                      label, **aux_info)
        return self.forward_test(sentence_features, sentence_masks, sentence_length, 
                                 video_features, video_masks, video_length, **aux_info)
