# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np

from .base import BaseLocalizer
from .. import builder
from ..builder import LOCALIZERS


@LOCALIZERS.register_module()
class BaseStreamingTSGV(BaseLocalizer):
    """Base class for streaming temporal sentence grounding methods.
    
    All streaming temporal sentence grounding methods should subclass it. 
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

    def forward_train(self, sentence_features, sentence_length, 
                      short_memories, short_memory_masks,
                      long_memories, long_memory_masks,
                      future_memories, future_memory_masks,
                      start_label, end_label, semantic_label, 
                      future_start_label, future_end_label, future_semantic_label,
                      **train_aux_info):
        """Defines the computation performed at training.

        Note that the output future_memories of ``video_encoder()`` should be 
            the real future frame sequence, an anticipated one (OadTR, LSTR),
            or ``None``.
            However, the output future_memories of ``interactor()`` must be either
            an anticipated future frame sequence or ``None``.
        """
        # Sentence Encoding
        sentence_features = self.sentence_encoder(sentence_features=sentence_features, 
                                                  sentence_length=sentence_length, 
                                                  **train_aux_info)
        # [B, L, h]
        # Video Encoding
        short_memories, long_memories, future_memories, *aux_info = \
            self.video_encoder(short_memories=short_memories, 
                               short_memory_masks=short_memory_masks, 
                               long_memories=long_memories, 
                               long_memory_masks=long_memory_masks,
                               future_memories=future_memories, 
                               future_memory_masks=future_memory_masks,
                               **train_aux_info)
        # [B, T_s, h], [B, T_l, h], [B, T_f, h]
        train_aux_info = self.merge_aux_info(aux_info, train_aux_info)
        # Multi-modal Interaction
        short_memories, anticipated_future_memories, sentence_features, *aux_info = \
            self.interactor(sentence_features=sentence_features, 
                            sentence_length=sentence_length,
                            short_memories=short_memories, 
                            short_memory_masks=short_memory_masks, 
                            long_memories=long_memories, 
                            long_memory_masks=long_memory_masks,
                            future_memories=future_memories, 
                            future_memory_masks=future_memory_masks,
                            **train_aux_info)
        # [B, T_s, h], [B, T_f, h]
        train_aux_info = self.merge_aux_info(aux_info, train_aux_info)
        # Prediction
        short_logits, future_logits, *aux_info = \
            self.predictor(short_memories=short_memories, 
                           short_memory_masks=short_memory_masks,
                           anticipated_future_memories=anticipated_future_memories, 
                           future_memory_masks=future_memory_masks, 
                           sentence_features=sentence_features, 
                           **train_aux_info)
        train_aux_info = self.merge_aux_info(aux_info, train_aux_info)
        # [B, T_s, 3], [B, T_f, 3]
        # Loss
        losses = self.predictor.loss(short_logits=short_logits, 
                                     short_memory_masks=short_memory_masks,
                                     future_logits=future_logits, 
                                     future_memory_masks=future_memory_masks,
                                     sentence_features=sentence_features, 
                                     start_label=start_label, 
                                     end_label=end_label, 
                                     semantic_label=semantic_label, 
                                     future_start_label=future_start_label, 
                                     future_end_label=future_end_label, 
                                     future_semantic_label=future_semantic_label, **train_aux_info)
        return losses

    def forward_test(self, sentence_features, sentence_length, 
                     short_memories, short_memory_masks,
                     long_memories, long_memory_masks,
                     **test_aux_info):
        """Defines the computation performed at testing."""
        # Sentence Encoding
        sentence_features = self.sentence_encoder(sentence_features=sentence_features, 
                                                  sentence_length=sentence_length, 
                                                  **test_aux_info)
        # Video Encoding
        short_memories, long_memories, future_memories, *aux_info = \
            self.video_encoder(short_memories=short_memories, 
                               short_memory_masks=short_memory_masks, 
                               long_memories=long_memories, 
                               long_memory_masks=long_memory_masks,
                               future_memories=None, 
                               future_memory_masks=None,
                               **test_aux_info)
        test_aux_info = self.merge_aux_info(aux_info, test_aux_info)
        future_memory_masks = torch.zeros(short_memories.shape[:2], 
                                          device=short_memories.device).bool()
        # Multi-modal Interaction
        short_memories, anticipated_future_memories, sentence_features, *aux_info = \
            self.interactor(sentence_features=sentence_features, 
                            sentence_length=sentence_length,
                            short_memories=short_memories, 
                            short_memory_masks=short_memory_masks, 
                            long_memories=long_memories, 
                            long_memory_masks=long_memory_masks,
                            future_memories=future_memories, 
                            future_memory_masks=future_memory_masks,
                            **test_aux_info)
        test_aux_info = self.merge_aux_info(aux_info, test_aux_info)
        # Prediction
        short_logits, future_logits, *aux_info = \
            self.predictor(short_memories=short_memories, 
                           short_memory_masks=short_memory_masks,
                           anticipated_future_memories=anticipated_future_memories, 
                           future_memory_masks=future_memory_masks, 
                           sentence_features=sentence_features, 
                           **test_aux_info)
        test_aux_info = self.merge_aux_info(aux_info, test_aux_info)
        pred_scores = torch.sigmoid(short_logits[:,-1,:])
        return pred_scores.cpu().numpy().astype(np.float16) # save RAM

    def forward(self, sentence_features, sentence_length, 
                short_memories, short_memory_masks,
                long_memories=None, long_memory_masks=None,
                future_memories=None, future_memory_masks=None,
                start_label=None, end_label=None, semantic_label=None, 
                future_start_label=None, future_end_label=None, future_semantic_label=None,
                return_loss=True, **aux_info):
        """Define the computation performed at every call.
        
        Args:
            sentence_features (torch.Tensor): Shape `(B, L, h_s)`. Containing 
                the zero-padded sentence feature.
            sentence_length (torch.Tensor): Shape `(B, 1)`. Containing the 
                length of each sentence.
            short_memories (torch.Tensor):  Shape `(B, T_s, h_v)`. Containing 
                the zero-padded feature of short memory.
            short_memory_masks (torch.Tensor): Shape `(B, T_s)`. Containing 
                the mask of invalid feature frames. Value ``True`` for invalid frames.
            long_memories (torch.Tensor | None):  Shape `(B, T_l, h_v)`. Containing 
                the zero-padded feature of long memory. None if long-memory is disabled.
            long_memory_masks (torch.Tensor | None): Shape `(B, T_l)`. Containing 
                the mask of invalid feature frames. Value ``True`` for invalid frames.  
                None if long-memory is disabled.
            future_memories (torch.Tensor | None):  Shape `(B, T_f, h_v)`. Containing 
                the zero-padded feature of future memory. None if future-memory is disabled.
            future_memory_masks (torch.Tensor | None): Shape `(B, T_f)`. Containing 
                the mask of invalid feature frames. Value ``True`` for invalid frames.  
                None if future-memory is disabled.
            start_label (torch.Tensor): Shape `(B, T_s)`.
            end_label (torch.Tensor): Shape `(B, T_s)`.
            semantic_label (torch.Tensor): Shape `(B, T_s)`.
            future_start_label (torch.Tensor): Shape `(B, T_f)`.
            future_end_label (torch.Tensor): Shape `(B, T_f)`.
            future_semantic_label (torch.Tensor): Shape `(B, T_f)`.
        """
        if return_loss:
            return self.forward_train(sentence_features, sentence_length, 
                                      short_memories, short_memory_masks,
                                      long_memories, long_memory_masks,
                                      future_memories, future_memory_masks,
                                      start_label, end_label, semantic_label, 
                                      future_start_label, future_end_label, 
                                      future_semantic_label, **aux_info)
        return self.forward_test(sentence_features, sentence_length, 
                                 short_memories, short_memory_masks,
                                 long_memories, long_memory_masks, **aux_info)
