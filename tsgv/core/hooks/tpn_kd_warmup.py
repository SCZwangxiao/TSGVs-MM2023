import functools
import warnings
import math

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class TPNKnowledgeDistillationWarmUpHook(Hook):
    """Warm up the weight of knowledge distillation for TPN in training.

    Args:
        gamma (float):
            Default: 10
    """
    def __init__(self, gamma=10):
        super().__init__()
        self.gamma = gamma

    def before_train_iter(self, runner):
        model = runner.model.module
        distillation_loss_weight0 = model.predictor.distillation_loss_weight0
        p = runner._iter / runner._max_iters
        ratio = 1 + math.exp(-self.gamma*p)
        ratio = 2 / ratio - 1
        model.predictor.distillation_loss_weight = ratio * distillation_loss_weight0