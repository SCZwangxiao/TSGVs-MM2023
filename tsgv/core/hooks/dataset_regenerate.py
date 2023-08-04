import functools
import warnings
import math

import torch
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DatasetRegenerateHook(Hook):
    """Dataset regenerate hook. After each epoch, the hook will call the
        ``re_generate_dataset()`` method of the train dataset.

    """

    def after_train_epoch(self, runner):
        if runner.epoch < runner._max_epochs - 1:
            # Expalanation of the inequality:
            # In mmcv ``EpochBasedRunner``, ``epoch`` start with 0.
            # After calling ``after_train_epoch``, ``epoch``++.  
            # Therefore ``runner.epoch`` generally returns the previous epoch number.
            runner.logger.info('Regenerating training dataset...')
            # Regenerate dataset
            runner.data_loader.dataset.re_generate_dataset()
            # Re-calculate sampler total_size
            len_dataset = len(runner.data_loader.dataset)
            drop_last = runner.data_loader.sampler.drop_last
            num_replicas = runner.data_loader.sampler.num_replicas
            if drop_last and len_dataset % num_replicas != 0:
                # See ./tsgv/datasets/samplers/distributed_sampler.py
                num_samples = math.ceil(
                    (len_dataset - num_replicas) / num_replicas
                )
            else:
                num_samples = math.ceil(len_dataset / num_replicas)
            total_size = num_samples * num_replicas
            runner.data_loader.sampler.num_samples = num_samples
            runner.data_loader.sampler.total_size = total_size