# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseWeightedLoss
from .binary_logistic_regression_loss import BinaryLogisticRegressionLoss
from .cross_entropy_loss import (BCELossWithLogits, CBFocalLoss,
                                 CrossEntropyLoss)

__all__ = [
    'BaseWeightedLoss', 'CrossEntropyLoss', 'BCELossWithLogits',
    'BinaryLogisticRegressionLoss', 'CBFocalLoss'
]
