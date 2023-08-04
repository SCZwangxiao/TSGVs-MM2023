# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

from tsgv.utils import import_module_error_func

MODELS = Registry('models', parent=MMCV_MODELS)
INTERACTORS = MODELS
LOCALIZERS = MODELS
LOSSES = MODELS
PREDICTORS = MODELS
SENTENCE_ENCODERS = MODELS
VIDEO_ENCODERS = MODELS

try:
    from mmdet.models.builder import DETECTORS, build_detector
except (ImportError, ModuleNotFoundError):
    # Define an empty registry and building func, so that can import
    DETECTORS = MODELS

    @import_module_error_func('mmdet')
    def build_detector(cfg, train_cfg, test_cfg):
        pass


def build_interactor(cfg):
    """Build interactors."""
    return INTERACTORS.build(cfg)


def build_localizer(cfg):
    """Build localizer."""

    return LOCALIZERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_predictor(cfg):
    """Build predictor."""
    return PREDICTORS.build(cfg)


def build_sentence_encoder(cfg):
    """Build sentence encoder."""
    return SENTENCE_ENCODERS.build(cfg)


def build_video_encoder(cfg):
    """Build video encoder."""
    return VIDEO_ENCODERS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in LOCALIZERS:
        return build_localizer(cfg)
    if obj_type in DETECTORS:
        if train_cfg is not None or test_cfg is not None:
            warnings.warn(
                'train_cfg and test_cfg is deprecated, '
                'please specify them in model. Details see this '
                'PR: https://github.com/open-mmlab/mmaction2/pull/629',
                UserWarning)
        return build_detector(cfg, train_cfg, test_cfg)
    model_in_mmdet = ['FastRCNN']
    if obj_type in model_in_mmdet:
        raise ImportError(
            'Please install mmdet for spatial temporal detection tasks.')
    raise ValueError(f'{obj_type} is not registered in '
                     'LOCALIZERS, or DETECTORS')
