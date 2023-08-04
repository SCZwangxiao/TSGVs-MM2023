_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
num_segments = 32
test_interval = 16
extention_ratio = 0.1
model = dict(
    type='BaseTSGV',
    sentence_encoder=dict(
        type='BaseSentenceEncoder',
        sent_dim=512, 
        hidden_dim=128, 
        pool_strategy=None,
        linear_options=dict(norm_before=True,
                            activation='gelu')),
    video_encoder=dict(
        type='BaseVideoEncoder',
        video_dim=512, 
        hidden_dim=128,
        linear_options=dict(norm_before=True,
                            activation='gelu')),
    interactor=dict(
        type='VSLNetInteractor',
        num_segments=num_segments,
        hidden_dim=128,
        v_kernel_size=7,
        s_kernel_size=7,
        num_layers=4,
        num_heads=8,
        dropout_rate=0.2),
    predictor=dict(
        type='ConditionedSpanPredictor',
        hidden_dim=128))

# dataset settings
dataset_type = 'MADTSGVDataset'
data_root = 'data/mad/feature/'
data_root_val = 'data/mad/feature/'
ann_file_train = 'data/mad/train.json'
ann_file_val = 'data/mad/val.json'
ann_file_test = 'data/mad/test.json'
train_pipeline = [
    dict(type='Collect',
         keys=['video_features', 'sentence_features', 
               'video_masks', 'sentence_length', 'sentence_masks', 'label'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['video_features', 'sentence_features', 
               'video_masks', 'sentence_length', 'sentence_masks', 'label'])
]
val_pipeline = [
    dict(type='Collect',
         keys=['video_features', 'sentence_features', 
               'video_masks', 'sentence_length', 'sentence_masks'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['video_features', 'sentence_features', 
               'video_masks', 'sentence_length', 'sentence_masks'])
]
test_pipeline = [
    dict(type='Collect',
         keys=['video_features', 'sentence_features', 
               'video_masks', 'sentence_length', 'sentence_masks'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['video_features', 'sentence_features', 
               'video_masks', 'sentence_length', 'sentence_masks'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(videos_per_gpu=64, pin_memory=False),
    test_dataloader=dict(videos_per_gpu=64, pin_memory=False),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        split='train',
        video_feat_filename='CLIP_frames_features_5fps.h5',
        language_model='CLIP',  
        num_segments=num_segments,
        window_sample_options=dict(neg_sample_ratio=0.7,
                                   sample_stride=1),
        proposal_options=dict(type='span_based',
                              extension_ratio=extention_ratio),
        nms_thresh=0.3,
        video_in_memory=True,
        sent_in_memory=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root,
        split='val',
        video_feat_filename='CLIP_frames_features_5fps_val.hdf5',
        language_model='CLIP',  
        num_segments=num_segments,
        window_sample_options=dict(sample_stride=2,
                                   test_interval=test_interval),
        proposal_options=dict(type='span_based'),
        nms_thresh=0.3,
        video_in_memory=False,
        sent_in_memory=False,
        portion=0.1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root,
        split='test',
        video_feat_filename='CLIP_frames_features_5fps_test.hdf5',
        language_model='CLIP',  
        num_segments=num_segments,
        window_sample_options=dict(sample_stride=2,
                                   test_interval=test_interval),
        proposal_options=dict(type='span_based'),
        nms_thresh=0.3,
        video_in_memory=True,
        sent_in_memory=False))

# evaluation
evaluation = dict(
    interval=1, 
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1, 5, 10, 50, 100],
             iou_at=[0.1,0.3,0.5])},
    save_best='R@1,IoU=0.3',
    rule='greater')

eval_config = dict(
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1, 5, 10, 50, 100],
             iou_at=[0.1,0.3,0.5])})

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=2.5e-4,
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(
    policy='FlatCosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=4,
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    start_percent=0.4,
    min_lr=.0)
total_epochs = 25

# logger
checkpoint_config = dict(interval=-1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])