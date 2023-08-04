_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
short_memory_sample_length = 8
long_memory_sample_length = 64
future_memory_length = 64
short_memory_stride = 1
long_memory_stride = 1
future_memory_stride = 1
model = dict(
    type='BaseStreamingTSGV',
    train_aux_info=['video_name', 'anno_framestamp', 'memory_framestamp'],
    sentence_encoder=dict(
        type='RNNSentenceEncoder',
        sent_dim=768, 
        hidden_dim=1024, 
        num_layers=2,
        rnn_type='GRU',
        dropout=0.5,
        bidirectional=True,
        pool_strategy=None),
    video_encoder=dict(
        type='BaseStreamingVideoEncoder',
        video_dim=500, 
        hidden_dim=1024, 
        use_long=True,
        use_future=True,
        norm_after=False),
    interactor=dict(
        type='TPNInteractor',
        hidden_dim=1024, 
        feedforward_dim=1024,
        short_memory_length=short_memory_sample_length//short_memory_stride,
        long_memory_length=long_memory_sample_length//long_memory_stride,
        future_memory_length=future_memory_length//future_memory_stride,
        memory_compressor='MultimodalTokenLearner',
        num_compressor_tokens=[16, 16],
        num_decoder_layers=2,
        sent_norm_before=False, 
        num_heads=8,
        dropout=0.5,
        future_usage='TemporalPincer'),
    predictor=dict(
        type='DotProductPredictorTPN',
        hidden_dim=1024,
        tau=16.0,
        gamma=3.0,
        num_layers=1,
        tpn_type='KD',
        tpn_distillation_weight=0.4))

# dataset settings
dataset_regenerate = True
dataset_type = 'OnlineTSGVDataset'
data_root = 'data/activitynet/feature/'
data_root_val = 'data/activitynet/feature/'
ann_file_train = 'data/activitynet/train.json'
ann_file_val = 'data/activitynet/val.json'
ann_file_test = 'data/activitynet/test.json'
train_pipeline = [
    dict(type='Collect',
         keys=['short_memories', 'short_memory_masks',
               'sentence_features', 'sentence_length',
               'long_memories', 'long_memory_masks',
               'future_memories', 'future_memory_masks', 
               'start_label', 'end_label', 'semantic_label',
               'video_name', 'anno_framestamp', 'memory_framestamp'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToDataContainer',
         fields=[
             dict(key='video_name', cpu_only=True),
             dict(key='anno_framestamp', cpu_only=True),
             dict(key='memory_framestamp', cpu_only=True)]),
    dict(type='ToTensor', 
         keys=['short_memories', 'short_memory_masks',
               'sentence_features', 'sentence_length',
               'long_memories', 'long_memory_masks',
               'future_memories', 'future_memory_masks',
               'start_label', 'end_label', 'semantic_label'])
]
val_pipeline = [
    dict(type='Collect',
         keys=['short_memories', 'short_memory_masks',
               'sentence_features', 'sentence_length',
               'long_memories', 'long_memory_masks'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['short_memories', 'short_memory_masks',
               'sentence_features', 'sentence_length',
               'long_memories', 'long_memory_masks'])
]
test_pipeline = [
    dict(type='Collect',
         keys=['short_memories', 'short_memory_masks',
               'sentence_features', 'sentence_length',
               'long_memories', 'long_memory_masks'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['short_memories', 'short_memory_masks',
               'sentence_features', 'sentence_length',
               'long_memories', 'long_memory_masks'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=6,
    val_dataloader=dict(videos_per_gpu=256, pin_memory=True),
    test_dataloader=dict(videos_per_gpu=256, pin_memory=True),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        short_memory_sample_length=short_memory_sample_length,
        data_prefix=data_root,
        split='train',
        video_feat_filename='activitynet_v1-3_c3d_train.hdf5',
        video_in_memory=False, # enable if OOM
        long_memory_sample_length=long_memory_sample_length,
        short_memory_stride=short_memory_stride,
        long_memory_stride=long_memory_stride,
        future_memory_sample_length=future_memory_length,
        future_memory_stride=future_memory_stride, 
        load_future_memory=True,
        gaussian_label=True),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        short_memory_sample_length=short_memory_sample_length,
        data_prefix=data_root,
        split='val',
        video_feat_filename='activitynet_v1-3_c3d_val.hdf5',
        long_memory_sample_length=long_memory_sample_length,
        short_memory_stride=short_memory_stride,
        long_memory_stride=long_memory_stride,
        portion=0.5),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        short_memory_sample_length=short_memory_sample_length,
        data_prefix=data_root,
        split='test',
        video_feat_filename='activitynet_v1-3_c3d_test.hdf5',
        # video_in_memory=False,
        long_memory_sample_length=long_memory_sample_length,
        short_memory_stride=short_memory_stride,
        long_memory_stride=long_memory_stride))

# evaluation
evaluation = dict(
    interval=1, 
    metrics=['R@N,IoU=M', 'mcAP'],
    metric_options={'R@N,IoU=M': 
        dict(
            recall_at=[1,5],
            iou_at=[0.3,0.5,0.7],
            nms_thresh=0.5)},
    save_best='R@1,IoU=0.5',
    rule='greater')

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=1e-5,
    weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

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
total_epochs = 10

# logger
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])