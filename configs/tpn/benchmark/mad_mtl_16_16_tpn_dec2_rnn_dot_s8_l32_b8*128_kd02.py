_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
short_memory_sample_length = 8
long_memory_sample_length = 32
future_memory_length = 32
short_memory_stride = 1
long_memory_stride = 1
future_memory_stride = 1
model = dict(
    type='BaseStreamingTSGV',
    train_aux_info=['video_name', 'anno_framestamp', 'memory_framestamp'],
    sentence_encoder=dict(
        type='RNNSentenceEncoder',
        sent_dim=512, 
        hidden_dim=512, 
        linear_before=False, # for CLIP features
        num_layers=2,
        rnn_type='GRU',
        dropout=0.5,
        bidirectional=True,
        pool_strategy=None),
    video_encoder=dict(
        type='BaseStreamingVideoEncoder',
        video_dim=512, 
        hidden_dim=512, 
        linear_before=False, # for CLIP features
        use_long=True,
        use_future=True,
        norm_after=False),
    interactor=dict(
        type='TPNInteractor',
        hidden_dim=512, 
        feedforward_dim=1024,
        short_memory_length=short_memory_sample_length//short_memory_stride,
        long_memory_length=long_memory_sample_length//long_memory_stride,
        future_memory_length=future_memory_length//future_memory_stride,
        memory_compressor='MultimodalTokenLearner',
        num_compressor_tokens=[16, 16],
        num_decoder_layers=2,
        sent_norm_before=False, 
        num_heads=8,
        dropout=0.2,
        future_usage='TemporalPincer'),
    predictor=dict(
        type='DotProductPredictorTPN',
        hidden_dim=512,
        tau=16.0,
        gamma=3.0,
        num_layers=1,
        tpn_type='KD',
        tpn_distillation_weight=0.2))

# dataset settings
dataset_regenerate = True
dataset_type = 'OnlineTSGVDataset'
data_root = 'data/mad/feature/'
data_root_val = 'data/mad/feature/'
ann_file_train = 'data/mad/train.json'
ann_file_val = 'data/mad/val.json'
ann_file_test = 'data/mad/test.json'
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
    videos_per_gpu=128,
    workers_per_gpu=6,
    val_dataloader=dict(videos_per_gpu=256, pin_memory=False),
    test_dataloader=dict(videos_per_gpu=256, pin_memory=True),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        short_memory_sample_length=short_memory_sample_length,
        chunk_options=dict(chunk_interval=1,
                           pos_expansion_ratio=5.,
                           neg_pos_ratio=25.),
        data_prefix=data_root,
        video_feat_filename='CLIP_frames_features_5fps_train.hdf5',
        split='train',
        language_model='CLIP',  
        long_memory_sample_length=long_memory_sample_length,
        short_memory_stride=short_memory_stride,
        long_memory_stride=long_memory_stride,
        future_memory_sample_length=future_memory_length,
        future_memory_stride=future_memory_stride, 
        load_future_memory=True,
        video_in_memory=False,
        sent_in_memory=True, 
        cache_anno_label=False,
        gaussian_label=True),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        short_memory_sample_length=short_memory_sample_length,
        chunk_options=dict(chunk_interval=4,
                           cache_chunk_info=False),
        proposal_options=dict(smooth_filter_win_len=3,
                              downsample_stride=1,
                              min_num_segments=128,
                              max_proposal_len=128,
                              group_size_2d=2),
        nms_thresh=0.3,
        data_prefix=data_root,
        video_feat_filename='CLIP_frames_features_5fps_val.hdf5',
        split='val',
        language_model='CLIP', 
        long_memory_sample_length=long_memory_sample_length,
        short_memory_stride=short_memory_stride,
        long_memory_stride=long_memory_stride,
        video_in_memory=True,
        sent_in_memory=True, 
        cache_anno_label=False,
        portion=0.1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        short_memory_sample_length=short_memory_sample_length,
        chunk_options=dict(chunk_interval=4,
                           cache_chunk_info=False), 
        proposal_options=dict(smooth_filter_win_len=3,
                              downsample_stride=1,
                              min_num_segments=128,
                              max_proposal_len=128,
                              group_size_2d=2),
        nms_thresh=0.3,
        data_prefix=data_root,
        video_feat_filename='CLIP_frames_features_5fps_test.hdf5',
        split='test',
        language_model='CLIP', 
        long_memory_sample_length=long_memory_sample_length,
        short_memory_stride=short_memory_stride,
        long_memory_stride=long_memory_stride,
        video_in_memory=True,
        sent_in_memory=True, 
        cache_anno_label=False))

# evaluation
evaluation = dict(
    interval=1, 
    metrics=['R@N,IoU=M', 'mcAP'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1,5,10,50,100],
             iou_at=[0.1,0.3,0.5])},
    save_best='R@1,IoU=0.3',
    rule='greater')

eval_config = dict(
    metrics=['R@N,IoU=M', 'mcAP'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1,5,10,50,100],
             iou_at=[0.1,0.3,0.5])},
)

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=6e-5, 
    weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(
    policy='FlatCosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=6,
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    start_percent=0.4,
    min_lr=.0)
total_epochs = 15

# logger
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])