_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
num_segments = 32
test_interval = 16
num_clips = 16
group_size = 8
model = dict(
    type='BaseTSGV',
    sentence_encoder=dict(
        type='RNNSentenceEncoder',
        sent_dim=512, 
        hidden_dim=512, 
        num_layers=3,
        rnn_type='LSTM',
        dropout=0.2,
        bidirectional=False,
        pool_strategy='Last'),
    video_encoder=dict(
        type='TANVideoEncoder',
        video_dim=512, 
        hidden_dim=512,
        num_segments=num_segments,
        num_clips=num_clips,
        group_size=group_size),
    interactor=dict(
        type='TANInteractor',
        hidden_dim=512,
        kernel_size=7,
        num_layers=4,
        num_clips=num_clips,
        group_size=group_size),
    predictor=dict(
        type='TANPredictor',
        hidden_dim=512,
        min_iou=0.1,
        max_iou=1.0))

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
               'sentence_length', 'label'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['video_features', 'sentence_features', 
               'sentence_length', 'label'])
]
val_pipeline = [
    dict(type='Collect',
         keys=['video_features', 'sentence_features', 
               'sentence_length'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['video_features', 'sentence_features', 
               'sentence_length'])
]
test_pipeline = [
    dict(type='Collect',
         keys=['video_features', 'sentence_features', 
               'sentence_length'],
         meta_keys=(),
         meta_name="tsgv_metas"),
    dict(type='ToTensor', 
         keys=['video_features', 'sentence_features', 
               'sentence_length'])
]
data = dict(
    videos_per_gpu=32,
    workers_per_gpu=6,
    val_dataloader=dict(videos_per_gpu=64, pin_memory=True),
    test_dataloader=dict(videos_per_gpu=64, pin_memory=True),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        split='train',
        video_feat_filename='CLIP_frames_features_5fps_train.hdf5',
        language_model='CLIP',  
        num_segments=num_segments,
        window_sample_options=dict(neg_sample_ratio=0.7,
                                   sample_stride=1),
        proposal_options=dict(type='2dtan',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
        nms_thresh=0.3,
        video_in_memory=False,
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
        proposal_options=dict(type='2dtan',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
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
        proposal_options=dict(type='2dtan',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
        nms_thresh=0.3,
        video_in_memory=False))

# evaluation
evaluation = dict(
    interval=1, 
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1,5,10,50,100],
             iou_at=[0.1,0.3,0.5])},
    save_best='R@1,IoU=0.3',
    rule='greater')

eval_config = dict(
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1,5,10,50,100],
             iou_at=[0.1,0.3,0.5])})

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=5e-4,
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
total_epochs = 25

# logger
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])