_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
num_segments = 64
test_interval = 32
num_clips = 32
group_size = 8
model = dict(
    type='BaseTSGV',
    sentence_encoder=dict(
        type='RNNSentenceEncoder',
        sent_dim=768, 
        hidden_dim=512, 
        num_layers=3,
        rnn_type='LSTM',
        dropout=0.0,
        bidirectional=False,
        pool_strategy='Last',
        linear_before=False,
        norm_after=False),
    video_encoder=dict(
        type='TANVideoEncoder',
        video_dim=500, 
        hidden_dim=512,
        num_segments=num_segments,
        num_clips=num_clips,
        group_size=group_size),
    interactor=dict(
        type='TANInteractor',
        hidden_dim=512,
        kernel_size=9,
        num_layers=4,
        num_clips=num_clips,
        group_size=group_size),
    predictor=dict(
        type='TANPredictor',
        hidden_dim=512,
        min_iou=0.5,
        max_iou=1.0))

# dataset settings
dataset_type = 'MADTSGVDataset'
data_root = 'data/activitynet/feature/'
data_root_val = 'data/activitynet/feature/'
ann_file_train = 'data/activitynet/train.json'
ann_file_val = 'data/activitynet/val.json'
ann_file_test = 'data/activitynet/test.json'
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
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=32, pin_memory=True),
    test_dataloader=dict(videos_per_gpu=32, pin_memory=True),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        split='train',
        video_feat_filename='activitynet_v1-3_c3d_train.hdf5',
        num_segments=num_segments,
        window_sample_options=dict(neg_sample_ratio=0.7,
                                   sample_stride=1),
        proposal_options=dict(type='2dtan',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
        nms_thresh=0.5,
        video_in_memory=False),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root,
        split='val',
        video_feat_filename='activitynet_v1-3_c3d_val.hdf5',
        num_segments=num_segments,
        window_sample_options=dict(sample_stride=2,
                                   test_interval=test_interval),
        proposal_options=dict(type='2dtan',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
        nms_thresh=0.5,
        video_in_memory=True),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline,
        data_prefix=data_root,
        split='test',
        video_feat_filename='activitynet_v1-3_c3d_test.hdf5',
        num_segments=num_segments,
        window_sample_options=dict(sample_stride=2,
                                   test_interval=test_interval),
        proposal_options=dict(type='2dtan',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
        nms_thresh=0.5,
        video_in_memory=False))

# evaluation
evaluation = dict(
    interval=1, 
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(
            recall_at=[1,5],
            iou_at=[0.3,0.5,0.7],
            nms_thresh=0.5)},
    save_best='R@1,IoU=0.5',
    rule='greater')

# optimizer
optimizer = dict(
    type='Adam', 
    lr=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))

# learning policy
lr_config = dict(
    policy='step', 
    step=[3, 4])
total_epochs = 5

# logger
checkpoint_config = dict(interval=-1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])