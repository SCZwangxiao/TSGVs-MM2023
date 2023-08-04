_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
num_segments = 128
num_clips = 64
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
        pool_strategy=None,
        linear_before=False,
        norm_after=False),
    video_encoder=dict(
        type='BaseVideoEncoder',
        video_dim=500, 
        hidden_dim=512,
        linear_before=True,
        linear_options=dict(dropout_rate=0.0,
                            norm_before=False,
                            activation='relu'),
        norm_after=False),
    interactor=dict(
        type='SMINInteractor',
        hidden_dim=512,
        d_l=128,
        kernel_size=5,
        num_layers=3,
        num_segments=num_segments,
        num_clips=num_clips,
        group_size=group_size,
        C=4),
    predictor=dict(
        type='SMINPredictor',
        hidden_dim=512,
        binary_thresh=0.5))

# dataset settings
dataset_type = 'TSGVDataset'
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
    videos_per_gpu=8,
    workers_per_gpu=8,
    val_dataloader=dict(videos_per_gpu=64, pin_memory=True),
    test_dataloader=dict(videos_per_gpu=64, pin_memory=True),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root,
        split='train',
        video_feat_filename='activitynet_v1-3_c3d_train.hdf5',
        max_sentence_length=20,
        num_segments=num_segments,
        proposal_options=dict(type='smin',
                              group_size=group_size,
                              num_clips=num_clips,
                              gaussian_label=True,
                              gaussian_label_width=[0.2,0.2,0.2],
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
        max_sentence_length=20,
        num_segments=num_segments,
        proposal_options=dict(type='smin',
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
        max_sentence_length=20,
        num_segments=num_segments,
        proposal_options=dict(type='smin',
                              group_size=group_size,
                              num_clips=num_clips,
                              cache_anno_label=False),
        nms_thresh=0.5,
        video_in_memory=True))

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
    lr=5e-4,
    weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# learning policy
lr_config = dict(
    policy='FlatCosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.1,
    warmup_by_epoch=True,
    start_percent=0.4,
    min_lr=.0)
total_epochs = 15

# logger
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])