_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
num_segments = 64
test_interval = 32
extention_ratio = 0.1
model = dict(
    type='BaseTSGV',
    sentence_encoder=dict(
        type='BaseSentenceEncoder',
        sent_dim=768, 
        hidden_dim=128, 
        pool_strategy=None,
        linear_options=dict(norm_before=True,
                            activation='gelu')),
    video_encoder=dict(
        type='BaseVideoEncoder',
        video_dim=500, 
        hidden_dim=128,
        linear_options=dict(norm_before=True,
                            activation='gelu')),
    interactor=dict(
        type='SeqPANInteractor',
        num_segments=num_segments,
        hidden_dim=128,
        v_kernel_size=7,
        s_kernel_size=7,
        num_encoder_layers=4,
        num_sgpa_layers=2,
        num_heads=8,
        tau=0.3,
        dropout_rate=0.2),
    predictor=dict(
        type='SeqPANPredictor',
        hidden_dim=128, 
        num_head=8, 
        dropout_ratio=0.2,
        seq_loss_weigth=1.0))

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
    videos_per_gpu=256,
    workers_per_gpu=6,
    val_dataloader=dict(videos_per_gpu=128, pin_memory=True),
    test_dataloader=dict(videos_per_gpu=128, pin_memory=True),
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
        proposal_options=dict(type='span_based',
                              extension_ratio=extention_ratio),
        nms_thresh=0.3,
        video_in_memory=True,
        sent_in_memory=True),
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
        proposal_options=dict(type='span_based',
                              extension_ratio=extention_ratio),
        nms_thresh=0.3,
        video_in_memory=True,
        sent_in_memory=True),
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
        proposal_options=dict(type='span_based',
                              extension_ratio=extention_ratio),
        nms_thresh=0.3,
        video_in_memory=False,
        sent_in_memory=False))

# evaluation
evaluation = dict(
    interval=1, 
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1],
             iou_at=[0.3,0.5,0.7])},
    save_best='R@1,IoU=0.5',
    rule='greater')

eval_config = dict(
    metrics=['R@N,IoU=M'],
    metric_options={'R@N,IoU=M': 
        dict(recall_at=[1],
             iou_at=[0.3,0.5,0.7])})

# optimizer
optimizer = dict(
    type='AdamW', 
    lr=1.5e-3,
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))

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
total_epochs = 20

# logger
checkpoint_config = dict(interval=-1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])