# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='Exp', 
    gamma=0.98, 
    by_epoch=False)
total_epochs = 1
# 333 iters for tacos OadTR of 512 batch size
# 2657 iters for tacos LSTR of 512 batch size