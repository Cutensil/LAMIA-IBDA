max_epochs = 100
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=2,
        start_factor=0.1,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        type='CosineAnnealingLR',
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        eta_min_ratio=0.5,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0),
    clip_grad=dict(max_norm=10, norm_type=2),
)
auto_scale_lr = dict(enable=True, base_batch_size=8)
