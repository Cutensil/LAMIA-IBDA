default_scope = 'det'
custom_imports = dict(
    imports=[
        'mm.det.cd_pipeline',
        'mm.det.cd_dataset',
        'mm.det.cd_backbone',
        'mm.det.cd_metric',
        'mm.det.cd_hook',
        'mm.det.lamia',
    ],
    allow_failed_imports=False,
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='SafeCheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/segm_mAP',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='BiDetVisualizationHook',
        pre_dir='pre_event',
        post_dir='post_event',
        interval=50,
        draw=False,
    ),
)
visualizer = dict(
    type='BiDetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
randomness = dict(seed=42)
log_level = 'INFO'
load_from = None
resume = False

val_evaluator = dict(
    type='ScaledCocoMetric',
    ann_file=None,
    metric=['segm', 'bbox'],
    prefix='coco',
    classwise=True,
)
test_evaluator = val_evaluator
