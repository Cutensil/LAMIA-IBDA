_base_ = '../paths.py'

data_root = _base_.paths.xbd_root
metainfo = dict(
    classes=('no-damage', 'minor-damage', 'major-damage', 'destroyed'),
    palette=[(239, 239, 239), (217, 234, 210), (249, 179, 101), (237, 154, 152)],
)

image_size = (1024, 1024)
crop_size = (512, 512)
dataset_defaults = dict(
    type='BiCocoDataset',
    data_root=data_root,
    metainfo=metainfo,
    pre_dir='pre_event',
    post_dir='post_event',
)
load_pair = dict(type='LoadBiImageFromFile', post_color_type='color')
train_pipeline = [
    load_pair,
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='RandomCrop', crop_size=crop_size, recompute_bbox=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1.0, 1.0), by_mask=True),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs'),
]


def evaluation_pipeline(tile_index):
    return [
        load_pair,
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='Resize', scale=image_size, keep_ratio=True),
        dict(
            type='FixedQuadCrop',
            tile_index=tile_index,
            recompute_bbox=True,
            allow_negative_crop=True,
            image_size=image_size,
        ),
        dict(
            type='PackDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'pre_img_path', 'post_img_path',
                'ori_shape', 'img_shape', 'scale_factor', 'instances',
            ),
        ),
    ]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='ClassAwareSampler', num_sample_class=1),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        **dataset_defaults,
        ann_file='train_annotations.json',
        data_prefix=dict(img='train', seg='train/target'),
        seg_map_suffix='.tif',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)
valid_dataset = dict(
    **dataset_defaults,
    ann_file='valid_annotations.json',
    data_prefix=dict(img='valid'),
    filter_cfg=dict(filter_empty_gt=False),
    test_mode=True,
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(**valid_dataset, pipeline=evaluation_pipeline(0)),
)
test_dataset = dict(
    **dataset_defaults,
    ann_file='test_annotations.json',
    data_prefix=dict(img='test'),
    filter_cfg=dict(filter_empty_gt=False),
    test_mode=True,
)
test_dataloader = dict(
    val_dataloader,
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(**test_dataset, pipeline=evaluation_pipeline(tile_index))
            for tile_index in range(4)
        ],
    ),
)
