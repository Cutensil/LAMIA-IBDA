_base_ = [
    './_base_/dataset/xbd.py',
    './_base_/model.py',
    './_base_/schedule.py',
    './_base_/runtime.py',
]

class_prompts = (
    (  # no-damage: 3 descriptions
        'building retaining its pre-event footprint and overall roof shape',
        'building with continuous roof covering and no newly exposed roof sections',
        'building with a continuous outer wall outline and no collapsed sections',
    ),
    (  # minor-damage: 5 descriptions
        'building with patches of roof covering missing while the roof shape is retained',
        'building with displaced roof-covering panels and the main roof shape still preserved',
        'building with visible cracks in its roof or walls but no collapsed sections',
        'building with localized burn marks and its main roof still in place',
        'building with floodwater around its base and its roof still exposed',
    ),
    (  # major-damage: 5 descriptions
        'building with a collapsed roof section exposing the interior while other sections remain',
        'building with a collapsed outer wall section and part of the roof still standing',
        'building with a tilted or shifted main roof plane and surviving standing sections',
        'building enclosed by floodwater or mud while its roof remains visible',
        'building with volcanic material reaching its footprint while part of the structure remains exposed',
    ),
    (  # destroyed: 7 descriptions
        'building replaced by a compact pile of collapsed roof and wall rubble',
        'building reduced to widely scattered structural debris with no main structure standing',
        'building burned down to a foundation with only charred wall remnants',
        'building absent from its pre-event footprint with an exposed foundation remaining',
        'building covered by floodwater at its pre-event location',
        'building buried under mud and transported debris at its pre-event location',
        'building buried beneath volcanic deposits at its pre-event location',
    ),
)

num_classes = len(_base_.metainfo.classes)

model = dict(
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53] * 2,
        std=[58.395, 57.12, 57.375] * 2,
    ),
    backbone=dict(
        type='SiamBackboneSharedWeights',
        init_cfg=dict(checkpoint=_base_.paths.pvt_checkpoint),
    ),
    text_bank=dict(
        text_encoder_path=_base_.paths.bge_dir,
        class_prompts=class_prompts,
    ),
    roi_head=dict(
        bbox_head=[
            dict(head, num_classes=num_classes)
            for head in _base_.model.roi_head.bbox_head
        ],
        mask_head=[
            dict(head, num_classes=num_classes)
            for head in _base_.model.roi_head.mask_head
        ],
        semantic_head=dict(num_classes=num_classes + 1),
    ),
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone.siam_backbone': dict(decay_mult=1.0, lr_mult=0.1),
        },
    ),
)

default_hooks = dict(visualization=dict(post_colortype='color'))
