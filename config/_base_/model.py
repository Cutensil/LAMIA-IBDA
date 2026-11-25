text_dims = 128
pvt_channels = [64, 128, 320, 512]
fpn_channels = 256
fpn_strides = [4, 8, 16, 32]
bbox_target_stds = (
    [0.1, 0.1, 0.2, 0.2],
    [0.05, 0.05, 0.1, 0.1],
    [0.033, 0.033, 0.067, 0.067],
)
rpn_proposals = dict(
    nms_pre=2500,
    max_per_img=1500,
    nms=dict(type='nms', iou_threshold=0.7),
    min_bbox_size=0,
)

model = dict(
    type='TextHTC',
    use_text_fusion=True,
    use_text_classification=True,
    attn_use_cls_prompt=True,
    data_preprocessor=dict(
        type='BiDetDataPreprocessor',
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_seg=True,
    ),
    backbone=dict(
        btype='pvt_v2_b3',
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(
            type='Pretrained',
        ),
    ),
    text_bank=dict(
        type='TextBank',
        noise_prompts=None,
        embed_dims=text_dims,
    ),
    neck=dict(
        type='TextGuidedFusionNeck',
        lambda_attn=0.1,
        attn_loss_type='instance',
        lambda_sem=0.2,
        semantic_channels=64,
        channels=pvt_channels,
        embed_dims=text_dims,
        use_outer_residual=False,
        encoder_layers=3,
        attn_level=3,
        attn_loss_weights=(0.1, 0.3, 0.6),
        stride_level=3,
        stride_loss_weights=(0.6, 0.3, 0.1),
        use_shifted_window=False,
    ),
    fpn=dict(
        type='FPN',
        in_channels=pvt_channels,
        out_channels=fpn_channels,
        num_outs=5,
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=fpn_channels,
        feat_channels=fpn_channels,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[*fpn_strides, 64],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1.],
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
    ),
    roi_head=dict(
        type='TextRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=fpn_channels,
            featmap_strides=fpn_strides,
        ),
        bbox_head=[
            dict(
                type='TextBBoxHead',
                in_channels=fpn_channels,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=target_stds,
                ),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                ),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                text_dims=text_dims,
                temperature=0.07,
                learnable_temperature=False,
                lambda_cls=1.0,
                lambda_align=1.0,
                random_align_sampling=True,
                sampling_temperature=0.05,
            )
            for target_stds in bbox_target_stds
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=fpn_channels,
            featmap_strides=fpn_strides,
        ),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=stage > 0,
                num_convs=4,
                in_channels=fpn_channels,
                conv_out_channels=fpn_channels,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            )
            for stage in range(3)
        ],
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=fpn_channels,
            featmap_strides=[8],
        ),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            seg_scale_factor=1 / 8,
            num_convs=4,
            in_channels=fpn_channels,
            conv_out_channels=fpn_channels,
            loss_seg=dict(type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2),
        ),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=rpn_proposals,
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=iou_threshold,
                    neg_iou_thr=iou_threshold,
                    min_pos_iou=iou_threshold,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            )
            for iou_threshold in (0.5, 0.6, 0.7)
        ],
    ),
    test_cfg=dict(
        rpn=rpn_proposals,
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=500,
            mask_thr_binary=0.5,
        ),
    ),
)
