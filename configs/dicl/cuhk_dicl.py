_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_reid_norm_unsu_nor5.py',
    '../_base_/datasets/coco_reid_unsup.py',
    '../_base_/schedules/schedule_1x_reid_norm_base.py', '../_base_/default_runtime.py'
]
model = dict(
    type='TwoStageDetectorsiamese',
    mask_ratio=0.2,
    pixel_mask=False,
    num_mask_patch=2,
    pro_mask=0.5,
    use_mask=True,
    mask_up=True,
    mask_down=False,
    gt_assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.6,  
        neg_iou_thr=0.1,  
        min_pos_iou=0.5,
        match_low_quality=False,
        ignore_iof_thr=-1),
    backbone=dict(
        type='ResNet',
        depth=50,
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        strides=(1, 2, 2, 1),
        num_stages=4,
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    neck=dict(
        type='FPNs16C45add',
        in_channels=[512, 1024, 2048],
        out_channels=1024,
        use_dconv=True,
        kernel1=True, ),
    roi_head=dict(
        type='ReidRoIHeadsiamese',
        bbox_head=dict(
            type='DICLHead',
            in_channels=1024,
        ),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='DeformRoIPoolPack',
                output_size=tuple([14, 6]),
                output_channels=1024),
            out_channels=1024,
            featmap_strides=[16]),
    )
)
# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize',
        img_scale=[(667, 400),(1000, 600), (1333, 800), (1500,900), (1666, 1000),],#
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1500, 900),
        # img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
query_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals', num_max_proposals=None),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1500, 900),
        # img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['proposals']),
            dict(type='Collect', keys=['img', 'proposals']),
        ])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1500, 900),
        # img_scale=(1500, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline, query_test_pipeline=None),
    train_cluster=dict(pipeline=val_pipeline, query_test_pipeline=None),
    val=dict(pipeline=test_pipeline, query_test_pipeline=None),
    test=dict(pipeline=test_pipeline,
        query_test_pipeline=query_test_pipeline,
    ))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)#0.003

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2242,
    warmup_ratio=1.0 / 200,
    step=[16, 22])
total_epochs = 26

SPCL=True
PSEUDO_LABELS = dict(
    freq=1, # epochs
    use_outliers=True,
    norm_feat=True,
    norm_center=True,
    cluster='dbscan_context',
    eps=[0.68, 0.7, 0.72],
    min_samples=2, # for dbscan
    dist_metric='jaccard',
    k1=30, # for jaccard distance
    k2=6, # for jaccard distance
    search_type=0,#search_option=0,# # 0,1,2 for GPU, 3 for CPU (work for faiss)
    cluster_num=None
)