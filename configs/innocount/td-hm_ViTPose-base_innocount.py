_base_ = ['../../third_party/mmpose/configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['mmpose.engine.optim_wrappers.layer_decay_optim_wrapper'],
    allow_failed_imports=False,
)

# Training schedule
train_cfg = dict(max_epochs=100, val_interval=5)

# Optimizer
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            'bias': dict(decay_multi=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
        },
    ),
    constructor='LayerDecayOptimWrapperConstructor',
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        milestones=[80, 95],
        gamma=0.1,
        by_epoch=True,
    ),
]

# Checkpoints
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2),
)

# Six global keypoint slots
codec = dict(
    type='UDPHeatmap',
    input_size=(192, 256),
    heatmap_size=(48, 64),
    sigma=2,
)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='mmpretrain.VisionTransformer',
        arch='base',
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        with_cls_token=False,
        out_type='featmap',
        patch_cfg=dict(padding=2),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                '/home/owner/.cache/torch/hub/checkpoints/'
                'mae_pretrain_vit_base_20230913.pth'
            ),
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=768,
        out_channels=6,
        deconv_out_channels=(256, 256),
        deconv_kernel_sizes=(4, 4),
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,
    ),
)

# Dataset
dataset_type = 'CocoDataset'
data_root = 'datasets/ViTPose_COCO/'
metainfo = dict(from_file='configs/innocount/innocount_metainfo.py')

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size'], use_udp=True),
    dict(type='PackPoseInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/valid.json',
        data_prefix=dict(img='images/valid/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

# Training-time validation metric only; use the project's common evaluator
# for final test metrics.
val_evaluator = dict(type='PCKAccuracy', thr=0.05)
test_evaluator = dict(type='PCKAccuracy', thr=0.05)
