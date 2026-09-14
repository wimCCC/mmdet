app = 0
auto_scale_lr = dict(base_batch_size=2, enable=False)
backend_args = None
base_lr_lp = 0.002
base_lr_tea = 0.0001
batch_size = 2
bit = 4
data_root = ''
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=None,
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
dorefa = 0
dsq = 0
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
extra_config = dict(
    extra_qconfig_dict=dict(
        a_fakequantize='LearnableFakeQuantize',
        a_observer='EMAMinMaxObserver',
        a_qscheme=dict(
            bit=4, per_channel=False, pot_scale=False, symmetry=True),
        w_fakequantize='LearnableFakeQuantize',
        w_observer='EMAMinMaxObserver',
        w_qscheme=dict(
            bit=4, per_channel=False, pot_scale=False, symmetry=True)))
fixed = 0
img_scale = (
    800,
    800,
)
img_scales = [
    (
        1333,
        800,
    ),
    (
        666,
        400,
    ),
    (
        2000,
        1200,
    ),
]
kd = 0
launcher = 'none'
load_from = '/home/caiwm/mmdetection-3.0.0/ckpt/retina18_epoch_50.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lsq = 1
max_epochs = 300
mcqd = 0
method = 'lsq'
mine = 0
model = dict(
    backbone=dict(
        depth=18,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=64,
        in_channels=64,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=0.5, type='CrossEntropyLoss', use_sigmoid=True),
        num_classes=1,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            64,
            128,
            256,
            512,
        ],
        num_outs=5,
        out_channels=64,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.3, type='nms'),
        nms_pre=1000,
        score_thr=0.3),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet_quanti')
nnie = 0
num_classes = 1
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
optim_wrapper_lp = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
pact = 0
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            30,
        ],
        type='MultiStepLR'),
]
progress = True
qdcl = 0
qdrop = 0
qfd = 0
qkd = 0
qkd_p1 = 10
qkd_p2 = 15
quanti = 1
resume = False
resume_from = None
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/home2/caizy/cwm/IRSTD-1k/',
        metainfo=dict(classes=('target', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home2/caizy/cwm/IRSTD-1k/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
tqt = 0
train_cfg = dict(
    max_epochs=300, type='EpochBasedTrainLoop_quanti', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/home2/caizy/cwm/IRSTD-1k/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('target', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='annotations/test.json',
        backend_args=None,
        data_prefix=dict(img='images/'),
        data_root='/home2/caizy/cwm/IRSTD-1k/',
        metainfo=dict(classes=('target', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/home2/caizy/cwm/IRSTD-1k/annotations/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
weight_channel_loss = 0.64
weight_feat = 0.05
weight_feat_loss = 0.08
weight_kdori = 9
weight_mine_feat = 0.18
weight_relation_loss = 2.2
weight_spatial_loss = 0.61
work_dir = 'quanti/retinanet/SSDD/std/w4a4/std35_06_02_03_23'
