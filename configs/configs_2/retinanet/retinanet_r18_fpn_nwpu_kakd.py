_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/nwpu.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model_t_cfg = 'NWPU_t/retinanet/retinanet_r50_fpn_nwpu.py'
model_t_ckpt = 'NWPU_t/retinanet/best_bbox_mAP_50_epoch_27_9010.pth'

# metanet_1 是细粒度元学习器，用于为不同尺寸特征图的蒸馏损失分配权重
metanet_1_hidden_layer_sizes = [256, 128]
# 5个不同尺度特征图经过平均池化后的shape是256，concat在一起是 1280，还有5个是每个特征图的损失值 (per instance)
# metanet_1_input_size = 1285
metanet_1_input_size = 10
metanet_1_output_size = 5
meta_1_lr = 0.02
meta_1_weight_decay = 0.

# metanet_2 是粗粒度元学习器，用于为学生的不同种类的损失分配权重
metanet_2_hidden_layer_sizes = [256, 128]
# 45个不同尺度特征图经过平均池化后的shape是256，concat在一起是 1280，还有5个是 5 中不同类别损失 (per instance)
# metanet_2_input_size = 1285
metanet_2_input_size = 10
metanet_2_output_size = 5
meta_2_lr = 0.02
meta_2_weight_decay = 0.
# 是否使用metaset
metaset = 0

# 按照 iter 为单位, 不是按照epoch
meta_net_update_interval = 10

model = dict(
    type='RetinaNet_TAED',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        # out_channels=256,
        out_channels=64,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=10,
        # in_channels=256,
        in_channels=64,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        # loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
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
        img_scale=(1000, 600),
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
dataset_type = 'NWPUDataset'
data_root = '/root/autodl-tmp/nwpu/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'instances_train2017.json',
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'instances_val2017.json',
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    meta=dict(
        type=dataset_type,
        # ann_file=data_root + 'instances_val2017.json',
        ann_file=data_root + 'metaset.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'instances_val2017.json',
        ann_file=data_root + 'test.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))


# optimizer
# 当前最好:0.005
optimizer = dict(
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,  
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=30,
    gamma=0.1)
runner = dict(type='EpochBasedRunner_TAED', max_epochs=40)

checkpoint_config = dict(interval=-1)  # 保存的间隔是 1
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')

# work_dir = 'NWPU_s/retinanet_3_quarter'
# work_dir = 'NWPU_taed/retinanet_2'
# work_dir = 'NWPU_taed/retinanet_taed_newBL_3'
work_dir = 'NWPU_taed/retinanet_taed_KAKD_test_1'

fp16 = dict(loss_scale='dynamic')

num_classes = 10

seed = 3407
deterministic = False