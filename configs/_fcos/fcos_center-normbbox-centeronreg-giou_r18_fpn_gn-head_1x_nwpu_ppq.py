_base_ = [
    '../_base_/default_runtime.py'
]

quanti = 1
img_scale = (1024, 1024)
load_from = '/root/autodl-tmp/2.26model/NWPU_s/mine/fcos/best_bbox_mAP_epoch_29_8830.pth'
# work_dir = 'models/NWPU_s/fcos_resize_pad_from_2.26_2'
work_dir = f'models/NWPU_s/fcos_ppq_rknnPerchannel_{img_scale[0]}_{img_scale[1]}'

model = dict(
    type='FCOS_quanti',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            # checkpoint='open-mmlab://detectron/resnet50_caffe')),
            checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        # out_channels=256,
        out_channels=64,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=10,
        # in_channels=256,
        in_channels=64,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        # loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
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

# dataset settings
dataset_type = 'NWPUDataset'  # 数据集类型，这将被用来定义数据集。
data_root = '/root/autodl-tmp/nwpu/'  # 数据的根路径。
backend_args = None




# img_scale = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=0),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]



train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop_quanti', max_epochs=13, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')




base_lr = 5e-5
# 低精度模型学习率相关设置
param_scheduler_lp = [
    # dict(
        # type='LinearLR', start_factor=0.0001, by_epoch=True, begin=0, end=1),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=20,
    #     by_epoch=True,
    #     milestones=[14,],
    #     gamma=0.1),
    dict(
        type='CosineAnnealingLR',
        T_max=13,
        eta_min=base_lr * 0.01,
        begin=0,
        end=13,
        by_epoch=True)
]
# optimizer
# optim_wrapper_lp=dict(
#     type='OptimWrapper',
#     optimizer=dict(type='DAdaptAdaGrad', lr=0.001, momentum=0.9),
#     clip_grad=dict(max_norm=35, norm_type=2))
optim_wrapper_lp = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))







# 全精度模型学习率相关设置
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=300),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[14,],
        gamma=0.1)
]
# optimizer
# optim_wrapper=dict(
#     type='OptimWrapper',
#     optimizer=dict(type='DAdaptAdaGrad', lr=0.001, momentum=0.9),
#     clip_grad=dict(max_norm=35, norm_type=2))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP_50',
        interval=-1))
