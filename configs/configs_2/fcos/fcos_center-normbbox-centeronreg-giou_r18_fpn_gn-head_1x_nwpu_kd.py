_base_ = [
    '../_base_/datasets/nwpu.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model_t_cfg = 'NWPU_t/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_nwpu.py'
model_t_ckpt = 'NWPU_t/fcos/best_bbox_mAP_50_epoch_21_9040.pth'


model = dict(
    type='FCOS_KD',
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



optimizer_config = dict(_delete_=True, grad_clip=None)
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.003,
    step=30,
    gamma=0.1)

runner = dict(type='EpochBasedRunner_TAED', max_epochs=40)
checkpoint_config = dict(interval=-1)  # 保存的间隔是 1
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')

work_dir = 'NWPU_s/mine/fcos'

num_classes = 10

seed = 3407
deterministic = False
