_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/nwpu.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
kd = 0
quanti = 1
bit = 4
progress = True
# load_from = '/root/autodl-tmp/2.26model/NWPU_s/modified_retinanet_mine_r18_nwpu_8730.pth'
load_from = '/root/autodl-tmp/2.26model/NWPU_s/mine/retinanet/best_bbox_mAP_epoch_31_8930.pth'
# work_dir = f'models/quanti/NWPU/w{bit}a{bit}/retinanet_dorefa_2'


base_lr_lp = 0.0005
base_lr_tea = 0.0005
img_scale = (1024, 608)

weight_feat_loss = 0.08
weight_channel_loss = 0.64
weight_spatial_loss = 0.61
weight_relation_loss = 2.2

num_classes = 10
mine = 1
dorefa = 0
lsq = 0
app = 0
mcqd = 0
qkd = 0
qfd = 0
qdcl = 1

qkd_p1 = 10
qkd_p2 = 15

weight_mine_feat = 0.18
weight_kdori = 9
weight_feat = 0.1

method = ''
if mine:
    method = 'mine'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'LearnableFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'LearnableFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': False,                                    # custom whether quant is symmetric for weight,
            'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif dorefa:
    method = 'dorefa'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'DoReFaFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'DoReFaFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif lsq:
    method = 'lsq'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'LearnableFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'LearnableFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif app:
    method = 'app'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif mcqd:
    method = 'mcqd'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif qkd:
    method = 'qkd'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif qfd:
    method = 'qfd'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif qdcl:
    method = 'qdcl'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
else:
    method = 'only_qat'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'MinMaxObserver',                              # custom weight observer
        'a_observer': 'MinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': False,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }

work_dir = f'quanti/retinanet/NWPU/w{bit}a{bit}/{method}'


# model
model = dict(
    type='RetinaNet_quanti',
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



backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size_divisor=32),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


dataset_type = 'NWPUDataset'  # 数据集类型，这将被用来定义数据集。
data_root = '/root/autodl-tmp/nwpu/'  # 数据的根路径。
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='remaining_data.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
meta_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='sampled_data.json',
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



# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop_quanti', max_epochs=20, val_interval=1)
# train_cfg = dict(type='EpochBasedTrainLoop_quanti_meta', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')



# 低精度模型学习率相关设置
# optimizer
optim_wrapper_lp = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr = base_lr_lp, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler_lp = [
    # dict(
    #     type='LinearLR', start_factor=0.0005, by_epoch=True, begin=0, end=1),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[10, 15],
        gamma=0.5)
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=20,
    #     eta_min= base_lr_lp * 0.2,
    #     begin=1,
    #     end=20,
    #     by_epoch=True)
]

# 全精度模型学习率相关设置
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.00001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=20,
        by_epoch=True,
        milestones=[10,],
        gamma=0.1)
]
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr_tea, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))




default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='coco/bbox_mAP_50',
        interval=-1))

