default_scope = 'mmdet'


kd = 0
quanti = 1
bit = 4
progress = True
load_from = '/home1/zhangyn/code/mmdetection-3.0.0/tools/ckp/air/faster/epoch_490_student.pth'

# if bit == 8 or bit == 6:
#     max_epochs = 10
# else:
#     max_epochs = 20

max_epochs = 300
batch_size = 8

base_lr_lp = 0.002

base_lr_tea = 0.0001



img_scale = (800, 800)

weight_feat_loss = 0.08
weight_channel_loss = 0.64
weight_spatial_loss = 0.61
weight_relation_loss = 2.2

num_classes = 1
mine = 0
dorefa = 0
lsq = 1
app = 0
mcqd = 0
qkd = 0
qfd = 0
qdcl = 0
tqt = 0
nnie = 0
qdrop = 0
dsq = 0
pact = 0
fixed = 0

qkd_p1 = 10
qkd_p2 = 15

weight_mine_feat = 0.18
weight_kdori = 9
weight_feat = 0.05

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
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
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
        'w_observer': 'ClipStdObserver',                              # custom weight observer
        'a_observer': 'ClipStdObserver',                              # custom activation observer
        # 'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        # 'a_observer': 'EMAMinMaxObserver',      
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
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
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
elif fixed:
    method = 'fixed'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'ClipStdObserver',                              # custom weight observer
        'a_observer': 'ClipStdObserver',                              # custom activation observer
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
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
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
            'symmetry': True,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif qkd:
    method = 'qkd'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
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
elif qfd:
    method = 'qfd'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
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
elif tqt:
    method = 'tqt'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'ClipStdObserver',                              # custom weight observer
        'a_observer': 'ClipStdObserver',                              # custom activation observer
        'w_fakequantize': 'TqtFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'TqtFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': True,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
    # quantize = {
    #     'quantize_type': naive_ptq, # support naive_ptq or advanced_ptq
    #     'cali_batchnum': 256,  # 越多越好？？似乎是的
    #     'quant_algorithm': tqt,
    # }
    

# training:
#     my_buff_flag: True
#     qloss_flag: True
#     fold_bn_flag: False
# misc:
#     resume: False
elif nnie:
    method = 'nnie'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'NNIEFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'NNIEFakeQuantize',                    # custom activation fake quantize function
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
elif pact:
    method = 'pact'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'PACTFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'PACTFakeQuantize',                    # custom activation fake quantize function
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
elif qdrop:
    method = 'qdrop'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'QDropFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'QDropFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': False,                                # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': True,                                    # custom whether quant is symmetric for activation,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for activation,
            'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }
elif dsq:
    method = 'dsq'
    extra_config = {
    'extra_qconfig_dict': {
        'w_observer': 'EMAMinMaxObserver',                              # custom weight observer
        'a_observer': 'EMAMinMaxObserver',                              # custom activation observer
        'w_fakequantize': 'DSQFakeQuantize',                    # custom weight fake quantize function
        'a_fakequantize': 'DSQFakeQuantize',                    # custom activation fake quantize function
        'w_qscheme': {
            'bit': bit,                                             # custom bitwidth for weight,
            'symmetry': True,                                    # custom whether quant is symmetric for weight,
            'per_channel': False,                                  # custom whether quant is per-channel or per-tensor for weight,
            'pot_scale': False,                                   # custom whether scale is power of two for weight.
        },
        'a_qscheme': {
            'bit': bit,                                             # custom bitwidth for activation,
            'symmetry': True,                                    # custom whether quant is symmetric for activation,
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

work_dir = f'quanti/faster/AIR/w{bit}a{bit}/vp_qod'
# work_dir = f'quanti/fcos/SSDD/w{bit}a{bit}/qdrop_lsq_clip'

auto_scale_lr = dict(base_batch_size=2, enable=False)
backend_args = None
data_root = ''
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=0, type='CheckpointHook'),
    # checkpoint=None,
    logger=dict(interval=50, type='LoggerHook'),
    # logger=None,
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
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
launcher = 'none'
# load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
# model settings
model = dict(
    type='FasterRCNN_quanti',
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
        in_channels=[64,128,256,512],
        out_channels=64,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=64,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=64,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.9),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.9,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))



# optim_wrapper = dict(
#     clip_grad=dict(max_norm=35, norm_type=2),
#     optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
#     paramwise_cfg=dict(bias_decay_mult=0.0, bias_lr_mult=2.0),
#     type='AmpOptimWrapper')
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr_tea, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

optim_wrapper_lp = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr = base_lr_lp, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

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
resume = False
test_cfg = dict(type='TestLoop')



# test_dataloader = dict(
#     batch_size=16,
#     dataset=dict(
#         ann_file='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/annotations/test.json',
#         backend_args=None,
#         data_prefix=dict(
#             img='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test/'),
#         data_root='',
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(keep_ratio=True, scale=(
#                 1333,
#                 800,
#             ), type='Resize'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         test_mode=True,
#         type='CocoDataset'),
#     drop_last=False,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/annotations/test.json',
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
# train_cfg = dict( type='EpochBasedTrainLoop')
train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=1)
# train_dataloader = dict(
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     batch_size=16,
#     dataset=dict(
#         ann_file='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/annotations/train.json',
#         backend_args=None,
#         data_prefix=dict(
#             img='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/train/'),
#         data_root='',
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(keep_ratio=True, scale=(
#                 1333,
#                 800,
#             ), type='Resize'),
#             dict(prob=0.5, type='RandomFlip'),
#             dict(type='PackDetInputs'),
#         ],
#         type='CocoDataset'),
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(shuffle=True, type='DefaultSampler'))
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
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.5, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1333,
                    800,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    666,
                    400,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    2000,
                    1200,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')

# val_dataloader = dict(
#     batch_size=16,
#     dataset=dict(
#         ann_file='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/annotations/test.json',
#         backend_args=None,
#         data_prefix=dict(
#             img='/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test/'),
#         data_root='',
#         pipeline=[
#             dict(backend_args=None, type='LoadImageFromFile'),
#             dict(keep_ratio=True, scale=(
#                 1333,
#                 800,
#             ), type='Resize'),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 meta_keys=(
#                     'img_id',
#                     'img_path',
#                     'ori_shape',
#                     'img_shape',
#                     'scale_factor',
#                 ),
#                 type='PackDetInputs'),
#         ],
#         test_mode=True,
#         type='CocoDataset'),
#     drop_last=False,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/test.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/images/'
        ),
        data_root='',
        metainfo=dict(classes='ship', palette=[
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
    ann_file=
    '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/test.json',
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
train_cfg = dict(
    max_epochs=300, type='EpochBasedTrainLoop_quanti', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file=
        '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/train.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/images/'
        ),
        data_root='',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes='ship', palette=[
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
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.5, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1333,
                    800,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    666,
                    400,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    2000,
                    1200,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/test.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/images/'
        ),
        data_root='',
        metainfo=dict(classes='ship', palette=[
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
    ann_file=
    '/home1/zhangyn/code/mmdetection-3.0.0/data/airsarship/voc/test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])


train_cfg = dict(type='EpochBasedTrainLoop_quanti', max_epochs=max_epochs, val_interval=1)
# train_cfg = dict(type='EpochBasedTrainLoop_quanti_meta', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
