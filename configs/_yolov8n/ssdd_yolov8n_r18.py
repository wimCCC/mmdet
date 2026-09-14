custom_imports = dict(
    imports=['mmyolo'],
    allow_failed_imports=False
)

default_scope = 'mmyolo'


kd = 0
quanti = 1
bit = 4
progress = True
load_from = '/home1/zhangyn/code/mmdetection-3.0.0/mmyolo/tools/ckp/ssdd/yolov8n/best_coco_bbox_mAP_50_epoch_95.pth'

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

# work_dir = f'quanti/fcos/SSDD/w{bit}a{bit}/{method}'
# work_dir = f'quanti/fcos/SSDD/prob/w{bit}a{bit}/0/{method}'
work_dir = f'quanti/yolov8n/SSDD/prob/w{bit}a{bit}/0/{method}'

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
# default_scope = 'mmdet'
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
data_root = '' # Root path of data
# Path of train annotation file
train_ann_file = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/annotations/train.json'
train_data_prefix = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/annotations/test.json'
val_data_prefix = '/home1/zhangyn/code/mmdetection-3.0.0/data/SSDD/images/test/'  # Prefix of val image path


# ========================Frequently modified parameters======================

num_classes = 1  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.001
max_epochs = 100 # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

model_test_cfg = dict(
    multi_label=True,
    nms_pre=1000,
    score_thr=0.3,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=300
)



# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.25
# Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals in stage 1
save_epoch_intervals = 1
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 2
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg)

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']
        ),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'ori_shape',
            'img_shape',
            'flip',
            'flip_direction'
        )
    )
]



train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id',  'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=lr_factor,
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='coco/bbox_mAP_50',
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        val_interval_stage2)])

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


train_cfg = dict(type='EpochBasedTrainLoop_quanti', max_epochs=max_epochs, val_interval=1)
# train_cfg = dict(type='EpochBasedTrainLoop_quanti_meta', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
