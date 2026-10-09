_base_ = ['./ssdd_retinanet_r50_w6a6_ssi_lsq_200e.py']

# W6A6 QAT fine-tuned from a *converged* FP32 detector instead of from scratch.
#
# Motivation: the from-scratch W6A6 run (ssdd_retinanet_r50_w6a6_ssi_lsq_200e_
# lr3e-3.py) lost 0.197 mAP / 0.117 mAP_50 against the matched 200-epoch FP32
# control, which is far more than 6-bit quantization usually costs. This run
# isolates the cause: if fine-tuning recovers most of the gap, the from-scratch
# recipe was the problem; if it does not, the loss is inherent to quantizing
# this architecture.
#
# Unlike the 200-epoch from-scratch run, activation ranges here are calibrated
# on a trained network, so the frozen SSI ranges are meaningful rather than
# being fit to an untrained model's activations.
qat_pretrained = ('work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_200e_lr3e-3/'
                  'best_coco_bbox_mAP_50_epoch_130.pth')

# The repo's ft_fp32 convention is lr = 0.001 (3x below the from-scratch
# 0.003). Override with --cfg-options to probe sensitivity.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={'fake_quant': dict(decay_mult=0.0)}),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=30,
        milestones=[20, 27],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=5)

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_30e_ft_fp32_200e'
