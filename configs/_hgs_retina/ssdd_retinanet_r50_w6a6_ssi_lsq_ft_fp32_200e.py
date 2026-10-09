_base_ = ['./ssdd_retinanet_r50_w6a6_ssi_lsq_200e.py']

# W6A6 QAT fine-tuned from a converged FP32 detector, for the *full* 200-epoch
# budget used by the from-scratch run and the FP32 control.
#
# The 30-epoch variant (ssdd_retinanet_r50_w6a6_ssi_lsq_30e_ft_fp32_200e.py)
# already recovered 0.448 -> 0.549 mAP over the from-scratch W6A6 and was still
# improving when its schedule ended, so this run hands the same recipe the
# 200-epoch budget with the matching [120, 170] milestones.
qat_pretrained = ('work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_200e_lr3e-3/'
                  'best_coco_bbox_mAP_50_epoch_130.pth')

# Non-quantization hyperparameters mirror ssdd_retinanet_r50_w6a6_ssi_lsq_200e_
# lr3e-3.py exactly (same warmup length, milestones, epoch budget, val
# interval) so the only remaining difference from the from-scratch run is the
# starting point and the learning rate. Probe the rate with
# --cfg-options optim_wrapper.optimizer.lr=...
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
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=200,
        milestones=[120, 170],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=10)

work_dir = 'work_dirs/hgs_retina/ft_w6a6_200e_lr1e-3'
