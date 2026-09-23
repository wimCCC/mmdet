_base_ = [
    './ssdd_retinanet_r50_w6a6_ssi_lsq_30e_lr3e-3.py',
]

# QAT fine-tuning from the converged FP32 detector. Activation ranges are
# calibrated on a trained network, so the frozen SSI ranges are meaningful and
# HGS-style compensation targets a real W_ref. Same 30-epoch schedule as the
# FP32 baseline, at a 3x lower learning rate because we start from a solution.
qat_pretrained = ('work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_30e_lr3e-3/'
                  'best_coco_bbox_mAP_50_epoch_30.pth')

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

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_30e_ft_fp32'
