_base_ = ['./ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_200e.py']

# SUPERSEDED -- do not use for new runs. Kept only to document the reasoning
# that led to the compressed schedules.
#
# This was written before the 60-epoch schedule was validated. It keeps the
# milestone at absolute epoch 120 because relocating it was still an
# unverified assumption at the time. That assumption has since been tested:
# ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_60e.py moves the milestones to
# [30, 50] and matches the 200-epoch result at 30% of the cost. Use that
# instead; 150e only trims the plateau without compressing the adaptation
# phase, so it is strictly worse value.
#
# Original note follows.
#
# Fine-tune schedule of 150 epochs instead of 200.
#
# Justification is empirical, from ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_200e:
# validation is flat from the first milestone onward. The 10x LR drop at
# milestone 120 is what produces the final jump (ep120 mAP 0.6220 -> ep130
# 0.6270 for lr 3e-3; 0.5410 -> 0.6360 for lr 5e-3), and the last 70 epochs
# then add 0.005 mAP in total, which is noise. The [170] milestone never does
# anything measurable.
#
# The milestone is therefore deliberately kept at the *absolute* epoch 120
# rather than scaled to 60% of the schedule. The wandering before 120 is not
# needed for accuracy, but relocating the milestone is an assumption that the
# adaptation phase itself can be compressed -- 100e / 60e variants are being
# measured separately, and if those hold up this can move earlier still.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
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
        end=150,
        milestones=[120],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=150,
    val_interval=10)

work_dir = 'work_dirs/hgs_retina/ft_150e_lr5e-3_std8.0'
