_base_ = ['./ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_200e.py']

# Compressed fine-tune schedule, 60 epochs instead of 200.
#
# Why: the 200-epoch runs all show the same shape. Adaptation to quantization
# is essentially finished within ~10-30 epochs at the high LR; the model then
# wanders for ~90 epochs without improving (lr 5e-3 spends ep30-ep120 between
# mAP 0.50 and 0.56), and the real gain appears only at the ep120 milestone,
# where the 10x LR drop takes mAP from 0.5410 to 0.6360 in a single validation
# interval. The [170] milestone adds nothing: ep130 -> ep200 moves mAP by
# ~0.005.
#
# So the high-LR phase exists to adapt, and the decay exists to settle; the
# wandering in between is wasted. This schedule keeps the same shape at 30%
# of the length. Warmup stays at 500 iterations for consistency with the
# other runs even though that is a larger fraction of a 60-epoch budget.
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
        end=60,
        milestones=[30, 50],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=60,
    val_interval=5)

work_dir = 'work_dirs/hgs_retina/ft_60e_lr5e-3_std8.0'
