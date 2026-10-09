_base_ = ['./ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_200e.py']

# Mid-length fine-tune schedule, 100 epochs. Companions:
#   ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_60e.py   (aggressive)
#   ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_200e.py  (original)
#
# Milestones keep the 200-epoch schedule's proportions (60% / 85%), so this is
# the same shape compressed 2x rather than a new recipe. Running 60/100/200
# against each other locates the shortest schedule that still reaches the
# plateau; see ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_60e.py for why the long
# high-LR phase is suspected to be mostly wasted.
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
        end=100,
        milestones=[60, 85],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=5)

work_dir = 'work_dirs/hgs_retina/ft_100e_lr5e-3_std8.0'
