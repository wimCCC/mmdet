_base_ = ['./ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_200e.py']

# Minimum-schedule probe, 40 epochs, milestones [20, 33] at the 60%/85% shape.
#
# Context: ssdd_retinanet_r50_w6a6_ssi_lsq_ft_fp32_60e.py already matched the
# 200-epoch result at 30% of the cost (mAP 0.6290 / mAP_50 0.9240 / best
# mAP_50 0.9270, versus 0.6330 / 0.9190 / 0.9220 for 200 epochs at the same lr
# and std_scale). Its validation shows the whole shape in 60 epochs:
# ep30 = 0.5450 at the high lr, then the ep30 milestone drops lr 10x and
# ep35 jumps to 0.6210, plateauing by ep40.
#
# If 20 epochs of high lr is still enough to finish the quantization
# adaptation, this should land in the same place. If it undershoots, the
# adaptation phase is the binding constraint and 60 epochs is the floor.
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
        end=40,
        milestones=[20, 33],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=5)

work_dir = 'work_dirs/hgs_retina/ft_40e_lr5e-3_std8.0'
