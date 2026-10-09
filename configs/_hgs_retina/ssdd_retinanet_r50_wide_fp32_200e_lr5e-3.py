_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r50_wide.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

# FP32 control at upstream FPN/head width (256 instead of the inherited 64),
# everything else identical to
# ssdd_retinanet_r50_fp32_200e_lr3e-3.py -- same data pipeline, schedule,
# warmup, seed and epoch budget. The only variable against the narrow baseline
# is the detection-head capacity.
#
# lr is 5e-3 rather than the narrow model's 3e-3 because the 2026-10-09 lr
# sweep showed 5e-3 >= 3e-3 within noise and 1e-2 collapses, so 5e-3 is the top
# of the usable range and is also upstream's own batch-8 RetinaNet value
# (configs/retinanet/retinanet_r18_fpn_1xb8-1x_coco.py:19). Note a wider model
# may not share the narrow one's lr ceiling -- if this run underperforms, lr is
# the first thing to re-probe.
runner_type = 'Runner'
model = dict(type='RetinaNet')
extra_config = None
qat_compensation = dict(enabled=False)
custom_hooks = []

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
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

work_dir = 'work_dirs/hgs_retina/fp32_wide_200e_lr5e-3'
