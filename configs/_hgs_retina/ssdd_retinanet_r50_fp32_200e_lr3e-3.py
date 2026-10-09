_base_ = ['./ssdd_retinanet_r50_fp32_100e.py']

# Matched FP32 control for the validated 200-epoch QAT run. Everything except
# quantization is identical to
# ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3.py: same data pipeline, same
# seed (randomness in _base_/runtime.py), same SGD recipe, same warmup and
# milestone schedule, same epoch budget. Comparing the two gives the actual
# W6A6 quantization drop.
#
# The optimizer's `paramwise_cfg` for `fake_quant` is intentionally absent:
# it exists only to stop weight decay from pulling LSQ scales/zero-points
# towards zero, and there are no such parameters without QAT.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001),
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

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_200e_lr3e-3'
