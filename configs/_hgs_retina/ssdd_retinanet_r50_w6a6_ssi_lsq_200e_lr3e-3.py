_base_ = [
    './ssdd_retinanet_r50_w6a6_ssi_lsq_200e.py',
]

# Long QAT run using the validated FP32 optimization recipe.
# No qat_pretrained, SRQ, or HGS compensation is used.
# LSQ scales/zero-points must not be pulled towards zero by weight decay.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.003, momentum=0.9, weight_decay=0.0001),
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

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e_lr3e-3'
