_base_ = ['./ssdd_retinanet_r50_fp32_30e_diagnose.py']

# Match the repository's working RetinaNet optimization scale.
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
        end=30,
        milestones=[20, 27],
        gamma=0.1),
]

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_30e_lr3e-3'
