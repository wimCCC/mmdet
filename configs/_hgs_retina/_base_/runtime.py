_base_ = ['../../_base_/default_runtime.py']

# Shared runtime policy for HGS RetinaNet experiments.
default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',
        rule='greater'),
    logger=dict(type='LoggerHook', interval=50))

randomness = dict(seed=2026, deterministic=False)
