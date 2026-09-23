# Shared SSDD optimization and 100-epoch W8A8 training schedule.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=100,
        milestones=[60, 90],
        gamma=0.1),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
