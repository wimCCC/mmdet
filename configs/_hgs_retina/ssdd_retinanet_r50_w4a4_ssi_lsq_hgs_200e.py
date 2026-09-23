_base_ = ['./ssdd_retinanet_r50_w4a4_ssi_lsq_hgs_100e.py']

# Optional continuation target; use --resume from the 100-epoch checkpoint.
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=5)

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
        end=200,
        milestones=[60, 90],
        gamma=0.1),
]

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w4a4_ssi_lsq_hgs_200e'
