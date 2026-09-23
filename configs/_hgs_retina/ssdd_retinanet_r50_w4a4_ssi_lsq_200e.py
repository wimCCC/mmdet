_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r50.py',
    './_base_/quantization/ssi_lsq_w4a4.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=5)

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=200,
        milestones=[60, 90],
        gamma=0.1),
]

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w4a4_ssi_lsq_200e'
