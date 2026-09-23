_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r50.py',
    './_base_/quantization/ssi_lsq_w6a6.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

# W6A6 is the first diagnostic baseline after the failed W4A4 run.
# It intentionally uses the config's default initialization: no qat_pretrained.
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=5)

# Keep the initial learning rate longer so QAT can adapt before decay.
param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=200,
        milestones=[120, 170],
        gamma=0.1),
]

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w6a6_ssi_lsq_200e'
