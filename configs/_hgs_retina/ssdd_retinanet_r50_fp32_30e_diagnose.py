_base_ = ['./ssdd_retinanet_r50_fp32_100e.py']

# Diagnostic only: verify the detector/data/evaluator without MQBench QAT.
# Uses the config's normal initialization and does not load qat_pretrained.
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=5)

param_scheduler = [
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=30,
        milestones=[20, 27],
        gamma=0.1),
]

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_30e_diagnose'
