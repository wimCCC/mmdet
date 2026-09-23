_base_ = ['./ssdd_retinanet_r50_w4a4_ssi_lsq_srq_100e.py']

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

work_dir = 'work_dirs/hgs_retina/ssdd_retinanet_r50_w4a4_ssi_lsq_srq_200e'

# QAT must start from the trained FP32 detector when using the very small
# W4A4 learning rate; otherwise the randomly initialized RetinaHead cannot
# recover from the large initial quantization loss.
qat_pretrained = 'work_dirs/ssdd_retina_r50_fp32_100e/best_coco_bbox_mAP_50_epoch_100.pth'
