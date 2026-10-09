_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r50.py',
    './_base_/quantization/ssi_lsq_w4a4.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

# W4A4 counterpart of ssdd_retinanet_r50_w8a8_ssi_lsq_ft_fp32_60e.py: the same
# W6A6-derived fine-tune recipe, so the two bit widths are directly comparable
# and sit on one curve with FP32 and the existing W6A6 runs.
#
# The recipe (see HGS_W6A6_ABLATION_RESULTS.md for the evidence):
#   * start from the converged FP32 checkpoint -- the single largest effect;
#   * lr 5e-3 with an early decay -- the high-lr phase buys compensation of the
#     quantization error, the decay settles it;
#   * milestones [30, 50] over 60 epochs -- the 200-epoch plateau carries no
#     measurable accuracy, so it is cut.
qat_pretrained = ('work_dirs/hgs_retina/ssdd_retinanet_r50_fp32_200e_lr3e-3/'
                  'best_coco_bbox_mAP_50_epoch_130.pth')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
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
        end=60,
        milestones=[30, 50],
        gamma=0.1),
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=60,
    val_interval=5)

# Same observer reasoning as the W8A8 config: the std_scale sweep is monotonic
# up to no-clipping, and no-clipping is exactly MinMaxObserver. Supersedes the
# SSIClipObserver default inherited from the base.
#
# Activation per-channel is left as an override -- it is the open question at
# low bit width (W6A6 showed +0.009/+0.011 over two seeds, W4A4 showed -0.009 on
# one seed). To run it:
#   extra_config.extra_qconfig_dict.a_qscheme.per_channel=True
#   extra_config.extra_qconfig_dict.a_observer_extra_args.ch_axis=1
extra_config = dict(
    extra_qconfig_dict=dict(
        a_observer='MinMaxObserver'))

work_dir = 'work_dirs/hgs_retina/ft_60e_w4a4_pt'
