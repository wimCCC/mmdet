_base_ = [
    './_base_/datasets/ssdd.py',
    './_base_/models/retinanet_r50.py',
    './_base_/quantization/ssi_lsq_w8a8.py',
    './_base_/schedules/ssdd_100e.py',
    './_base_/runtime.py',
]

# W8A8 QAT fine-tuned from a converged FP32 detector, using the recipe the
# W6A6 ablation settled on (see HGS_W6A6_ABLATION_RESULTS.md):
#
#   * start from the FP32 checkpoint, not from scratch -- by far the largest
#     single effect (+0.142 mAP on W6A6);
#   * lr 5e-3 with an early decay. The high-lr phase exists to push the weights
#     into a region that compensates quantization error, and the decay exists to
#     settle them; 5e-3..1e-2 is a plateau and 2e-2 starts to degrade;
#   * milestones [30, 50] over 60 epochs. The 200-epoch runs spend ep30-ep120
#     wandering without improving, and the real gain appears only at the
#     milestone, so the plateau is simply cut.
#
# The recipe is inlined here rather than left to --cfg-options so this file is
# reproducible as-is. (mmengine's DictAction also cannot express the nested dict
# form, which is how the ablation's sweeps had to spell out leaf keys.)
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

# The ablation's key observer finding: the SSIClipObserver std_scale sweep is
# monotonic up to no-clipping, because the brightest SAR pixels *are* the ships
# and clipping them discards the targets. At large k the observer's range
# degenerates to [min, max] anyway, so MinMaxObserver is the same thing stated
# directly. This supersedes the SSIClipObserver default inherited from the base.
#
# Activation per-channel is deliberately NOT baked in -- it is the open question
# being measured at each bit width. To run it, override:
#   extra_config.extra_qconfig_dict.a_qscheme.per_channel=True
#   extra_config.extra_qconfig_dict.a_observer_extra_args.ch_axis=1
# (ch_axis=1 is required: MQBench maps a_qscheme.per_channel to ch_axis=0, which
# is the batch axis of an (N, C, H, W) activation.)
extra_config = dict(
    extra_qconfig_dict=dict(
        a_observer='MinMaxObserver'))

work_dir = 'work_dirs/hgs_retina/ft_60e_w8a8_pt'
