_base_ = [
    './ssdd_retinanet_r50_w6a6_ssi_lsq_30e_ft_fp32.py',
]

# Matched W6A6 + SSI/LSQ fine-tuning baseline with only HGS compensation added.
# Keep the same FP32 checkpoint, optimizer, schedule, calibration, seed and
# data pipeline as the no-compensation run.
qat_compensation = dict(
    enabled=True,
    method='hgs_low_rank',
    rank=4,
    init='hgs',
    init_alpha=0.01,
    target='all_existing_weight_quantized_conv2d_and_linear',
    trainable=True,
    compensation_dtype='model')

# Preserve the same SSI/LSQ calibration, then initialize HGS from activations.
custom_hooks = [
    dict(type='QATCalibrationHook', num_batches=8),
    dict(
        type='HGSCalibrationHook',
        num_batches=4,
        max_samples_per_layer=128),
]

work_dir = (
    'work_dirs/hgs_retina/'
    'ssdd_retinanet_r50_w6a6_ssi_lsq_hgs_rank4_30e_ft_fp32')
