# Shared W4A4 SSI + LSQ + low-rank HGS quantization recipe.
runner_type = 'QATCompensationRunner'
bit = 4

extra_config = dict(
    extra_qconfig_dict=dict(
        w_observer='ClipStdObserver',
        a_observer='SSIClipObserver',
        w_fakequantize='LearnableFakeQuantize',
        a_fakequantize='LearnableFakeQuantize',
        w_qscheme=dict(
            bit=bit,
            symmetry=True,
            per_channel=True,
            pot_scale=False),
        a_qscheme=dict(
            bit=bit,
            symmetry=False,
            per_channel=False,
            pot_scale=False)))

qat_compensation = dict(
    enabled=True,
    method='hgs_low_rank',
    rank=4,
    init='hgs',
    init_alpha=0.01,
    target='all_existing_weight_quantized_conv2d_and_linear',
    trainable=True,
    compensation_dtype='model')

# QATCalibrationHook (HIGHEST) must run before HGSCalibrationHook (VERY_HIGH)
# so HGS sees the final quantizer scales with observers already frozen.
custom_hooks = [
    dict(type='QATCalibrationHook', num_batches=8),
    dict(
        type='HGSCalibrationHook',
        num_batches=4,
        max_samples_per_layer=128),
]

# The network input is already an 8-bit image; keep it unquantized so the
# clipping observer does not clip bright (ship) pixels.
qat_fp32_activation_patterns = [r'^inputs_post_act_fake_quantizer$']
