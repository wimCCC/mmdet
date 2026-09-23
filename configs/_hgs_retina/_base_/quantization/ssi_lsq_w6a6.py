# Shared W6A6 SSI + LSQ quantization recipe.
runner_type = 'QATCompensationRunner'
bit = 6

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

# SSI + LSQ only; no SRQ and no HGS compensation.
qat_compensation = dict(enabled=False)

# SSI observers run on a few FP32 batches, then observers are frozen so the
# LSQ scales are actually learned instead of being overwritten every step.
custom_hooks = [
    dict(type='QATCalibrationHook', num_batches=8),
]

# The network input is already an 8-bit image; keep it unquantized so the
# clipping observer does not clip bright (ship) pixels.
qat_fp32_activation_patterns = [r'^inputs_post_act_fake_quantizer$']
