# Shared W4A4 SSI + LSQ + SRQ quantization recipe.
# SRQ is applied to activations during training; weights use standard LSQ.
runner_type = 'QATCompensationRunner'
bit = 4
srq_prob = 0.25

extra_config = dict(
    extra_qconfig_dict=dict(
        w_observer='ClipStdObserver',
        a_observer='SSIClipObserver',
        w_fakequantize='LearnableFakeQuantize',
        a_fakequantize='LSQSRQFakeQuantize',
        w_qscheme=dict(
            bit=bit,
            symmetry=True,
            per_channel=True,
            pot_scale=False),
        a_qscheme=dict(
            bit=bit,
            symmetry=False,
            per_channel=False,
            pot_scale=False),
        a_fakeq_params=dict(prob=srq_prob)))

# No HGS compensation in this ablation. HGS is a separate compensation
# mechanism and is intentionally not combined with this SSI + LSQ + SRQ base.
qat_compensation = dict(enabled=False)

custom_hooks = [
    dict(type='QATCalibrationHook', num_batches=8),
]

# The network input is already an 8-bit image; keep it unquantized so the
# clipping observer does not clip bright (ship) pixels.
qat_fp32_activation_patterns = [r'^inputs_post_act_fake_quantizer$']
