import pytest
import torch

mqbench = pytest.importorskip('mqbench')

from mqbench.observer import ClipStdObserver  # noqa: E402

from mmdet.models.quantization import LSQSRQFakeQuantize  # noqa: E402
from mmdet.models.quantization import (  # noqa: E402
    register_mqbench_fake_quantizers)


def _build(prob):
    return LSQSRQFakeQuantize(
        observer=ClipStdObserver,
        prob=prob,
        quant_min=-8,
        quant_max=7,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
        ch_axis=-1)


def test_lsq_srq_prob_zero_bypasses_activation_during_training():
    fake_quant = _build(prob=0.0).train()
    x = torch.randn(2, 3, requires_grad=True)
    output = fake_quant(x)
    torch.testing.assert_close(output, x)


def test_lsq_srq_is_available_to_mqbench_string_config():
    register_mqbench_fake_quantizers()
    from mqbench.prepare_by_platform import FakeQuantizeDict
    assert FakeQuantizeDict['LSQSRQFakeQuantize'] is LSQSRQFakeQuantize
