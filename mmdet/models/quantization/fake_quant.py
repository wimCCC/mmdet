# Copyright (c) OpenMMLab. All rights reserved.
"""Fake quantizers combining SSI/LSQ with stochastic rounding bypass."""

import torch

from mmdet.registry import MODELS

try:
    from mqbench.fake_quantize import LearnableFakeQuantize
except ImportError:  # pragma: no cover - MQBench is an optional dependency.
    LearnableFakeQuantize = None


if LearnableFakeQuantize is not None:

    @MODELS.register_module()
    class LSQSRQFakeQuantize(LearnableFakeQuantize):
        """LSQ fake quantization followed by SRQ/QDrop stochastic bypass.

        The observer is still selected by MQBench. Using ``ClipStdObserver``
        therefore keeps SSI, while this class keeps LSQ's gradient-scaled
        learnable scale and applies SRQ only to activations during training.
        """

        def __init__(self, observer, prob: float = 1.0, **observer_kwargs):
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f'prob must be in [0, 1], but got {prob}')
            super().__init__(observer, **observer_kwargs)
            self.prob = float(prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            quantized = super().forward(x)
            if (self.training and self.fake_quant_enabled[0] == 1
                    and self.prob < 1.0):
                mask = torch.rand_like(quantized) < self.prob
                return torch.where(mask, quantized, x)
            return quantized

else:

    class LSQSRQFakeQuantize:  # pragma: no cover
        """Import-time placeholder producing an actionable error."""

        def __init__(self, *args, **kwargs):
            raise ImportError('LSQSRQFakeQuantize requires MQBench')


def register_mqbench_fake_quantizers() -> None:
    """Expose local fake quantizers/observers to MQBench's string registry."""
    if LearnableFakeQuantize is None:
        raise ImportError('QAT preparation requires MQBench')
    from mqbench.prepare_by_platform import FakeQuantizeDict, ObserverDict

    from .observer import SSIClipObserver

    FakeQuantizeDict.setdefault('LSQSRQFakeQuantize', LSQSRQFakeQuantize)
    ObserverDict.setdefault('SSIClipObserver', SSIClipObserver)
