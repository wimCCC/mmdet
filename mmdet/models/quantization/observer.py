# Copyright (c) OpenMMLab. All rights reserved.
"""Statistics-based clipping observer for SSI initialization."""

import torch

try:
    from mqbench.observer import ObserverBase
except ImportError:  # pragma: no cover - MQBench is an optional dependency.
    ObserverBase = None


if ObserverBase is not None:

    class SSIClipObserver(ObserverBase):
        """Clip the quantization range to ``mean +- std_scale * std``.

        MQBench's ``ClipStdObserver`` *widens* the range to
        ``min(mean - k*std, min)`` / ``max(mean + k*std, max)``. For post-ReLU
        activations this puts the lower bound below zero (wasting levels) and
        keeps the upper bound at the batch outlier maximum, so most values
        collapse onto a handful of levels. This observer performs the actual
        SSI clipping instead, intersected with the observed min/max so that
        the range never exceeds the data.
        """

        def __init__(self, dtype=torch.quint8,
                     qscheme=torch.per_tensor_affine, reduce_range=False,
                     quant_min=None, quant_max=None, ch_axis=-1,
                     pot_scale=False, std_scale=3.0, factory_kwargs=None):
            super().__init__(dtype, qscheme, reduce_range, quant_min,
                             quant_max, ch_axis, pot_scale,
                             factory_kwargs=None)
            self.std_scale = float(std_scale)

        def forward(self, x_orig):
            if x_orig.numel() == 0:
                return x_orig
            x = x_orig.detach().to(self.min_val.dtype)
            if self.ch_axis == -1:
                min_cur, max_cur = torch.aminmax(x)
                mean = x.mean()
                std = x.std()
            else:
                axes = list(range(x.dim()))
                axes[self.ch_axis], axes[0] = 0, self.ch_axis
                y = torch.flatten(x.permute(axes), start_dim=1)
                min_cur, max_cur = torch.aminmax(y, dim=1)
                mean = y.mean(1)
                std = y.std(1)
            self.min_val = torch.maximum(mean - self.std_scale * std, min_cur)
            self.max_val = torch.minimum(mean + self.std_scale * std, max_cur)
            return x_orig

else:

    class SSIClipObserver:  # pragma: no cover
        """Import-time placeholder producing an actionable error."""

        def __init__(self, *args, **kwargs):
            raise ImportError('SSIClipObserver requires MQBench')
