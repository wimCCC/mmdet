# Copyright (c) OpenMMLab. All rights reserved.
"""Injection utilities for QAT low-rank compensation."""

import re
import types
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from mmdet.registry import MODELS


def _forward_with_qat_compensation(module: nn.Module,
                                   x: torch.Tensor) -> torch.Tensor:
    """Run the existing QAT operation and add a residual at its output."""
    output = module._forward_without_qat_compensation(x)
    return output + module.qat_compensation(x)


@MODELS.register_module()
class QATCompensationInjector:
    """Inject trainable compensation into existing weight-quantized layers.

    A target must be a Conv2d/Linear-like module with ``weight_fake_quant``.
    The module itself is not wrapped, so existing checkpoint keys for weights
    and LSQ parameters remain unchanged.
    """

    def __init__(self,
                 enabled: bool = True,
                 method: str = 'hgs_low_rank',
                 rank: int = 4,
                 init: str = 'hgs',
                 init_alpha: float = 0.01,
                 target: str =
                 'all_existing_weight_quantized_conv2d_and_linear',
                 trainable: bool = True,
                 compensation_dtype: str = 'model',
                 exclude: Optional[list] = None):
        if method != 'hgs_low_rank':
            raise ValueError(f'Unsupported compensation method: {method}')
        if init not in ('hgs', 'fallback'):
            raise ValueError(f'Unsupported initialization: {init}')
        if target != 'all_existing_weight_quantized_conv2d_and_linear':
            raise ValueError(f'Unsupported target policy: {target}')
        if compensation_dtype != 'model':
            raise ValueError('Only compensation_dtype="model" is supported')
        self.enabled = enabled
        self.rank = rank
        self.init = init
        self.init_alpha = init_alpha
        self.trainable = trainable
        self.exclude = [re.compile(pattern) for pattern in (exclude or [])]

    def __call__(self, model: nn.Module) -> int:
        if not self.enabled:
            return 0
        injected = 0
        # named_modules removes duplicate shared-module references by default.
        for name, module in list(model.named_modules()):
            if self._excluded(name) or hasattr(
                    module, 'qat_compensation'):
                continue
            if not hasattr(module, 'weight_fake_quant') or not hasattr(
                    module, 'weight'):
                continue
            weight = module.weight
            if weight.ndim == 4:
                compensation = MODELS.build(dict(
                    type='LowRankConv2dCompensation',
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    padding_mode=module.padding_mode,
                    rank=self.rank))
            elif weight.ndim == 2:
                compensation = MODELS.build(dict(
                    type='LowRankLinearCompensation',
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=self.rank))
            else:
                continue
            compensation.to(device=weight.device, dtype=weight.dtype)
            compensation.requires_grad_(self.trainable)
            module.add_module('qat_compensation', compensation)
            module._forward_without_qat_compensation = module.forward
            module.forward = types.MethodType(
                _forward_with_qat_compensation, module)
            injected += 1
        return injected

    def initialize_hgs(self, model: nn.Module,
                       moments: Mapping[str, torch.Tensor]) -> Dict[str, str]:
        """Initialize injected layers from uncentered input second moments.

        ``moments`` maps module names to ``[in, in]`` tensors, or to
        ``[groups, in_per_group * kh * kw, ...]`` for grouped convolutions.
        Layers without a supplied moment retain the explicit Xavier/zero
        fallback and are reported as ``fallback``.
        """
        solver = MODELS.build(dict(type='HGSSolver', alpha=self.init_alpha))
        status = {}
        for name, module in model.named_modules():
            compensation = getattr(module, 'qat_compensation', None)
            if compensation is None:
                continue
            moment = moments.get(name)
            if self.init != 'hgs' or moment is None:
                status[name] = 'fallback'
                continue
            with torch.no_grad():
                quantized = module.weight_fake_quant(module.weight.detach())
                error = (quantized - module.weight.detach()).flatten(1)
                compensation.initialize_hgs(error, moment, solver)
            status[name] = 'hgs'
        return status

    def initialize_hgs_from_samples(
            self, model: nn.Module,
            samples: Mapping[str, torch.Tensor]) -> Dict[str, str]:
        """Initialize compensation from sampled layer-input rows."""
        solver = MODELS.build(dict(type='HGSSolver', alpha=self.init_alpha))
        status = {}
        for name, module in model.named_modules():
            compensation = getattr(module, 'qat_compensation', None)
            if compensation is None:
                continue
            layer_samples = samples.get(name)
            if self.init != 'hgs' or layer_samples is None:
                status[name] = 'fallback'
                continue
            with torch.no_grad():
                quantized = module.weight_fake_quant(module.weight.detach())
                error = (quantized - module.weight.detach()).flatten(1)
                compensation.initialize_hgs_from_samples(
                    error, layer_samples, solver)
            status[name] = 'hgs'
        return status

    def _excluded(self, name: str) -> bool:
        return any(pattern.search(name) for pattern in self.exclude)
