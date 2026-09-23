# Copyright (c) OpenMMLab. All rights reserved.
"""Data-driven HGS initialization for QAT compensation branches."""

import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from mmengine.dist import broadcast, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class HGSCalibrationHook(Hook):
    """Initialize low-rank compensation from sampled training activations.

    Input rows are sampled rather than forming full second-moment matrices.
    The HGS solver uses their compact SVD, which is equivalent on the
    non-zero eigenspace of the empirical uncentered moment.
    """

    priority = 'VERY_HIGH'

    def __init__(self,
                 num_batches: int = 4,
                 max_samples_per_layer: int = 128,
                 enabled: bool = True,
                 skip_resumed: bool = True):
        if num_batches <= 0:
            raise ValueError('num_batches must be positive')
        if max_samples_per_layer <= 0:
            raise ValueError('max_samples_per_layer must be positive')
        self.num_batches = int(num_batches)
        self.max_samples_per_layer = int(max_samples_per_layer)
        self.enabled = bool(enabled)
        self.skip_resumed = bool(skip_resumed)

    def before_train(self, runner) -> None:
        if not self.enabled:
            return
        if self.skip_resumed and runner.iter > 0:
            runner.logger.info('Skip HGS calibration for resumed training')
            return

        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        injector = getattr(model, '_qat_compensation_injector', None)
        if injector is None:
            raise RuntimeError(
                'HGS calibration requires a compensation injector')

        targets = {
            name: module
            for name, module in model.named_modules()
            if hasattr(module, 'qat_compensation')
            and hasattr(module, 'weight_fake_quant')
        }
        if not targets:
            raise RuntimeError('No QAT compensation target found for HGS')
        if all(bool(module.qat_compensation.initialized)
               for module in targets.values()):
            runner.logger.info('All HGS compensation branches are initialized')
            return

        runner.logger.info(
            'Collecting HGS samples from %d batches for %d layers',
            self.num_batches, len(targets))
        samples = self._collect_samples(runner, model, targets)
        status = injector.initialize_hgs_from_samples(model, samples)
        if get_world_size() > 1:
            for module in targets.values():
                compensation = module.qat_compensation
                broadcast(compensation.A.data, src=0)
                broadcast(compensation.B.data, src=0)
                broadcast(compensation.initialized, src=0)
        initialized = sum(value == 'hgs' for value in status.values())
        fallback = sum(value == 'fallback' for value in status.values())
        if initialized == 0:
            raise RuntimeError('HGS calibration did not initialize any layer')
        runner.logger.info(
            'HGS initialization complete: %d initialized, %d fallback',
            initialized, fallback)

    def _collect_samples(self, runner, model,
                         targets) -> Dict[str, torch.Tensor]:
        chunks: Dict[str, List[torch.Tensor]] = {
            name: [] for name in targets
        }
        samples_per_batch = math.ceil(
            self.max_samples_per_layer / self.num_batches)
        handles = []

        def make_hook(name):
            def collect(module, inputs, output):
                del output
                x = inputs[0].detach()
                if module.weight.ndim == 4:
                    rows = self._conv_rows(x, module)
                elif module.weight.ndim == 2:
                    rows = x.reshape(-1, x.shape[-1]).unsqueeze(0)
                else:
                    return
                row_count = rows.shape[1]
                count = min(samples_per_batch, row_count)
                indices = torch.randperm(row_count, device=rows.device)[:count]
                chunks[name].append(rows[:, indices].float().cpu())
            return collect

        for name, module in targets.items():
            handles.append(module.register_forward_hook(make_hook(name)))

        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                for batch_index, data_batch in enumerate(
                        runner.train_dataloader):
                    if batch_index >= self.num_batches:
                        break
                    data = model.data_preprocessor(data_batch, training=False)
                    model(**data, mode='tensor')
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        result = {}
        for name, layer_chunks in chunks.items():
            if not layer_chunks:
                continue
            layer_samples = torch.cat(layer_chunks, dim=1)
            if layer_samples.shape[1] > self.max_samples_per_layer:
                indices = torch.randperm(layer_samples.shape[1])[
                    :self.max_samples_per_layer]
                layer_samples = layer_samples[:, indices]
            result[name] = layer_samples
        return result

    @staticmethod
    def _conv_rows(x: torch.Tensor, module) -> torch.Tensor:
        padding = module.padding
        if module.padding_mode != 'zeros':
            if isinstance(padding, int):
                padding = (padding, padding)
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]),
                      mode=module.padding_mode)
            padding = 0
        patches = F.unfold(
            x,
            kernel_size=module.kernel_size,
            dilation=module.dilation,
            padding=padding,
            stride=module.stride)
        batch, full_dim, locations = patches.shape
        groups = module.groups
        group_dim = full_dim // groups
        return patches.reshape(batch, groups, group_dim, locations).permute(
            1, 0, 3, 2).reshape(groups, batch * locations, group_dim)
