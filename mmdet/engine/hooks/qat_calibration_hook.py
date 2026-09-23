# Copyright (c) OpenMMLab. All rights reserved.
"""Observer calibration followed by fixed-scale QAT for MQBench models."""

from typing import Dict, List

import torch
from mmengine.dist import broadcast, get_world_size
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from torch.ao.quantization import FakeQuantizeBase

from mmdet.registry import HOOKS


@HOOKS.register_module()
class QATCalibrationHook(Hook):
    """Calibrate quantizer ranges once, then freeze observers for QAT.

    MQBench leaves ``observer_enabled=1`` after ``prepare_by_platform``. With
    ``LearnableFakeQuantize`` this overwrites the learnable ``scale`` from the
    observer on every forward pass, so LSQ never actually learns. This hook
    runs the SSI observers on ``num_batches`` FP32 forward passes, aggregates
    their min/max across batches, writes the resulting qparams into the fake
    quantizers and then calls ``enable_quantization`` (observer off, fake
    quant on) before the first training iteration.
    """

    priority = 'HIGHEST'

    def __init__(self, num_batches: int = 8, skip_resumed: bool = True):
        if num_batches <= 0:
            raise ValueError('num_batches must be positive')
        self.num_batches = int(num_batches)
        self.skip_resumed = bool(skip_resumed)

    def before_train(self, runner) -> None:
        from mqbench.utils.state import enable_calibration, enable_quantization

        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        quantizers: Dict[str, FakeQuantizeBase] = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, FakeQuantizeBase)
        }
        if not quantizers:
            raise RuntimeError('QATCalibrationHook found no fake quantizer')

        if self.skip_resumed and runner.iter > 0:
            # Scales were restored from the checkpoint; only freeze observers.
            enable_quantization(model)
            runner.logger.info(
                'Resumed run: skip QAT calibration, observers disabled')
            return

        runner.logger.info('Calibrating %d fake quantizers on %d batches',
                           len(quantizers), self.num_batches)
        enable_calibration(model)
        min_vals: Dict[str, torch.Tensor] = {}
        max_vals: Dict[str, torch.Tensor] = {}
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
                    self._accumulate_ranges(quantizers, min_vals, max_vals)
        finally:
            model.train(was_training)

        skipped = self._write_qparams(quantizers, min_vals, max_vals)
        if skipped:
            runner.logger.warning(
                '%d fake quantizers received no data during calibration, '
                'e.g. %s', len(skipped), skipped[:5])
        enable_quantization(model)
        if get_world_size() > 1:
            for module in quantizers.values():
                broadcast(module.scale.data, src=0)
                broadcast(module.zero_point.data, src=0)
        runner.logger.info(
            'QAT calibration complete: observers disabled, fake quant enabled')

    @staticmethod
    def _accumulate_ranges(quantizers, min_vals, max_vals) -> None:
        for name, module in quantizers.items():
            observer = module.activation_post_process
            min_val = observer.min_val.detach()
            max_val = observer.max_val.detach()
            if min_val.numel() == 0:
                continue
            if name in min_vals and min_vals[name].shape == min_val.shape:
                min_vals[name] = torch.minimum(min_vals[name], min_val)
                max_vals[name] = torch.maximum(max_vals[name], max_val)
            else:
                min_vals[name] = min_val.clone()
                max_vals[name] = max_val.clone()

    @staticmethod
    @torch.no_grad()
    def _write_qparams(quantizers, min_vals, max_vals) -> List[str]:
        skipped: List[str] = []
        for name, module in quantizers.items():
            if name not in min_vals:
                skipped.append(name)
                continue
            observer = module.activation_post_process
            # Buffers are re-bound (not copied) because ClipStdObserver
            # rebinds them on every forward as well.
            observer.min_val = min_vals[name]
            observer.max_val = max_vals[name]
            scale, zero_point = observer.calculate_qparams()
            scale = scale.to(module.scale.device)
            zero_point = zero_point.to(module.zero_point.device)
            if module.scale.shape != scale.shape:
                module.scale.data = torch.ones_like(scale)
                module.zero_point.data = torch.zeros_like(zero_point.float())
            module.scale.data.copy_(scale)
            module.zero_point.data.copy_(zero_point.float())
        return skipped
