# Copyright (c) OpenMMLab. All rights reserved.
"""Registry-based MMEngine runner for MQBench QAT and compensation."""

import copy
import inspect
import re

import torch.nn as nn
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.runner import Runner
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS, RUNNERS
from mmdet.models.quantization import register_mqbench_fake_quantizers


def _patch_mqbench_swap_module() -> None:
    """Adapt older MQBench calls to the torch 2.1 ``swap_module`` API."""
    from mqbench.custom_quantizer import model_quantizer

    swap_module = model_quantizer.swap_module
    if len(inspect.signature(swap_module).parameters) == 3:
        def compatible_swap_module(module, mapping, custom_mapping,
                                   use_precomputed_fake_quant=False):
            del use_precomputed_fake_quant
            return swap_module(module, mapping, custom_mapping)

        model_quantizer.swap_module = compatible_swap_module


def _restore_head_api(converted_head, original_head) -> None:
    """Restore state used only by loss/predict and therefore omitted by FX."""
    converted_head.__class__ = type(original_head)
    for name, module in original_head._modules.items():
        if name not in converted_head._modules:
            converted_head.add_module(name, module)
    for name, parameter in original_head._parameters.items():
        if name not in converted_head._parameters:
            converted_head.register_parameter(name, parameter)
    for name, buffer in original_head._buffers.items():
        if name not in converted_head._buffers:
            converted_head.register_buffer(name, buffer)
    ignored = {'_modules', '_parameters', '_buffers'}
    for name, value in original_head.__dict__.items():
        if name not in ignored and not hasattr(converted_head, name):
            setattr(converted_head, name, value)


def _disable_weight_fake_quant(model, patterns) -> int:
    """Keep numerically sensitive QAT modules at full-precision weights."""
    compiled = [re.compile(pattern) for pattern in patterns]
    disabled = 0
    for name, module in model.named_modules():
        if (hasattr(module, 'weight_fake_quant')
                and any(pattern.search(name) for pattern in compiled)):
            module.weight_fake_quant = nn.Identity()
            disabled += 1
    return disabled


def _disable_activation_fake_quant(model, patterns) -> int:
    """Keep selected FX activation boundaries in full precision."""
    compiled = [re.compile(pattern) for pattern in patterns]
    disabled = 0
    for name, module in list(model.named_modules()):
        if (name and hasattr(module, 'fake_quant_enabled')
                and any(pattern.search(name) for pattern in compiled)):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, nn.Identity())
            disabled += 1
    return disabled


@RUNNERS.register_module()
class QATCompensationRunner(Runner):
    """Build MQBench QAT before optimizer construction, then inject HGS."""

    @classmethod
    def from_cfg(cls, cfg):
        cfg = copy.deepcopy(cfg)
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        model = MODELS.build(cfg['model'])
        # FX tracing replaces the backbone/neck/head containers with plain
        # modules, so their ``init_cfg`` (e.g. torchvision pretrained backbone)
        # is lost after conversion. Initialize the FP32 model here, before
        # ``prepare_by_platform``; Runner.init_weights() later becomes a no-op.
        model.init_weights()
        qat_pretrained = cfg.get('qat_pretrained')
        if qat_pretrained:
            # QAT changes parameter names (and FX duplicates reused detector
            # heads), so an FP32 checkpoint must be loaded before conversion.
            load_checkpoint(
                model, qat_pretrained, map_location='cpu', strict=False)
        data_preprocessor = model.data_preprocessor
        original_bbox_head = model.bbox_head

        qat_config = cfg.get('extra_config')
        if qat_config:
            register_mqbench_fake_quantizers()
            _patch_mqbench_swap_module()
            from mqbench.prepare_by_platform import BackendType
            from mqbench.prepare_by_platform import prepare_by_platform

            backend_name = cfg.get('qat_backend', 'Academic')
            try:
                backend = BackendType[backend_name]
            except KeyError as error:
                message = f'Unknown MQBench backend: {backend_name}'
                raise ValueError(message) from error
            model = prepare_by_platform(model, backend, dict(qat_config))
            _restore_head_api(model.bbox_head, original_bbox_head)
            fp32_weight_patterns = cfg.get('qat_fp32_weight_patterns', [])
            if fp32_weight_patterns:
                disabled = _disable_weight_fake_quant(
                    model, fp32_weight_patterns)
                if disabled == 0:
                    raise RuntimeError(
                        'No QAT weight quantizer matched '
                        'qat_fp32_weight_patterns')
            fp32_activation_patterns = cfg.get(
                'qat_fp32_activation_patterns', [])
            if fp32_activation_patterns:
                disabled = _disable_activation_fake_quant(
                    model, fp32_activation_patterns)
                if disabled == 0:
                    raise RuntimeError(
                        'No QAT activation quantizer matched '
                        'qat_fp32_activation_patterns')
            model = MODELS.build(dict(
                type='QATSingleStageGraphModel',
                graph=model,
                data_preprocessor=data_preprocessor))

        compensation_cfg = cfg.get('qat_compensation', {})
        if compensation_cfg.get('enabled', False):
            injector = MODELS.build(dict(
                type='QATCompensationInjector', **dict(compensation_cfg)))
            injected = injector(model)
            if injected == 0:
                raise RuntimeError(
                    'QAT compensation is enabled, but no weight-quantized '
                    'Conv2d/Linear layer was found after QAT preparation')
            # Kept as a plain attribute for optional pre-training HGS init.
            object.__setattr__(model, '_qat_compensation_injector', injector)

        cfg.pop('runner_type', None)
        return cls(
            model=model,
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmdet'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg)
