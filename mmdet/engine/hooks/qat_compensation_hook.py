# Copyright (c) OpenMMLab. All rights reserved.
"""Runtime controls and diagnostics for QAT compensation branches."""

from mmengine.hooks import Hook

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SetQATCompensationEnabledHook(Hook):
    """Enable or disable all low-rank branches after checkpoint loading."""

    priority = 'VERY_HIGH'

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled)

    def _set_enabled(self, runner) -> None:
        count = 0
        for module in runner.model.modules():
            compensation = getattr(module, 'qat_compensation', None)
            if compensation is not None:
                compensation.enabled.fill_(self.enabled)
                count += 1
        runner.logger.info(
            'Set %d QAT compensation branches enabled=%s',
            count, self.enabled)

    def before_train(self, runner) -> None:
        self._set_enabled(runner)

    def before_val(self, runner) -> None:
        self._set_enabled(runner)

    def before_test(self, runner) -> None:
        self._set_enabled(runner)
