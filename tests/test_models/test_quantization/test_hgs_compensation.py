import torch
import torch.nn as nn

from mmdet.engine.hooks import HGSCalibrationHook
from mmdet.models.quantization import (HGSSolver, QATCompensationInjector)


class FakeQuantizedConv(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fake_quant = nn.Identity()


class FakeQuantizedLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_fake_quant = nn.Identity()


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = FakeQuantizedConv(4, 6, 3, padding=1, groups=2)
        self.linear = FakeQuantizedLinear(5, 3)
        self.float_conv = nn.Conv2d(3, 3, 1)


def test_injection_preserves_shapes_keys_and_is_idempotent():
    model = ToyModel()
    original_keys = set(model.state_dict())
    injector = QATCompensationInjector(rank=2)

    assert injector(model) == 2
    assert injector(model) == 0
    assert model.conv(torch.randn(2, 4, 8, 8)).shape == (2, 6, 8, 8)
    assert model.linear(torch.randn(2, 5)).shape == (2, 3)
    keys = set(model.state_dict())
    assert original_keys <= keys
    assert 'conv.qat_compensation.A' in keys
    assert 'linear.qat_compensation.B' in keys
    assert not hasattr(model.float_conv, 'qat_compensation')


def test_disabled_path_does_not_modify_model():
    model = ToyModel()
    assert QATCompensationInjector(enabled=False)(model) == 0
    assert not hasattr(model.conv, 'qat_compensation')


def test_hgs_initialization_is_finite_for_linear_and_grouped_conv():
    model = ToyModel()
    injector = QATCompensationInjector(rank=2, init_alpha=0.01)
    injector(model)
    moments = {
        'linear': torch.eye(5),
        'conv': torch.stack((torch.eye(18), torch.eye(18))),
    }
    status = injector.initialize_hgs(model, moments)

    assert status == {'conv': 'hgs', 'linear': 'hgs'}
    for name in ('conv', 'linear'):
        compensation = getattr(model, name).qat_compensation
        assert compensation.initialized
        assert torch.isfinite(compensation.A).all()
        assert torch.isfinite(compensation.B).all()


def test_solver_uses_requested_fixed_rank():
    error = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    moment = torch.diag(torch.tensor([3.0, 1.0]))
    a, b = HGSSolver(alpha=0.01)(error, moment, rank=1)
    assert a.shape == (2, 1)
    assert b.shape == (1, 2)


def test_sample_solver_matches_full_empirical_moment():
    torch.manual_seed(0)
    samples = torch.randn(6, 4)
    error = torch.randn(3, 4)
    moment = samples.T @ samples / samples.shape[0]
    solver = HGSSolver(alpha=0.01)
    full_a, full_b = solver(error, moment, rank=4)
    sample_a, sample_b = solver.from_samples(error, samples, rank=4)

    assert torch.allclose(full_a @ full_b, sample_a @ sample_b, atol=1e-5)


def test_sample_solver_reduces_calibration_error():
    torch.manual_seed(1)
    samples = torch.randn(16, 7)
    error = torch.randn(5, 7)
    a, b = HGSSolver(alpha=0.01).from_samples(error, samples, rank=2)
    before = samples @ error.T
    after = samples @ (error + a @ b).T

    assert after.square().mean() < before.square().mean()


def test_hgs_conv_rows_preserve_group_dimensions():
    conv = nn.Conv2d(4, 6, 3, padding=1, groups=2)
    rows = HGSCalibrationHook._conv_rows(torch.randn(2, 4, 5, 5), conv)

    assert rows.shape == (2, 50, 18)


def test_sample_based_injector_initializes_all_layers():
    model = ToyModel()
    injector = QATCompensationInjector(rank=2)
    injector(model)
    samples = {
        'linear': torch.randn(8, 5),
        'conv': torch.randn(2, 8, 18),
    }

    status = injector.initialize_hgs_from_samples(model, samples)

    assert status == {'conv': 'hgs', 'linear': 'hgs'}
    assert model.conv.qat_compensation.initialized
    assert model.linear.qat_compensation.initialized
