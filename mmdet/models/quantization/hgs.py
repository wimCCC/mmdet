# Copyright (c) OpenMMLab. All rights reserved.
"""Trainable low-rank compensation and full-spectrum HGS initialization."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class HGSSolver:
    """Solve fixed-rank HGS factors using a complete eigendecomposition."""

    def __init__(self, alpha: float = 0.01):
        if alpha < 0:
            raise ValueError(f'alpha must be non-negative, but got {alpha}')
        self.alpha = float(alpha)

    @torch.no_grad()
    def __call__(self, error: torch.Tensor, moment: torch.Tensor,
                 rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return factors ``A, B`` such that ``A @ B`` approximates ``-error``.

        Args:
            error: Matrix ``W_q - W_ref`` with shape ``[out, in]``.
            moment: Uncentered second moment with shape ``[in, in]``.
            rank: Requested fixed rank.
        """
        if error.ndim != 2 or moment.shape != (error.shape[1], error.shape[1]):
            raise ValueError('error/moment shapes are inconsistent')
        work_dtype = (
            torch.float64 if error.dtype == torch.float64 else torch.float32)
        error_work = error.to(dtype=work_dtype)
        moment_work = moment.to(device=error.device, dtype=work_dtype)
        diagonal_mean = moment_work.diagonal().mean()
        regularized = moment_work + self.alpha * diagonal_mean * torch.eye(
            moment_work.shape[0], device=moment_work.device, dtype=work_dtype)
        eigenvalues, eigenvectors = torch.linalg.eigh(regularized)
        projected = error_work @ eigenvectors
        scores = eigenvalues.clamp_min(0) * projected.square().sum(dim=0)
        effective_rank = min(rank, error.shape[0], error.shape[1])
        indices = torch.topk(scores, k=effective_rank, largest=True).indices
        directions = eigenvectors[:, indices]
        b = directions.transpose(0, 1).contiguous()
        a = -(error_work @ directions)
        return a.to(error.dtype), b.to(error.dtype)

    @torch.no_grad()
    def from_samples(self, error: torch.Tensor, samples: torch.Tensor,
                     rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve HGS from sampled input rows without forming ``X.T @ X``.

        This is algebraically equivalent to eigendecomposing the empirical
        uncentered moment, restricted to its non-zero eigenspace. It keeps
        calibration practical for convolutions whose flattened kernels have
        thousands of input dimensions.
        """
        if error.ndim != 2 or samples.ndim != 2:
            raise ValueError('error and samples must both be matrices')
        if samples.shape[1] != error.shape[1] or samples.shape[0] == 0:
            raise ValueError('error/sample shapes are inconsistent')
        work_dtype = (
            torch.float64 if error.dtype == torch.float64 else torch.float32)
        error_work = error.to(dtype=work_dtype)
        samples_work = samples.to(device=error.device, dtype=work_dtype)
        _, singular_values, vh = torch.linalg.svd(
            samples_work, full_matrices=False)
        eigenvalues = singular_values.square() / samples_work.shape[0]
        diagonal_mean = eigenvalues.sum() / samples_work.shape[1]
        eigenvalues = eigenvalues + self.alpha * diagonal_mean
        directions = vh.transpose(0, 1)
        projected = error_work @ directions
        scores = eigenvalues * projected.square().sum(dim=0)
        effective_rank = min(rank, error.shape[0], directions.shape[1])
        indices = torch.topk(scores, k=effective_rank, largest=True).indices
        directions = directions[:, indices]
        b = directions.transpose(0, 1).contiguous()
        a = -(error_work @ directions)
        return a.to(error.dtype), b.to(error.dtype)


@MODELS.register_module()
class LowRankLinearCompensation(nn.Module):
    """FP trainable low-rank residual for a quantized linear operation."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'rank must be positive, but got {rank}')
        self.rank = int(rank)
        self.B = nn.Parameter(torch.empty(rank, in_features))
        self.A = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.xavier_uniform_(self.B)
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('enabled', torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.enabled):
            return x.new_zeros((*x.shape[:-1], self.A.shape[0]))
        hidden = F.linear(x, self.B.to(dtype=x.dtype))
        return F.linear(hidden, self.A.to(dtype=x.dtype))

    @torch.no_grad()
    def initialize_hgs(self, error: torch.Tensor, moment: torch.Tensor,
                       solver: HGSSolver) -> None:
        a, b = solver(error, moment, self.rank)
        self.A.zero_()
        self.B.zero_()
        self.A[:, :a.shape[1]].copy_(a.to(self.A))
        self.B[:b.shape[0]].copy_(b.to(self.B))
        self.initialized.fill_(True)

    @torch.no_grad()
    def initialize_hgs_from_samples(self, error: torch.Tensor,
                                    samples: torch.Tensor,
                                    solver: HGSSolver) -> None:
        a, b = solver.from_samples(error, samples, self.rank)
        self.A.zero_()
        self.B.zero_()
        self.A[:, :a.shape[1]].copy_(a.to(self.A))
        self.B[:b.shape[0]].copy_(b.to(self.B))
        self.initialized.fill_(True)


@MODELS.register_module()
class LowRankConv2dCompensation(nn.Module):
    """Grouped low-rank residual at a quantized Conv2d linear boundary."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1,
                 padding_mode: str = 'zeros', rank: int = 4):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'rank must be positive, but got {rank}')
        self.rank = int(rank)
        self.groups = int(groups)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        kernel_size = (kernel_size, kernel_size) if isinstance(
            kernel_size, int) else tuple(kernel_size)
        in_per_group = in_channels // groups
        self.B = nn.Parameter(torch.empty(
            groups * rank, in_per_group, *kernel_size))
        self.A = nn.Parameter(torch.zeros(out_channels, rank, 1, 1))
        nn.init.xavier_uniform_(self.B)
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('enabled', torch.tensor(True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.enabled):
            out_channels = self.A.shape[0]
            # This path is mainly used for ablation; preserve exact output
            # shape.
            hidden = self._project_b(x)
            shape = (hidden.shape[0], out_channels, *hidden.shape[2:])
            return x.new_zeros(shape)
        hidden = self._project_b(x)
        return F.conv2d(hidden, self.A.to(dtype=x.dtype), groups=self.groups)

    def _project_b(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != 'zeros':
            if isinstance(padding, int):
                padding = (padding, padding)
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]),
                      mode=self.padding_mode)
            padding = 0
        return F.conv2d(
            x, self.B.to(dtype=x.dtype), stride=self.stride,
            padding=padding, dilation=self.dilation, groups=self.groups)

    @torch.no_grad()
    def initialize_hgs(self, error: torch.Tensor, moment: torch.Tensor,
                       solver: HGSSolver) -> None:
        grouped_error = error.reshape(self.groups, -1, error.shape[1])
        grouped_moment = moment.unsqueeze(0) if moment.ndim == 2 else moment
        if grouped_moment.shape[0] != self.groups:
            raise ValueError('grouped moment must have one matrix per group')
        self.A.zero_()
        self.B.zero_()
        out_per_group = grouped_error.shape[1]
        for group in range(self.groups):
            a, b = solver(grouped_error[group], grouped_moment[group],
                          self.rank)
            rank = b.shape[0]
            self.B[group * self.rank:group * self.rank + rank].copy_(
                b.reshape(rank, *self.B.shape[1:]).to(self.B))
            start = group * out_per_group
            self.A[start:start + out_per_group, :rank, 0, 0].copy_(
                a.to(self.A))
        self.initialized.fill_(True)

    @torch.no_grad()
    def initialize_hgs_from_samples(self, error: torch.Tensor,
                                    samples: torch.Tensor,
                                    solver: HGSSolver) -> None:
        grouped_error = error.reshape(self.groups, -1, error.shape[1])
        grouped_samples = (
            samples.unsqueeze(0) if samples.ndim == 2 else samples)
        if grouped_samples.shape[0] != self.groups:
            raise ValueError('grouped samples must have one matrix per group')
        self.A.zero_()
        self.B.zero_()
        out_per_group = grouped_error.shape[1]
        for group in range(self.groups):
            a, b = solver.from_samples(
                grouped_error[group], grouped_samples[group], self.rank)
            effective_rank = b.shape[0]
            self.B[group * self.rank:group * self.rank + effective_rank].copy_(
                b.reshape(effective_rank, *self.B.shape[1:]).to(self.B))
            start = group * out_per_group
            self.A[start:start + out_per_group, :effective_rank, 0, 0].copy_(
                a.to(self.A))
        self.initialized.fill_(True)
