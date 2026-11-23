"""Tversky and Focal Tversky.

* ``TverskyLoss`` reproduces ``monai.losses.TverskyLoss(to_onehot_y=True, softmax=True)``.
* ``FocalTverskyLoss`` is ``Tversky ** (1 / gamma)`` (gamma = 4/3 in the CATMIL paper).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .._utils import resolve_activation, squeeze_target
from ._common import _one_hot, _region_target


class TverskyLoss(nn.Module):
    """Soft Tversky loss, ``1 - TP / (TP + alpha FP + beta FN)`` per sample and class, averaged.

    Equivalent to ``monai.losses.TverskyLoss(to_onehot_y=True, softmax=True)`` (background
    included, smooth 1e-5) for softmax outputs; per channel on sigmoid outputs.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, include_background: bool = True,
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5, ignore_index: Optional[int] = None,
                 activation: str = "auto") -> None:
        super().__init__()
        self.alpha, self.beta = float(alpha), float(beta)
        self.include_background = include_background
        self.smooth_nr, self.smooth_dr = float(smooth_nr), float(smooth_dr)
        self.ignore_index, self.activation = ignore_index, activation

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        act = resolve_activation(net_output, self.activation)
        if act == "sigmoid":
            p = torch.sigmoid(net_output)
            g, mask = _region_target(net_output, target, self.ignore_index)
        else:
            tgt = squeeze_target(target)
            mask = None
            if self.ignore_index is not None:
                mask = (tgt != int(self.ignore_index)).unsqueeze(1).float()
                tgt = torch.where(mask[:, 0] > 0, tgt, torch.zeros_like(tgt))
            p = torch.softmax(net_output, 1)
            g = _one_hot(tgt, int(net_output.shape[1]))
            if not self.include_background:
                p, g = p[:, 1:], g[:, 1:]
        if mask is not None:
            p, g = p * mask, g * mask
        dims = tuple(range(2, p.ndim))
        tp = (p * g).sum(dims)
        fp = self.alpha * (p * (1 - g)).sum(dims)
        fn = self.beta * ((1 - p) * g).sum(dims)
        return (1.0 - (tp + self.smooth_nr) / (tp + fp + fn + self.smooth_dr)).mean()


class FocalTverskyLoss(nn.Module):
    """``TverskyLoss ** (1 / gamma)`` (paper: alpha 0.3, beta 0.7, gamma 4/3)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 4.0 / 3.0, **tversky_kwargs) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.tversky(net_output, target), 1.0 / self.gamma)
