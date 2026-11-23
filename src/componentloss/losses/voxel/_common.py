"""Target handling shared by the voxel-wise losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .._utils import squeeze_target


def _one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(B, *spatial) long -> (B, C, *spatial) float."""
    return F.one_hot(labels.clamp_min(0), n_classes).movedim(-1, 1).to(torch.float32)


def _region_target(logits: torch.Tensor, target: torch.Tensor, ignore_index):
    """Region targets (B, R, *sp) and loss mask (B, 1, *sp) or None (nnU-Net: ignore = last channel)."""
    if logits.shape[1] == 1 and (target.ndim == logits.ndim - 1 or target.shape[1] == 1):
        tgt = squeeze_target(target)
        mask = None if ignore_index is None else (tgt != int(ignore_index)).unsqueeze(1).float()
        return (tgt > 0).unsqueeze(1).float() * (1.0 if mask is None else mask), mask
    if ignore_index is not None:
        return target[:, :-1].float(), (1 - target[:, -1:]).float()
    return target.float(), None
