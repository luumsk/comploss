"""Soft Dice and DiceCE.

``DiceCELoss`` reproduces nnU-Net v2's default ``DC_and_CE_loss`` (memory-efficient soft Dice
without background + cross entropy). Like nnU-Net, the Dice term is ``-Dice``, so the loss is
``CE - Dice`` (a constant shift from ``CE + 1 - Dice``; gradients are identical).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._utils import resolve_activation, squeeze_target
from ._common import _one_hot, _region_target


class SoftDiceLoss(nn.Module):
    """nnU-Net ``MemoryEfficientSoftDiceLoss`` on probabilities; returns ``-mean Dice``."""

    def __init__(self, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.0) -> None:
        super().__init__()
        self.batch_dice, self.do_bg, self.smooth = bool(batch_dice), bool(do_bg), float(smooth)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        axes = tuple(range(2, x.ndim))
        if not self.do_bg:
            x, y_onehot = x[:, 1:], y_onehot[:, 1:]
        with torch.no_grad():
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)
        if self.batch_dice:
            intersect, sum_pred, sum_gt = intersect.sum(0), sum_pred.sum(0), sum_gt.sum(0)
        dc = (2 * intersect + self.smooth) / torch.clip(sum_gt + sum_pred + self.smooth, 1e-8)
        return -dc.mean()


class DiceCELoss(nn.Module):
    """nnU-Net v2 default loss (soft Dice + CE for softmax; soft Dice + BCE for sigmoid/regions).

    Defaults match nnU-Net: smooth 1e-5, background excluded from Dice for softmax, included for
    sigmoid. ``batch_dice`` should follow the nnU-Net plans (True for 3d_fullres usually).
    """

    def __init__(self, batch_dice: bool = False, smooth: float = 1e-5, do_bg: Optional[bool] = None,
                 weight_ce: float = 1.0, weight_dice: float = 1.0, ignore_index: Optional[int] = None,
                 activation: str = "auto") -> None:
        super().__init__()
        self.batch_dice, self.smooth, self.do_bg = batch_dice, smooth, do_bg
        self.weight_ce, self.weight_dice = float(weight_ce), float(weight_dice)
        self.ignore_index, self.activation = ignore_index, activation

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        act = resolve_activation(net_output, self.activation)
        if act == "sigmoid":
            dice = SoftDiceLoss(self.batch_dice, True if self.do_bg is None else self.do_bg, self.smooth)
            tgt, mask = _region_target(net_output, target, self.ignore_index)
            dc = dice(torch.sigmoid(net_output), tgt, loss_mask=mask)
            if mask is None:
                ce = F.binary_cross_entropy_with_logits(net_output, tgt)
            else:
                ce = (F.binary_cross_entropy_with_logits(net_output, tgt, reduction="none") * mask).sum() \
                    / torch.clip(mask.sum(), 1e-8)
            return self.weight_ce * ce + self.weight_dice * dc

        dice = SoftDiceLoss(self.batch_dice, False if self.do_bg is None else self.do_bg, self.smooth)
        tgt = squeeze_target(target)
        n_cls = int(net_output.shape[1])
        mask, tgt_dice = None, tgt
        if self.ignore_index is not None:
            mask = (tgt != int(self.ignore_index)).unsqueeze(1)
            tgt_dice = torch.where(mask[:, 0], tgt, torch.zeros_like(tgt))
        dc = dice(torch.softmax(net_output, 1), _one_hot(tgt_dice, n_cls), loss_mask=mask)
        if self.ignore_index is not None and not bool(mask.any()):
            ce = net_output.new_zeros(())
        else:
            ce = F.cross_entropy(net_output, tgt,
                                 ignore_index=-100 if self.ignore_index is None else int(self.ignore_index))
        return self.weight_ce * ce + self.weight_dice * dc
