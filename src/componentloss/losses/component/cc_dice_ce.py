"""CC-DiceCE (Bouteille et al., ISBI 2026, arXiv:2511.17146), ported from the official code.

``CCDiceCELoss`` is the Voronoi component-wise DiceCE; ``CCDiceCEGlobalLoss`` adds the global
DiceCE (smooth = 0) 1:1, which is the "CC-DiceCE" objective of that paper.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ...components.voronoi import voronoi_components
from .._utils import no_compile, squeeze_target
from ..voxel.dice import DiceCELoss

_EPS = 1e-7


class CCDiceCELoss(nn.Module):
    """Voronoi component-wise DiceCE (official ``VectorizedCCLoss``, mode ``dice_ce``).

    For each GT component, its Voronoi region gets ``1 - 2 sum(p y) / sum(p + y)`` plus the mean
    voxel CE in the region; the sample loss is the mean over components. A sample without
    foreground gets ``1 + mean CE``. Binary softmax (2 channels) only, as in the official code.

    spacing : voxel spacing at full resolution (default 1 per axis).
    patch_size : full-resolution patch size; used to scale the spacing at deep-supervision levels.
        If None, every input is treated as full resolution.
    """

    def __init__(self, spacing: Optional[Sequence[float]] = None,
                 patch_size: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.spacing = None if spacing is None else tuple(float(s) for s in spacing)
        self.patch_size = None if patch_size is None else tuple(int(s) for s in patch_size)

    def _sampling(self, shape):
        spacing = self.spacing or (1.0,) * len(shape)
        if self.patch_size is None:
            return spacing
        return tuple(s * full / cur for s, full, cur in zip(spacing, self.patch_size, shape))

    @no_compile
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if net_output.shape[1] != 2:
            raise ValueError("CCDiceCELoss supports binary softmax outputs (2 channels) only")
        y = squeeze_target(target)
        probs = torch.softmax(net_output.float(), 1)
        p_fg = probs[:, 1]
        y_fg = (y == 1).float()
        ce_map = -torch.log(probs.clamp_min(_EPS)).gather(1, y.unsqueeze(1)).squeeze(1)
        sampling = self._sampling(y.shape[1:])

        losses = []
        for b in range(y.shape[0]):
            vor = voronoi_components(y[b] == 1, sampling)
            k = int(vor.max())
            if k == 0:  # official fallback: Dice term 1, CE over the whole patch
                losses.append(ce_map[b].mean() + 1.0)
                continue
            ids = vor.reshape(-1)

            def seg_sum(v):
                return torch.zeros(k + 1, device=v.device, dtype=v.dtype).scatter_add_(0, ids, v.reshape(-1))[1:]

            inter = seg_sum(p_fg[b] * y_fg[b])
            denom = seg_sum(p_fg[b]) + seg_sum(y_fg[b])
            count = seg_sum(torch.ones_like(p_fg[b]))
            comp = (1.0 - 2.0 * inter / denom.clamp_min(_EPS)) + seg_sum(ce_map[b]) / count.clamp_min(1.0)
            valid = (count > 0).float()
            losses.append((comp * valid).sum() / valid.sum().clamp_min(1.0))
        return torch.stack(losses).mean()


class CCDiceCEGlobalLoss(nn.Module):
    """The paper's CC-DiceCE objective: ``w_cc * CCDiceCE + w_global * DiceCE(smooth=0)``.

    In nnU-Net it is applied at every deep-supervision output (wrap it in ``DeepSupervisionLoss``).
    """

    def __init__(self, spacing=None, patch_size=None, batch_dice: bool = False,
                 w_cc: float = 1.0, w_global: float = 1.0, global_loss: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.cc = CCDiceCELoss(spacing, patch_size)
        self.glob = global_loss if global_loss is not None else DiceCELoss(batch_dice=batch_dice, smooth=0.0)
        self.w_cc, self.w_global = float(w_cc), float(w_global)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_cc * self.cc(net_output, target) + self.w_global * self.glob(net_output, target)
