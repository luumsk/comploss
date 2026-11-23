"""Voronoi partition of a mask by its connected components (used by CC-DiceCE)."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

try:  # GPU Voronoi as in the official code
    import cupy as _cp
    from cupyx.scipy.ndimage import distance_transform_edt as _cp_edt
    from cupyx.scipy.ndimage import label as _cp_label
    _HAS_CUPY = True
except Exception:  # scipy on CPU, same result
    _HAS_CUPY = False


def _voronoi_cupy(fg: torch.Tensor, sampling) -> torch.Tensor:
    m = _cp.from_dlpack(fg.to(torch.uint8).contiguous()) > 0  # uint8: old torch cannot export bool
    lab, _ = _cp_label(m, structure=_cp.ones((3,) * m.ndim, dtype=bool))
    if int(lab.max()) == 0:
        return torch.zeros(fg.shape, dtype=torch.int64, device=fg.device)
    idx = _cp_edt(lab == 0, sampling=sampling, return_distances=False, return_indices=True,
                  float64_distances=False)
    return torch.from_dlpack(lab[tuple(idx)].astype(_cp.int64)).to(fg.device)


def voronoi_components(fg: torch.Tensor, sampling=None) -> torch.Tensor:
    """Voronoi map of a binary mask: every voxel gets the id of its nearest GT component.

    Fully connected components (26 in 3D), Euclidean distance with ``sampling`` (voxel spacing).
    0 = no component in the patch. Returns int64 on ``fg.device``.
    """
    global _HAS_CUPY
    if _HAS_CUPY and fg.is_cuda:
        try:
            return _voronoi_cupy(fg, sampling)
        except Exception as e:  # e.g. cupy/CUDA mismatch: scipy for the rest of the run
            print(f"CC-DiceCE: cupy Voronoi failed ({e!r}); falling back to scipy on CPU", flush=True)
            _HAS_CUPY = False
    m = fg.detach().cpu().numpy()
    lab, n = ndimage.label(m, structure=np.ones((3,) * m.ndim, dtype=bool))
    if n == 0:
        return torch.zeros(fg.shape, dtype=torch.int64, device=fg.device)
    idx = ndimage.distance_transform_edt(lab == 0, sampling=sampling, return_distances=False,
                                         return_indices=True)
    return torch.from_numpy(lab[tuple(idx)].astype(np.int64)).to(fg.device)
