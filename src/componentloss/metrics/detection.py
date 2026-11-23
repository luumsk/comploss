"""Object-level (component-wise) evaluation of binary segmentations.

A ground-truth object is a connected component of the GT mask (face connectivity by default); it
is detected if any predicted voxel overlaps it. A predicted component is a true positive if it
overlaps any GT object. All functions work on numpy arrays (2D or 3D).

The ``lesion_*`` functions are aliases with the key names of the CATMIL paper
(``lesion_recall``, ``gt_lesion_count``, ...); the values are identical.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..components import label_components

EPS = 1e-8
SIZE_BINS: Tuple[Tuple[str, float], ...] = (("tiny", 10), ("small", 50), ("medium", 200), ("large", np.inf))


def _match(gt: np.ndarray, pred: np.ndarray, connectivity: int):
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape} vs pred {pred.shape}")
    gt_lab, gt_n = label_components(gt, connectivity)
    pr_lab, pr_n = label_components(pred, connectivity)
    gt_hit = np.zeros(gt_n + 1, dtype=bool)
    pr_hit = np.zeros(pr_n + 1, dtype=bool)
    if gt_n and pr_n:
        both = gt & pred
        gt_hit[np.unique(gt_lab[both])] = True
        pr_hit[np.unique(pr_lab[both])] = True
    gt_hit[0] = pr_hit[0] = False
    return gt, gt_lab, gt_n, pr_lab, pr_n, gt_hit, pr_hit


def object_detection_metrics(gt: np.ndarray, pred: np.ndarray, spacing: Optional[Sequence[float]] = None,
                             small_voxels: int = 50, connectivity: int = 1) -> Dict[str, float]:
    """Object counts, object precision / recall / F1, small-object recall, missed volume.

    small_voxels : GT objects with at most this many voxels count as small (CATMIL paper: 50).
    spacing : voxel spacing (e.g. mm) for volumes (default 1 per axis).
    """
    gt, gt_lab, gt_n, _, pr_n, gt_hit, pr_hit = _match(gt, pred, connectivity)
    sizes = np.bincount(gt_lab.ravel(), minlength=gt_n + 1)
    tp_gt, tp_pr = int(gt_hit.sum()), int(pr_hit.sum())
    recall = tp_gt / (gt_n + EPS)
    precision = tp_pr / (pr_n + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    small = np.arange(1, gt_n + 1)[sizes[1:] <= small_voxels]
    small_recall = float(gt_hit[small].mean()) if small.size else float("nan")
    vox_mm3 = float(np.prod(spacing)) if spacing is not None else 1.0
    missed_vox = float(sizes[1:][~gt_hit[1:]].sum())
    total_vox = float(gt.sum())
    return {
        "gt_object_count": float(gt_n),
        "pred_object_count": float(pr_n),
        "fn_object_count": float(gt_n - tp_gt),
        "fp_object_count": float(pr_n - tp_pr),
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_f1": float(f1),
        "miss_rate": float((gt_n - tp_gt) / (gt_n + EPS)),
        "small_object_recall": small_recall,
        "fn_volume_mm3": missed_vox * vox_mm3,
        "fn_volume_fraction": missed_vox / (total_vox + EPS) if total_vox > 0 else float("nan"),
    }


def recall_by_size(gt: np.ndarray, pred: np.ndarray, bins: Sequence[Tuple[str, float]] = SIZE_BINS,
                   connectivity: int = 1) -> Dict[str, Dict[str, float]]:
    """Detected / total GT objects per size bin (upper bounds in voxels, inclusive).

    Default bins as in the CATMIL paper: tiny <= 10, small <= 50, medium <= 200, large > 200 voxels.
    """
    _, gt_lab, gt_n, _, _, gt_hit, _ = _match(gt, pred, connectivity)
    sizes = np.bincount(gt_lab.ravel(), minlength=gt_n + 1)[1:]
    hit = gt_hit[1:]
    out, lo = {}, 0
    for name, hi in bins:
        sel = (sizes > lo) & (sizes <= hi)
        n = int(sel.sum())
        out[name] = {"count": n, "detected": int(hit[sel].sum()),
                     "recall": float(hit[sel].mean()) if n else float("nan")}
        lo = hi
    return out


def object_table(gt: np.ndarray, pred: np.ndarray, prob: Optional[np.ndarray] = None,
                 connectivity: int = 1):
    """One row per GT object: object_id, size_vox, detected, coverage, max_prob (if ``prob`` is given).

    Returns a dict of numpy arrays (``pandas.DataFrame(object_table(...))`` makes it a table).
    """
    gt, gt_lab, gt_n, _, _, gt_hit, _ = _match(gt, pred, connectivity)
    ids = np.arange(1, gt_n + 1)
    sizes = np.bincount(gt_lab.ravel(), minlength=gt_n + 1)[1:]
    covered = np.bincount(gt_lab[np.asarray(pred) > 0].ravel(), minlength=gt_n + 1)[1:]
    row = {"object_id": ids, "size_vox": sizes, "detected": gt_hit[1:], "coverage": covered / np.maximum(sizes, 1)}
    if prob is not None:
        from scipy import ndimage
        row["max_prob"] = np.asarray(ndimage.maximum(prob, gt_lab, ids)) if gt_n else np.zeros(0)
    return row


def remove_small_components(mask: np.ndarray, min_size: int, connectivity: int = 1) -> np.ndarray:
    """Drop predicted components smaller than ``min_size`` voxels (simple FP filter)."""
    lab, n = label_components(mask, connectivity)
    if n == 0 or min_size <= 1:
        return np.asarray(mask) > 0
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lab]


def object_size_profile(labels: Sequence[np.ndarray], bins: Sequence[Tuple[str, float]] = SIZE_BINS,
                        connectivity: int = 1) -> Dict[str, float]:
    """Object-size distribution of a dataset (list of GT masks): counts and share per size bin."""
    sizes = []
    for m in labels:
        lab, n = label_components(np.asarray(m) > 0, connectivity)
        if n:
            sizes.append(np.bincount(lab.ravel())[1:])
    s = np.concatenate(sizes) if sizes else np.zeros(0, dtype=int)
    out: Dict[str, float] = {"cases": float(len(labels)), "objects": float(s.size),
                             "median_size_vox": float(np.median(s)) if s.size else float("nan")}
    lo = 0
    for name, hi in bins:
        n = int(((s > lo) & (s <= hi)).sum())
        out[f"n_{name}"] = float(n)
        out[f"frac_{name}"] = n / s.size if s.size else float("nan")
        lo = hi
    return out


# --------------------------------------------------------------- lesion-named aliases

def _to_lesion_keys(d: dict) -> dict:
    return {k.replace("object", "lesion"): v for k, v in d.items()}


def lesion_detection_metrics(gt: np.ndarray, pred: np.ndarray, spacing: Optional[Sequence[float]] = None,
                             small_voxels: int = 50, connectivity: int = 1) -> Dict[str, float]:
    """``object_detection_metrics`` with lesion key names (``lesion_recall``, ``gt_lesion_count``, ...)."""
    return _to_lesion_keys(object_detection_metrics(gt, pred, spacing, small_voxels, connectivity))


def lesion_table(gt: np.ndarray, pred: np.ndarray, prob: Optional[np.ndarray] = None, connectivity: int = 1):
    """``object_table`` with a ``lesion_id`` column instead of ``object_id``."""
    return _to_lesion_keys(object_table(gt, pred, prob, connectivity))


def lesion_size_profile(labels: Sequence[np.ndarray], bins: Sequence[Tuple[str, float]] = SIZE_BINS,
                        connectivity: int = 1) -> Dict[str, float]:
    """``object_size_profile`` with the count under ``lesions`` instead of ``objects``."""
    out = object_size_profile(labels, bins, connectivity)
    out["lesions"] = out.pop("objects")
    return out
