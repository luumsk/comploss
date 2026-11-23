"""Frozen copy of the lesion-wise detection metric of the CATMIL paper's evaluation script.

Source: multiscale_eval.py in https://github.com/luumsk/SmallLesionMRI (commit cd16748), copied
verbatim (only the functions needed by lesion_detection_metrics). Used by test_metrics.py to
check that comploss.metrics reproduces the paper's numbers. Do not edit.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

EPS = 1e-8


def _label_cc(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    lab, n = ndimage.label(mask, structure=struct)
    return lab.astype(np.int32), int(n)


def lesion_detection_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
    small_voxels_thresh: int,
) -> Dict[str, float]:
    """Lesion-wise metrics using connected components."""
    gt_lab, gt_n = _label_cc(gt)
    pr_lab, pr_n = _label_cc(pred)

    gt_sizes = np.bincount(gt_lab.ravel())
    gt_ids = list(range(1, gt_n + 1))

    gt_hit = set()
    pr_hit = set()

    if gt_n > 0 and pr_n > 0:
        overlap_pairs = np.unique(
            np.stack([gt_lab[gt], pr_lab[gt]], axis=1), axis=0
        )
        for gt_id, pr_id in overlap_pairs:
            if gt_id == 0 or pr_id == 0:
                continue
            gt_hit.add(int(gt_id))
            pr_hit.add(int(pr_id))

    tp_gt = len(gt_hit)
    tp_pr = len(pr_hit)
    fn_gt = gt_n - tp_gt

    recall = tp_gt / (gt_n + EPS)
    precision = tp_pr / (pr_n + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    miss_rate = fn_gt / (gt_n + EPS)

    small_gt_ids = [
        i for i in gt_ids if int(gt_sizes[i]) <= int(small_voxels_thresh)
    ]
    if len(small_gt_ids) == 0:
        small_recall = float("nan")
    else:
        small_hit = sum([1 for i in small_gt_ids if i in gt_hit])
        small_recall = small_hit / (len(small_gt_ids) + EPS)

    missed_gt_ids = [i for i in gt_ids if i not in gt_hit]
    missed_fn_voxels = float(sum(int(gt_sizes[i]) for i in missed_gt_ids))
    voxel_volume_mm3 = _voxel_volume_mm3(spacing)
    total_gt_voxels = float(int(gt.sum()))
    fn_volume_mm3 = missed_fn_voxels * voxel_volume_mm3
    total_gt_volume_mm3 = total_gt_voxels * voxel_volume_mm3
    fn_volume_fraction = (
        fn_volume_mm3 / (total_gt_volume_mm3 + EPS)
        if total_gt_volume_mm3 > 0.0
        else float("nan")
    )

    return {
        "gt_lesion_count": float(gt_n),
        "pred_lesion_count": float(pr_n),
        "fn_lesion_count": float(fn_gt),
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "miss_rate": float(miss_rate),
        "small_lesion_recall": float(small_recall),
        "fn_volume_mm3": float(fn_volume_mm3),
        "fn_volume_fraction": float(fn_volume_fraction),
    }


def _voxel_volume_mm3(spacing: Tuple[float, float, float]) -> float:
    return float(spacing[0] * spacing[1] * spacing[2])

