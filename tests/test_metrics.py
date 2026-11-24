import math

import numpy as np
import pytest
from scipy import ndimage

import _reference_metrics as ref

from comploss.cli import main as cli_main
from comploss.metrics import (lesion_detection_metrics, lesion_size_profile, lesion_table,
                                   object_detection_metrics, object_size_profile, object_table, recall_by_size,
                                   remove_small_components)


def _masks(seed, shape=(40, 40, 40)):
    rng = np.random.default_rng(seed)
    gt = ndimage.binary_dilation(rng.random(shape) < 0.002, iterations=int(rng.integers(1, 3)))
    pred = ndimage.binary_dilation(rng.random(shape) < 0.002, iterations=1) | (gt & (rng.random(shape) < 0.5))
    return gt, pred


@pytest.mark.parametrize("seed", range(5))
def test_detection_matches_paper_script(seed):
    gt, pred = _masks(seed)
    old = ref.lesion_detection_metrics(gt, pred, (1.0, 1.0, 1.2), small_voxels_thresh=50)
    new = lesion_detection_metrics(gt, pred, (1.0, 1.0, 1.2), small_voxels=50)
    for k, v in old.items():
        assert (math.isnan(v) and math.isnan(new[k])) or math.isclose(v, new[k], rel_tol=1e-6, abs_tol=1e-9), k
    obj = object_detection_metrics(gt, pred, (1.0, 1.0, 1.2), small_voxels=50)
    assert {k.replace("object", "lesion"): v for k, v in obj.items()}.keys() == new.keys()


def test_default_small_threshold_is_paper_150():
    gt = np.zeros((20, 20, 20), bool)
    gt[2:7, 2:7, 2:6] = True      # 100 voxels: small at 150, not at 50
    gt[12:14, 12:14, 12:14] = True
    pred = np.zeros_like(gt)
    pred[12, 12, 12] = True       # only the 8-voxel object is found
    default = lesion_detection_metrics(gt, pred)
    assert default["small_lesion_recall"] == 0.5
    paper = ref.lesion_detection_metrics(gt, pred, (1.0, 1.0, 1.0), small_voxels_thresh=150)
    assert math.isclose(default["small_lesion_recall"], paper["small_lesion_recall"], abs_tol=1e-6)
    assert lesion_detection_metrics(gt, pred, small_voxels=50)["small_lesion_recall"] == 1.0


def test_simple_case():
    gt = np.zeros((10, 10, 10), bool)
    gt[1, 1, 1] = True            # tiny object, missed
    gt[5:8, 5:8, 5:8] = True      # 27 voxels, found
    pred = np.zeros_like(gt)
    pred[6, 6, 6] = True
    pred[0, 9, 9] = True          # false positive
    m = object_detection_metrics(gt, pred)
    assert m["gt_object_count"] == 2 and m["fn_object_count"] == 1 and m["fp_object_count"] == 1
    assert math.isclose(m["object_recall"], 0.5, rel_tol=1e-6)
    assert lesion_detection_metrics(gt, pred)["lesion_recall"] == m["object_recall"]
    r = recall_by_size(gt, pred)
    assert r["tiny"] == {"count": 1, "detected": 0, "recall": 0.0}
    assert r["small"]["count"] == 1 and r["small"]["recall"] == 1.0
    t = object_table(gt, pred, prob=gt.astype(float) * 0.7)
    assert list(lesion_table(gt, pred)["lesion_id"]) == list(t["object_id"]) == [1, 2]
    assert list(t["size_vox"]) == [1, 27] and list(t["detected"]) == [False, True]
    assert np.allclose(t["max_prob"], 0.7)
    assert remove_small_components(pred, 2).sum() == 0


def test_profile_cli(tmp_path, capsys):
    for i in range(3):
        gt, _ = _masks(i, (24, 24, 24))
        np.save(tmp_path / f"case{i}.npy", gt.astype(np.uint8))
    assert cli_main(["profile", str(tmp_path), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"objects"' in out
    masks = [np.load(p) for p in sorted(tmp_path.glob("*.npy"))]
    prof = object_size_profile(masks)
    assert prof["cases"] == 3 and prof["objects"] > 0
    assert lesion_size_profile(masks)["lesions"] == prof["objects"]


def test_bench_cli(capsys):
    assert cli_main(["bench", "--shape", "1", "2", "16", "16", "16", "--iters", "1", "--warmup", "0",
                     "--device", "cpu"]) == 0
    assert "catmil" in capsys.readouterr().out
