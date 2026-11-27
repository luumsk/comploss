# comploss

**Component-aware loss functions and object-level metrics for segmentation in PyTorch.**

Voxel-wise losses such as Dice, cross entropy and Tversky sum over voxels, so a few large
objects dominate the gradient. A missed 5-voxel object barely changes the loss. That matters
whenever an image holds many objects of very different sizes: small lesions in MRI, nodules in
CT, cells and nuclei in microscopy, defects in industrial inspection, small buildings or
vehicles in remote sensing.

The losses in `comploss` are defined over the **connected components** of the ground
truth, so every object counts. The package includes:

* **Component-aware losses**: CATMIL and its two terms, CAT and MIL (Luu et al.), and CC-DiceCE
  (Bouteille et al., ISBI 2026).
* **Voxel-wise baselines** with the same interface: DiceCE (identical to nnU-Net's default),
  Tversky (identical to MONAI's), Focal Tversky.
* **Object-level metrics**: object recall, precision and F1, recall by object size, a per-object
  table, size profiling of a dataset.
* **Integrations**: nnU-Net v2 trainers for every loss, MONAI-style transforms, deep supervision.

All losses take raw logits `(B, C, *spatial)` and integer targets `(B, 1, *spatial)` (the
nnU-Net layout), in 2D or 3D. Tests check every port against its reference implementation
(values and gradients).

## Install

```bash
pip install comploss                  # torch, numpy, scipy
pip install "comploss[nnunet]"        # + nnU-Net v2 trainers
pip install "comploss[fast-cc]"       # + faster connected components (cc3d)
```

Tested with Python 3.10–3.12, torch 2.0.1–2.15 and nnU-Net v2 2.1.1–2.8 (CI runs the newest
torch and the oldest supported versions). With torch older than 2.3, install `numpy<2`: those
torch builds were compiled against numpy 1. nnU-Net v2 2.1 is not supported (its Dice+CE differs
from later versions); the CATMIL paper used a fork of nnU-Net v2 2.1.1.

From source (for development):

```bash
git clone https://github.com/luumsk/comploss && cd comploss
pip install -e ".[dev,nnunet,fast-cc]"
pytest
```

## Available losses

| name (`get_loss`) | class | type | reference |
|---|---|---|---|
| `catmil` | `CATMILLoss` | DiceCE + CAT + MIL | Luu et al., arXiv:2604.08015 |
| `cat` | `ComponentAdaptiveTverskyLoss` | component-weighted Tversky | Luu et al. |
| `mil` | `ComponentMILLoss` | one bag per object, `-log max p` | Luu et al. |
| `cc_dice_ce` | `CCDiceCEGlobalLoss` | Voronoi component DiceCE + global DiceCE | Bouteille et al., arXiv:2511.17146 |
| `cc_dice_ce_component` | `CCDiceCELoss` | Voronoi component DiceCE only | Bouteille et al. |
| `dice_ce` | `DiceCELoss` | voxel-wise (nnU-Net default) | Isensee et al. |
| `tversky` | `TverskyLoss` | voxel-wise | Salehi et al. |
| `focal_tversky` | `FocalTverskyLoss` | voxel-wise | Abraham & Khan |

## Usage in PyTorch or MONAI

```python
from comploss import get_loss

loss_fn = get_loss("catmil")      # DiceCE + 0.1*CAT (linear warm-up over 50 epochs) + 0.2*MIL
for epoch in range(n_epochs):
    loss_fn.set_epoch(epoch)      # drives the CAT warm-up
    for image, label in loader:              # label: (B, 1, D, H, W) integer map
        logits = model(image)                # (B, C, D, H, W); or a list for deep supervision
        out = loss_fn(logits, label, return_components=True)
        out["loss"].backward()
        log(base=out["base"], cat=out["cat"], mil=out["mil"])
```

You can pick any loss by name, for example to compare losses from a config file:

```python
from comploss import get_loss, list_losses, register_loss
list_losses()
loss_fn = get_loss("focal_tversky", deep_supervision=True)   # takes lists of outputs/targets
register_loss("my_loss", MyLoss)                             # compare your own loss the same way
```

A runnable example with synthetic data is in [examples/train_pytorch.py](examples/train_pytorch.py).

### Options shared by the component losses

| option | meaning |
|---|---|
| `mode="union"` / `"per_class"` / `"auto"` | one object mask = any foreground, or one mask per class (multi-class) |
| `activation="sigmoid"` | region-based / multi-label outputs (binary target per channel) |
| `ignore_index=k` | voxels labelled `k` are ignored (never part of an object) |
| `connectivity=1` | 1 = face neighbours (6 in 3D, 4 in 2D), up to `ndim` = all neighbours |
| `components=` | precomputed GT components from `LabelComponentsd` in the data loader (avoids a GPU→CPU copy per step) |

CATMIL also takes `base_loss=` (your own voxel loss instead of DiceCE), `ds_mode="highest"` or
`"all"` (component terms on the full-resolution output only, or on every deep-supervision output),
and `cat_kwargs=dict(alpha, beta, gamma, eps_cc, w_bg)`.

The CATMIL defaults are the paper's values: alpha 0.3, beta 0.7, gamma 1, eps_cc 5, w_bg 0.01,
lambda_cat 0.1 (50-epoch warm-up), lambda_mil 0.2, connectivity 1.

## nnU-Net v2

```bash
pip install "comploss[nnunet]"
comploss nnunet install     # puts a stub in nnunetv2/training/nnUNetTrainer/ (nnU-Net only looks there)
comploss nnunet status      # shows which module each trainer resolves to
nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerCATMIL
```

The trainers are `nnUNetTrainerCATMIL`, `nnUNetTrainerCAT`, `nnUNetTrainerMIL`,
`nnUNetTrainerCCDiceCE`, `nnUNetTrainerDiceCE`, `nnUNetTrainerTversky` and
`nnUNetTrainerFocalTversky`.

* The base term is nnU-Net's own loss (DiceCE, or Dice+BCE for regions), with nnU-Net's deep
  supervision, DDP batch Dice and ignore label.
* The CAT warm-up follows `current_epoch`, including when you resume from a checkpoint.
* The component terms run in float32 even under AMP, and stay out of `torch.compile`.
* Tested with nnU-Net 2.1.1 and 2.8.1.

To change hyperparameters, write a subclass and put it in a file inside
`nnunetv2/training/nnUNetTrainer/`:

```python
from comploss.integrations.nnunet import nnUNetTrainerCATMIL

class nnUNetTrainerCATMIL_lmil05(nnUNetTrainerCATMIL):
    loss_lambda_mil = 0.5          # any loss_* attribute, see comploss/integrations/nnunet/__init__.py
```

For a quick test you can set environment variables instead, for example
`COMPLOSS_LAMBDA_MIL=0.5 nnUNetv2_train ...`. `COMPLOSS_NUM_EPOCHS` overrides the
epoch count.

`comploss nnunet uninstall` removes the stub. Run `install` again after you upgrade nnunetv2.

## Object-level evaluation

Voxel Dice hides missed small objects. A ground-truth object is a connected component, and it
counts as detected if any predicted voxel overlaps it.

```python
from comploss.metrics import object_detection_metrics, recall_by_size, object_table, object_size_profile
object_detection_metrics(gt, pred, spacing=(1, 1, 1))   # object F1/precision/recall, small-object recall, FP count, missed volume
recall_by_size(gt, pred)                                 # tiny <=10, small <=50, medium <=200, large voxels
object_table(gt, pred, prob)                             # one row per object: size, detected, coverage, max prob
object_size_profile([gt1, gt2, ...])                     # object-size distribution of a dataset
```

For medical imaging, `lesion_detection_metrics`, `lesion_table` and `lesion_size_profile` return
the same values with lesion key names (`lesion_recall`, `gt_lesion_count`, ...), as in the CATMIL
paper. `SmallLesionMILLoss`, the paper's name for `ComponentMILLoss`, also still works.

## Command line

```bash
comploss profile path/to/labelsTr     # object-size distribution: does your dataset have many small objects?
comploss bench --shape 2 2 128 128 128 --device cuda   # ms per forward+backward for each loss
comploss losses
```

## Tests

```bash
pip install -e ".[test]"
pytest            # the nnU-Net tests run when nnunetv2 is installed
```

`tests/_reference_losses.py` and `tests/_reference_metrics.py` hold frozen copies of the
research code. The parity tests compare values and gradients against them, against nnU-Net's
`DC_and_CE_loss` / `DC_and_BCE_loss`, and against MONAI's `TverskyLoss`.

## Adding a loss

1. Put a component-aware loss in `src/comploss/losses/component/`, or a voxel-wise one in
   `losses/voxel/`, using the `forward(logits, target)` interface. Use
   `comploss.components` for labelling and `losses._utils.binary_problems` for the
   softmax / sigmoid / ignore-label handling.
2. Register it in `losses/registry.py`.
3. If you port it from published code, add a frozen copy of that code to `tests/` and a parity
   test. Cite the paper in the docstring and in the table above.

## Citation

If you use this package, please cite it (see [CITATION.cff](CITATION.cff)):
[doi:10.5281/zenodo.24590690](https://doi.org/10.5281/zenodo.24590690) covers all versions, and each
release has its own DOI on Zenodo. If you use CATMIL,
CAT or MIL, please also cite:

```bibtex
@article{luu2026catmil,
  title   = {Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI},
  author  = {Luu, Minh Sao Khue and Pavlovskiy, Evgeniy N. and Tuchinov, Bair N.},
  journal = {arXiv preprint arXiv:2604.08015},
  year    = {2026}
}
```

If you use CC-DiceCE, please cite Bouteille et al., ISBI 2026 (arXiv:2511.17146).

The code to reproduce the CATMIL paper is at
[github.com/luumsk/SmallLesionMRI](https://github.com/luumsk/SmallLesionMRI).

## License

MIT, see [LICENSE](LICENSE).
