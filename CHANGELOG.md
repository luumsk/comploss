# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-26

### Changed
- `CATMILLoss` labels the ground-truth components once per output and shares them between the
  CAT and MIL terms, instead of labelling them in each term. Loss values are unchanged.
- Packaging: the licence is an SPDX expression (`license = "MIT"`, `license-files`) instead of
  the deprecated licence table and classifier; building now needs setuptools 77 or newer.
- The `nnunet` extra needs nnU-Net v2 2.1.1 or newer (`nnunetv2>=2.1.1`). Version 2.1 computes
  Dice+CE differently, so the trainers do not match the paper's trainers with it.

### Added
- README states the tested versions: Python 3.10–3.12, torch 2.0.1–2.15, nnU-Net v2 2.1.1–2.8,
  and `numpy<2` with torch older than 2.3.
- CI job with the oldest supported versions (Python 3.10, torch 2.0.1, numpy<2, nnU-Net v2 2.1.2).

## [0.1.0] - 2026-11-24

First release, extracted from the `catmil` package of
[SmallLesionMRI](https://github.com/luumsk/SmallLesionMRI) (arXiv:2604.08015).

### Added
- Component-aware losses: `CATMILLoss`, `ComponentAdaptiveTverskyLoss` (CAT),
  `ComponentMILLoss` (MIL), `CCDiceCELoss` and `CCDiceCEGlobalLoss` (CC-DiceCE).
- `ComponentMILLoss` is the general name of the MIL term; `SmallLesionMILLoss` is kept as an alias.
- Voxel-wise losses: `DiceCELoss` (nnU-Net default), `TverskyLoss`, `FocalTverskyLoss`, `SoftDiceLoss`.
- `get_loss` / `list_losses` / `register_loss`, `DeepSupervisionLoss`.
- Connected-component labelling (scipy, or cc3d when installed) and Voronoi partitions.
- Object-level metrics: `object_detection_metrics`, `recall_by_size`, `object_table`,
  `object_size_profile`, `remove_small_components`. The `lesion_*` functions of `catmil` are kept
  as aliases and return lesion key names (`lesion_recall`, ...) with identical values.
- `object_detection_metrics` and `lesion_detection_metrics` count objects of at most 150 voxels
  as small by default (`small_voxels=150`), the threshold of the CATMIL paper.
- `LabelComponents` / `LabelComponentsd` data-loader transforms.
- nnU-Net v2 trainers for every loss and the `comploss nnunet` installer.
- `comploss` command line (`losses`, `profile`, `bench`, `nnunet install | uninstall | status`).

### Changed (compared to `catmil` in SmallLesionMRI)
- Package and CLI renamed: `catmil` -> `comploss`, `catmil-nnunet` -> `comploss nnunet`.
- nnU-Net trainer hyperparameters: `catmil_<name>` -> `loss_<name>`, environment variables
  `CATMIL_<NAME>` -> `COMPLOSS_<NAME>`. Trainer class names are unchanged.
- Modules reorganised into `components/`, `losses/voxel/`, `losses/component/`, `metrics/` and
  `integrations/nnunet/`. Loss values are unchanged (parity tests).
