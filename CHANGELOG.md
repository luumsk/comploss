# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

First release, extracted from the `catmil` package of
[SmallLesionMRI](https://github.com/luumsk/SmallLesionMRI) (arXiv:2604.08015).

### Added
- Component-aware losses: `CATMILLoss`, `ComponentAdaptiveTverskyLoss` (CAT),
  `SmallLesionMILLoss` (MIL), `CCDiceCELoss` and `CCDiceCEGlobalLoss` (CC-DiceCE).
- Voxel-wise losses: `DiceCELoss` (nnU-Net default), `TverskyLoss`, `FocalTverskyLoss`, `SoftDiceLoss`.
- `get_loss` / `list_losses` / `register_loss`, `DeepSupervisionLoss`.
- Connected-component labelling (scipy, or cc3d when installed) and Voronoi partitions.
- Object-level metrics: `lesion_detection_metrics`, `recall_by_size`, `lesion_table`,
  `lesion_size_profile`, `remove_small_components`.
- `LabelComponents` / `LabelComponentsd` data-loader transforms.
- nnU-Net v2 trainers for every loss and the `componentloss-nnunet` installer.
- `componentloss` command line (`losses`, `profile`, `bench`).

### Changed (compared to `catmil` in SmallLesionMRI)
- Package and CLI renamed: `catmil` -> `componentloss`, `catmil-nnunet` -> `componentloss-nnunet`.
- nnU-Net trainer hyperparameters: `catmil_<name>` -> `loss_<name>`, environment variables
  `CATMIL_<NAME>` -> `COMPONENTLOSS_<NAME>`. Trainer class names are unchanged.
- Modules reorganised into `components/`, `losses/voxel/`, `losses/component/`, `metrics/` and
  `integrations/nnunet/`. Loss values are unchanged (parity tests).
