"""All losses take raw logits ``(B, C, *spatial)`` and nnU-Net-style targets ``(B, 1, *spatial)``."""
from ..components import voronoi_components
from ._utils import foreground_probability, nnunet_ds_weights, squeeze_target
from .component import (CATMILLoss, CCDiceCEGlobalLoss, CCDiceCELoss, ComponentAdaptiveTverskyLoss,
                        SmallLesionMILLoss, component_max, component_weight_map)
from .deep_supervision import DeepSupervisionLoss
from .registry import get_loss, list_losses, register_loss
from .voxel import DiceCELoss, FocalTverskyLoss, SoftDiceLoss, TverskyLoss

__all__ = [
    "CATMILLoss", "ComponentAdaptiveTverskyLoss", "SmallLesionMILLoss",
    "DiceCELoss", "SoftDiceLoss", "TverskyLoss", "FocalTverskyLoss", "CCDiceCELoss", "CCDiceCEGlobalLoss",
    "DeepSupervisionLoss", "get_loss", "list_losses", "register_loss",
    "component_weight_map", "component_max", "voronoi_components",
    "foreground_probability", "squeeze_target", "nnunet_ds_weights",
]
