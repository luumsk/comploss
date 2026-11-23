"""Component-aware losses: terms defined over the connected components of the ground truth."""
from .cat import ComponentAdaptiveTverskyLoss, component_weight_map
from .catmil import CATMILLoss
from .cc_dice_ce import CCDiceCEGlobalLoss, CCDiceCELoss
from .mil import SmallLesionMILLoss, component_max

__all__ = [
    "CATMILLoss", "ComponentAdaptiveTverskyLoss", "SmallLesionMILLoss", "CCDiceCELoss", "CCDiceCEGlobalLoss",
    "component_weight_map", "component_max",
]
