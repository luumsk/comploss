"""comploss: component-aware losses and object-level metrics for segmentation.

Voxel-wise losses (Dice, cross entropy, Tversky) are dominated by large objects, so small
structures can be missed at almost no cost. The losses here are defined over the connected
components of the ground truth instead, so every object counts.

Quick start::

    from comploss import get_loss
    loss_fn = get_loss("catmil")      # DiceCE + 0.1 * CAT (50-epoch warm-up) + 0.2 * MIL
    for epoch in range(n_epochs):
        loss_fn.set_epoch(epoch)
        ...
        loss = loss_fn(logits, labels)  # logits (B, C, *spatial), labels (B, 1, *spatial)
"""
from .components import label_components, set_backend
from .losses import *  # noqa: F401,F403
from .losses import __all__ as _losses_all
from .transforms import LabelComponents, LabelComponentsd

__version__ = "0.1.1"
__all__ = list(_losses_all) + ["LabelComponents", "LabelComponentsd", "label_components", "set_backend"]
