"""
Loss Functions for Text Detection.
"""

from .builder import LOSSES, build_loss
from .bce_loss import BalanceCrossEntropyLoss
from .l1_loss import MaskedSmoothL1Loss
from .dice_loss import DiceLoss
from .db_loss import DBLoss

__all__ = [
    "LOSSES",
    "build_loss",
    "BalanceCrossEntropyLoss",
    "MaskedSmoothL1Loss",
    "DiceLoss",
    "DBLoss",
]
