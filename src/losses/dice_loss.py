"""
Dice Loss for Text Region Segmentation.
Measures contour area overlap between prediction and target.
"""

from typing import Optional
import torch
import torch.nn as nn
from src.losses.builder import LOSSES


@LOSSES.register_module(name="DiceLoss")
@LOSSES.register_module(name="dice_loss")
class DiceLoss(nn.Module):
    """Dice Loss for continuous probability segmentation."""

    def __init__(
        self,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probability tensor (B, 1, H, W)
            target: Ground truth binary target (B, 1, H, W)
            mask: Valid region mask (B, 1, H, W)
        Returns:
            Scalar Dice loss (1 - Dice Coefficient)
        """
        pred = pred.float()
        target = target.float()
        if mask is not None:
            mask = mask.float()
            pred = pred * mask
            target = target * mask

        intersection = 2.0 * torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)

        dice_score = (intersection + self.eps) / (union + self.eps)
        return 1.0 - dice_score
