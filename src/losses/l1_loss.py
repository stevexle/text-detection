"""
Masked Smooth L1 Loss for Threshold Map Supervision.
Calculates L1 regression loss strictly over the boundary mask region.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.builder import LOSSES


@LOSSES.register_module(name="MaskedSmoothL1Loss")
@LOSSES.register_module(name="masked_l1")
class MaskedSmoothL1Loss(nn.Module):
    """Masked Smooth L1 / L1 Loss computed exclusively over masked regions."""

    def __init__(
        self,
        beta: float = 1.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted threshold map (B, 1, H, W)
            target: Ground truth threshold map (B, 1, H, W)
            mask: Threshold boundary mask (B, 1, H, W)
        Returns:
            Scalar masked L1 loss
        """
        pred = pred.float()
        target = target.float()
        if mask is None:
            mask = torch.ones_like(target)
        else:
            mask = mask.float()

        loss = F.smooth_l1_loss(pred, target, reduction="none", beta=self.beta)
        masked_loss = loss * mask

        mask_sum = mask.sum()
        if mask_sum == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return masked_loss.sum() / (mask_sum + self.eps)
