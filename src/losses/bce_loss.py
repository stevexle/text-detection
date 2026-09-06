"""
Balanced Cross-Entropy Loss with Online Hard Example Mining (OHEM).
Maintains positive:negative ratio (default 1:3) to address extreme class imbalance.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.builder import LOSSES


@LOSSES.register_module(name="BalanceCrossEntropyLoss")
@LOSSES.register_module(name="bce_ohem")
class BalanceCrossEntropyLoss(nn.Module):
    """Balanced Binary Cross-Entropy Loss with OHEM mining."""

    def __init__(
        self,
        negative_ratio: float = 3.0,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted probability tensor (B, 1, H, W) in range [0, 1]
            target: Ground truth binary target (B, 1, H, W)
            mask: Valid region mask (B, 1, H, W) where 1 is valid, 0 is ignored
        Returns:
            Scalar balanced BCE loss
        """
        # Cast to float32 for numerical stability in AMP mixed precision
        pred = pred.float()
        target = target.float()
        if mask is not None:
            mask = mask.float()
        else:
            mask = torch.ones_like(target)

        # Clip predictions for numerical stability
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        bce = -(target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))

        pos_mask = (target == 1.0) & (mask == 1.0)
        neg_mask = (target == 0.0) & (mask == 1.0)

        pos_count = pos_mask.sum().float()
        neg_count = neg_mask.sum().float()

        if pos_count == 0:
            # Fallback when no positive text pixels exist in batch
            neg_losses = bce[neg_mask]
            k = min(len(neg_losses), 100)
            if k == 0:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
            topk_neg_losses, _ = torch.topk(neg_losses, k=k)
            return topk_neg_losses.mean()

        pos_loss = bce[pos_mask].sum()

        # OHEM: Select hardest negatives
        num_neg_target = min(int(neg_count.item()), int(pos_count.item() * self.negative_ratio))
        if num_neg_target > 0:
            neg_losses = bce[neg_mask]
            topk_neg_losses, _ = torch.topk(neg_losses, k=num_neg_target)
            neg_loss = topk_neg_losses.sum()
        else:
            neg_loss = torch.tensor(0.0, device=pred.device)

        total_loss = (pos_loss + neg_loss) / (pos_count + num_neg_target + self.eps)
        return total_loss
