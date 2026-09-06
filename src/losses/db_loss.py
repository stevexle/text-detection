"""
Multi-Task DBLoss for DBNet Text Detection.
Combines Probability Map BCE/OHEM Loss, Binary Map Dice/BCE Loss, and Masked Smooth L1 Threshold Loss.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
from src.losses.builder import LOSSES
from src.losses.bce_loss import BalanceCrossEntropyLoss
from src.losses.l1_loss import MaskedSmoothL1Loss
from src.losses.dice_loss import DiceLoss


@LOSSES.register_module(name="DBLoss")
@LOSSES.register_module(name="db_loss")
class DBLoss(nn.Module):
    """
    Consolidated DBNet Multi-Task Loss:
    L = L_prob + alpha * L_binary + beta * L_thresh
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 10.0,
        ohem_ratio: float = 3.0,
        use_dice_for_binary: bool = True,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_dice_for_binary = use_dice_for_binary

        self.prob_loss_fn = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, eps=eps)
        if use_dice_for_binary:
            self.binary_loss_fn = DiceLoss(eps=eps)
        else:
            self.binary_loss_fn = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio, eps=eps)
        self.thresh_loss_fn = MaskedSmoothL1Loss(beta=1.0, eps=eps)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: Dict containing 'prob_map', 'thresh_map', 'binary_map' (B, 1, H, W)
            targets: Dict containing 'gt_prob_map', 'gt_mask', 'gt_thresh_map', 'gt_thresh_mask'
        Returns:
            Dict containing 'loss' (total loss) and individual loss components
        """
        prob_pred = preds["prob_map"]
        if prob_pred.ndim == 4 and prob_pred.shape[1] == 1:
            prob_pred = prob_pred.squeeze(1)

        gt_prob = targets["gt_prob_map"]
        if gt_prob.ndim == 4 and gt_prob.shape[1] == 1:
            gt_prob = gt_prob.squeeze(1)

        gt_mask = targets.get("gt_mask", torch.ones_like(gt_prob))
        if gt_mask.ndim == 4 and gt_mask.shape[1] == 1:
            gt_mask = gt_mask.squeeze(1)

        # 1. Probability Map Loss (BCE + OHEM)
        loss_prob = self.prob_loss_fn(prob_pred, gt_prob, gt_mask)

        # 2. Binary Map Loss (Dice or BCE)
        if "binary_map" in preds:
            binary_pred = preds["binary_map"]
            if binary_pred.ndim == 4 and binary_pred.shape[1] == 1:
                binary_pred = binary_pred.squeeze(1)
            loss_binary = self.binary_loss_fn(binary_pred, gt_prob, gt_mask)
        else:
            loss_binary = torch.tensor(0.0, device=prob_pred.device)

        # 3. Threshold Map Loss (Masked Smooth L1)
        if "thresh_map" in preds and "gt_thresh_map" in targets:
            thresh_pred = preds["thresh_map"]
            if thresh_pred.ndim == 4 and thresh_pred.shape[1] == 1:
                thresh_pred = thresh_pred.squeeze(1)

            gt_thresh = targets["gt_thresh_map"]
            if gt_thresh.ndim == 4 and gt_thresh.shape[1] == 1:
                gt_thresh = gt_thresh.squeeze(1)

            gt_thresh_mask = targets.get("gt_thresh_mask", torch.ones_like(gt_thresh))
            if gt_thresh_mask.ndim == 4 and gt_thresh_mask.shape[1] == 1:
                gt_thresh_mask = gt_thresh_mask.squeeze(1)

            loss_thresh = self.thresh_loss_fn(thresh_pred, gt_thresh, gt_thresh_mask)
        else:
            loss_thresh = torch.tensor(0.0, device=prob_pred.device)

        # Total Weighted Multi-Task Loss
        total_loss = loss_prob + self.alpha * loss_binary + self.beta * loss_thresh

        return {
            "loss": total_loss,
            "loss_prob": loss_prob,
            "loss_binary": loss_binary,
            "loss_thresh": loss_thresh,
        }
