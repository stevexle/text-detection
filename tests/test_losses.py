"""
Unit tests for Loss Functions (BalanceCrossEntropyLoss, MaskedSmoothL1Loss, DiceLoss, DBLoss).
"""

import pytest
import torch

from src.losses import (
    BalanceCrossEntropyLoss,
    MaskedSmoothL1Loss,
    DiceLoss,
    DBLoss,
    build_loss,
)


def test_balance_bce_loss():
    """Test OHEM Balanced BCE loss computation."""
    loss_fn = BalanceCrossEntropyLoss(negative_ratio=3.0)

    # 1. Prediction and target matching
    pred = torch.tensor([[[[0.9, 0.1], [0.1, 0.9]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    mask = torch.ones_like(target)

    loss = loss_fn(pred, target, mask)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None


def test_masked_l1_loss():
    """Test Masked Smooth L1 loss computed strictly on mask."""
    loss_fn = MaskedSmoothL1Loss(beta=1.0)

    pred = torch.tensor([[[[0.7, 0.3], [0.3, 0.7]]]], requires_grad=True)
    target = torch.tensor([[[[0.7, 0.3], [0.3, 0.5]]]])
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = loss_fn(pred, target, mask)
    assert loss.item() > 0.0
    loss.backward()
    assert pred.grad is not None


def test_dice_loss():
    """Test Dice loss segmentation overlap."""
    loss_fn = DiceLoss()

    pred = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss_perfect = loss_fn(pred, target)
    assert pytest.approx(loss_perfect.item(), abs=1e-4) == 0.0

    pred_wrong = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
    loss_wrong = loss_fn(pred_wrong, target)
    assert loss_wrong.item() > 0.8


def test_full_db_loss():
    """Test consolidated Multi-Task DBLoss and backpropagation."""
    db_loss = build_loss({
        "type": "DBLoss",
        "alpha": 1.0,
        "beta": 10.0,
        "ohem_ratio": 3.0,
    })

    B, H, W = 2, 64, 64
    preds = {
        "prob_map": torch.rand(B, 1, H, W, requires_grad=True),
        "thresh_map": torch.rand(B, 1, H, W, requires_grad=True),
        "binary_map": torch.rand(B, 1, H, W, requires_grad=True),
    }
    targets = {
        "gt_prob_map": (torch.rand(B, 1, H, W) > 0.7).float(),
        "gt_mask": torch.ones(B, 1, H, W),
        "gt_thresh_map": torch.full((B, 1, H, W), 0.3) + 0.4 * torch.rand(B, 1, H, W),
        "gt_thresh_mask": (torch.rand(B, 1, H, W) > 0.8).float(),
    }

    loss_dict = db_loss(preds, targets)
    assert "loss" in loss_dict
    assert "loss_prob" in loss_dict
    assert "loss_binary" in loss_dict
    assert "loss_thresh" in loss_dict

    total_loss = loss_dict["loss"]
    assert total_loss.item() > 0.0

    # Backpropagation test
    total_loss.backward()
    assert preds["prob_map"].grad is not None
    assert preds["thresh_map"].grad is not None
    assert preds["binary_map"].grad is not None
