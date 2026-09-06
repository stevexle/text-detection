"""
Unit tests for Model Architectures (Backbones, Necks, Heads, DBNet, and Pretrained Weights).
"""

from pathlib import Path
import pytest
import torch

from src.models.backbones import ResNet, MobileNetV3
from src.models.necks import FPN, ASF
from src.models.heads import DBHead
from src.models.detectors import DBNet
from src.models.builder import build_model, MODELS, BACKBONES, NECKS, HEADS


def test_resnet_backbone():
    """Test ResNet backbone feature extraction at multi-scale resolutions."""
    # Test ResNet-18
    r18 = ResNet(depth=18, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    feats_18 = r18(x)
    assert len(feats_18) == 4
    assert feats_18[0].shape == (2, 64, 64, 64)
    assert feats_18[1].shape == (2, 128, 32, 32)
    assert feats_18[2].shape == (2, 256, 16, 16)
    assert feats_18[3].shape == (2, 512, 8, 8)

    # Test ResNet-50
    r50 = ResNet(depth=50, pretrained=False)
    feats_50 = r50(x)
    assert len(feats_50) == 4
    assert feats_50[0].shape == (2, 256, 64, 64)
    assert feats_50[1].shape == (2, 512, 32, 32)
    assert feats_50[2].shape == (2, 1024, 16, 16)
    # Test frozen stages
    r50_frozen = ResNet(depth=50, pretrained=False, frozen_stages=2)
    assert not any(p.requires_grad for p in r50_frozen.layer1.parameters())
    assert not any(p.requires_grad for p in r50_frozen.layer2.parameters())
    assert any(p.requires_grad for p in r50_frozen.layer3.parameters())


def test_mobilenetv3_backbone():
    """Test MobileNetV3 Large and Small feature extraction."""
    x = torch.randn(2, 3, 256, 256)

    # MobileNetV3 Large
    mb_large = MobileNetV3(arch="large", pretrained=False, frozen_stages=1)
    feats_large = mb_large(x)
    assert len(feats_large) == 4
    assert feats_large[0].shape == (2, 24, 64, 64)
    assert feats_large[1].shape == (2, 40, 32, 32)
    assert feats_large[2].shape == (2, 112, 16, 16)
    assert feats_large[3].shape == (2, 960, 8, 8)
    assert not any(p.requires_grad for p in mb_large.stage1.parameters())
    assert any(p.requires_grad for p in mb_large.stage2.parameters())

    # MobileNetV3 Small
    mb_small = MobileNetV3(arch="small", pretrained=False)
    feats_small = mb_small(x)
    assert len(feats_small) == 4
    assert feats_small[0].shape == (2, 16, 64, 64)
    assert feats_small[1].shape == (2, 24, 32, 32)
    assert feats_small[2].shape == (2, 48, 16, 16)
    assert feats_small[3].shape == (2, 576, 8, 8)


def test_fpn_and_asf_necks():
    """Test FPN and ASF feature fusion neck modules."""
    feats = [
        torch.randn(2, 256, 64, 64),
        torch.randn(2, 512, 32, 32),
        torch.randn(2, 1024, 16, 16),
        torch.randn(2, 2048, 8, 8),
    ]

    # Test FPN
    fpn = FPN(in_channels=[256, 512, 1024, 2048], inner_channels=256, out_channels=256)
    fused_fpn = fpn(feats)
    assert fused_fpn.shape == (2, 256, 64, 64)

    # Test ASF
    asf = ASF(in_channels=[256, 512, 1024, 2048], inner_channels=256, out_channels=256)
    fused_asf = asf(feats)
    assert fused_asf.shape == (2, 256, 64, 64)


def test_db_head():
    """Test DBHead output maps and Differentiable Binarization step function."""
    head = DBHead(in_channels=256, k=50.0)
    x = torch.randn(2, 256, 64, 64)
    out = head(x)

    assert "prob_map" in out
    assert "thresh_map" in out
    assert "binary_map" in out
    assert out["prob_map"].shape == (2, 1, 256, 256)
    assert out["thresh_map"].shape == (2, 1, 256, 256)
    assert out["binary_map"].shape == (2, 1, 256, 256)

    # Check value ranges in [0.0, 1.0]
    assert (out["prob_map"] >= 0.0).all() and (out["prob_map"] <= 1.0).all()
    assert (out["thresh_map"] >= 0.0).all() and (out["thresh_map"] <= 1.0).all()
    assert (out["binary_map"] >= 0.0).all() and (out["binary_map"] <= 1.0).all()


def test_dbnet_full_pipeline():
    """Test full DBNet end-to-end construction and inference."""
    cfg = {
        "type": "DBNet",
        "backbone": {"type": "ResNet", "depth": 50, "pretrained": False},
        "neck": {"type": "FPN", "inner_channels": 256, "out_channels": 256},
        "head": {"type": "DBHead", "in_channels": 256, "k": 50.0},
    }
    model = build_model(cfg)
    assert isinstance(model, DBNet)

    x = torch.randn(2, 3, 256, 256)
    # Test training mode returns all 3 maps
    out = model(x)
    assert out["prob_map"].shape == (2, 1, 256, 256)
    assert out["thresh_map"].shape == (2, 1, 256, 256)
    assert out["binary_map"].shape == (2, 1, 256, 256)

    # Test eval mode bypasses threshold branch for lightweight inference
    model.eval()
    out_eval = model(x)
    assert "prob_map" in out_eval
    assert "thresh_map" not in out_eval
    assert out_eval["prob_map"].shape == (2, 1, 256, 256)


def test_pretrained_weights_loading():
    """Test loading pretrained OCR checkpoint into DBNet."""
    weights_path = Path("weights/dbnet/dbnet_r50_icdar2015.pth")
    if weights_path.exists():
        model = DBNet(
            backbone={"type": "ResNet", "depth": 50, "pretrained": False},
            neck={"type": "FPN", "in_channels": [256, 512, 1024, 2048], "inner_channels": 256, "out_channels": 256},
            head={"type": "DBHead", "in_channels": 256, "k": 50.0},
            pretrained_weights=str(weights_path),
        )
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out["prob_map"].shape == (1, 1, 256, 256)


def test_yolo_wrapper_unified_detect():
    """Test YOLOWrapper.detect output conforming to unified schema."""
    from unittest.mock import MagicMock
    from src.models.wrappers.yolo_wrapper import YOLOWrapper

    wrapper = YOLOWrapper(model_path="dummy.pt")

    # Mock ultralytics YOLO prediction
    mock_box = MagicMock()
    mock_box.conf = 0.9750
    mock_box.cls = 0
    mock_box.xyxy = [torch.tensor([10.0, 20.0, 100.0, 50.0])]

    mock_res = MagicMock()
    mock_res.names = {0: "id", 1: "name"}
    mock_res.masks = None
    mock_res.boxes = [mock_box]

    mock_yolo = MagicMock()
    mock_yolo.predict.return_value = [mock_res]
    wrapper._model = mock_yolo

    detections = wrapper.detect("dummy.jpg", min_conf=0.5)
    assert len(detections) == 1
    assert detections[0]["label"] == "id"
    assert detections[0]["confidence"] == 0.9750
    assert detections[0]["polygon"] == [[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]]

