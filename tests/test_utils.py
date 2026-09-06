"""
Unit tests for Core Utilities (Registry, Config, Logger, Yolo_utils, Checkpoint).
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.utils.registry import Registry
from src.utils.config import Config, ConfigDict
from src.utils.logger import get_logger
from src.utils.yolo_utils import (
    CLASS_NAMES,
    CLASS_TO_ID,
    polygon_to_bbox,
    polygon_to_normalized_coords,
)
from src.utils.checkpoint import save_checkpoint, load_checkpoint, load_pretrained_weights


def test_registry_registration_and_build():
    TEST_REG = Registry("test_blocks")

    @TEST_REG.register_module()
    class DummyBlock(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x):
            return self.linear(x)

    assert "DummyBlock" in TEST_REG
    assert "dummyblock" in TEST_REG
    assert len(TEST_REG) == 2  # standard + lowercase

    # Test building instance from dict
    cfg = {"type": "DummyBlock", "in_features": 10, "out_features": 5}
    instance = TEST_REG.build(cfg)
    assert isinstance(instance, DummyBlock)
    assert instance.linear.in_features == 10
    assert instance.linear.out_features == 5

    # Test unregistered error
    with pytest.raises(KeyError):
        TEST_REG.build({"type": "NonExistentBlock"})


def test_config_parser():
    sample_yaml = """
    model:
      type: "DBNet"
      backbone:
        depth: 50
        pretrained: true
    data:
      target_size: [800, 512]
      batch_size: 8
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml)
        f_path = f.name

    cfg = Config.fromfile(f_path)
    assert cfg.model.type == "DBNet"
    assert cfg.model.backbone.depth == 50
    assert cfg.model.backbone.pretrained is True
    assert cfg.data.target_size == [800, 512]
    assert cfg.data.batch_size == 8

    # Test modification
    cfg.data.batch_size = 16
    assert cfg.data.batch_size == 16

    # Test conversion to native dict
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["data"]["batch_size"] == 16


def test_yolo_coordinate_conversions():
    assert len(CLASS_NAMES) == 11
    assert CLASS_TO_ID["id"] == 0
    assert CLASS_TO_ID["mrz"] == 10

    # Polygon to bbox
    poly = [[100.0, 50.0], [200.0, 50.0], [200.0, 150.0], [100.0, 150.0]]
    xc, yc, w, h = polygon_to_bbox(poly, img_w=640.0, img_h=480.0)
    assert pytest.approx(xc, 1e-4) == 150.0 / 640.0
    assert pytest.approx(yc, 1e-4) == 100.0 / 480.0
    assert pytest.approx(w, 1e-4) == 100.0 / 640.0
    assert pytest.approx(h, 1e-4) == 100.0 / 480.0

    # Polygon to normalized flat coords
    coords = polygon_to_normalized_coords(poly, img_w=500.0, img_h=500.0)
    assert len(coords) == 8
    assert coords == [0.2, 0.1, 0.4, 0.1, 0.4, 0.3, 0.2, 0.3]


def test_checkpoint_save_and_load():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

    model1 = SimpleModel()
    model2 = SimpleModel()

    with tempfile.NamedTemporaryFile("w", suffix=".pth", delete=False) as f:
        save_path = f.name

    # Save checkpoint
    save_checkpoint(model1, save_path, epoch=5, meta={"best_f1": 0.96})

    # Load into model2
    loaded_meta = load_checkpoint(save_path, model=model2)
    assert loaded_meta["epoch"] == 5
    assert loaded_meta["meta"]["best_f1"] == 0.96

    # Verify weights match
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
