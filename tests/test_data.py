"""
Unit tests for Data Pipeline and Augmentations (DBTargetGenerator, Transforms, Dataset, Builder).
"""

import numpy as np
import pytest
import torch
from src.data.db_target_generator import DBTargetGenerator
from src.data.transforms import (
    ColorJitter,
    Compose,
    NormalizeImage,
    RandomPerspective,
    RandomRotate,
    Resize,
    ToTensor,
)
from src.data.dataset import TextDetectionDataset
from src.data.builder import build_dataloader


def test_db_target_generator():
    generator = DBTargetGenerator(shrink_ratio=0.4, min_thresh=0.3, max_thresh=0.7)

    poly = np.array([
        [50, 50],
        [200, 50],
        [200, 100],
        [50, 100],
    ], dtype=np.float32)

    polygons = [poly]
    ignore_tags = [False]

    targets = generator.generate_targets(
        image_shape=(256, 256),
        polygons=polygons,
        ignore_tags=ignore_tags,
    )

    assert "gt_prob_map" in targets
    assert "gt_thresh_map" in targets
    assert "gt_thresh_mask" in targets
    assert "gt_mask" in targets

    assert targets["gt_prob_map"].shape == (256, 256)
    assert targets["gt_thresh_map"].shape == (256, 256)
    assert targets["gt_thresh_mask"].shape == (256, 256)
    assert targets["gt_mask"].shape == (256, 256)

    # Values within bounds
    assert targets["gt_prob_map"].max() == 1.0
    assert targets["gt_thresh_map"].min() >= 0.3 - 1e-5
    assert targets["gt_thresh_map"].max() <= 0.7 + 1e-5


def test_transforms_coordinate_scaling():
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    poly = np.array([[100.0, 100.0], [300.0, 100.0], [300.0, 200.0], [100.0, 200.0]], dtype=np.float32)
    data = {"image": img, "polygons": [poly]}

    # Test Resize
    resize = Resize(size=(300, 200))
    res_data = resize(data)
    assert res_data["image"].shape == (200, 300, 3)
    p_scaled = res_data["polygons"][0]
    assert pytest.approx(p_scaled[0, 0], 1e-3) == 50.0   # 100 * (300/600)
    assert pytest.approx(p_scaled[0, 1], 1e-3) == 50.0   # 100 * (200/400)

    # Test Pipeline Compose
    pipeline = Compose([
        Resize(size=(640, 640)),
        RandomRotate(max_angle=5.0, prob=1.0),
        RandomPerspective(distortion_scale=0.1, prob=1.0),
        ColorJitter(brightness=0.2, prob=1.0),
        NormalizeImage(),
        ToTensor(),
    ])
    out = pipeline({"image": img, "polygons": [poly]})
    assert isinstance(out["image"], torch.Tensor)
    assert out["image"].shape == (3, 640, 640)


def test_real_dataset_and_dataloader():
    train_cfg = {
        "type": "TextDetectionDataset",
        "is_training": True,
        "data_dir": "data",
        "ann_file": "labels/dbnet/dataset.jsonl",
        "val_ratio": 0.15,
        "split": "train",
        "split_seed": 42,
        "target_size": [256, 256],
    }

    val_cfg = {
        "type": "TextDetectionDataset",
        "is_training": False,
        "data_dir": "data",
        "ann_file": "labels/dbnet/dataset.jsonl",
        "val_ratio": 0.15,
        "split": "val",
        "split_seed": 42,
        "target_size": [256, 256],
    }

    train_loader = build_dataloader(train_cfg, batch_size=4, shuffle=False, num_workers=0)
    val_loader = build_dataloader(val_cfg, batch_size=4, shuffle=False, num_workers=0)

    # 912 total samples: 85% train = 776, 15% val = 136
    assert len(train_loader.dataset) == 776
    assert len(val_loader.dataset) == 136

    # Test batch structure
    batch = next(iter(train_loader))
    assert batch["image"].shape == (4, 3, 256, 256)
    assert batch["gt_prob_map"].shape == (4, 1, 256, 256)
    assert batch["gt_thresh_map"].shape == (4, 1, 256, 256)
    assert batch["gt_thresh_mask"].shape == (4, 1, 256, 256)
    assert batch["gt_mask"].shape == (4, 1, 256, 256)
    assert len(batch["image_path"]) == 4
    assert len(batch["polygons"]) == 4
