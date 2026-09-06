"""
Unit tests for Post-Processing (DBPostprocessor).
"""

import cv2
import numpy as np
import pytest
import torch

from src.postprocess import DBPostprocessor, build_postprocessor


def test_db_postprocessor_synthetic():
    """Test polygon extraction in unified detection format from synthetic probability map."""
    postprocessor = build_postprocessor({
        "type": "DBPostprocessor",
        "thresh": 0.3,
        "box_thresh": 0.5,
        "unclip_ratio": 1.5,
    })

    H, W = 200, 200
    prob_map = np.zeros((1, 1, H, W), dtype=np.float32)

    # Draw two high-confidence text boxes
    # Box 1: [20, 20] to [100, 50]
    prob_map[0, 0, 20:50, 20:100] = 0.95
    # Box 2: [50, 100] to [160, 140]
    prob_map[0, 0, 100:140, 50:160] = 0.90

    # Single image inference returns unified list
    detections = postprocessor(prob_map)
    assert isinstance(detections, list)
    assert len(detections) == 2

    for det in detections:
        assert det["label"] == "text"
        assert det["confidence"] >= 0.5
        assert isinstance(det["polygon"], list)
        assert len(det["polygon"]) == 4

        # Verify Clockwise vertex ordering: [TL, TR, BR, BL]
        poly = det["polygon"]
        # Top points have smaller y than bottom points
        assert poly[0][1] <= poly[3][1]
        assert poly[1][1] <= poly[2][1]


def test_db_postprocessor_extract_detections():
    """Test extract_detections helper method."""
    postprocessor = DBPostprocessor(thresh=0.3, box_thresh=0.5)
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[20:50, 20:80] = 0.95

    detections = postprocessor.extract_detections(prob_map, orig_shape=(200, 200), label="id")
    assert len(detections) == 1
    assert detections[0]["label"] == "id"
    assert detections[0]["confidence"] >= 0.5
    assert len(detections[0]["polygon"]) == 4


def test_db_postprocessor_empty_image():
    """Test postprocessing when no text exists."""
    postprocessor = DBPostprocessor()
    empty_prob = np.zeros((100, 100), dtype=np.float32)

    detections = postprocessor(empty_prob)
    assert isinstance(detections, list)
    assert len(detections) == 0

    # Batch empty images
    batch_empty = np.zeros((2, 1, 100, 100), dtype=np.float32)
    batch_detections = postprocessor(batch_empty)
    assert len(batch_detections) == 2
    assert len(batch_detections[0]) == 0
    assert len(batch_detections[1]) == 0
