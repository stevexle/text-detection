"""
Unit tests for Text Detection Evaluation Metrics (ICDAR 2015 Protocol).
"""

import unittest
import numpy as np
from src.metrics.evaluator import ICDAREvaluator, polygon_iou
from src.metrics.builder import build_evaluator


class TestPolygonIoU(unittest.TestCase):
    """Test polygon IoU calculation."""

    def test_identical_squares(self):
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        poly2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        iou = polygon_iou(poly1, poly2)
        self.assertAlmostEqual(iou, 1.0, places=4)

    def test_non_overlapping_squares(self):
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        poly2 = np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32)
        iou = polygon_iou(poly1, poly2)
        self.assertAlmostEqual(iou, 0.0, places=4)

    def test_half_overlapping_squares(self):
        poly1 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
        poly2 = np.array([[5, 0], [15, 0], [15, 10], [5, 10]], dtype=np.float32)
        # Inter = 50, Union = 150, IoU = 50 / 150 = 1/3
        iou = polygon_iou(poly1, poly2)
        self.assertAlmostEqual(iou, 1.0 / 3.0, places=4)


class TestICDAREvaluator(unittest.TestCase):
    """Test ICDAR 2015 evaluation protocol and metric calculations."""

    def setUp(self):
        self.evaluator = ICDAREvaluator(iou_thresh=0.5)

    def test_perfect_match(self):
        self.evaluator.reset()
        gt = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32),
        ]
        pred = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]], dtype=np.float32),
        ]
        scores = [0.95, 0.90]

        res = self.evaluator.evaluate_image(pred, scores, gt, [False, False])
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)

        metrics = self.evaluator.compute_metrics()
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["hmean"], 1.0)

    def test_zero_predictions(self):
        self.evaluator.reset()
        gt = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]
        pred = []

        res = self.evaluator.evaluate_image(pred, [], gt, [False])
        self.assertEqual(res["tp"], 0)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 1)

        metrics = self.evaluator.compute_metrics()
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["hmean"], 0.0)

    def test_ignore_tags(self):
        self.evaluator.reset()
        gt = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32),  # Ignored
        ]
        pred = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32),
            np.array([[50, 50], [60, 50], [60, 60], [50, 60]], dtype=np.float32),
        ]
        ignore = [False, True]

        res = self.evaluator.evaluate_image(pred, [0.9, 0.8], gt, ignore)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 0)  # Ignored match doesn't count as FP
        self.assertEqual(res["fn"], 0)

        metrics = self.evaluator.compute_metrics()
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_builder(self):
        evaluator = build_evaluator(dict(type="ICDAREvaluator", iou_thresh=0.6))
        self.assertIsInstance(evaluator, ICDAREvaluator)
        self.assertEqual(evaluator.iou_thresh, 0.6)


if __name__ == "__main__":
    unittest.main()
