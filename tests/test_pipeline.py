"""
Unit tests for End-to-End Pipeline & Spatial Matching Fusion.
"""

import unittest
from src.pipeline.spatial_matcher import match_text_to_fields


class TestSpatialMatcher(unittest.TestCase):
    """Test matching DBNet text detections with YOLO semantic fields."""

    def test_empty_inputs(self):
        self.assertEqual(match_text_to_fields([], []), [])
        self.assertEqual(
            match_text_to_fields([{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "confidence": 0.9}], []),
            [
                {
                    "label": "text",
                    "text_confidence": 0.9,
                    "field_confidence": 0.0,
                    "confidence": 0.9,
                    "overlap_ratio": 0.0,
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                }
            ],
        )

    def test_overlap_matching(self):
        text_dets = [
            {
                "polygon": [[10, 10], [50, 10], [50, 30], [10, 30]],
                "confidence": 0.88,
            },
            {
                "polygon": [[100, 100], [200, 100], [200, 150], [100, 150]],
                "confidence": 0.92,
            },
            {
                "polygon": [[500, 500], [550, 500], [550, 550], [500, 550]],
                "confidence": 0.75,
            },
        ]

        field_dets = [
            {
                "label": "id",
                "polygon": [[8, 8], [55, 8], [55, 35], [8, 35]],
                "confidence": 0.99,
            },
            {
                "label": "name",
                "polygon": [[95, 95], [205, 95], [205, 155], [95, 155]],
                "confidence": 0.95,
            },
        ]

        fused = match_text_to_fields(text_dets, field_dets, min_overlap_ratio=0.30)
        self.assertEqual(len(fused), 3)

        # First text polygon is inside "id"
        self.assertEqual(fused[0]["label"], "id")
        self.assertEqual(fused[0]["text_confidence"], 0.88)
        self.assertEqual(fused[0]["field_confidence"], 0.99)
        self.assertGreaterEqual(fused[0]["overlap_ratio"], 0.9)

        # Second text polygon is inside "name"
        self.assertEqual(fused[1]["label"], "name")
        self.assertEqual(fused[1]["text_confidence"], 0.92)
        self.assertEqual(fused[1]["field_confidence"], 0.95)

        # Third text polygon has no overlapping field -> "other_text"
        self.assertEqual(fused[2]["label"], "other_text")
        self.assertEqual(fused[2]["field_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
