"""
Unit tests for End-to-End Pipeline & Spatial Matching Fusion.
"""

import asyncio
from pathlib import Path
import unittest
import numpy as np
from src.pipeline.spatial_matcher import match_text_to_fields
from src.pipeline.cccd_pipeline import CCCDDetectionPipeline


class TestSpatialMatcher(unittest.TestCase):
    """Test matching DBNet text detections with YOLO semantic fields."""

    def test_empty_inputs(self):
        self.assertEqual(match_text_to_fields([], []), [])
        self.assertEqual(
            match_text_to_fields([{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "confidence": 0.9}], []),
            [
                {
                    "label": "text",
                    "confidence": 0.9,
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
        self.assertEqual(fused[0]["confidence"], 0.88)

        # Second text polygon is inside "name"
        self.assertEqual(fused[1]["label"], "name")
        self.assertEqual(fused[1]["confidence"], 0.92)

        # Third text polygon has no overlapping field -> "other_text"
        self.assertEqual(fused[2]["label"], "other_text")
        self.assertEqual(fused[2]["confidence"], 0.75)


class TestCCCDPipelineExecution(unittest.TestCase):
    """Test CCCDDetectionPipeline synchronous vs async vs sequential execution."""

    def test_pipeline_concurrent_vs_sequential(self):
        test_img_path = Path("data/thidong_F.png")
        if not test_img_path.exists():
            return

        pipe_concurrent = CCCDDetectionPipeline(concurrent=True)
        pipe_sequential = CCCDDetectionPipeline(concurrent=False)

        res_con = pipe_concurrent.predict(str(test_img_path), min_conf=0.4)
        res_seq = pipe_sequential.predict(str(test_img_path), min_conf=0.4)

        # Ensure field counts and structure match identically
        self.assertEqual(res_con["total_texts"], res_seq["total_texts"])
        self.assertEqual(res_con["classification"], res_seq["classification"])
        self.assertEqual(len(res_con["detections"]), len(res_seq["detections"]))

        pipe_concurrent.close()
        pipe_sequential.close()

    def test_pipeline_async_predict(self):
        test_img_path = Path("data/thidong_F.png")
        if not test_img_path.exists():
            return

        async def _run_async_test():
            pipe = CCCDDetectionPipeline(concurrent=True)
            res_async = await pipe.predict_async(str(test_img_path), min_conf=0.4)
            pipe.close()
            return res_async

        res_async = asyncio.run(_run_async_test())
        self.assertIsNotNone(res_async)
        self.assertIn("detections", res_async)
        self.assertIn("classification", res_async)
        self.assertIn("latency_ms", res_async)
        self.assertGreater(res_async["total_texts"], 0)


if __name__ == "__main__":
    unittest.main()
