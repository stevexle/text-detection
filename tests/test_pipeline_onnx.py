"""
Unit tests for the Pure ONNX CCCD Detection Pipeline.
Validates end-to-end execution across synchronous, asynchronous, and batched inference.
"""

from pathlib import Path
import numpy as np
import pytest

from src.pipeline.cccd_pipeline_onnx import CCCDDetectionPipelineONNX


@pytest.fixture(scope="module")
def onnx_pipeline():
    """Module-level fixture for shared ONNX pipeline instance."""
    dbnet_p = "weights/onnx/dbnet.onnx"
    seg_p = "weights/onnx/yolo26_seg.onnx"
    cls_p = "weights/onnx/yolo26_cls.onnx"

    if not Path(dbnet_p).exists() or not Path(seg_p).exists():
        pytest.skip("ONNX weights not available in weights/onnx/. Run tools/export_onnx.py first.")

    pipeline = CCCDDetectionPipelineONNX(
        dbnet_onnx=dbnet_p,
        yolo_seg_onnx=seg_p,
        yolo_cls_onnx=cls_p if Path(cls_p).exists() else None,
        concurrent=True,
    )
    yield pipeline
    pipeline.close()


class TestCCCDPipelineONNXExecution:
    def test_predict_single_dummy_image(self, onnx_pipeline):
        """Test single image prediction with synchronous API."""
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        res = onnx_pipeline.predict(dummy_img, min_conf=0.25)

        assert isinstance(res, dict)
        assert "classification" in res
        assert "total_texts" in res
        assert "detections" in res
        assert "latency_ms" in res
        assert isinstance(res["detections"], list)
        assert res["latency_ms"] > 0

    @pytest.mark.anyio
    async def test_predict_async_api(self, onnx_pipeline):
        """Test non-blocking asynchronous prediction API."""
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        res = await onnx_pipeline.predict_async(dummy_img, min_conf=0.25)

        assert isinstance(res, dict)
        assert "classification" in res
        assert "total_texts" in res
        assert "detections" in res

    def test_predict_batch_api(self, onnx_pipeline):
        """Test batch prediction API across multiple images."""
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(2)]
        results = onnx_pipeline.predict_batch(images, batch_size=2, min_conf=0.25)

        assert len(results) == 2
        for res in results:
            assert isinstance(res, dict)
            assert "detections" in res
