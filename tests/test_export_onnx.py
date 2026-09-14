"""
Unit tests for ONNX Export and ONNX Runtime Validation.
Ensures exported models load correctly and support dynamic shapes and batched inputs.
"""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from tools.export_onnx import DBNetExportWrapper, export_dbnet_onnx
from src.models.builder import build_model
from src.utils.config import Config


@pytest.fixture
def onnx_dir(tmp_path):
    d = tmp_path / "onnx_export_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestONNXExport:
    def test_dbnet_export_and_inference(self, onnx_dir):
        """Test DBNet export to ONNX and multi-resolution dynamic batching with ONNX Runtime."""
        config_path = "configs/dbnet/dbnet.yaml"
        output_path = str(onnx_dir / "test_dbnet.onnx")

        exported_path = export_dbnet_onnx(
            config_path=config_path,
            weights_path=None,  # test with initialized weights for speed and isolation
            output_path=output_path,
            imgsz=(256, 256),
            opset=18,
            simplify=True,
            verify=True,
        )

        assert Path(exported_path).exists()
        assert Path(exported_path).stat().st_size > 10 * 1024 * 1024  # > 10MB

        # Verify with onnx checker
        onnx_model = onnx.load(str(exported_path))
        onnx.checker.check_model(onnx_model)

        # Verify dynamic shapes with onnxruntime
        session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        assert inputs[0].name == "input"
        assert outputs[0].name == "prob_map"

        # Check dynamic shape support: Batch 1, 384x384
        x1 = np.random.randn(1, 3, 384, 384).astype(np.float32)
        out1 = session.run(["prob_map"], {"input": x1})[0]
        assert out1.shape == (1, 1, 384, 384)

        # Check dynamic shape support: Batch 3, 512x512
        x2 = np.random.randn(3, 3, 512, 512).astype(np.float32)
        out2 = session.run(["prob_map"], {"input": x2})[0]
        assert out2.shape == (3, 1, 512, 512)

    def test_production_onnx_models_exist_and_run(self):
        """Verify the 3 pre-exported production models in weights/onnx/ if present."""
        onnx_files = {
            "dbnet": Path("weights/onnx/dbnet.onnx"),
            "yolo_seg": Path("weights/onnx/yolo26_seg.onnx"),
            "yolo_cls": Path("weights/onnx/yolo26_cls.onnx"),
        }

        # If models have been exported, verify them with onnxruntime
        for name, p in onnx_files.items():
            if p.exists():
                session = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                input_meta = session.get_inputs()[0]
                assert input_meta.type == "tensor(float)"
                if name == "dbnet":
                    test_tensor = np.random.randn(1, 3, 640, 640).astype(np.float32)
                    res = session.run(None, {input_meta.name: test_tensor})[0]
                    assert res.shape == (1, 1, 640, 640)
                elif name == "yolo_cls":
                    test_tensor = np.random.randn(1, 3, 224, 224).astype(np.float32)
                    res = session.run(None, {input_meta.name: test_tensor})[0]
                    assert res.shape[1] == 4  # 4 classes
