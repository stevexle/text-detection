"""
Unit tests for YOLO Document Classification module.
Tests:
- ClassificationResult dataclass and schema integrity.
- YOLOClassifier._parse_card_type parsing logic.
- Dynamic in-memory Train/Val staging creation.
- Model registry integration.
- Classification inference pipeline and output formatting.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.models.builder import build_model
from src.models.wrappers.yolo_classifier import ClassificationResult, YOLOClassifier
from tools.train_yolo_cls import create_dynamic_cls_staging


class TestClassificationResult(unittest.TestCase):
    """Test ClassificationResult schema and serialization."""

    def test_schema_fields(self):
        res = ClassificationResult(
            card_type="front_2021",
            confidence=0.9982,
        )
        d = res.to_dict()
        self.assertEqual(
            d,
            {
                "card_type": "front_2021",
                "confidence": 0.9982,
            },
        )
        self.assertEqual(len(d), 2)

    def test_unknown_values(self):
        res = ClassificationResult(
            card_type="unknown",
            confidence=0.12,
        )
        d = res.to_dict()
        self.assertEqual(d["card_type"], "unknown")
        self.assertEqual(d["confidence"], 0.12)
        self.assertEqual(len(d), 2)


class TestYOLOClassifier(unittest.TestCase):
    """Test YOLOClassifier registry and inference wrapper."""

    def test_registry_build(self):
        cfg = dict(
            type="YOLOClassifier",
            model_path="weights/yolo/yolo26_cls_best.pt",
        )
        model = build_model(cfg)
        self.assertIsInstance(model, YOLOClassifier)
        self.assertEqual(model.model_path, "weights/yolo/yolo26_cls_best.pt")

    @patch.object(YOLOClassifier, "_init_model")
    def test_classify_with_mocked_model(self, mock_init):
        classifier = YOLOClassifier()

        # Mock ultralytics YOLO result
        mock_probs = MagicMock()
        mock_probs.top1 = 2
        mock_probs.top1conf = 0.9945

        mock_result = MagicMock()
        mock_result.probs = mock_probs
        mock_result.names = {0: "back_2021", 1: "back_2024", 2: "front_2021", 3: "front_2024"}

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.return_value = [mock_result]
        classifier._model = mock_yolo_instance

        # Run inference
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        output = classifier.classify(dummy_img, min_conf=0.5)

        self.assertEqual(output["card_type"], "front_2021")
        self.assertEqual(output["confidence"], 0.9945)
        self.assertEqual(len(output), 2)

    @patch.object(YOLOClassifier, "_init_model")
    def test_classify_low_confidence_fallback(self, mock_init):
        classifier = YOLOClassifier()

        mock_probs = MagicMock()
        mock_probs.top1 = 2
        mock_probs.top1conf = 0.35  # Below threshold

        mock_result = MagicMock()
        mock_result.probs = mock_probs
        mock_result.names = {0: "back_2021", 1: "back_2024", 2: "front_2021", 3: "front_2024"}

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.return_value = [mock_result]
        classifier._model = mock_yolo_instance

        output = classifier.classify("dummy.jpg", min_conf=0.5)

        self.assertEqual(output["card_type"], "unknown")
        self.assertEqual(output["confidence"], 0.35)
        self.assertEqual(len(output), 2)

    @patch.object(YOLOClassifier, "_init_model")
    def test_classify_batch(self, mock_init):
        classifier = YOLOClassifier()

        mock_probs1 = MagicMock()
        mock_probs1.top1 = 0
        mock_probs1.top1conf = 0.98

        mock_probs2 = MagicMock()
        mock_probs2.top1 = 3
        mock_probs2.top1conf = 0.99

        mock_res1 = MagicMock()
        mock_res1.probs = mock_probs1
        mock_res1.names = {0: "back_2021", 1: "back_2024", 2: "front_2021", 3: "front_2024"}

        mock_res2 = MagicMock()
        mock_res2.probs = mock_probs2
        mock_res2.names = {0: "back_2021", 1: "back_2024", 2: "front_2021", 3: "front_2024"}

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.predict.return_value = [mock_res1, mock_res2]
        classifier._model = mock_yolo_instance

        batch_outputs = classifier.classify_batch(["img1.jpg", "img2.jpg"], batch_size=2)
        self.assertEqual(len(batch_outputs), 2)
        self.assertEqual(batch_outputs[0]["card_type"], "back_2021")
        self.assertEqual(batch_outputs[0]["confidence"], 0.98)
        self.assertEqual(batch_outputs[1]["card_type"], "front_2024")
        self.assertEqual(batch_outputs[1]["confidence"], 0.99)

    @patch.object(YOLOClassifier, "_init_model")
    def test_export(self, mock_init):
        classifier = YOLOClassifier()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.export.return_value = "weights/yolo/model.onnx"
        classifier._model = mock_yolo_instance

        out = classifier.export(format="onnx")
        self.assertEqual(out, "weights/yolo/model.onnx")
        mock_yolo_instance.export.assert_called_once_with(format="onnx", imgsz=224, half=False)


class TestDynamicStaging(unittest.TestCase):
    """Test dynamic in-memory staging for YOLO classification."""

    def test_dynamic_cls_staging_creation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            img_dir = tmp_p / "images"
            img_dir.mkdir()

            # Create dummy images
            dummy_images = []
            for i in range(10):
                img_path = img_dir / f"img_{i}.jpg"
                img_path.write_text("fake image content")
                dummy_images.append(img_path)

            # Create dummy jsonl annotations
            anno_file = tmp_p / "dataset.jsonl"
            with open(anno_file, "w", encoding="utf-8") as f:
                for i, img_p in enumerate(dummy_images):
                    std = "CCCD_2024" if i % 2 == 0 else "CCCD_2021_CHIP"
                    side = "front" if i < 5 else "back"
                    line_data = {
                        "img_path": str(img_p),
                        "card_standard": std,
                        "card_side": side,
                    }
                    f.write(json.dumps(line_data) + "\n")

            # Run dynamic staging
            stage_dir = tmp_p / "staging"
            res_dir = create_dynamic_cls_staging(
                anno_file=str(anno_file),
                val_ratio=0.2,
                split_seed=42,
                staging_dir=str(stage_dir),
            )

            self.assertTrue(stage_dir.exists())
            self.assertTrue((stage_dir / "train").exists())
            self.assertTrue((stage_dir / "val").exists())

            # Count train and val symlinks
            train_links = list((stage_dir / "train").rglob("*.*"))
            val_links = list((stage_dir / "val").rglob("*.*"))

            self.assertEqual(len(train_links), 8)
            self.assertEqual(len(val_links), 2)

            # Check that files are symlinks pointing to originals
            first_link = val_links[0]
            self.assertTrue(first_link.is_symlink())
            self.assertTrue(first_link.resolve().exists())


if __name__ == "__main__":
    unittest.main()
