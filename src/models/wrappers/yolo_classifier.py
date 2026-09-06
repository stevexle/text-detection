"""
YOLO26 Document Classifier Wrapper.
Classifies CCCD/Identity Cards into 4 categories: front_2021, back_2021, front_2024, back_2024.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.builder import MODELS


@dataclass
class ClassificationResult:
    """Clean and compact classification output schema."""
    card_type: str    # "front_2021" | "back_2021" | "front_2024" | "back_2024" | "unknown"
    confidence: float # Probability score in [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@MODELS.register_module(name="YOLOClassifier")
@MODELS.register_module(name="yolo_classifier")
class YOLOClassifier(nn.Module):
    """Adapter wrapper for Ultralytics YOLO classification models."""

    CLASS_NAMES = ["back_2021", "back_2024", "front_2021", "front_2024"]

    def __init__(
        self,
        model_path: str = "weights/yolo/yolo26_cls_best.pt",
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self._model = None

    def _init_model(self):
        """Lazy loader for Ultralytics YOLO classification model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
            target_path = Path(self.model_path)
            if not target_path.exists() and Path(f"weights/yolo/{self.model_path}").exists():
                target_path = Path(f"weights/yolo/{self.model_path}")
            elif not target_path.exists() and Path("weights/yolo/yolo26n-cls.pt").exists():
                target_path = Path("weights/yolo/yolo26n-cls.pt")

            if target_path.exists():
                self._model = YOLO(str(target_path), task="classify")
            else:
                # Fallback to downloading or initializing default
                self._model = YOLO("yolo11n-cls.pt", task="classify")
        except Exception as e:
            print(f"[YOLOClassifier] Notice: Ultralytics initialization deferred: {e}")

    @property
    def model(self):
        self._init_model()
        return self._model

    def classify(
        self,
        image: Union[str, Path, np.ndarray, Any],
        min_conf: float = 0.5,
        imgsz: int = 224,
    ) -> Dict[str, Any]:
        """
        Classify document image into 4 card types.
        Args:
            image: Image file path, numpy BGR array, or PIL image
            min_conf: Minimum confidence threshold
            imgsz: Inference input resolution
        Returns:
            Dict conforming to the clean schema: {"card_type": str, "confidence": float}
        """
        if self.model is None:
            raise RuntimeError("YOLO classification model is not available.")

        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return ClassificationResult(
                card_type="unknown",
                confidence=0.0,
            ).to_dict()

        res = results[0]
        probs = res.probs

        if probs is not None:
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            class_name = str(res.names[top1_idx])
        else:
            return ClassificationResult(
                card_type="unknown",
                confidence=0.0,
            ).to_dict()

        if top1_conf < min_conf:
            card_type = "unknown"
        else:
            card_type = class_name

        result_obj = ClassificationResult(
            card_type=card_type,
            confidence=round(top1_conf, 4),
        )
        return result_obj.to_dict()

    def classify_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Any]],
        batch_size: int = 32,
        min_conf: float = 0.5,
        imgsz: int = 224,
    ) -> List[Dict[str, Any]]:
        """
        Classify a batch of document images with high-throughput vectorization.
        Args:
            images: List of image paths, numpy BGR arrays, or PIL images
            batch_size: Batch size for parallel inference
            min_conf: Minimum confidence threshold
            imgsz: Inference input resolution
        Returns:
            List of Dicts conforming to {"card_type": str, "confidence": float}
        """
        if not images:
            return []
        if self.model is None:
            raise RuntimeError("YOLO classification model is not available.")

        outputs = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            results = self.model.predict(
                source=chunk,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
                batch=len(chunk),
            )
            for res in results:
                probs = res.probs
                if probs is not None:
                    top1_idx = int(probs.top1)
                    top1_conf = float(probs.top1conf)
                    class_name = str(res.names[top1_idx])
                    if top1_conf < min_conf:
                        card_type = "unknown"
                    else:
                        card_type = class_name
                else:
                    card_type = "unknown"
                    top1_conf = 0.0

                outputs.append(
                    ClassificationResult(
                        card_type=card_type,
                        confidence=round(top1_conf, 4),
                    ).to_dict()
                )
        return outputs

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 224,
        half: bool = False,
        **kwargs,
    ) -> str:
        """
        Export trained YOLO classifier to target runtime (ONNX, TensorRT, CoreML, etc.).
        """
        if self.model is None:
            raise RuntimeError("YOLO classification model is not available.")
        return str(
            self.model.export(
                format=format,
                imgsz=imgsz,
                half=half,
                **kwargs,
            )
        )
