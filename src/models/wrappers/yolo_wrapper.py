"""
YOLO26 Adapter Wrapper for Ultralytics Detection and Segmentation Models.
Provides a unified interface conforming to the project architecture.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch.nn as nn
from src.models.builder import MODELS


@MODELS.register_module(name="YOLOWrapper")
@MODELS.register_module(name="yolo_wrapper")
class YOLOWrapper(nn.Module):
    """Adapter Wrapper for Ultralytics YOLO26 detection and segmentation."""

    def __init__(
        self,
        model_path: str = "weights/yolo/yolo26n.pt",
        task: str = "detect",
        **kwargs,
    ):
        super().__init__()
        self.model_path = model_path
        self.task = task
        self._model = None

    def _init_model(self):
        """Lazy initialization of Ultralytics YOLO model."""
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
            target_path = Path(self.model_path)
            if not target_path.exists() and Path(f"weights/yolo/{self.model_path}").exists():
                target_path = Path(f"weights/yolo/{self.model_path}")
            if target_path.exists():
                self._model = YOLO(str(target_path), task=self.task)
            elif Path("weights/yolo/yolo26n.pt").exists():
                self._model = YOLO("weights/yolo/yolo26n.pt", task=self.task)
        except Exception as e:
            print(f"[YOLOWrapper] Notice: 'ultralytics' initialization deferred: {e}")

    @property
    def model(self):
        if self._model is None:
            self._init_model()
        return self._model

    def predict(self, source: Any, **kwargs) -> Any:
        """Run YOLO inference on source image(s)."""
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not initialized.")
        return self.model.predict(source, **kwargs)

    def _extract_detections(self, res: Any, min_conf: float = 0.5) -> List[Dict[str, Any]]:
        """Extract detections from an Ultralytics Results object."""
        if res is None:
            return []

        detections = []
        names = res.names

        # Case 1: Segmentation Masks available
        if res.masks is not None and len(res.masks) > 0:
            for mask_xy, box in zip(res.masks.xy, res.boxes):
                conf = float(box.conf)
                if conf < min_conf:
                    continue
                cls_id = int(box.cls)
                label = names.get(cls_id, str(cls_id))

                if len(mask_xy) >= 4:
                    poly_list = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in mask_xy]
                else:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    poly_list = [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)], [round(x1, 2), round(y2, 2)]]

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "polygon": poly_list,
                })
        # Case 2: Bounding Boxes only
        elif res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                conf = float(box.conf)
                if conf < min_conf:
                    continue
                cls_id = int(box.cls)
                label = names.get(cls_id, str(cls_id))
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                poly_list = [
                    [round(x1, 2), round(y1, 2)],
                    [round(x2, 2), round(y1, 2)],
                    [round(x2, 2), round(y2, 2)],
                    [round(x1, 2), round(y2, 2)],
                ]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "polygon": poly_list,
                })

        return detections

    def detect(
        self,
        image: Union[str, Path, Any],
        min_conf: float = 0.5,
        imgsz: int = 640,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO detection / segmentation on a single image.
        """
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not initialized.")

        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=min_conf,
            verbose=False,
            **kwargs,
        )
        if not results or len(results) == 0:
            return []

        return self._extract_detections(results[0], min_conf=min_conf)

    def detect_batch(
        self,
        images: List[Union[str, Path, Any]],
        min_conf: float = 0.5,
        imgsz: int = 640,
        batch_size: int = 8,
        **kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run high-throughput batched YOLO detection / segmentation.
        """
        if not images:
            return []
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not initialized.")

        all_detections = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            results = self.model.predict(
                source=chunk,
                imgsz=imgsz,
                conf=min_conf,
                batch=len(chunk),
                verbose=False,
                **kwargs,
            )
            for res in results:
                all_detections.append(self._extract_detections(res, min_conf=min_conf))

        return all_detections

    def train(self, data: str, epochs: int = 100, **kwargs) -> Any:
        """Train YOLO model on dataset."""
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not initialized.")
        return self.model.train(data=data, epochs=epochs, **kwargs)

    def val(self, data: Optional[str] = None, **kwargs) -> Any:
        """Validate YOLO model."""
        if self.model is None:
            raise RuntimeError("Ultralytics YOLO is not initialized.")
        return self.model.val(data=data, **kwargs)

    def forward(self, *args, **kwargs):
        """Pass-through forward to underlying YOLO network."""
        if self.model is not None and hasattr(self.model, "model"):
            return self.model.model(*args, **kwargs)
        raise NotImplementedError("Direct PyTorch forward pass requires initialized model.model.")
