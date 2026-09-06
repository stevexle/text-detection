"""
End-to-End Hybrid CCCD Processing Pipeline.
Combines:
1. YOLO Document Classification (front/back 2021/2024).
2. YOLO Field Segmentation (11 semantic CCCD fields).
3. DBNet Text Detection (high-precision character boundary unclipping).
4. Spatial Fusion (maps field labels to DBNet polygon boxes).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import torch

from src.models.builder import build_model
from src.models.wrappers.yolo_classifier import YOLOClassifier
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.pipeline.spatial_matcher import match_text_to_fields
from src.postprocess.builder import build_postprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("CCCDPipeline")


class CCCDDetectionPipeline:
    """
    Unified eKYC Detection Pipeline for Vietnamese Citizen Identity Cards.
    """

    def __init__(
        self,
        dbnet_config: str = "configs/dbnet/dbnet.yaml",
        dbnet_weights: str = "weights/dbnet/dbnet_cccd_best.pth",
        yolo_seg_weights: str = "weights/yolo/yolo26_seg_best.pt",
        yolo_cls_weights: Optional[str] = "weights/yolo/yolo26_cls_best.pt",
        device: str = "",
    ):
        # 1. Device resolution
        if device:
            self.device = torch.device(device)
            self.device_str = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_str = "cuda"
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_str = "mps"
        else:
            self.device = torch.device("cpu")
            self.device_str = "cpu"

        # 2. Build DBNet model
        logger.info(f"Loading DBNet model from {dbnet_weights} on {self.device}...")
        self.db_cfg = Config.fromfile(dbnet_config)
        self.db_model = build_model(self.db_cfg.model).to(self.device)

        w_path = Path(dbnet_weights)
        if not w_path.exists() and Path(f"weights/dbnet/{dbnet_weights}").exists():
            w_path = Path(f"weights/dbnet/{dbnet_weights}")

        if w_path.exists():
            try:
                ckpt = torch.load(w_path, map_location=self.device, weights_only=False)
            except Exception:
                ckpt = torch.load(w_path, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.db_model.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                self.db_model.load_state_dict(ckpt)
        else:
            logger.warning(f"DBNet weights not found at '{dbnet_weights}'.")

        self.db_model.eval()
        self.postprocessor = build_postprocessor(self.db_cfg.postprocess)

        # 3. Build YOLO Segmentation model
        logger.info(f"Loading YOLO-seg model from {yolo_seg_weights}...")
        self.yolo_seg = YOLOWrapper(model_path=yolo_seg_weights, task="segment")

        # 4. Build YOLO Classification model (optional)
        self.yolo_cls = None
        if yolo_cls_weights and Path(yolo_cls_weights).exists():
            logger.info(f"Loading YOLO-cls model from {yolo_cls_weights}...")
            self.yolo_cls = YOLOClassifier(model_path=yolo_cls_weights, device=self.device_str)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.target_size = tuple(self.db_cfg.data.target_size)

    def _run_dbnet(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Run DBNet detection on a single BGR image."""
        orig_h, orig_w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.target_size)
        norm_img = ((img_resized / 255.0) - self.mean) / self.std
        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            preds = self.db_model(tensor_img)
            prob_map = preds["prob_map"][0]
            detections = self.postprocessor(prob_map, orig_shape=(orig_h, orig_w))

        return detections

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        min_conf: float = 0.4,
        min_overlap: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Process single image through full eKYC hybrid pipeline:
        Classification -> Field Segmentation -> DBNet Text Detection -> Spatial Fusion.
        """
        if isinstance(image, (str, Path)):
            img_path = Path(image)
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise ValueError(f"Could not load image from: {image}")
            img_name = img_path.name
        else:
            img_bgr = image
            img_name = "in_memory_image.jpg"

        # Step 1: Document Classification
        card_classification = None
        if self.yolo_cls is not None:
            cls_res = self.yolo_cls.classify(img_bgr)
            card_classification = cls_res if isinstance(cls_res, dict) else cls_res.to_dict()

        # Step 2: YOLO Field Segmentation
        yolo_fields = self.yolo_seg.detect(
            image=img_bgr,
            min_conf=min_conf,
            device=self.device_str if self.device_str != "mps" else None,
        )

        # Step 3: DBNet Text Detection
        dbnet_texts = self._run_dbnet(img_bgr)

        # Step 4: Spatial Fusion
        fused_texts = match_text_to_fields(
            text_detections=dbnet_texts,
            field_detections=yolo_fields,
            min_overlap_ratio=min_overlap,
        )

        return {
            "image": img_name,
            "classification": card_classification,
            "total_texts": len(fused_texts),
            "detections": fused_texts,
            "raw_yolo_fields": yolo_fields,
        }
