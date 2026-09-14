"""
High-Performance End-to-End ONNX CCCD Processing Pipeline.
Features:
- Pure ONNX Runtime Inference for DBNet, YOLO-seg, and YOLO-cls.
- 3-Way Concurrent Multi-threaded Execution (releases Python GIL during native C++ inference).
- AsyncIO Integration (`predict_async`, `predict_batch_async`) for FastAPI and asynchronous event loops.
- Aspect-Ratio Preserving Dynamic Scaling (divisible by 32).
- Hardware Auto-Configuring: CUDAExecutionProvider, CoreMLExecutionProvider, or CPUExecutionProvider.
- Guaranteed Zero-Lost-Field Fallback with Single-Pass Spatial Matching.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from src.pipeline.spatial_matcher import match_text_to_fields
from src.postprocess.builder import build_postprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("CCCDPipelineONNX")


class CCCDDetectionPipelineONNX:
    """
    Production ONNX Runtime Detection Pipeline for Vietnamese Citizen Identity Cards (CCCD).
    Executes Document Classification, Semantic Segmentation, and DBNet Text Detection.
    """

    def __init__(
        self,
        dbnet_onnx: str = "weights/onnx/dbnet.onnx",
        yolo_seg_onnx: str = "weights/onnx/yolo26_seg.onnx",
        yolo_cls_onnx: Optional[str] = "weights/onnx/yolo26_cls.onnx",
        dbnet_config: str = "configs/dbnet/dbnet.yaml",
        dbnet_box_thresh: Optional[float] = None,
        dbnet_unclip_ratio: Optional[float] = None,
        max_side_len: int = 960,
        providers: Optional[List[str]] = None,
        concurrent: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.max_side_len = max_side_len
        self.concurrent = concurrent

        # 1. Resolve ONNX Runtime Execution Providers
        if providers is None:
            available = ort.get_available_providers()
            resolved = []
            if "CUDAExecutionProvider" in available:
                resolved.append("CUDAExecutionProvider")
            if "CoreMLExecutionProvider" in available:
                resolved.append("CoreMLExecutionProvider")
            resolved.append("CPUExecutionProvider")
            self.providers = resolved
        else:
            self.providers = providers

        logger.info(f"Configured ONNX Runtime Execution Providers: {self.providers}")

        # 2. Initialize DBNet ONNX Session
        dbnet_p = Path(dbnet_onnx)
        if not dbnet_p.exists():
            raise FileNotFoundError(f"DBNet ONNX model not found at: {dbnet_p}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.dbnet_session = ort.InferenceSession(str(dbnet_p), sess_options=sess_options, providers=self.providers)
        self.dbnet_input_name = self.dbnet_session.get_inputs()[0].name
        self.dbnet_output_name = self.dbnet_session.get_outputs()[0].name
        logger.info(f"Loaded DBNet ONNX: {dbnet_p.name} (input='{self.dbnet_input_name}', output='{self.dbnet_output_name}')")

        # 3. Initialize YOLO-seg ONNX Model
        yolo_seg_p = Path(yolo_seg_onnx)
        if not yolo_seg_p.exists():
            raise FileNotFoundError(f"YOLO-seg ONNX model not found at: {yolo_seg_p}")
        self.yolo_seg = YOLO(str(yolo_seg_p), task="segment")
        logger.info(f"Loaded YOLO-seg ONNX: {yolo_seg_p.name}")

        # 4. Initialize YOLO-cls ONNX Model (Optional)
        self.yolo_cls = None
        if yolo_cls_onnx:
            cls_p = Path(yolo_cls_onnx)
            if cls_p.exists():
                self.yolo_cls = YOLO(str(cls_p), task="classify")
                logger.info(f"Loaded YOLO-cls ONNX: {cls_p.name}")
            else:
                logger.warning(f"YOLO-cls ONNX model not found at: {cls_p}. Skipping classification.")

        # 5. Initialize DBPostProcessor from config
        self.cfg = Config.fromfile(dbnet_config) if Path(dbnet_config).exists() else None
        post_cfg = self.cfg.postprocess.copy() if self.cfg and hasattr(self.cfg, "postprocess") else {"type": "DBPostProcessor"}

        if dbnet_box_thresh is not None:
            post_cfg["box_thresh"] = dbnet_box_thresh
        elif "box_thresh" not in post_cfg:
            post_cfg["box_thresh"] = 0.6

        if dbnet_unclip_ratio is not None:
            post_cfg["unclip_ratio"] = dbnet_unclip_ratio
        elif "unclip_ratio" not in post_cfg:
            post_cfg["unclip_ratio"] = 1.75

        self.postprocessor = build_postprocessor(post_cfg)
        logger.info(
            f"Initialized DBNet PostProcessor: box_thresh={post_cfg.get('box_thresh')}, unclip_ratio={post_cfg.get('unclip_ratio')}"
        )

        # 6. Normalization constants (ImageNet standard)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # 7. Worker auto-sizing
        try:
            available_cores = len(os.sched_getaffinity(0))
        except (AttributeError, NotImplementedError):
            available_cores = os.cpu_count() or 1

        if max_workers is None:
            self.num_workers = min(3, max(1, available_cores))
        else:
            self.num_workers = min(3, max(1, max_workers))

        logger.info(f"Initialized ONNX Pipeline with {self.num_workers} parallel workers (Concurrent={self.concurrent}).")
        self.executor = (
            ThreadPoolExecutor(max_workers=self.num_workers, thread_name_prefix="onnx_worker") if self.concurrent else None
        )

    def close(self):
        """Shutdown thread pool executor gracefully."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def warmup(self, num_runs: int = 2):
        """Warm up ONNX Runtime execution graphs to eliminate first-request cold-start latency."""
        logger.info("Warming up ONNX pipeline models...")
        dummy_img = np.zeros((448, 960, 3), dtype=np.uint8)
        for _ in range(num_runs):
            _ = self.predict(dummy_img, min_conf=0.25)
        logger.info("ONNX Pipeline warmup complete.")

    def _get_dynamic_dims(self, h: int, w: int) -> Tuple[int, int]:
        """Compute aspect-ratio preserved dimensions divisible by 32."""
        scale = self.max_side_len / max(h, w)
        dyn_w = max(32, int(round(w * scale / 32) * 32))
        dyn_h = max(32, int(round(h * scale / 32) * 32))
        return dyn_h, dyn_w

    def _preprocess_single(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Preprocess single image for DBNet ONNX."""
        h, w = img_bgr.shape[:2]
        dyn_h, dyn_w = self._get_dynamic_dims(h, w)

        img_resized = cv2.resize(img_bgr, (dyn_w, dyn_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_img = (img_rgb - self.mean) / self.std

        # Transpose HWC -> CHW and add batch dim (1, 3, dyn_h, dyn_w)
        tensor_np = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return tensor_np, (dyn_h, dyn_w)

    def _preprocess_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Vectorized batch preprocessing with uniform max dimensions."""
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        dyn_h, dyn_w = self._get_dynamic_dims(max_h, max_w)

        batch_arrays = []
        orig_shapes = []
        for img_bgr in images:
            orig_shapes.append(img_bgr.shape[:2])
            img_resized = cv2.resize(img_bgr, (dyn_w, dyn_h), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            norm_img = (img_rgb - self.mean) / self.std
            batch_arrays.append(np.transpose(norm_img, (2, 0, 1)))

        batch_tensor = np.stack(batch_arrays, axis=0).astype(np.float32)
        return batch_tensor, orig_shapes

    # =========================================================================
    # Step Execution Helpers
    # =========================================================================
    def _run_classification(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Step 1: Document classification via YOLO-cls ONNX."""
        if self.yolo_cls is None:
            return {"card_type": "unknown", "confidence": 0.0}

        res = self.yolo_cls(img_bgr, verbose=False)[0]
        top1_idx = int(res.probs.top1)
        card_type = str(res.names[top1_idx])
        conf = float(res.probs.top1conf.item())
        return {"card_type": card_type, "confidence": round(conf, 4)}

    def _run_field_segmentation(self, img_bgr: np.ndarray, min_conf: float) -> List[Dict[str, Any]]:
        """Step 2: Semantic field segmentation via YOLO-seg ONNX."""
        res = self.yolo_seg(img_bgr, conf=min_conf, verbose=False)[0]
        fields = []

        if res.masks is not None:
            for poly_pts, cls_idx, conf in zip(res.masks.xy, res.boxes.cls, res.boxes.conf):
                c_idx = int(cls_idx.item())
                label = str(res.names.get(c_idx, f"field_{c_idx}"))
                fields.append(
                    {
                        "label": label,
                        "confidence": round(float(conf.item()), 4),
                        "polygon": poly_pts.tolist(),
                    }
                )
        return fields

    def _run_dbnet(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Step 3: Text character detection via DBNet ONNX."""
        orig_h, orig_w = img_bgr.shape[:2]
        tensor_np, _ = self._preprocess_single(img_bgr)

        # Forward pass on ONNX Runtime
        prob_map = self.dbnet_session.run([self.dbnet_output_name], {self.dbnet_input_name: tensor_np})[0]

        boxes = self.postprocessor(prob_map, orig_shape=(orig_h, orig_w))
        return boxes if isinstance(boxes, list) else []

    # =========================================================================
    # Synchronous API
    # =========================================================================
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
        fallback_unmatched: bool = True,
    ) -> Dict[str, Any]:
        """
        Run end-to-end detection on a single CCCD card using ONNX Runtime.
        """
        t0 = time.perf_counter()

        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not load image from: {image}")
        elif isinstance(image, np.ndarray):
            img_bgr = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Parallel 3-way concurrent execution
        if self.concurrent and self.executor is not None:
            f_cls = self.executor.submit(self._run_classification, img_bgr)
            f_seg = self.executor.submit(self._run_field_segmentation, img_bgr, min_conf)
            f_db = self.executor.submit(self._run_dbnet, img_bgr)

            cls_result = f_cls.result()
            yolo_fields = f_seg.result()
            dbnet_boxes = f_db.result()
        else:
            cls_result = self._run_classification(img_bgr)
            yolo_fields = self._run_field_segmentation(img_bgr, min_conf)
            dbnet_boxes = self._run_dbnet(img_bgr)

        # Step 4: Fast Spatial Matcher
        fused_detections = match_text_to_fields(
            text_detections=dbnet_boxes,
            field_detections=yolo_fields,
            min_overlap_ratio=min_overlap,
            fallback_unmatched_fields=fallback_unmatched,
        )

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "classification": cls_result,
            "total_texts": len(fused_detections),
            "detections": fused_detections,
            "latency_ms": latency_ms,
        }

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        batch_size: int = 8,
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        High-throughput batched inference over multiple images.
        """
        results = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            for img in chunk:
                results.append(self.predict(img, min_conf=min_conf, min_overlap=min_overlap))
        return results

    # =========================================================================
    # Asynchronous API
    # =========================================================================
    async def predict_async(
        self,
        image: Union[str, Path, np.ndarray],
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Asynchronously execute prediction without blocking the main event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, image, min_conf, min_overlap)

    async def predict_batch_async(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        batch_size: int = 8,
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously execute batched predictions concurrently across an event loop.
        """
        tasks = [self.predict_async(img, min_conf=min_conf, min_overlap=min_overlap) for img in images]
        return await asyncio.gather(*tasks)
