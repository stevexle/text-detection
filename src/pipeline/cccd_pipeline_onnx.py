"""
High-Performance End-to-End ONNX CCCD Processing Pipeline.

Architecture Overview:
  Input Images
       │
       ├─► YOLO-cls ONNX ──────────────────────────────────────────┐ (Classification)
       ├─► YOLO-seg ONNX ──────────────────────────────────────────┼─► [Spatial Matcher] ──► Unified Output
       └─► Vectorized Preprocessing ──► DBNet ONNX ──► DBPostProc ─┘ (Text Polygons)
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
    Executes Document Classification, Semantic Field Segmentation, and DBNet Text Detection.
    """

    # =========================================================================
    # Section 1: Class Setup & Resource Lifecycle
    # =========================================================================
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

        # 1. Resolve Hardware Execution Providers
        self.providers = self._resolve_providers(providers)
        logger.info(f"Configured ONNX Runtime Execution Providers: {self.providers}")

        # 2. Initialize DBNet ONNX Session
        self.dbnet_session, self.dbnet_input_name, self.dbnet_output_name = self._init_dbnet_session(dbnet_onnx)

        # 3. Initialize YOLO-seg ONNX Model
        seg_path = Path(yolo_seg_onnx)
        if not seg_path.exists():
            raise FileNotFoundError(f"YOLO-seg ONNX model not found at: {seg_path}")
        self.yolo_seg = YOLO(str(seg_path), task="segment")
        logger.info(f"Loaded YOLO-seg ONNX: {seg_path.name}")

        # 4. Initialize YOLO-cls ONNX Model (Optional)
        self.yolo_cls = None
        if yolo_cls_onnx and Path(yolo_cls_onnx).exists():
            self.yolo_cls = YOLO(str(yolo_cls_onnx), task="classify")
            logger.info(f"Loaded YOLO-cls ONNX: {Path(yolo_cls_onnx).name}")
        elif yolo_cls_onnx:
            logger.warning(f"YOLO-cls ONNX model not found at: {yolo_cls_onnx}. Skipping classification.")

        # 5. Initialize DBPostProcessor
        self.postprocessor = self._init_postprocessor(dbnet_config, dbnet_box_thresh, dbnet_unclip_ratio)

        # 6. Precomputed fast scale and bias for single-pass fused normalization (ImageNet standard)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.scale = (1.0 / (255.0 * self.std)).astype(np.float32)
        self.bias = (-self.mean / self.std).astype(np.float32)

        # 7. Concurrent Worker Thread Pool
        self.executor = self._init_executor(concurrent, max_workers)

    def _resolve_providers(self, requested: Optional[List[str]]) -> List[str]:
        """Detect and return highest priority available execution providers."""
        if requested is not None:
            return requested
        available = ort.get_available_providers()
        resolved = []
        if "CUDAExecutionProvider" in available:
            resolved.append("CUDAExecutionProvider")
        if "CoreMLExecutionProvider" in available:
            resolved.append("CoreMLExecutionProvider")
        resolved.append("CPUExecutionProvider")
        return resolved

    def _init_dbnet_session(self, dbnet_onnx: str) -> Tuple[ort.InferenceSession, str, str]:
        """Configure and instantiate ONNX Runtime session for DBNet."""
        dbnet_p = Path(dbnet_onnx)
        if not dbnet_p.exists():
            raise FileNotFoundError(f"DBNet ONNX model not found at: {dbnet_p}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = True
        sess_options.intra_op_num_threads = min(4, max(1, (os.cpu_count() or 4) // 2))
        sess_options.inter_op_num_threads = 1

        session = ort.InferenceSession(str(dbnet_p), sess_options=sess_options, providers=self.providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        logger.info(f"Loaded DBNet ONNX: {dbnet_p.name} (input='{input_name}', output='{output_name}')")
        return session, input_name, output_name

    def _init_postprocessor(
        self,
        config_path: str,
        box_thresh: Optional[float],
        unclip_ratio: Optional[float],
    ):
        """Construct DBNet contour extractor and polygon postprocessor."""
        cfg = Config.fromfile(config_path) if Path(config_path).exists() else None
        post_cfg = cfg.postprocess.copy() if cfg and hasattr(cfg, "postprocess") else {"type": "DBPostProcessor"}

        if box_thresh is not None:
            post_cfg["box_thresh"] = box_thresh
        elif "box_thresh" not in post_cfg:
            post_cfg["box_thresh"] = 0.6

        if unclip_ratio is not None:
            post_cfg["unclip_ratio"] = unclip_ratio
        elif "unclip_ratio" not in post_cfg:
            post_cfg["unclip_ratio"] = 1.75

        postprocessor = build_postprocessor(post_cfg)
        logger.info(f"Initialized DBNet PostProcessor: box_thresh={post_cfg.get('box_thresh')}, unclip_ratio={post_cfg.get('unclip_ratio')}")
        return postprocessor

    def _init_executor(self, concurrent: bool, max_workers: Optional[int]) -> Optional[ThreadPoolExecutor]:
        """Configure multi-threaded worker pool for concurrent model inference."""
        if not concurrent:
            return None
        try:
            available_cores = len(os.sched_getaffinity(0))
        except (AttributeError, NotImplementedError):
            available_cores = os.cpu_count() or 1

        workers = min(3, max(1, max_workers if max_workers is not None else available_cores))
        logger.info(f"Initialized ONNX Pipeline with {workers} parallel workers.")
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="onnx_worker")

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
        """Warm up ONNX Runtime graphs to eliminate first-request cold-start latency."""
        logger.info("Warming up ONNX pipeline models...")
        dummy_img = np.zeros((448, 960, 3), dtype=np.uint8)
        for _ in range(num_runs):
            _ = self.predict(dummy_img, min_conf=0.25)
        logger.info("ONNX Pipeline warmup complete.")

    # =========================================================================
    # Section 2: Image I/O & Preprocessing
    # =========================================================================
    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load and validate BGR image from filesystem path or numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from: {image}")
            return img
        if isinstance(image, np.ndarray):
            return image
        raise TypeError(f"Unsupported image type: {type(image)}. Expected str, Path, or np.ndarray.")

    def _get_dynamic_dims(self, h: int, w: int) -> Tuple[int, int]:
        """Compute aspect-ratio preserved dimensions divisible by 32."""
        scale = self.max_side_len / max(h, w)
        dyn_w = max(32, int(round(w * scale / 32) * 32))
        dyn_h = max(32, int(round(h * scale / 32) * 32))
        return dyn_h, dyn_w

    def _preprocess_batch(self, images: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Vectorized batch preprocessing for DBNet:
        Resizes to uniform dynamic dimensions and applies fused multiply-add normalization.
        """
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        dyn_h, dyn_w = self._get_dynamic_dims(max_h, max_w)

        batch_arrays = []
        orig_shapes = []
        for img_bgr in images:
            orig_shapes.append(img_bgr.shape[:2])
            img_resized = cv2.resize(img_bgr, (dyn_w, dyn_h), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm_img = img_rgb * self.scale + self.bias
            batch_arrays.append(np.transpose(norm_img, (2, 0, 1)))

        batch_tensor = np.ascontiguousarray(np.stack(batch_arrays, axis=0))
        return batch_tensor, orig_shapes

    # =========================================================================
    # Section 3: Parallel Model Inferences
    # =========================================================================
    def _run_classification(self, images: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        """Run document classification across an image batch via YOLO-cls ONNX."""
        if self.yolo_cls is None:
            return [{"card_type": "unknown", "confidence": 0.0} for _ in images]

        results = self.yolo_cls(list(images), imgsz=224, verbose=False)
        cls_outputs = []
        for r in results:
            top1 = int(r.probs.top1)
            cls_outputs.append({
                "card_type": str(r.names[top1]),
                "confidence": round(float(r.probs.top1conf.item()), 4),
            })
        return cls_outputs

    def _run_field_segmentation(self, images: Sequence[np.ndarray], min_conf: float) -> List[List[Dict[str, Any]]]:
        """Run semantic field segmentation across an image batch via YOLO-seg ONNX."""
        results = self.yolo_seg(
            list(images),
            conf=min_conf,
            max_det=30,
            retina_masks=False,
            verbose=False,
        )
        batch_fields = []
        for r in results:
            fields = []
            if r.masks is not None:
                for poly_pts, cls_idx, conf in zip(r.masks.xy, r.boxes.cls, r.boxes.conf):
                    c_idx = int(cls_idx.item())
                    fields.append({
                        "label": str(r.names.get(c_idx, f"field_{c_idx}")),
                        "confidence": round(float(conf.item()), 4),
                        "polygon": poly_pts.tolist(),
                    })
            batch_fields.append(fields)
        return batch_fields

    def _run_dbnet(self, images: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Run text line detection across an image batch via DBNet ONNX."""
        batch_tensor, orig_shapes = self._preprocess_batch(images)
        prob_maps = self.dbnet_session.run([self.dbnet_output_name], {self.dbnet_input_name: batch_tensor})[0]
        boxes_batch = self.postprocessor(prob_maps, shape_list=orig_shapes)

        if not isinstance(boxes_batch, list):
            return [[]]
        if boxes_batch and isinstance(boxes_batch[0], dict):
            return [boxes_batch]
        return boxes_batch

    # =========================================================================
    # Section 4: Core Pipeline Orchestration
    # =========================================================================
    def _execute_chunk(
        self,
        images: List[np.ndarray],
        min_conf: float,
        min_overlap: float,
        fallback_unmatched: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute concurrent multi-model inference and spatial matching on an image chunk."""
        if not images:
            return []

        # 3-Way Concurrent multi-threading (releases Python GIL during ONNX C++ inference)
        if self.concurrent and self.executor is not None:
            f_cls = self.executor.submit(self._run_classification, images)
            f_seg = self.executor.submit(self._run_field_segmentation, images, min_conf)
            f_db = self.executor.submit(self._run_dbnet, images)

            cls_results = f_cls.result()
            seg_results = f_seg.result()
            db_results = f_db.result()
        else:
            cls_results = self._run_classification(images)
            seg_results = self._run_field_segmentation(images, min_conf)
            db_results = self._run_dbnet(images)

        # Spatial Matching & Field Association
        chunk_outputs = []
        for i in range(len(images)):
            fused = match_text_to_fields(
                text_detections=db_results[i] if i < len(db_results) else [],
                field_detections=seg_results[i] if i < len(seg_results) else [],
                min_overlap_ratio=min_overlap,
                fallback_unmatched_fields=fallback_unmatched,
            )
            chunk_outputs.append({
                "classification": cls_results[i] if i < len(cls_results) else {"card_type": "unknown", "confidence": 0.0},
                "total_texts": len(fused),
                "detections": fused,
                "latency_ms": 0.0,
            })
        return chunk_outputs

    # =========================================================================
    # Section 5: Public Synchronous & Asynchronous APIs
    # =========================================================================
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
        fallback_unmatched: bool = True,
    ) -> Dict[str, Any]:
        """
        Run end-to-end CCCD detection on a single card image.
        Returns dictionary with classification, detected fields, polygons, and latency.
        """
        t0 = time.perf_counter()
        img = self._load_image(image)
        results = self._execute_chunk(
            images=[img],
            min_conf=min_conf,
            min_overlap=min_overlap,
            fallback_unmatched=fallback_unmatched,
        )
        output = results[0]
        output["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return output

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        batch_size: int = 8,
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
        fallback_unmatched: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute high-throughput true batched inference across multiple card images.
        """
        results = []
        for i in range(0, len(images), batch_size):
            chunk_items = images[i : i + batch_size]
            chunk_imgs = [self._load_image(item) for item in chunk_items]

            t0 = time.perf_counter()
            chunk_res = self._execute_chunk(
                images=chunk_imgs,
                min_conf=min_conf,
                min_overlap=min_overlap,
                fallback_unmatched=fallback_unmatched,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_item_ms = round(elapsed_ms / len(chunk_imgs), 2) if chunk_imgs else 0.0

            for r in chunk_res:
                r["latency_ms"] = per_item_ms
            results.extend(chunk_res)

        return results

    async def predict_async(
        self,
        image: Union[str, Path, np.ndarray],
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
        fallback_unmatched: bool = True,
    ) -> Dict[str, Any]:
        """Asynchronously execute prediction without blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, image, min_conf, min_overlap, fallback_unmatched)

    async def predict_batch_async(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        batch_size: int = 8,
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
        fallback_unmatched: bool = True,
    ) -> List[Dict[str, Any]]:
        """Asynchronously execute batched predictions across an event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.predict_batch,
            images,
            batch_size,
            min_conf,
            min_overlap,
            fallback_unmatched,
        )
