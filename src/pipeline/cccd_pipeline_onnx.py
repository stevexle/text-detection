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
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Optimize worker thread allocation to prevent thrashing
        cpu_cores = os.cpu_count() or 4
        sess_options.intra_op_num_threads = min(4, max(1, cpu_cores // 2))
        sess_options.inter_op_num_threads = 1
        sess_options.enable_cpu_mem_arena = True

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

        # 6. Precomputed fast scale and bias for single-pass fused normalization (ImageNet standard)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.scale = (1.0 / (255.0 * self.std)).astype(np.float32)
        self.bias = (-self.mean / self.std).astype(np.float32)

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
        """Preprocess single image for DBNet ONNX using fused multiply-add normalization."""
        h, w = img_bgr.shape[:2]
        dyn_h, dyn_w = self._get_dynamic_dims(h, w)

        img_resized = cv2.resize(img_bgr, (dyn_w, dyn_h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm_img = img_rgb * self.scale + self.bias

        # Transpose HWC -> CHW and add batch dim (1, 3, dyn_h, dyn_w) in contiguous memory
        tensor_np = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...])
        return tensor_np, (dyn_h, dyn_w)

    def _preprocess_batch(self, images: Sequence[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Vectorized batch preprocessing with uniform max dimensions and fused normalization."""
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
    # Step Execution Helpers
    # =========================================================================
    def _run_classification(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Step 1: Document classification via YOLO-cls ONNX."""
        if self.yolo_cls is None:
            return {"card_type": "unknown", "confidence": 0.0}

        res = self.yolo_cls(img_bgr, imgsz=224, verbose=False)[0]
        top1_idx = int(res.probs.top1)
        card_type = str(res.names[top1_idx])
        conf = float(res.probs.top1conf.item())
        return {"card_type": card_type, "confidence": round(conf, 4)}

    def _run_field_segmentation(self, img_bgr: np.ndarray, min_conf: float) -> List[Dict[str, Any]]:
        """Step 2: Semantic field segmentation via YOLO-seg ONNX with optimized NMS."""
        res = self.yolo_seg(
            img_bgr,
            conf=min_conf,
            max_det=30,
            retina_masks=False,
            verbose=False,
        )[0]
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

    def _predict_chunk(
        self,
        chunk_images: List[np.ndarray],
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Execute true batched inference on a chunk of images."""
        if not chunk_images:
            return []

        def _batch_cls():
            if self.yolo_cls is None:
                return [{"card_type": "unknown", "confidence": 0.0}] * len(chunk_images)
            res_list = self.yolo_cls(chunk_images, imgsz=224, verbose=False)
            res_cls = []
            for r in res_list:
                top1 = int(r.probs.top1)
                res_cls.append({
                    "card_type": str(r.names[top1]),
                    "confidence": round(float(r.probs.top1conf.item()), 4),
                })
            return res_cls

        def _batch_seg():
            res_list = self.yolo_seg(
                chunk_images,
                conf=min_conf,
                max_det=30,
                retina_masks=False,
                verbose=False,
            )
            res_fields = []
            for res in res_list:
                fields = []
                if res.masks is not None:
                    for poly_pts, cls_idx, conf in zip(res.masks.xy, res.boxes.cls, res.boxes.conf):
                        c_idx = int(cls_idx.item())
                        label = str(res.names.get(c_idx, f"field_{c_idx}"))
                        fields.append({
                            "label": label,
                            "confidence": round(float(conf.item()), 4),
                            "polygon": poly_pts.tolist(),
                        })
                res_fields.append(fields)
            return res_fields

        def _batch_dbnet():
            batch_tensor, orig_shapes = self._preprocess_batch(chunk_images)
            prob_maps = self.dbnet_session.run(
                [self.dbnet_output_name],
                {self.dbnet_input_name: batch_tensor},
            )[0]
            boxes_batch = self.postprocessor(prob_maps, shape_list=orig_shapes)
            if not isinstance(boxes_batch, list):
                boxes_batch = [[]]
            elif boxes_batch and isinstance(boxes_batch[0], dict):
                boxes_batch = [boxes_batch]
            return boxes_batch

        if self.concurrent and self.executor is not None:
            f_cls = self.executor.submit(_batch_cls)
            f_seg = self.executor.submit(_batch_seg)
            f_db = self.executor.submit(_batch_dbnet)

            cls_results = f_cls.result()
            seg_results = f_seg.result()
            db_results = f_db.result()
        else:
            cls_results = _batch_cls()
            seg_results = _batch_seg()
            db_results = _batch_dbnet()

        chunk_out = []
        for i in range(len(chunk_images)):
            fused = match_text_to_fields(
                text_detections=db_results[i] if i < len(db_results) else [],
                field_detections=seg_results[i] if i < len(seg_results) else [],
                min_overlap_ratio=min_overlap,
                fallback_unmatched_fields=True,
            )
            chunk_out.append({
                "classification": cls_results[i] if i < len(cls_results) else {"card_type": "unknown", "confidence": 0.0},
                "total_texts": len(fused),
                "detections": fused,
                "latency_ms": 0.0,
            })
        return chunk_out

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        batch_size: int = 8,
        min_conf: float = 0.25,
        min_overlap: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """
        High-throughput true batched inference over multiple images.
        """
        results = []
        for i in range(0, len(images), batch_size):
            chunk_raw = images[i : i + batch_size]
            chunk_imgs = []
            for item in chunk_raw:
                if isinstance(item, (str, Path)):
                    im = cv2.imread(str(item))
                    if im is None:
                        raise ValueError(f"Could not load image: {item}")
                    chunk_imgs.append(im)
                elif isinstance(item, np.ndarray):
                    chunk_imgs.append(item)
                else:
                    raise TypeError(f"Unsupported image type: {type(item)}")

            t0 = time.perf_counter()
            chunk_res = self._predict_chunk(chunk_imgs, min_conf=min_conf, min_overlap=min_overlap)
            chunk_elapsed = (time.perf_counter() - t0) * 1000.0
            per_item_ms = round(chunk_elapsed / len(chunk_imgs), 2) if chunk_imgs else 0.0
            for r in chunk_res:
                r["latency_ms"] = per_item_ms
            results.extend(chunk_res)
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
