"""
High-Performance End-to-End NVIDIA TensorRT CCCD Processing Pipeline.

Architecture Overview:
  Input Images
       │
       ├─► YOLO-cls Engine ────────────────────────────────────────┐ (Classification)
       ├─► YOLO-seg Engine ────────────────────────────────────────┼─► [Spatial Matcher] ──► Unified Output
       └─► Vectorized Preprocessing ──► DBNet Engine ──► DBPostProc ┘ (Text Polygons)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Auto-detect and register NVIDIA CUDA, cuDNN, and TensorRT shared libraries on Linux
if sys.platform == "linux":
    import ctypes
    import site
    try:
        for site_pkg in site.getsitepackages():
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "tensorrt"]:
                    lib_dir = os.path.join(nvidia_dir, sub, "lib")
                    if os.path.isdir(lib_dir):
                        if "LD_LIBRARY_PATH" in os.environ:
                            if lib_dir not in os.environ["LD_LIBRARY_PATH"]:
                                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
                        else:
                            os.environ["LD_LIBRARY_PATH"] = lib_dir
                        for f in sorted(os.listdir(lib_dir)):
                            if f.endswith(".so") or ".so." in f:
                                try:
                                    ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                                except Exception:
                                    pass
            # Also check direct tensorrt package directory
            trt_dir = os.path.join(site_pkg, "tensorrt")
            if os.path.isdir(trt_dir):
                for f in sorted(os.listdir(trt_dir)):
                    if f.endswith(".so") or ".so." in f:
                        try:
                            ctypes.CDLL(os.path.join(trt_dir, f), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception:
        pass

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.pipeline.spatial_matcher import match_text_to_fields
from src.postprocess.builder import build_postprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("CCCDPipelineTRT")


class DBNetTensorRTRunner:
    """
    Direct NVIDIA TensorRT Execution Runner for DBNet Text Detection.
    Executes compiled .engine files via TensorRT C++ bindings and PyTorch GPU streams.
    Falls back to ONNX Runtime with TensorrtExecutionProvider if engine is an ONNX file.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"DBNet model not found at: {self.model_path}")

        self.is_engine = self.model_path.suffix == ".engine"
        self.session = None
        self.engine = None
        self.context = None
        self.trt = None
        self.stream = None

        if self.is_engine:
            self._init_tensorrt_engine()
        else:
            self._init_onnx_trt_session()

    def _init_tensorrt_engine(self):
        """Initialize native TensorRT runtime engine."""
        try:
            import tensorrt as trt
            self.trt = trt
        except ImportError as e:
            raise RuntimeError(
                "tensorrt package is required to load .engine files. "
                "Install with: uv pip install tensorrt tensorrt-cu12"
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA is required for TensorRT inference, but torch.cuda is unavailable.")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        logger.info(f"Loading native TensorRT engine from: {self.model_path}")
        with open(self.model_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.model_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        logger.info(f"Successfully instantiated DBNet native TensorRT engine ({self.model_path.name})")

    def _init_onnx_trt_session(self):
        """Initialize ONNX Runtime session with TensorrtExecutionProvider."""
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        cache_dir = Path("weights/tensorrt")
        cache_dir.mkdir(parents=True, exist_ok=True)

        trt_options = {
            "device_id": 0,
            "trt_max_workspace_size": 2147483648,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
        }

        available = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in available:
            providers.append(("TensorrtExecutionProvider", trt_options))
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        logger.info(f"Loading DBNet ONNX via ORT providers: {providers}")
        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Execute DBNet inference on input tensor (B, 3, H, W).
        Returns probability map numpy array (B, 1, H, W).
        """
        if not self.is_engine:
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            return outputs[0]

        # Native TensorRT execution with PyTorch CUDA tensors
        b, c, h, w = input_tensor.shape
        gpu_input = torch.from_numpy(input_tensor).to("cuda", non_blocking=True)
        gpu_output = torch.empty((b, 1, h, w), dtype=torch.float32, device="cuda")

        # Set dynamic shape and tensor bindings for TensorRT 8.5+ / 10+
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape("input", (b, c, h, w))
            self.context.set_tensor_address("input", gpu_input.data_ptr())
            # Output binding can be 'prob_map' or 'output'
            output_name = "prob_map" if "prob_map" in self.engine else "output"
            self.context.set_tensor_address(output_name, gpu_output.data_ptr())

            with torch.cuda.stream(self.stream):
                self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
        else:
            # Fallback for older TensorRT 8.0-8.4 execute_async_v2 API
            bindings = [int(gpu_input.data_ptr()), int(gpu_output.data_ptr())]
            self.context.set_binding_shape(0, (b, c, h, w))
            with torch.cuda.stream(self.stream):
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()

        return gpu_output.cpu().numpy()


class CCCDDetectionPipelineTRT:
    """
    Production NVIDIA TensorRT Detection Pipeline for Vietnamese Citizen Identity Cards (CCCD).
    Executes Document Classification, Semantic Field Segmentation, and DBNet Text Detection
    at sub-20ms latency.
    """

    def __init__(
        self,
        dbnet_engine: str = "weights/tensorrt/dbnet.engine",
        yolo_seg_engine: str = "weights/tensorrt/yolo26_seg.engine",
        yolo_cls_engine: Optional[str] = "weights/tensorrt/yolo26_cls.engine",
        dbnet_config: str = "configs/dbnet/dbnet.yaml",
        dbnet_box_thresh: Optional[float] = None,
        dbnet_unclip_ratio: Optional[float] = None,
        max_side_len: int = 960,
        concurrent: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.max_side_len = max_side_len
        self.concurrent = concurrent

        # 1. Resolve model file paths (with auto-fallback to .onnx if .engine is not yet compiled)
        dbnet_path = self._resolve_model_path(dbnet_engine, "weights/onnx/dbnet.onnx")
        seg_path = self._resolve_model_path(
            yolo_seg_engine,
            "weights/yolo/yolo26_seg_best.engine",
            "weights/onnx/yolo26_seg.onnx",
        )
        cls_path = None
        if yolo_cls_engine:
            cls_path = self._resolve_model_path(
                yolo_cls_engine,
                "weights/yolo/yolo26_cls_best.engine",
                "weights/onnx/yolo26_cls.onnx",
                optional=True,
            )

        # 2. Initialize DBNet Runner
        self.dbnet_runner = DBNetTensorRTRunner(dbnet_path)

        # 3. Initialize YOLO-seg Model
        if not seg_path or not Path(seg_path).exists():
            raise FileNotFoundError(f"YOLO-seg model not found: {seg_path}")
        self.yolo_seg = YOLO(str(seg_path), task="segment")
        logger.info(f"Loaded YOLO-seg: {Path(seg_path).name}")

        # 4. Initialize YOLO-cls Model (Optional)
        self.yolo_cls = None
        if cls_path and Path(cls_path).exists():
            self.yolo_cls = YOLO(str(cls_path), task="classify")
            logger.info(f"Loaded YOLO-cls: {Path(cls_path).name}")
        else:
            logger.warning("YOLO-cls model not found. Skipping document classification.")

        # 5. Initialize DBPostProcessor
        self.postprocessor = self._init_postprocessor(dbnet_config, dbnet_box_thresh, dbnet_unclip_ratio)

        # 6. Precomputed fast scale and bias for fused ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.scale = (1.0 / (255.0 * self.std)).astype(np.float32)
        self.bias = (-self.mean / self.std).astype(np.float32)

        # 7. Concurrent Worker Thread Pool
        self.executor = self._init_executor(concurrent, max_workers)

    @staticmethod
    def _resolve_model_path(*candidates: str, optional: bool = False) -> Optional[str]:
        """Find the first existing model file among candidates."""
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        if optional:
            return None
        return candidates[0] if candidates else None

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
        logger.info(
            f"Initialized DBNet PostProcessor: box_thresh={post_cfg.get('box_thresh')}, "
            f"unclip_ratio={post_cfg.get('unclip_ratio')}"
        )
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
        logger.info(f"Initialized TRT Pipeline with {workers} parallel workers.")
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trt_worker")

    def close(self):
        """Shutdown thread pool executor gracefully."""
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def warmup(self, num_runs: int = 3):
        """
        Warm up TensorRT execution engines across standard CCCD aspect ratios
        (Landscape 608x960 and Portrait 960x704) to eliminate GPU allocation overhead.
        """
        logger.info("Warming up TensorRT pipeline engines (Landscape & Portrait)...")
        dummy_shapes = [(608, 960), (960, 704)]
        for dyn_h, dyn_w in dummy_shapes:
            dummy_img = np.zeros((dyn_h, dyn_w, 3), dtype=np.uint8)
            for _ in range(num_runs):
                _ = self.predict(dummy_img, min_conf=0.25)
        logger.info("TensorRT Pipeline warmup complete.")

    # =========================================================================
    # Section 2: Image I/O & Preprocessing
    # =========================================================================
    @staticmethod
    def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load and validate BGR image from filesystem path or numpy array."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from path: {image}")
            return img
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Input numpy array must have shape (H, W, 3), got {image.shape}")
            return image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def _preprocess_dbnet(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Preprocess image for DBNet inference with aspect-ratio preserving resize and ImageNet norm.
        """
        orig_h, orig_w = image.shape[:2]

        scale = float(self.max_side_len) / max(orig_h, orig_w)
        new_h = int(round(orig_h * scale / 32) * 32)
        new_w = int(round(orig_w * scale / 32) * 32)
        new_h = max(32, new_h)
        new_w = max(32, new_w)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Fused normalization: (rgb * scale) + bias
        norm_img = rgb.astype(np.float32)
        norm_img = np.multiply(norm_img, self.scale, out=norm_img)
        norm_img = np.add(norm_img, self.bias, out=norm_img)

        # (H, W, C) -> (1, C, H, W)
        chw = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[None, ...])
        scale_factors = (orig_h / float(new_h), orig_w / float(new_w))
        return chw, scale_factors, (orig_h, orig_w)

    # =========================================================================
    # Section 3: Model Execution Branches
    # =========================================================================
    def _run_yolo_cls(self, image: np.ndarray) -> Dict[str, Any]:
        """Execute document classification on input image."""
        if self.yolo_cls is None:
            return {"card_type": "unknown", "confidence": 0.0}
        results = self.yolo_cls(image, verbose=False)
        top1_idx = results[0].probs.top1
        card_type = results[0].names[top1_idx]
        conf = float(results[0].probs.top1conf.cpu().item())
        return {"card_type": card_type, "confidence": conf}

    def _run_yolo_seg(self, image: np.ndarray, min_conf: float = 0.25) -> List[Dict[str, Any]]:
        """Execute semantic field segmentation on input image."""
        results = self.yolo_seg(image, conf=min_conf, verbose=False)
        boxes_info = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cls_id = clss[i]
                boxes_info.append({
                    "box": xyxy[i].tolist(),
                    "label": names[cls_id],
                    "confidence": float(confs[i]),
                })
        return boxes_info

    def _run_dbnet(
        self,
        input_tensor: np.ndarray,
        scale_factors: Tuple[float, float],
        orig_shape: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Execute DBNet inference and polygon extraction."""
        prob_map = self.dbnet_runner.infer(input_tensor)

        # Build prediction dictionary for DBPostProcessor
        preds = {"maps": prob_map}
        scale_h, scale_w = scale_factors
        orig_h, orig_w = orig_shape
        shape_list = [(orig_h, orig_w, scale_h, scale_w)]

        boxes_list, scores_list = self.postprocessor(preds, shape_list)

        text_detections = []
        if boxes_list and len(boxes_list) > 0:
            boxes = boxes_list[0]
            scores = scores_list[0]
            for poly, score in zip(boxes, scores):
                if isinstance(poly, np.ndarray):
                    poly_list = poly.tolist()
                else:
                    poly_list = [[float(pt[0]), float(pt[1])] for pt in poly]
                text_detections.append({
                    "polygon": poly_list,
                    "confidence": float(score),
                })
        return text_detections

    # =========================================================================
    # Section 4: Public Inference API
    # =========================================================================
    def predict(self, image: Union[str, Path, np.ndarray], min_conf: float = 0.25) -> Dict[str, Any]:
        """
        Synchronous end-to-end inference on a single image.
        Returns unified prediction dictionary with classifications, text polygons, and field labels.
        """
        start_time = time.perf_counter()
        img = self._load_image(image)

        input_tensor, scale_factors, orig_shape = self._preprocess_dbnet(img)

        if self.concurrent and self.executor is not None:
            fut_cls = self.executor.submit(self._run_yolo_cls, img)
            fut_seg = self.executor.submit(self._run_yolo_seg, img, min_conf)
            fut_dbnet = self.executor.submit(self._run_dbnet, input_tensor, scale_factors, orig_shape)

            classification = fut_cls.result()
            semantic_boxes = fut_seg.result()
            text_detections = fut_dbnet.result()
        else:
            classification = self._run_yolo_cls(img)
            semantic_boxes = self._run_yolo_seg(img, min_conf)
            text_detections = self._run_dbnet(input_tensor, scale_factors, orig_shape)

        # Match text bounding polygons to semantic field bounding boxes
        labeled_texts = match_text_to_fields(text_detections, semantic_boxes)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "classification": classification,
            "total_texts": len(labeled_texts),
            "detections": labeled_texts,
            "latency_ms": round(elapsed_ms, 2),
        }

    async def predict_async(self, image: Union[str, Path, np.ndarray], min_conf: float = 0.25) -> Dict[str, Any]:
        """
        Asynchronous non-blocking prediction API for high-throughput ASGI microservices.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.predict, image, min_conf)

    def predict_batch(
        self,
        images: Sequence[Union[str, Path, np.ndarray]],
        batch_size: int = 1,
        min_conf: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction API across an iterable of images.
        """
        results = []
        for img in images:
            results.append(self.predict(img, min_conf=min_conf))
        return results
