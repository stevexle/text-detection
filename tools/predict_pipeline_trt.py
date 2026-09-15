"""
High-Performance End-to-End NVIDIA TensorRT CCCD Detection & Segmentation CLI Tool.

Features:
  - Sub-20ms multi-threaded inference via compiled TensorRT engines (.engine).
  - High-precision transparent polygon visualization with badge labels.
  - Comprehensive statistical reporting (P50, P90, P95, P99, StdDev, FPS).
  - Production JSON payload export adhering to standardized CCCD schema.
"""

import argparse
import json
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

from src.pipeline.cccd_pipeline_trt import CCCDDetectionPipelineTRT
from src.utils.logger import get_logger

logger = get_logger("PredictPipelineTRT")

# Standardized Color Palette for CCCD Document Semantic Fields (BGR)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "id": (255, 105, 65),          # Royal Blue
    "name": (46, 184, 46),         # Vivid Green
    "dob": (0, 140, 255),          # Vibrant Orange
    "gender": (180, 60, 200),      # Violet
    "nationality": (220, 180, 0),  # Bright Cyan
    "origin_place": (180, 120, 30),# Deep Teal
    "current_place": (50, 60, 230),# Coral Red
    "expire_date": (140, 30, 240), # Neon Pink
    "issue_date": (230, 50, 160),  # Purple-Pink
    "features": (60, 220, 120),    # Emerald
    "mrz": (0, 215, 255),          # Gold / Yellow
    "other_text": (160, 160, 160), # Slate Gray
    "text": (0, 255, 0),           # Bright Green
}


# =============================================================================
# Section 1: High-Efficiency Visualization Renderer
# =============================================================================
def draw_labeled_polygons(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    title: Optional[str] = None,
    card_type: Optional[str] = None,
    alpha: float = 0.28,
) -> np.ndarray:
    """
    Render colored transparent polygon masks with crisp label badges in single pass.
    """
    vis_img = image.copy()
    overlay = image.copy()

    # Pass 1: Fill polygon areas on overlay and draw outlines on vis_img
    for d in detections:
        label = d.get("label", "text")
        poly = d.get("polygon", [])
        if not poly:
            continue

        color = CLASS_COLORS.get(label, (0, 255, 0))
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)

    # Fast alpha blend
    cv2.addWeighted(overlay, alpha, vis_img, 1.0 - alpha, 0, vis_img)

    # Pass 2: Render crisp label badges
    for d in detections:
        label = d.get("label", "text")
        conf = d.get("confidence", 0.0)
        poly = d.get("polygon", [])
        if not poly:
            continue

        color = CLASS_COLORS.get(label, (0, 255, 0))
        pts = np.array(poly, dtype=np.int32)
        top_idx = int(np.argmin(pts[:, 1]))
        top_left = pts[top_idx]
        x, y = int(top_left[0]), int(top_left[1])

        badge_text = f"{label} {conf:.2f}"
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        badge_y1 = max(0, y - th - 6)
        badge_y2 = y
        badge_x1 = max(0, x)
        badge_x2 = min(vis_img.shape[1], x + tw + 6)

        cv2.rectangle(vis_img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            vis_img,
            badge_text,
            (badge_x1 + 3, badge_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    # Pass 3: Header banner
    if card_type or title:
        header_text = f"TensorRT Pipeline | Type: {card_type or 'Unknown'}"
        if title:
            header_text = f"{title} | {header_text}"
        cv2.rectangle(vis_img, (0, 0), (vis_img.shape[1], 36), (24, 24, 24), -1)
        cv2.putText(
            vis_img,
            header_text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (240, 240, 240),
            1,
            lineType=cv2.LINE_AA,
        )

    return vis_img


# =============================================================================
# Section 2: Input Resolution & CLI Parsing
# =============================================================================
def collect_image_paths(source: str) -> List[Path]:
    """Collect image paths from file, directory, or pattern."""
    p = Path(source)
    if p.is_file():
        return [p]
    elif p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".PNG"}
        return [f for f in sorted(p.iterdir()) if f.suffix in exts]
    else:
        matched = sorted(Path().glob(source))
        if matched:
            return [m for m in matched if m.is_file()]
        raise FileNotFoundError(f"Source path not found: {source}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="High-Performance CCCD Detection via NVIDIA TensorRT Engines (.engine).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True, help="Path to image file, folder, or glob pattern.")
    parser.add_argument("--engine-dir", type=str, default="weights/tensorrt", help="TensorRT engine directory.")
    parser.add_argument("--dbnet-engine", type=str, default=None, help="Custom DBNet .engine file path.")
    parser.add_argument("--seg-engine", type=str, default=None, help="Custom YOLO-seg .engine file path.")
    parser.add_argument("--cls-engine", type=str, default=None, help="Custom YOLO-cls .engine file path.")
    parser.add_argument("--save-vis", type=str, default=None, help="Directory to save visualized predictions.")
    parser.add_argument("--save-json", type=str, default=None, help="File path to save JSON prediction results.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence threshold for YOLO.")
    parser.add_argument("--min-overlap", type=float, default=0.20, help="Minimum overlap ratio for field fusion.")
    parser.add_argument("--no-fallback", action="store_true", help="Disable fallback for unmatched YOLO fields.")
    parser.add_argument("--box-thresh", type=float, default=None, help="DBNet box score threshold.")
    parser.add_argument("--unclip-ratio", type=float, default=None, help="DBNet polygon unclip expansion ratio.")
    parser.add_argument("--max-side-len", type=int, default=960, help="DBNet maximum resize side length.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of benchmark iterations per image to measure stable steady-state latency.")
    parser.add_argument("--sequential", action="store_true", help="Run models sequentially instead of in parallel.")
    return parser.parse_args()


# =============================================================================
# Section 3: Benchmark Statistics Summary Formatter
# =============================================================================
def print_benchmark_table(image_count: int, latencies: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistical metrics and render an aligned ASCII table."""
    lat_arr = np.array(latencies, dtype=np.float32)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p90_lat = float(np.percentile(lat_arr, 90))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    std_lat = float(np.std(lat_arr))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print("\n" + "+" + "-" * 68 + "+")
    print("|" + "NVIDIA TensorRT CCCD Pipeline Benchmark Summary".center(68) + "|")
    print("+" + "-" * 32 + "+" + "-" * 35 + "+")
    print(f"| {'Metric':<30} | {'Value':<33} |")
    print("+" + "-" * 32 + "+" + "-" * 35 + "+")
    print(f"| {'Benchmark Inferences':<30} | {len(latencies):<33} |")
    print(f"| {'Unique Documents':<30} | {image_count:<33} |")
    print(f"| {'Mean Latency':<30} | {mean_lat:6.2f} ms                       |")
    print(f"| {'Median Latency (P50)':<30} | {median_lat:6.2f} ms                       |")
    print(f"| {'P90 Latency':<30} | {p90_lat:6.2f} ms                       |")
    print(f"| {'P95 Latency':<30} | {p95_lat:6.2f} ms                       |")
    print(f"| {'P99 Latency':<30} | {p99_lat:6.2f} ms                       |")
    print(f"| {'Min / Max Latency':<30} | {min_lat:5.2f} ms / {max_lat:5.2f} ms           |")
    print(f"| {'Standard Deviation':<30} | {std_lat:6.2f} ms                       |")
    print(f"| {'Overall Throughput':<30} | {fps:6.1f} FPS                      |")
    print("+" + "-" * 68 + "+\n")

    return {
        "total_inferences": len(latencies),
        "unique_documents": image_count,
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "p90_latency_ms": round(p90_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "min_latency_ms": round(min_lat, 2),
        "max_latency_ms": round(max_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "fps": round(fps, 1),
    }


# =============================================================================
# Section 4: Main Execution Driver
# =============================================================================
def main():
    args = parse_args()
    engine_dir = Path(args.engine_dir)

    dbnet_p = args.dbnet_engine or str(engine_dir / "dbnet.engine")
    seg_p = args.seg_engine or str(engine_dir / "yolo26_seg.engine")
    cls_p = args.cls_engine or str(engine_dir / "yolo26_cls.engine")

    image_paths = collect_image_paths(args.source)
    if not image_paths:
        logger.error(f"No valid image files discovered in: {args.source}")
        sys.exit(1)

    logger.info(f"Discovered {len(image_paths)} image(s) for TensorRT inference.")

    # Initialize Pipeline
    pipeline = CCCDDetectionPipelineTRT(
        dbnet_engine=dbnet_p,
        yolo_seg_engine=seg_p,
        yolo_cls_engine=cls_p,
        dbnet_box_thresh=args.box_thresh,
        dbnet_unclip_ratio=args.unclip_ratio,
        max_side_len=args.max_side_len,
        concurrent=not args.sequential,
    )

    # Multi-shape GPU Warmup
    if args.warmup > 0:
        pipeline.warmup(num_runs=args.warmup)

    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)

    # Pre-load images to isolate pure inference latency from disk I/O
    loaded_images: List[Tuple[Path, np.ndarray]] = []
    for img_p in image_paths:
        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is not None:
            loaded_images.append((img_p, raw_bgr))

    latencies: List[float] = []
    all_results: List[Dict[str, Any]] = []

    repeat_count = max(1, args.repeat)
    logger.info(f"Executing TensorRT inference (repeat={repeat_count}x)...")

    for r in range(repeat_count):
        for idx, (img_path, raw_bgr) in enumerate(loaded_images, 1):
            res = pipeline.predict(
                raw_bgr,
                min_conf=args.min_conf,
                min_overlap=args.min_overlap,
                fallback_unmatched=not args.no_fallback,
            )
            lat = res["latency_ms"]
            latencies.append(lat)

            rec = {
                "image": img_path.name,
                "path": str(img_path),
                "iteration": r + 1,
                **res,
            }
            if r == repeat_count - 1:
                all_results.append(rec)

            card_type = res["classification"].get("card_type", "Unknown")
            total_texts = res["total_texts"]
            rep_label = f" (Run {r+1}/{repeat_count})" if repeat_count > 1 else ""
            logger.info(
                f"[{idx:>3}/{len(loaded_images):<3}]{rep_label} {img_path.name:<22} | "
                f"Type: {card_type:<14} | Texts: {total_texts:<2} | Latency: {lat:5.1f}ms"
            )

            if args.save_vis and r == repeat_count - 1:
                vis_img = draw_labeled_polygons(
                    raw_bgr,
                    res["detections"],
                    title=img_path.name,
                    card_type=card_type,
                )
                out_vis_path = Path(args.save_vis) / f"trt_{img_path.stem}.jpg"
                cv2.imwrite(str(out_vis_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    pipeline.close()

    # Latency Statistical Benchmark
    summary_stats = print_benchmark_table(len(image_paths), latencies)

    # Save JSON Benchmark Report
    if args.save_json:
        json_out = Path(args.save_json)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "summary": summary_stats,
            "results": all_results,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON benchmark results to: {json_out}")

    if args.save_vis:
        logger.info(f"Saved visual annotations to: {args.save_vis}")


if __name__ == "__main__":
    main()
