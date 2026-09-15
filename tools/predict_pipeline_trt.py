"""
High-Performance End-to-End NVIDIA TensorRT CCCD Detection & Segmentation CLI Tool.
Executes DBNet, YOLO-seg, and YOLO-cls entirely via compiled TensorRT engines (.engine)
with sub-20ms latency, multi-threaded concurrent inference, colored polygon visualization,
and JSON output.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

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

CLASS_COLORS: Dict[str, tuple] = {
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


def draw_labeled_polygons(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    title: Optional[str] = None,
    card_type: Optional[str] = None,
    alpha: float = 0.28,
) -> np.ndarray:
    """
    Draw colored transparent polygon masks with crisp label badges.
    """
    vis_img = image.copy()
    overlay = image.copy()

    for d in detections:
        label = d.get("label", "text")
        poly = d.get("polygon", [])
        if not poly:
            continue

        color = CLASS_COLORS.get(label, (0, 255, 0))
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)

    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # Draw label badges
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

        text = f"{label} {conf:.2f}"
        font_scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        badge_y1 = max(0, y - th - 6)
        badge_y2 = y
        badge_x1 = max(0, x)
        badge_x2 = min(vis_img.shape[1], x + tw + 6)

        cv2.rectangle(vis_img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            vis_img,
            text,
            (badge_x1 + 3, badge_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )

    # Draw document header banner
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


def collect_image_paths(source: str) -> List[Path]:
    """Collect image paths from file, directory, or pattern."""
    p = Path(source)
    if p.is_file():
        return [p]
    elif p.is_dir():
        exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".PNG"]
        files = [f for f in sorted(p.iterdir()) if f.suffix in exts]
        return files
    else:
        matched = sorted(Path().glob(source))
        if matched:
            return [m for m in matched if m.is_file()]
        raise FileNotFoundError(f"Source path not found: {source}")


def parse_args():
    parser = argparse.ArgumentParser(description="High-Performance CCCD Detection via NVIDIA TensorRT Engines.")
    parser.add_argument("--source", type=str, required=True, help="Path to image file, folder, or glob pattern.")
    parser.add_argument("--engine-dir", type=str, default="weights/tensorrt", help="TensorRT engine directory.")
    parser.add_argument("--dbnet-engine", type=str, default=None, help="Custom DBNet .engine file path.")
    parser.add_argument("--seg-engine", type=str, default=None, help="Custom YOLO-seg .engine file path.")
    parser.add_argument("--cls-engine", type=str, default=None, help="Custom YOLO-cls .engine file path.")
    parser.add_argument("--save-vis", type=str, default=None, help="Directory to save visualized predictions.")
    parser.add_argument("--save-json", type=str, default=None, help="File path to save JSON prediction results.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence threshold for YOLO.")
    parser.add_argument("--box-thresh", type=float, default=None, help="DBNet box score threshold.")
    parser.add_argument("--unclip-ratio", type=float, default=None, help="DBNet polygon unclip expansion ratio.")
    parser.add_argument("--max-side-len", type=int, default=960, help="DBNet maximum resize side length.")
    parser.add_argument("--sequential", action="store_true", help="Run models sequentially instead of in parallel.")
    return parser.parse_args()


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

    # Warmup
    if args.warmup > 0:
        pipeline.warmup(num_runs=args.warmup)

    # Create output directories
    if args.save_vis:
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)

    latencies: List[float] = []
    all_results: List[Dict[str, Any]] = []

    logger.info("Executing TensorRT inference...")
    for idx, img_path in enumerate(image_paths, 1):
        res = pipeline.predict(img_path, min_conf=args.min_conf)
        lat = res["latency_ms"]
        latencies.append(lat)

        rec = {
            "image": img_path.name,
            "path": str(img_path),
            **res,
        }
        all_results.append(rec)

        card_type = res["classification"].get("card_type", "Unknown")
        total_texts = res["total_texts"]
        logger.info(
            f"[{idx}/{len(image_paths)}] {img_path.name:<25} | "
            f"Type: {card_type:<18} | Texts: {total_texts:<2} | Latency: {lat:.1f}ms"
        )

        if args.save_vis:
            raw_bgr = cv2.imread(str(img_path))
            vis_img = draw_labeled_polygons(
                raw_bgr,
                res["detections"],
                title=img_path.name,
                card_type=card_type,
            )
            out_vis_path = Path(args.save_vis) / f"trt_{img_path.stem}.jpg"
            cv2.imwrite(str(out_vis_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    pipeline.close()

    # Latency Statistics
    lat_arr = np.array(latencies)
    mean_lat = np.mean(lat_arr)
    median_lat = np.median(lat_arr)
    p95_lat = np.percentile(lat_arr, 95)
    min_lat = np.min(lat_arr)
    max_lat = np.max(lat_arr)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print("\n" + "=" * 70)
    print("           NVIDIA TensorRT CCCD Pipeline Benchmark Summary            ")
    print("=" * 70)
    print(f"  Processed Images : {len(image_paths)}")
    print(f"  Mean Latency     : {mean_lat:.2f} ms")
    print(f"  Median (P50)     : {median_lat:.2f} ms")
    print(f"  P95 Latency      : {p95_lat:.2f} ms")
    print(f"  Min / Max Latency: {min_lat:.2f} ms / {max_lat:.2f} ms")
    print(f"  Throughput (FPS) : {fps:.1f} FPS")
    print("=" * 70)

    if args.save_json:
        json_out = Path(args.save_json)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "summary": {
                "total_images": len(image_paths),
                "mean_latency_ms": round(float(mean_lat), 2),
                "median_latency_ms": round(float(median_lat), 2),
                "p95_latency_ms": round(float(p95_lat), 2),
                "min_latency_ms": round(float(min_lat), 2),
                "max_latency_ms": round(float(max_lat), 2),
                "fps": round(float(fps), 1),
            },
            "results": all_results,
        }
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON benchmark results to: {json_out}")

    if args.save_vis:
        logger.info(f"Saved visual annotations to directory: {args.save_vis}")


if __name__ == "__main__":
    main()
