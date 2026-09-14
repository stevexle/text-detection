"""
High-Performance End-to-End ONNX CCCD Detection & Segmentation CLI Tool.
Executes DBNet, YOLO-seg, and YOLO-cls entirely via ONNX Runtime with
multi-threaded concurrent inference, colored polygon visualization, and JSON output.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union

# Auto-detect and register NVIDIA CUDA/cuDNN shared libraries on Linux
if sys.platform == "linux":
    import ctypes
    import site
    try:
        for site_pkg in site.getsitepackages():
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand"]:
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
    except Exception:
        pass

import cv2
import numpy as np

from src.pipeline.cccd_pipeline_onnx import CCCDDetectionPipelineONNX
from src.utils.logger import get_logger

logger = get_logger("PredictPipelineONNX")

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
        top_left = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        x, y = int(top_left[0]), int(top_left[1])

        text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        badge_y1 = max(0, y - text_h - 6)
        badge_y2 = y
        badge_x1 = max(0, x)
        badge_x2 = x + text_w + 6

        cv2.rectangle(vis_img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(
            vis_img,
            text,
            (badge_x1 + 3, badge_y2 - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    # Draw title / card type badge if provided
    header_parts = []
    if title:
        header_parts.append(title)
    if card_type:
        header_parts.append(f"Type: {card_type}")

    if header_parts:
        header_text = " | ".join(header_parts)
        cv2.rectangle(vis_img, (10, 10), (15 + len(header_text) * 11, 42), (20, 20, 20), -1)
        cv2.putText(
            vis_img,
            header_text,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis_img


def predict_pipeline_onnx(
    source: str,
    dbnet_onnx: str = "weights/onnx/dbnet.onnx",
    yolo_seg_onnx: str = "weights/onnx/yolo26_seg.onnx",
    yolo_cls_onnx: Optional[str] = "weights/onnx/yolo26_cls.onnx",
    dbnet_config: str = "configs/dbnet/dbnet.yaml",
    batch_size: int = 8,
    min_conf: float = 0.25,
    min_overlap: float = 0.20,
    unclip_ratio: Optional[float] = None,
    box_thresh: Optional[float] = None,
    warmup: bool = True,
    save_json: str = "runs/pipeline_onnx/result.json",
    save_vis: Optional[str] = None,
    concurrent: bool = True,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run CCCD end-to-end inference using pure ONNX Runtime models.
    """
    p_src = Path(source)
    if not p_src.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if p_src.is_file():
        image_paths = [p_src]
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted([p for p in p_src.iterdir() if p.suffix.lower() in valid_exts])
        if not image_paths:
            raise ValueError(f"No valid image files found in directory: {source}")

    logger.info("=" * 60)
    logger.info(f"Initializing Pure ONNX CCCD Pipeline (Concurrent={concurrent})")
    logger.info(f"Images to process: {len(image_paths)}")
    logger.info("=" * 60)

    vis_dir = Path(save_vis) if save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    with CCCDDetectionPipelineONNX(
        dbnet_onnx=dbnet_onnx,
        yolo_seg_onnx=yolo_seg_onnx,
        yolo_cls_onnx=yolo_cls_onnx,
        dbnet_config=dbnet_config,
        dbnet_box_thresh=box_thresh,
        dbnet_unclip_ratio=unclip_ratio,
        concurrent=concurrent,
    ) as pipeline:
        if warmup:
            pipeline.warmup(num_runs=2)

        t_start = time.perf_counter()

        if len(image_paths) == 1:
            img_p = image_paths[0]
            img_bgr = cv2.imread(str(img_p))
            if img_bgr is None:
                raise ValueError(f"Could not read image: {img_p}")

            res = pipeline.predict(img_bgr, min_conf=min_conf, min_overlap=min_overlap)
            all_results = [res]

            card_type = res.get("classification", {}).get("card_type", "") if res.get("classification") else ""
            detections = res.get("detections", [])
            latency_ms = res.get("latency_ms", 0.0)

            vis_msg = ""
            if vis_dir:
                vis_img = draw_labeled_polygons(img_bgr, detections, title="CCCD ONNX Detection", card_type=card_type)
                vis_save_path = str(vis_dir / f"onnx_fused_{img_p.name}")
                cv2.imwrite(vis_save_path, vis_img)
                vis_msg = f" | Vis: {vis_save_path}"

            logger.info(
                f"Image: {img_p.name} | Type: '{card_type}' | {len(detections)} fields ({latency_ms:.1f}ms){vis_msg}"
            )
        else:
            all_results = pipeline.predict_batch(
                images=image_paths,
                batch_size=batch_size,
                min_conf=min_conf,
                min_overlap=min_overlap,
            )

            total_time_s = time.perf_counter() - t_start
            per_img_ms = (total_time_s / len(image_paths)) * 1000.0

            for res, img_p in zip(all_results, image_paths):
                res["latency_ms"] = round(per_img_ms, 2)
                card_type = res.get("classification", {}).get("card_type", "") if res.get("classification") else ""
                detections = res.get("detections", [])

                vis_msg = ""
                if vis_dir:
                    img_bgr = cv2.imread(str(img_p))
                    if img_bgr is not None:
                        vis_img = draw_labeled_polygons(img_bgr, detections, title="CCCD ONNX Detection", card_type=card_type)
                        vis_save_path = str(vis_dir / f"onnx_fused_{img_p.name}")
                        cv2.imwrite(vis_save_path, vis_img)
                        vis_msg = f" | Vis: {vis_save_path}"

                logger.info(
                    f"Image: {img_p.name} | Type: '{card_type}' | {len(detections)} fields ({per_img_ms:.1f}ms){vis_msg}"
                )

    total_time_s = time.perf_counter() - t_start
    avg_ms = (total_time_s / len(image_paths)) * 1000.0 if image_paths else 0.0
    fps = len(image_paths) / total_time_s if total_time_s > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"ONNX Pipeline Benchmark Summary:")
    logger.info(f"Total Images: {len(image_paths)} | Total Time: {total_time_s:.3f}s")
    logger.info(f"Average Latency: {avg_ms:.2f} ms/image | Throughput: {fps:.1f} FPS")
    logger.info("=" * 60)

    output_payload = all_results[0] if len(all_results) == 1 else all_results

    if save_json:
        json_path = Path(save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved ONNX pipeline results JSON to: {save_json}")

    return output_payload


def main():
    parser = argparse.ArgumentParser(description="End-to-End High-Performance Pure ONNX CCCD Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Input image or folder")
    parser.add_argument("--dbnet-onnx", type=str, default="weights/onnx/dbnet.onnx", help="Path to DBNet ONNX model")
    parser.add_argument("--yolo-seg-onnx", type=str, default="weights/onnx/yolo26_seg.onnx", help="Path to YOLO-seg ONNX model")
    parser.add_argument("--yolo-cls-onnx", type=str, default="weights/onnx/yolo26_cls.onnx", help="Path to YOLO-cls ONNX model")
    parser.add_argument("--dbnet-config", type=str, default="configs/dbnet/dbnet.yaml", help="DBNet config path for postprocessing")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for parallel processing")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--min-overlap", type=float, default=0.20, help="Minimum overlap ratio for field matching")
    parser.add_argument("--unclip-ratio", type=float, default=None, help="Vatti unclip expansion ratio override")
    parser.add_argument("--box-thresh", type=float, default=None, help="DBNet box score threshold override")
    parser.add_argument("--no-warmup", action="store_true", help="Disable warmup")
    parser.add_argument("--no-concurrent", dest="concurrent", action="store_false", default=True, help="Disable 3-step parallel execution")
    parser.add_argument("--save-json", type=str, default="runs/pipeline_onnx/result.json", help="Output JSON path")
    parser.add_argument("--save-vis", type=str, default=None, help="Optional output visualization directory")
    args = parser.parse_args()

    predict_pipeline_onnx(
        source=args.source,
        dbnet_onnx=args.dbnet_onnx,
        yolo_seg_onnx=args.yolo_seg_onnx,
        yolo_cls_onnx=args.yolo_cls_onnx,
        dbnet_config=args.dbnet_config,
        batch_size=args.batch_size,
        min_conf=args.min_conf,
        min_overlap=args.min_overlap,
        unclip_ratio=args.unclip_ratio,
        box_thresh=args.box_thresh,
        warmup=not args.no_warmup,
        save_json=args.save_json,
        save_vis=args.save_vis,
        concurrent=args.concurrent,
    )


if __name__ == "__main__":
    main()
