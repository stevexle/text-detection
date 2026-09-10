"""
Ultra-Fast Pure YOLO End-to-End Prediction CLI: Document Classification + Field Segmentation.
Runs with ultra-low latency (~60ms on Mac / ~5ms on GPU) without DBNet overhead.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from src.models.wrappers.yolo_classifier import YOLOClassifier
from src.models.wrappers.yolo_wrapper import YOLOWrapper
from src.utils.logger import get_logger

logger = get_logger("PredictYOLO")

# Distinct color palette for each CCCD field (BGR format)
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
    "text": (0, 255, 0),           # Green fallback
}


def draw_field_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    card_type: str = None,
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draw colored transparent polygon masks with crisp label badges.
    """
    vis_img = image.copy()
    overlay = image.copy()

    for d in detections:
        label = d.get("label", "field")
        poly = d.get("polygon", [])
        if not poly:
            continue

        color = CLASS_COLORS.get(label, (0, 255, 0))
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))

        # 1. Fill polygon mask on overlay
        cv2.fillPoly(overlay, [pts], color)
        # 2. Draw sharp contour lines
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=2)

    # Blend translucent masks
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

    # 3. Draw label badges on top of blended image
    for d in detections:
        label = d.get("label", "field")
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

        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
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

    # Draw card type header badge if provided
    if card_type:
        header_text = f"Type: {card_type}"
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


def predict_yolo(
    source: Union[str, Path],
    yolo_seg_weights: str = "weights/yolo/yolo26_seg_best.pt",
    yolo_cls_weights: str = "weights/yolo/yolo26_cls_best.pt",
    task: str = "segment",
    min_conf: float = 0.4,
    imgsz: int = 640,
    device: str = "",
    save_json: str = "runs/predict_yolo/result.json",
    save_vis: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run high-speed pure YOLO inference (Classification + Segmentation) on image(s).
    """
    yolo_seg = YOLOWrapper(model_path=yolo_seg_weights, task=task)
    yolo_cls = None
    if yolo_cls_weights and Path(yolo_cls_weights).exists():
        yolo_cls = YOLOClassifier(model_path=yolo_cls_weights, device=device if device else None)

    src_path = Path(source)
    if src_path.is_file():
        image_paths = [src_path]
    elif src_path.is_dir():
        image_paths = sorted(
            [p for p in src_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Source path not found: {source}")

    logger.info(f"Running Pure YOLO Pipeline on {len(image_paths)} image(s)...")

    # Warmup models
    dummy_img = np.zeros((448, 960, 3), dtype=np.uint8)
    if yolo_cls is not None:
        _ = yolo_cls.classify(dummy_img)
    _ = yolo_seg.detect(dummy_img)

    vis_dir = Path(save_vis) if save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    t_total_start = time.perf_counter()

    for img_p in image_paths:
        t0 = time.perf_counter()
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue

        # Step 1: Document Classification
        cls_res = None
        card_type = "unknown"
        if yolo_cls is not None:
            cls_res = yolo_cls.classify(img_bgr)
            card_type = cls_res.get("card_type", "unknown") if isinstance(cls_res, dict) else "unknown"

        # Step 2: YOLO Field Segmentation
        dets = yolo_seg.detect(image=img_bgr, min_conf=min_conf, imgsz=imgsz, device=device if device else None)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        item = {
            "classification": cls_res,
            "total_texts": len(dets),
            "detections": dets,
            "latency_ms": round(elapsed_ms, 2),
        }
        all_results.append(item)

        # Optional visualization
        vis_msg = ""
        if vis_dir:
            vis_img = draw_field_detections(img_bgr, dets, card_type=card_type)
            vis_save_path = str(vis_dir / f"fused_{img_p.name}")
            cv2.imwrite(vis_save_path, vis_img)
            vis_msg = f" | Vis: {vis_save_path}"

        logger.info(
            f"Image: {img_p.name} | Type: '{card_type}' | {len(dets)} fields ({elapsed_ms:.1f}ms){vis_msg}"
        )

    total_time_s = time.perf_counter() - t_total_start
    avg_ms = (total_time_s / len(image_paths)) * 1000.0 if image_paths else 0.0
    fps = len(image_paths) / total_time_s if total_time_s > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Pure YOLO Benchmark Summary:")
    logger.info(f"Total Images: {len(image_paths)} | Total Time: {total_time_s:.3f}s")
    logger.info(f"Average Latency: {avg_ms:.2f} ms/image | Throughput: {fps:.1f} FPS")
    logger.info("=" * 60)

    output_payload = all_results[0] if len(all_results) == 1 else all_results

    if save_json:
        out_p = Path(save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predictions to: {save_json}")

    return output_payload


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Pure YOLO End-to-End Prediction CLI")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--yolo-seg-weights", type=str, default="weights/yolo/yolo26_seg_best.pt", help="YOLO-seg weights")
    parser.add_argument("--yolo-cls-weights", type=str, default="weights/yolo/yolo26_cls_best.pt", help="YOLO-cls weights")
    parser.add_argument("--task", type=str, default="segment", choices=["segment", "detect"], help="Task type")
    parser.add_argument("--min-conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="", help="Device (mps, cuda, cpu)")
    parser.add_argument("--save-json", type=str, default="runs/predict_yolo/result.json", help="Output JSON path")
    parser.add_argument("--save-vis", type=str, default=None, help="Optional output visualization directory")
    args = parser.parse_args()

    predict_yolo(
        source=args.source,
        yolo_seg_weights=args.yolo_seg_weights,
        yolo_cls_weights=args.yolo_cls_weights,
        task=args.task,
        min_conf=args.min_conf,
        imgsz=args.imgsz,
        device=args.device,
        save_json=args.save_json,
        save_vis=args.save_vis,
    )


if __name__ == "__main__":
    main()

