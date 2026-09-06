"""
Prediction / Inference script for YOLO26 Field Detection & Segmentation on CCCD.
Accepts an image path or a directory and returns standard structured polygon/box outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import cv2
import numpy as np

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
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Draw colored transparent polygon masks with crisp label badges.
    """
    vis_img = image.copy()
    overlay = image.copy()

    for d in detections:
        label = d.get("label", "field")
        conf = d.get("confidence", 0.0)
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

        # Draw filled background badge
        cv2.rectangle(vis_img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Draw white text
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

    return vis_img


def predict_yolo(
    weights: str,
    source: Union[str, Path],
    task: str = "segment",
    min_conf: float = 0.5,
    imgsz: int = 640,
    device: str = "",
    save_json: str = None,
    save_visualizations: str = None,
) -> List[Dict[str, Any]]:
    """
    Run YOLO segmentation/detection inference on image(s).
    """
    wrapper = YOLOWrapper(model_path=weights, task=task)
    src_path = Path(source)

    if src_path.is_file():
        image_paths = [src_path]
    elif src_path.is_dir():
        image_paths = sorted(
            [p for p in src_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Source path not found: {source}")

    logger.info(f"Loaded YOLO ({task}) from '{weights}'. Running inference on {len(image_paths)} image(s)...")

    all_results = []
    vis_dir = Path(save_visualizations) if save_visualizations else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for img_p in image_paths:
        dets = wrapper.detect(image=str(img_p), min_conf=min_conf, imgsz=imgsz, device=device if device else None)
        vis_save_path = None
        if vis_dir:
            img = cv2.imread(str(img_p))
            if img is not None:
                vis_img = draw_field_detections(img, dets)
                vis_save_path = str(vis_dir / f"vis_{img_p.name}")
                cv2.imwrite(vis_save_path, vis_img)

        item = {
            "image": img_p.name,
            "detections": dets,
            "total_fields": len(dets),
            "saved_vis": vis_save_path,
        }
        all_results.append(item)
        logger.info(f"[{img_p.name}] -> Found {len(dets)} fields: {[d['label'] for d in dets]} | Vis: {vis_save_path}")

    if save_json:
        out_p = Path(save_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved predictions to: {save_json}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Inference for YOLO26 Field Detection / Segmentation")
    parser.add_argument("--weights", type=str, default="weights/yolo/yolo26_seg_best.pt", help="Path to weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--task", type=str, default="segment", choices=["segment", "detect"], help="Task type")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="", help="Device (mps, cuda, cpu)")
    parser.add_argument("--save-json", type=str, default=None, help="Output JSON path")
    parser.add_argument("--save-vis", type=str, default=None, help="Output visualization directory")
    args = parser.parse_args()

    predict_yolo(
        weights=args.weights,
        source=args.source,
        task=args.task,
        min_conf=args.min_conf,
        imgsz=args.imgsz,
        device=args.device,
        save_json=args.save_json,
        save_visualizations=args.save_vis,
    )


if __name__ == "__main__":
    main()
