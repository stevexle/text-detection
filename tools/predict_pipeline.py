"""
1-Click End-to-End High-Performance Prediction CLI: DBNet + YOLO-seg Hybrid Fusion.
Optimized with InferenceMode, FP16, Batched Pipeline Execution, and Zero-Redundant I/O.
"""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np

from src.pipeline.cccd_pipeline import CCCDDetectionPipeline
from src.utils.logger import get_logger

logger = get_logger("PredictPipeline")

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


def predict_pipeline(
    source: str,
    dbnet_config: str = "configs/dbnet/dbnet.yaml",
    dbnet_weights: str = "weights/dbnet/dbnet_cccd_best.pth",
    yolo_seg_weights: str = "weights/yolo/yolo26_seg_best.pt",
    yolo_cls_weights: str = "weights/yolo/yolo26_cls_best.pt",
    batch_size: int = 8,
    min_conf: float = 0.4,
    min_overlap: float = 0.20,
    unclip_ratio: Optional[float] = None,
    box_thresh: Optional[float] = None,
    device: str = "",
    fp16: bool = False,
    warmup: bool = True,
    save_json: str = "runs/pipeline/result.json",
    save_vis: Optional[str] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run end-to-end pipeline on input image(s) with high-speed batched execution and visualizations.
    """
    pipeline = CCCDDetectionPipeline(
        dbnet_config=dbnet_config,
        dbnet_weights=dbnet_weights,
        yolo_seg_weights=yolo_seg_weights,
        yolo_cls_weights=yolo_cls_weights,
        dbnet_box_thresh=box_thresh,
        dbnet_unclip_ratio=unclip_ratio,
        device=device,
        fp16=fp16,
    )

    if warmup:
        pipeline.warmup(num_runs=2)

    src_p = Path(source)
    if src_p.is_file():
        image_paths = [src_p]
    elif src_p.is_dir():
        image_paths = sorted(
            [p for p in src_p.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]]
        )
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    vis_dir = Path(save_vis) if save_vis else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running End-to-End Hybrid Pipeline on {len(image_paths)} image(s) (BatchSize={batch_size})...")

    t_start = time.perf_counter()

    # Single Image Optimized Path
    if len(image_paths) == 1:
        img_p = image_paths[0]
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            raise ValueError(f"Could not load image from: {img_p}")

        res = pipeline.predict(
            image=img_bgr,
            min_conf=min_conf,
            min_overlap=min_overlap,
        )
        all_results = [res]

        card_type = res.get("classification", {}).get("card_type", "") if res.get("classification") else ""
        detections = res.get("detections", [])
        latency_ms = res.get("latency_ms", 0.0)

        vis_msg = ""
        if vis_dir:
            vis_img = draw_labeled_polygons(img_bgr, detections, title="CCCD Detection", card_type=card_type)
            vis_save_path = str(vis_dir / f"fused_{img_p.name}")
            cv2.imwrite(vis_save_path, vis_img)
            vis_msg = f" | Vis: {vis_save_path}"

        logger.info(
            f"Image: {img_p.name} | Type: '{card_type}' | {len(detections)} fields ({latency_ms:.1f}ms){vis_msg}"
        )

    # Batched Execution Path for Multiple Images
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
                    vis_img = draw_labeled_polygons(img_bgr, detections, title="CCCD Detection", card_type=card_type)
                    vis_save_path = str(vis_dir / f"fused_{img_p.name}")
                    cv2.imwrite(vis_save_path, vis_img)
                    vis_msg = f" | Vis: {vis_save_path}"

            logger.info(
                f"Image: {img_p.name} | Type: '{card_type}' | {len(detections)} fields ({per_img_ms:.1f}ms){vis_msg}"
            )

    total_time_s = time.perf_counter() - t_start
    avg_ms = (total_time_s / len(image_paths)) * 1000.0 if image_paths else 0.0
    fps = len(image_paths) / total_time_s if total_time_s > 0 else 0.0

    logger.info("=" * 60)
    logger.info(f"Pipeline Benchmark Summary:")
    logger.info(f"Total Images: {len(image_paths)} | Total Time: {total_time_s:.3f}s")
    logger.info(f"Average Latency: {avg_ms:.2f} ms/image | Throughput: {fps:.1f} FPS")
    logger.info("=" * 60)

    output_payload = all_results[0] if len(all_results) == 1 else all_results

    if save_json:
        json_path = Path(save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved pipeline results JSON to: {save_json}")

    return output_payload


def main():
    parser = argparse.ArgumentParser(description="End-to-End High-Performance CCCD Hybrid Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Input image or folder")
    parser.add_argument("--dbnet-config", type=str, default="configs/dbnet/dbnet.yaml", help="DBNet config path")
    parser.add_argument("--dbnet-weights", type=str, default="weights/dbnet/dbnet_cccd_best.pth", help="DBNet weights path")
    parser.add_argument("--yolo-seg-weights", type=str, default="weights/yolo/yolo26_seg_best.pt", help="YOLO-seg weights")
    parser.add_argument("--yolo-cls-weights", type=str, default="weights/yolo/yolo26_cls_best.pt", help="YOLO-cls weights")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for parallel processing")
    parser.add_argument("--min-conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--min-overlap", type=float, default=0.20, help="Minimum overlap ratio for field matching")
    parser.add_argument("--unclip-ratio", type=float, default=None, help="Vatti unclip expansion ratio override (default: from dbnet.yaml)")
    parser.add_argument("--box-thresh", type=float, default=None, help="DBNet box score threshold override (default: from dbnet.yaml)")
    parser.add_argument("--device", type=str, default="", help="Device (mps, cuda, cpu)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 half precision")
    parser.add_argument("--no-warmup", action="store_true", help="Disable warmup")
    parser.add_argument("--save-json", type=str, default="runs/pipeline/result.json", help="Output JSON path")
    parser.add_argument("--save-vis", type=str, default=None, help="Optional output visualization directory")
    args = parser.parse_args()

    predict_pipeline(
        source=args.source,
        dbnet_config=args.dbnet_config,
        dbnet_weights=args.dbnet_weights,
        yolo_seg_weights=args.yolo_seg_weights,
        yolo_cls_weights=args.yolo_cls_weights,
        batch_size=args.batch_size,
        min_conf=args.min_conf,
        min_overlap=args.min_overlap,
        unclip_ratio=args.unclip_ratio,
        box_thresh=args.box_thresh,
        device=args.device,
        fp16=args.fp16,
        warmup=not args.no_warmup,
        save_json=args.save_json,
        save_vis=args.save_vis,
    )


if __name__ == "__main__":
    main()
