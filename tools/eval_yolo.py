"""
Evaluation script for YOLO26 Field Detection & Segmentation on CCCD.
Evaluates Precision, Recall, mAP50, and mAP50-95 on the validation split.
"""

import argparse
from pathlib import Path
from typing import Any, Dict
from ultralytics import YOLO

from src.utils.logger import get_logger
from tools.train_yolo import create_dynamic_yolo_staging

logger = get_logger("EvalYOLO")


def evaluate_yolo(
    weights: str = "weights/yolo/yolo26_seg_best.pt",
    config: str = "configs/yolo/yolo_seg.yaml",
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "",
) -> Dict[str, Any]:
    """
    Run evaluation on the validation split using dynamic staging.
    """
    p_weights = Path(weights)
    if not p_weights.exists() and Path(f"weights/yolo/{weights}").exists():
        p_weights = Path(f"weights/yolo/{weights}")

    if not p_weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    task = "segment" if "seg" in str(config).lower() else "detect"
    logger.info(f"Loading YOLO {task} model from {p_weights}...")
    model = YOLO(str(p_weights), task=task)

    # 1. Ensure dynamic staging exists for validation
    staged_data_yaml = create_dynamic_yolo_staging(
        config_path=config,
        val_ratio=0.15,
        split_seed=42,
    )

    # 2. Run validation
    logger.info(f"Evaluating model on validation set with imgsz={imgsz}, batch={batch_size}...")
    metrics = model.val(
        data=str(staged_data_yaml.resolve()),
        imgsz=imgsz,
        batch=batch_size,
        device=device if device else "",
        verbose=True,
    )

    # 3. Log results summary
    logger.info("=" * 60)
    logger.info("YOLO Evaluation Summary:")
    logger.info("-" * 60)

    results_dict = {}
    if task == "segment" and hasattr(metrics, "seg"):
        mp = float(metrics.seg.mp)
        mr = float(metrics.seg.mr)
        map50 = float(metrics.seg.map50)
        map95 = float(metrics.seg.map)
        logger.info(f"{'Mask Precision':<20}: {mp:.4f}")
        logger.info(f"{'Mask Recall':<20}: {mr:.4f}")
        logger.info(f"{'Mask mAP50':<20}: {map50:.4f}")
        logger.info(f"{'Mask mAP50-95':<20}: {map95:.4f}")
        results_dict = {"precision": mp, "recall": mr, "map50": map50, "map": map95}
    elif hasattr(metrics, "box"):
        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map95 = float(metrics.box.map)
        logger.info(f"{'Box Precision':<20}: {mp:.4f}")
        logger.info(f"{'Box Recall':<20}: {mr:.4f}")
        logger.info(f"{'Box mAP50':<20}: {map50:.4f}")
        logger.info(f"{'Box mAP50-95':<20}: {map95:.4f}")
        results_dict = {"precision": mp, "recall": mr, "map50": map50, "map": map95}

    logger.info("=" * 60)
    return results_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Field Detection / Segmentation")
    parser.add_argument("--config", type=str, default="configs/yolo/yolo_seg.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="weights/yolo/yolo26_seg_best.pt", help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image resolution")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device to use (mps, cuda, cpu)")
    args = parser.parse_args()

    evaluate_yolo(
        weights=args.weights,
        config=args.config,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
