"""
Evaluation script for YOLO26 Document Classifier.
Evaluates accuracy, per-class metrics, and confusion matrix on the validation split.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from tqdm import tqdm

from src.models.wrappers.yolo_classifier import YOLOClassifier
from src.utils.config import Config
from src.utils.logger import get_logger
from tools.train_yolo_cls import create_dynamic_cls_staging

logger = get_logger("EvalYOLOCls")


def evaluate_classifier(
    model_path: str,
    anno_file: str = "data/labels/dbnet/dataset.jsonl",
    val_ratio: float = 0.15,
    split_seed: int = 42,
    device: str = "",
    imgsz: int = 224,
) -> Dict[str, float]:
    """
    Evaluate trained classifier on validation split.
    """
    logger.info(f"Loading classifier from {model_path}...")
    classifier = YOLOClassifier(model_path=model_path, device=device if device else None)

    # 1. Create/Ensure dynamic staging exists
    staging_dir = create_dynamic_cls_staging(
        anno_file=anno_file,
        val_ratio=val_ratio,
        split_seed=split_seed,
    )
    val_dir = staging_dir / "val"

    classes = ["back_2021", "back_2024", "front_2021", "front_2024"]
    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []

    logger.info(f"Running high-throughput batch evaluation on validation set in {val_dir}...")
    cls_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    for d in tqdm(cls_dirs, desc="[Evaluating Classes]", unit="class"):
        cls_name = d.name
        img_files = list(d.glob("*.*"))
        if not img_files:
            continue
        batch_results = classifier.classify_batch(img_files, batch_size=32, min_conf=0.0, imgsz=imgsz)
        for res in batch_results:
            y_true.append(cls_name)
            y_pred.append(res["card_type"])
            confidences.append(res["confidence"])

    # Compute metrics
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    logger.info("=" * 60)
    logger.info(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    logger.info("-" * 60)
    for c in classes:
        metrics = report.get(c, {})
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0.0)
        sup = metrics.get("support", 0)
        logger.info(f"{c:<15} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {sup:<8}")

    logger.info("-" * 60)
    accuracy = report.get("accuracy", 0.0)
    macro_f1 = report.get("macro avg", {}).get("f1-score", 0.0)
    logger.info(f"{'Accuracy':<15} {accuracy:.4f}")
    logger.info(f"{'Macro F1':<15} {macro_f1:.4f}")
    logger.info(f"{'Avg Confidence':<15} {np.mean(confidences):.4f}")
    logger.info("=" * 60)

    logger.info("Confusion Matrix:")
    logger.info(f"Classes order: {classes}")
    for row in cm:
        logger.info(str(row))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_confidence": float(np.mean(confidences)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 Document Classifier")
    parser.add_argument("--config", type=str, default="configs/yolo/yolo_cls.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (defaults to config)")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="", help="Device (mps, cuda, cpu)")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    weights = args.weights or cfg.train.save_best_to or cfg.model.model_path
    imgsz = args.imgsz or cfg.data.imgsz

    evaluate_classifier(
        model_path=weights,
        anno_file=cfg.data.anno_file,
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
        device=args.device,
        imgsz=imgsz,
    )


if __name__ == "__main__":
    main()
