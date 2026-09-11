"""
Evaluation CLI Script for DBNet Text Detection on CCCD Dataset.
Measures ICDAR 2015 Protocol (Precision, Recall, Hmean/F1-score) and Inference Latency (FPS).
"""

import argparse
from pathlib import Path
import time
import numpy as np
import torch

from tqdm import tqdm

from src.data.builder import build_dataloader
from src.metrics.evaluator import ICDAREvaluator
from src.models.builder import build_model
from src.postprocess.builder import build_postprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("EvalDBNet")


def evaluate_dbnet(
    config_path: str = "configs/dbnet/dbnet.yaml",
    weights_path: str = "weights/dbnet/dbnet_cccd_best.pth",
    device: str = "",
    iou_thresh: float = 0.5,
    box_thresh: float = None,
    unclip_ratio: float = None,
):
    """Run full evaluation on validation dataset."""
    cfg = Config.fromfile(config_path)

    # 1. Device setup
    if device:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    logger.info(f"Using device: {dev}")

    # 2. Build Model & Load Weights
    model = build_model(cfg.model).to(dev)
    w_path = Path(weights_path)

    if not w_path.exists() and Path(f"work_dirs/dbnet/{weights_path}").exists():
        w_path = Path(f"work_dirs/dbnet/{weights_path}")

    if w_path.exists():
        logger.info(f"Loading checkpoint weights from: {w_path}")
        checkpoint = torch.load(w_path, map_location=dev)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
    else:
        logger.warning(f"Checkpoint not found at '{weights_path}'. Running with initialized weights.")

    model.eval()

    # 3. Postprocessor & Evaluator
    post_cfg = dict(getattr(cfg, "postprocess", {}))
    if "box_thresh" not in post_cfg:
        post_cfg["box_thresh"] = box_thresh if box_thresh is not None else 0.6
    elif box_thresh is not None:
        post_cfg["box_thresh"] = box_thresh

    if "unclip_ratio" not in post_cfg:
        post_cfg["unclip_ratio"] = unclip_ratio if unclip_ratio is not None else 1.75
    elif unclip_ratio is not None:
        post_cfg["unclip_ratio"] = unclip_ratio

    postprocessor = build_postprocessor(post_cfg)
    evaluator = ICDAREvaluator(iou_thresh=iou_thresh)

    # 4. DataLoader
    val_dataset_cfg = dict(
        type="TextDetectionDataset",
        anno_file=cfg.data.ann_file,
        data_root=cfg.data.data_dir,
        is_train=False,
        target_size=tuple(cfg.data.target_size),
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
    )
    val_loader = build_dataloader(
        val_dataset_cfg,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Running evaluation on {len(val_loader)} validation images (IoU Thresh: {iou_thresh})...")

    latencies = []
    t_start = time.time()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Evaluating DBNet]", unit="img")
        for batch in pbar:
            images = batch["image"].to(dev)
            gt_polygons = batch["polygons"][0]
            gt_ignore = batch["ignore_tags"][0] if "ignore_tags" in batch else None
            orig_shape = batch["shape"][0] if "shape" in batch else None

            t0 = time.time()
            preds = model(images)
            prob_map = preds["prob_map"][0]
            detections = postprocessor(prob_map, orig_shape=orig_shape)
            latencies.append(time.time() - t0)

            evaluator.evaluate_image(
                detections=detections,
                gt_polygons=gt_polygons,
                gt_ignore=gt_ignore,
            )

    total_time = time.time() - t_start
    metrics = evaluator.compute_metrics()
    avg_latency = np.mean(latencies) * 1000.0  # ms
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    logger.info("=" * 65)
    logger.info("  ICDAR 2015 Text Detection Evaluation Results")
    logger.info("=" * 65)
    logger.info(f"  Precision:         {metrics['precision'] * 100.0:.2f}%")
    logger.info(f"  Recall:            {metrics['recall'] * 100.0:.2f}%")
    logger.info(f"  Hmean (F1-score):  {metrics['hmean'] * 100.0:.2f}%")
    logger.info("-" * 65)
    logger.info(f"  True Positives (TP):  {metrics['tp']}")
    logger.info(f"  False Positives (FP): {metrics['fp']}")
    logger.info(f"  False Negatives (FN): {metrics['fn']}")
    logger.info(f"  Total Ground Truths:  {metrics['total_gt']}")
    logger.info(f"  Total Predictions:    {metrics['total_pred']}")
    logger.info("-" * 65)
    logger.info(f"  Avg Latency:       {avg_latency:.2f} ms / image")
    logger.info(f"  Inference Speed:   {fps:.1f} FPS")
    logger.info(f"  Total Time:        {total_time:.2f} s")
    logger.info("=" * 65)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate DBNet Text Detector on CCCD")
    parser.add_argument("--config", type=str, default="configs/dbnet/dbnet.yaml", help="Path to config file")
    parser.add_argument("--weights", type=str, default="weights/dbnet/dbnet_cccd_best.pth", help="Path to model weights")
    parser.add_argument("--device", type=str, default="", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for ICDAR matching")
    parser.add_argument("--box-thresh", type=float, default=None, help="Override box score threshold")
    parser.add_argument("--unclip-ratio", type=float, default=None, help="Override Vatti unclip ratio")
    args = parser.parse_args()

    evaluate_dbnet(
        config_path=args.config,
        weights_path=args.weights,
        device=args.device,
        iou_thresh=args.iou_thresh,
        box_thresh=args.box_thresh,
        unclip_ratio=args.unclip_ratio,
    )


if __name__ == "__main__":
    main()
