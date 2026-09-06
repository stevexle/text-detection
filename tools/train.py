"""
1-Click Training Script for DBNet Text Detection.
Supports Mixed Precision (AMP), Dynamic Train/Val Split, Cosine LR, and Auto Checkpointing.
"""

import argparse
import os
from pathlib import Path
import random
import numpy as np
import torch

from src.data.builder import build_dataloader
from src.engine.trainer import DBNetTrainer, build_lr_scheduler, build_optimizer
from src.losses.builder import build_loss
from src.metrics.evaluator import ICDAREvaluator
from src.models.builder import build_model
from src.postprocess.builder import build_postprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("TrainDBNet")


def set_seed(seed: int = 42):
    """Set random seed for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train DBNet Text Detector on CCCD")
    parser.add_argument("--config", type=str, default="configs/dbnet/dbnet.yaml", help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="", help="Device to use (mps, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--work-dir", type=str, default="work_dirs/dbnet", help="Directory to save checkpoints and logs")
    args = parser.parse_args()

    # 1. Load config
    cfg = Config.fromfile(args.config)
    set_seed(args.seed)

    # 2. Overrides
    epochs = args.epochs or cfg.get("epochs", 50)
    batch_size = args.batch_size or cfg.data.batch_size
    lr = args.lr or cfg.optimizer.lr
    opt_cfg = dict(cfg.optimizer)
    opt_cfg["lr"] = lr

    logger.info("=" * 70)
    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Training parameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, Seed={args.seed}")
    logger.info("=" * 70)

    # 3. Build Model, Loss, PostProcessor, Evaluator
    model = build_model(cfg.model)
    loss_fn = build_loss(cfg.loss)
    postprocessor = build_postprocessor(cfg.postprocess)
    evaluator = ICDAREvaluator(iou_thresh=0.5)

    # 4. Build DataLoaders (Deterministic In-Memory Split)
    train_dataset_cfg = dict(
        type="TextDetectionDataset",
        anno_file=cfg.data.ann_file,
        data_root=cfg.data.data_dir,
        is_train=True,
        target_size=tuple(cfg.data.target_size),
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
    )
    val_dataset_cfg = dict(
        type="TextDetectionDataset",
        anno_file=cfg.data.ann_file,
        data_root=cfg.data.data_dir,
        is_train=False,
        target_size=tuple(cfg.data.target_size),
        val_ratio=cfg.data.val_ratio,
        split_seed=cfg.data.split_seed,
    )

    train_loader = build_dataloader(
        train_dataset_cfg,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = build_dataloader(
        val_dataset_cfg,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # 5. Build Optimizer & LR Scheduler
    optimizer = build_optimizer(model, opt_cfg)
    lr_scheduler = build_lr_scheduler(optimizer, epochs=epochs, warmup_epochs=3)

    # 6. Instantiate Trainer
    trainer = DBNetTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        postprocessor=postprocessor,
        evaluator=evaluator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        epochs=epochs,
        device=args.device if args.device else None,
        work_dir=args.work_dir,
        best_weight_dest="weights/dbnet/dbnet_cccd_best.pth",
        cfg=dict(cfg),
    )

    # 7. Resume if specified
    if args.resume:
        trainer.resume(args.resume)

    # 8. Start Training
    trainer.fit()


if __name__ == "__main__":
    main()
