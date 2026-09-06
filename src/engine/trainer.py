"""
Comprehensive Training Engine for DBNet Text Detection.
Supports Mixed Precision (AMP FP16), Cosine Annealing with Warmup, Gradient Clipping,
and Automatic ICDAR 2015 Validation Checkpointing.
"""

import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler
from torch.utils.data import DataLoader

from tqdm import tqdm

from src.losses.builder import build_loss
from src.metrics.evaluator import ICDAREvaluator
from src.models.builder import build_model
from src.postprocess.builder import build_postprocessor
from src.utils.logger import get_logger
from src.utils.registry import Registry

ENGINE = Registry("engine")


def build_optimizer(model: nn.Module, opt_cfg: Dict[str, Any]) -> Optimizer:
    """Build optimizer from configuration."""
    opt_type = opt_cfg.get("type", "AdamW").lower()
    lr = opt_cfg.get("lr", 0.001)
    weight_decay = opt_cfg.get("weight_decay", 0.0001)

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_type in ["adamw", "adam_w"]:
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        momentum = opt_cfg.get("momentum", 0.9)
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def build_lr_scheduler(
    optimizer: Optimizer,
    epochs: int,
    warmup_epochs: int = 3,
    min_lr: float = 1e-6,
) -> _LRScheduler:
    """Build Cosine Annealing LR scheduler with linear warmup."""
    if warmup_epochs > 0:
        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            else:
                progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                import math
                return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)


@ENGINE.register_module(name="DBNetTrainer")
@ENGINE.register_module(name="dbnet_trainer")
class DBNetTrainer:
    """
    Production-ready Trainer for DBNet text detector.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        postprocessor: Optional[Any] = None,
        evaluator: Optional[ICDAREvaluator] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        epochs: int = 50,
        device: Optional[str] = None,
        work_dir: str = "work_dirs/dbnet",
        best_weight_dest: str = "weights/dbnet/dbnet_cccd_best.pth",
        grad_clip_norm: float = 5.0,
        eval_interval: int = 1,
        save_interval: int = 5,
        use_amp: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        self.logger = get_logger("DBNetTrainer")
        self.cfg = cfg or {}

        # 1. Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f"Using device: {self.device}")

        # 2. Model, Loss, Loaders
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 3. Postprocessor & Evaluator
        self.postprocessor = postprocessor
        self.evaluator = evaluator or ICDAREvaluator(iou_thresh=0.5)

        # 4. Optimizer & Scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.best_weight_dest = Path(best_weight_dest)
        self.best_weight_dest.parent.mkdir(parents=True, exist_ok=True)

        # 5. Mixed Precision Setup
        self.use_amp = use_amp and (self.device.type in ["cuda", "mps"])
        self.amp_device_type = "cuda" if self.device.type == "cuda" else ("mps" if self.device.type == "mps" else "cpu")
        self.scaler = torch.amp.GradScaler(device=self.amp_device_type, enabled=(self.use_amp and self.device.type == "cuda"))

        self.best_hmean = 0.0
        self.start_epoch = 1

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train model for a single epoch."""
        self.model.train()
        total_loss = 0.0
        total_prob_loss = 0.0
        total_thresh_loss = 0.0
        total_bin_loss = 0.0
        num_batches = len(self.train_loader)

        t0 = time.time()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch:03d}/{self.epochs:03d}]",
            unit="batch",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch tensors to device
            images = batch["image"].to(self.device)
            targets = {
                "gt_prob_map": batch["gt_prob_map"].to(self.device),
                "gt_mask": batch.get("gt_prob_mask", torch.ones_like(batch["gt_prob_map"])).to(self.device),
                "gt_thresh_map": batch["gt_thresh_map"].to(self.device),
                "gt_thresh_mask": batch.get("gt_thresh_mask", torch.ones_like(batch["gt_thresh_map"])).to(self.device),
            }

            self.optimizer.zero_grad()

            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                preds = self.model(images)
                loss_dict = self.loss_fn(preds, targets)
                loss = loss_dict["loss"]

            # Backward pass with scaler
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            loss_val = loss.item()
            prob_loss_val = loss_dict.get("loss_prob", torch.tensor(0.0)).item()
            thresh_loss_val = loss_dict.get("loss_thresh", torch.tensor(0.0)).item()
            bin_loss_val = loss_dict.get("loss_binary", torch.tensor(0.0)).item()

            total_loss += loss_val
            total_prob_loss += prob_loss_val
            total_thresh_loss += thresh_loss_val
            total_bin_loss += bin_loss_val

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "prob": f"{prob_loss_val:.4f}",
                "thresh": f"{thresh_loss_val:.4f}",
                "lr": f"{current_lr:.6f}",
            })

        elapsed = time.time() - t0
        avg_loss = total_loss / max(1, num_batches)
        current_lr = self.optimizer.param_groups[0]["lr"]

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        metrics = {
            "loss": avg_loss,
            "loss_prob": total_prob_loss / max(1, num_batches),
            "loss_thresh": total_thresh_loss / max(1, num_batches),
            "loss_bin": total_bin_loss / max(1, num_batches),
            "lr": current_lr,
            "time": elapsed,
        }
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation DataLoader using ICDAR 2015 protocol."""
        if self.val_loader is None or self.postprocessor is None:
            return {"hmean": 0.0, "precision": 0.0, "recall": 0.0}

        self.model.eval()
        self.evaluator.reset()

        pbar = tqdm(self.val_loader, desc="[Validating DBNet]", unit="batch", leave=False)
        for batch in pbar:
            images = batch["image"].to(self.device)
            gt_polygons_list = batch["polygons"]
            gt_ignore_list = batch.get("ignore_tags", None)
            orig_shapes = batch.get("shape", None)

            preds = self.model(images)
            prob_maps = preds["prob_map"]

            # Process each image in the batch
            for idx in range(len(images)):
                prob_map = prob_maps[idx]
                orig_shape = orig_shapes[idx] if orig_shapes is not None else None
                gt_polygons = gt_polygons_list[idx]
                gt_ignore = gt_ignore_list[idx] if gt_ignore_list is not None else None

                detections = self.postprocessor(prob_map, orig_shape=orig_shape)

                self.evaluator.evaluate_image(
                    detections=detections,
                    gt_polygons=gt_polygons,
                    gt_ignore=gt_ignore,
                )

        val_metrics = self.evaluator.compute_metrics()
        return val_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False, val_metrics: Optional[Dict[str, float]] = None):
        """Save training state checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "best_hmean": self.best_hmean,
            "val_metrics": val_metrics or {},
            "config": self.cfg,
        }

        # 1. Save latest checkpoint
        latest_path = self.work_dir / "latest.pth"
        torch.save(state, latest_path)

        # 2. Save periodic checkpoint
        if epoch % self.save_interval == 0:
            epoch_path = self.work_dir / f"epoch_{epoch}.pth"
            torch.save(state, epoch_path)

        # 3. Save best checkpoint
        if is_best:
            best_path = self.work_dir / "best_hmean.pth"
            torch.save(state, best_path)
            # Copy to weights/dbnet/dbnet_cccd_best.pth
            shutil.copy(best_path, self.best_weight_dest)
            self.logger.info(f"[Best Checkpoint] New best Hmean={self.best_hmean:.4f}! Saved to: {self.best_weight_dest}")

    def resume(self, checkpoint_path: Union[str, Path]):
        """Resume training state from a checkpoint file."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found for resume: {p}")

        self.logger.info(f"Resuming training from checkpoint: {p}")
        checkpoint = torch.load(p, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler and checkpoint.get("scheduler_state_dict"):
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_hmean = checkpoint.get("best_hmean", 0.0)
        self.logger.info(f"Resumed successfully at Epoch {self.start_epoch} (Previous Best Hmean: {self.best_hmean:.4f})")

    def fit(self):
        """Run complete multi-epoch training loop."""
        self.logger.info("=" * 70)
        self.logger.info(f"Starting DBNet Training | Total Epochs: {self.epochs} | Start Epoch: {self.start_epoch}")
        self.logger.info(f"Device: {self.device} | AMP Mixed Precision: {self.use_amp}")
        self.logger.info("=" * 70)

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            log_str = (
                f"Epoch [{epoch:03d}/{self.epochs:03d}] | "
                f"Loss: {train_metrics['loss']:.4f} "
                f"(Prob: {train_metrics['loss_prob']:.4f}, Thresh: {train_metrics['loss_thresh']:.4f}, Bin: {train_metrics['loss_bin']:.4f}) | "
                f"LR: {train_metrics['lr']:.6f} | "
                f"Time: {train_metrics['time']:.2f}s"
            )

            # Validation
            if self.val_loader is not None and (epoch % self.eval_interval == 0 or epoch == self.epochs):
                val_metrics = self.validate()
                hmean = val_metrics["hmean"]
                precision = val_metrics["precision"]
                recall = val_metrics["recall"]

                log_str += f" | Val Precision: {precision:.4f} Recall: {recall:.4f} Hmean: {hmean:.4f}"
                is_best = hmean > self.best_hmean
                if is_best:
                    self.best_hmean = hmean

                self.save_checkpoint(epoch, is_best=is_best, val_metrics=val_metrics)
            else:
                self.save_checkpoint(epoch, is_best=False)

            self.logger.info(log_str)

        self.logger.info("=" * 70)
        self.logger.info(f"Training Completed! Best Validation Hmean: {self.best_hmean:.4f}")
        self.logger.info(f"Production weights saved at: {self.best_weight_dest}")
        self.logger.info("=" * 70)
