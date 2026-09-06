"""
Unit tests for DBNet Training Engine, Optimizers, Schedulers, and Checkpoints.
"""

from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine.trainer import DBNetTrainer, build_lr_scheduler, build_optimizer
from src.losses.db_loss import DBLoss
from src.metrics.evaluator import ICDAREvaluator
from src.models.detectors.dbnet import DBNet
from src.postprocess.db_postprocessor import DBPostProcessor


class DummyDataset(torch.utils.data.Dataset):
    """Minimal dataset returning valid DBNet batches for testing."""

    def __init__(self, size: int = 4):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": torch.zeros(3, 64, 64, dtype=torch.float32),
            "gt_prob_map": torch.zeros(64, 64, dtype=torch.float32),
            "gt_prob_mask": torch.ones(64, 64, dtype=torch.float32),
            "gt_thresh_map": torch.zeros(64, 64, dtype=torch.float32),
            "gt_thresh_mask": torch.ones(64, 64, dtype=torch.float32),
            "polygons": [],
            "ignore_tags": [],
            "texts": [],
            "shape": (64, 64),
        }


def dummy_collate(batch):
    elem = batch[0]
    collated = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch], dim=0)
        else:
            collated[key] = [d[key] for d in batch]
    return collated


class TestEngine(unittest.TestCase):
    """Test optimizer, scheduler, and DBNetTrainer pipeline."""

    def setUp(self):
        self.model = DBNet(
            backbone=dict(type="MobileNetV3", arch="small", pretrained=False),
            neck=dict(type="FPN", in_channels=[16, 24, 48, 576], inner_channels=64),
            head=dict(type="DBHead", k=50.0),
        )
        self.loss_fn = DBLoss()
        self.postprocessor = DBPostProcessor()
        self.evaluator = ICDAREvaluator()

    def test_build_optimizer(self):
        opt_adamw = build_optimizer(self.model, dict(type="AdamW", lr=0.001))
        self.assertIsInstance(opt_adamw, torch.optim.AdamW)

        opt_adam = build_optimizer(self.model, dict(type="Adam", lr=0.0005))
        self.assertIsInstance(opt_adam, torch.optim.Adam)

        opt_sgd = build_optimizer(self.model, dict(type="SGD", lr=0.01))
        self.assertIsInstance(opt_sgd, torch.optim.SGD)

    def test_build_lr_scheduler(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        sched = build_lr_scheduler(opt, epochs=10, warmup_epochs=2)
        self.assertIsNotNone(sched)

        # Optimizer step then scheduler step
        opt.step()
        sched.step()
        self.assertTrue(opt.param_groups[0]["lr"] > 0)

    def test_trainer_train_epoch_and_save_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            train_loader = DataLoader(DummyDataset(size=2), batch_size=2, collate_fn=dummy_collate)
            val_loader = DataLoader(DummyDataset(size=2), batch_size=1, collate_fn=dummy_collate)

            optimizer = build_optimizer(self.model, dict(type="AdamW", lr=0.001))
            scheduler = build_lr_scheduler(optimizer, epochs=5, warmup_epochs=1)

            trainer = DBNetTrainer(
                model=self.model,
                loss_fn=self.loss_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                postprocessor=self.postprocessor,
                evaluator=self.evaluator,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                epochs=2,
                device="cpu",
                work_dir=str(tmp_p / "work_dir"),
                best_weight_dest=str(tmp_p / "best.pth"),
                use_amp=False,
            )

            # 1. Train 1 epoch
            metrics = trainer.train_epoch(epoch=1)
            self.assertIn("loss", metrics)
            self.assertIn("lr", metrics)
            self.assertTrue(metrics["loss"] >= 0.0)

            # 2. Save checkpoint
            trainer.save_checkpoint(epoch=1, is_best=True)
            best_ckpt_path = tmp_p / "work_dir" / "best_hmean.pth"
            self.assertTrue(best_ckpt_path.exists())
            self.assertTrue((tmp_p / "best.pth").exists())

            # 3. Resume from checkpoint
            trainer_new = DBNetTrainer(
                model=self.model,
                loss_fn=self.loss_fn,
                train_loader=train_loader,
                optimizer=optimizer,
                device="cpu",
                work_dir=str(tmp_p / "work_dir"),
            )
            trainer_new.resume(str(best_ckpt_path))
            self.assertEqual(trainer_new.start_epoch, 2)


if __name__ == "__main__":
    unittest.main()
