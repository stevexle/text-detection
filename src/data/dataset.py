"""
Text Detection Dataset loading real image annotations.
Supports JSONL / JSON annotations and deterministic dynamic Train/Val splitting.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.db_target_generator import DBTargetGenerator
from src.data.transforms import (
    ColorJitter,
    Compose,
    NormalizeImage,
    RandomPerspective,
    RandomRotate,
    Resize,
    ToTensor,
)
from src.utils.registry import Registry

DATASETS = Registry("dataset")


@DATASETS.register_module(name="TextDetectionDataset")
@DATASETS.register_module(name="text_detection_dataset")
class TextDetectionDataset(Dataset):
    """Dataset for Text Detection training and evaluation."""

    def __init__(
        self,
        data_root: Optional[str] = None,
        data_dir: Optional[str] = None,
        anno_file: Optional[str] = None,
        ann_file: Optional[str] = None,
        target_size: Union[Tuple[int, int], List[int]] = (640, 640),
        is_training: bool = True,
        val_ratio: float = 0.0,
        split: Optional[str] = None,
        split_seed: int = 42,
        shrink_ratio: float = 0.4,
        min_thresh: float = 0.3,
        max_thresh: float = 0.7,
        transforms: Optional[Sequence[Any]] = None,
        **kwargs,
    ):
        super().__init__()
        root = data_root or data_dir
        self.data_root = Path(root) if root else None

        file_path = anno_file or ann_file
        if file_path:
            p = Path(file_path)
            self.anno_file = p if p.is_absolute() or self.data_root is None else self.data_root / p
        else:
            raise ValueError("`ann_file` must be provided to load annotations.")

        if not self.anno_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.anno_file}")

        self.is_training = is_training
        self.target_size = target_size
        self.val_ratio = val_ratio
        self.split = split
        self.split_seed = split_seed

        self.target_generator = DBTargetGenerator(
            shrink_ratio=shrink_ratio,
            min_thresh=min_thresh,
            max_thresh=max_thresh,
        )

        self.samples: List[Dict[str, Any]] = []
        self._load_annotations()
        if self.val_ratio > 0.0 or self.split is not None:
            self._apply_dynamic_split()

        # Build transform pipeline
        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            if self.is_training:
                self.transforms = Compose([
                    Resize(size=target_size),
                    RandomRotate(max_angle=10.0, prob=0.4),
                    RandomPerspective(distortion_scale=0.12, prob=0.3),
                    ColorJitter(brightness=0.2, contrast=0.2, prob=0.4),
                    NormalizeImage(),
                    ToTensor(),
                ])
            else:
                self.transforms = Compose([
                    Resize(size=target_size),
                    NormalizeImage(),
                    ToTensor(),
                ])

    def _load_annotations(self):
        """Load annotations from JSON or JSONL file."""
        if self.anno_file.suffix == ".jsonl":
            with open(self.anno_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    raw_path = item.get("img_path") or item.get("image_path")
                    img_p = str(self.data_root / raw_path) if self.data_root and not Path(raw_path).is_absolute() else raw_path
                    polygons = [np.array(poly, dtype=np.float32) for poly in item.get("polygons", [])]
                    ignore_tags = item.get("ignore_tags", [False] * len(polygons))
                    texts = item.get("texts", [""] * len(polygons))
                    self.samples.append({
                        "image_path": img_p,
                        "polygons": polygons,
                        "ignore_tags": ignore_tags,
                        "texts": texts,
                    })
            return

        with open(self.anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "images" in data:
            for item in data["images"]:
                self.samples.append({
                    "image_path": str(self.data_root / item["file_name"]) if self.data_root else item["file_name"],
                    "polygons": [np.array(ann["polygon"], dtype=np.float32) for ann in item.get("annotations", [])],
                    "ignore_tags": [ann.get("ignore", False) for ann in item.get("annotations", [])],
                    "texts": [ann.get("text", "") for ann in item.get("annotations", [])],
                })
        elif isinstance(data, list):
            for item in data:
                raw_path = item.get("img_path") or item.get("image_path")
                img_p = str(self.data_root / raw_path) if self.data_root and not Path(raw_path).is_absolute() else raw_path
                polygons = [np.array(poly, dtype=np.float32) for poly in item.get("polygons", [])]
                ignore_tags = item.get("ignore_tags", [False] * len(polygons))
                texts = item.get("texts", [""] * len(polygons))
                self.samples.append({
                    "image_path": img_p,
                    "polygons": polygons,
                    "ignore_tags": ignore_tags,
                    "texts": texts,
                })

    def _apply_dynamic_split(self):
        """Split samples deterministically into train and validation partitions in memory."""
        if not self.samples:
            return

        rng = random.Random(self.split_seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)

        ratio = self.val_ratio if self.val_ratio > 0.0 else 0.15
        val_size = max(1, int(len(self.samples) * ratio))
        val_indices = set(indices[:val_size])

        target_split = self.split if self.split is not None else ("train" if self.is_training else "val")
        if target_split == "val":
            self.samples = [self.samples[i] for i in range(len(self.samples)) if i in val_indices]
        else:
            self.samples = [self.samples[i] for i in range(len(self.samples)) if i not in val_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        polygons = [p.copy() for p in sample["polygons"]]
        ignore_tags = list(sample["ignore_tags"])
        texts = list(sample["texts"])

        data = {
            "image": img,
            "polygons": polygons,
            "ignore_tags": ignore_tags,
            "texts": texts,
            "image_path": image_path,
        }

        # Apply spatial & photometric transforms
        data = self.transforms(data)

        # Generate DBNet target maps if training
        if self.is_training:
            target_h, target_w = data["image"].shape[1], data["image"].shape[2]
            target_maps = self.target_generator.generate_targets(
                image_shape=(target_h, target_w),
                polygons=data["polygons"],
                ignore_tags=data["ignore_tags"],
            )
            for k, v in target_maps.items():
                data[k] = torch.from_numpy(v[None, :, :].astype(np.float32))

        return data
