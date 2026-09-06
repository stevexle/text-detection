"""
Data Builders for Datasets and DataLoaders.
"""

from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset
from src.data.dataset import DATASETS


def build_dataset(cfg: Dict[str, Any], **default_args) -> Dataset:
    """Build a dataset from configuration dictionary using DATASETS registry."""
    return DATASETS.build(cfg, **default_args)


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to handle variable number of polygons/texts across batch.
    Tensor fields are stacked; polygon lists and string metadata are kept as lists.
    """
    elem = batch[0]
    collated = {}

    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch], dim=0)
        else:
            collated[key] = [d[key] for d in batch]

    return collated


def build_dataloader(
    dataset_cfg: Dict[str, Any],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = False,
    **kwargs,
) -> DataLoader:
    """Build PyTorch DataLoader from dataset configuration."""
    dataset = build_dataset(dataset_cfg)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
        **kwargs,
    )
