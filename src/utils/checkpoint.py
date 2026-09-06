"""
Checkpoint Management and Pretrained Weight Loading Utilities.
Supports non-strict, flexible state_dict mapping for transfer learning.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model checkpoint and training metadata."""
    save_path = Path(filepath)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k[7:] if k.startswith("module.") else k
        clean_state_dict[clean_key] = v.cpu()

    checkpoint = {
        "epoch": epoch,
        "state_dict": clean_state_dict,
        "meta": meta or {},
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    return save_path


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint and restore weights/optimizer state."""
    ckpt_path = Path(filepath)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    device = device or torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    if model is not None and "state_dict" in checkpoint:
        load_pretrained_weights(model, checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint


def load_pretrained_weights(
    model: nn.Module,
    weights_or_path: Union[str, Path, Dict[str, Any]],
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Flexibly load pretrained weights into model with key matching and shape validation.
    Handles 'state_dict', 'model', prefix stripping (module.), and non-matching head keys.
    """
    if isinstance(weights_or_path, (str, Path)):
        p = Path(weights_or_path)
        if not p.exists():
            raise FileNotFoundError(f"Pretrained weights file not found: {p}")
        try:
            data = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(p, map_location="cpu", weights_only=False)
    else:
        data = weights_or_path

    # Extract state dict from container dictionary if wrapped
    if isinstance(data, dict):
        if "state_dict" in data:
            raw_dict = data["state_dict"]
        elif "model" in data:
            raw_dict = data["model"]
        else:
            raw_dict = data
    else:
        raw_dict = data

    model_dict = model.state_dict()
    matched_dict = {}
    mismatched_keys = []

    for k, v in raw_dict.items():
        key = k[7:] if k.startswith("module.") else k

        # Match exact key or try stripped prefix
        target_key = None
        if key in model_dict:
            target_key = key
        elif f"backbone.{key}" in model_dict:
            target_key = f"backbone.{key}"

        if target_key:
            if model_dict[target_key].shape == v.shape:
                matched_dict[target_key] = v
            else:
                mismatched_keys.append((target_key, model_dict[target_key].shape, v.shape))

    # Load matched parameters
    model.load_state_dict(matched_dict, strict=False)
    loaded_count = len(matched_dict)
    total_count = len(model_dict)
    pct = (loaded_count / total_count * 100) if total_count > 0 else 0

    msg = f"Loaded {loaded_count}/{total_count} ({pct:.1f}%) parameters into model."
    if logger:
        logger.info(msg)
    else:
        print(f"[Checkpoint] {msg}")

    return loaded_count
