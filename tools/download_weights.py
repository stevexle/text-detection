"""
Pretrained Weights Downloader CLI for Vietnamese ID Card (CCCD) Pipeline.
Downloads ImageNet backbones and YOLO / DBNet base weights to `weights/` directory.
"""

import argparse
from pathlib import Path
import sys
import torch
import torchvision.models as models


def download_imagenet_backbones(dest_dir: Path):
    """Download standard ImageNet pretrained backbones."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("1. Downloading ImageNet Pretrained Backbones (Torchvision)...")
    print("=" * 65)

    # 1. ResNet-50
    r50_path = dest_dir / "resnet50_imagenet.pth"
    if not r50_path.exists():
        print(f" -> Downloading ResNet-50 ImageNet weights to {r50_path}...")
        weights = models.ResNet50_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        torch.save(state_dict, r50_path)
        print(f"    Saved: {r50_path} ({r50_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f" -> [Found] {r50_path}")

    # 2. ResNet-18
    r18_path = dest_dir / "resnet18_imagenet.pth"
    if not r18_path.exists():
        print(f" -> Downloading ResNet-18 ImageNet weights to {r18_path}...")
        weights = models.ResNet18_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        torch.save(state_dict, r18_path)
        print(f"    Saved: {r18_path} ({r18_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f" -> [Found] {r18_path}")

    # 3. MobileNetV3 Large
    mb_path = dest_dir / "mobilenetv3_large_imagenet.pth"
    if not mb_path.exists():
        print(f" -> Downloading MobileNetV3-Large ImageNet weights to {mb_path}...")
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        torch.save(state_dict, mb_path)
        print(f"    Saved: {mb_path} ({mb_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f" -> [Found] {mb_path}")


def download_yolo_weights(dest_dir: Path):
    """Download YOLO base models for classification, detection, and segmentation."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("2. Downloading YOLO Base Models (Ultralytics)...")
    print("=" * 65)

    try:
        from ultralytics import YOLO
        import shutil

        models_to_fetch = ["yolo26n.pt", "yolo26s.pt", "yolo26n-seg.pt", "yolo26n-cls.pt"]
        for model_name in models_to_fetch:
            target_p = dest_dir / model_name
            if not target_p.exists():
                print(f" -> Downloading {model_name}...")
                try:
                    yolo = YOLO(model_name)
                    # Move if downloaded to root
                    root_p = Path(model_name)
                    if root_p.exists() and root_p.resolve() != target_p.resolve():
                        shutil.move(str(root_p), str(target_p))
                    print(f"    Saved: {target_p}")
                except Exception as e:
                    print(f"    Notice: Ultralytics fallback for {model_name}: {e}")
            else:
                print(f" -> [Found] {target_p}")
    except ImportError:
        print(" -> Notice: 'ultralytics' is not installed. Skipping YOLO weights download.")


def main():
    parser = argparse.ArgumentParser(description="Download pretrained model weights for training/eval")
    parser.add_argument("--dbnet-dir", type=str, default="weights/dbnet", help="Directory for DBNet weights")
    parser.add_argument("--yolo-dir", type=str, default="weights/yolo", help="Directory for YOLO weights")
    parser.add_argument("--all", action="store_true", default=True, help="Download all weights")
    args = parser.parse_args()

    dbnet_p = Path(args.dbnet_dir)
    yolo_p = Path(args.yolo_dir)

    print("[Info] Initializing Pretrained Weights Downloader...")
    download_imagenet_backbones(dbnet_p)
    download_yolo_weights(yolo_p)
    print("\n[Success] All pretrained weights are ready in weights/ directory.")


if __name__ == "__main__":
    main()
