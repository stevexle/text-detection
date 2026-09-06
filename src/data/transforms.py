"""
Data Transforms and Augmentations for Text Detection.
Supports spatial transforms (Resize, Rotate, Perspective, Flip) with polygon coordinate syncing,
and photometric transforms (ColorJitter, Normalization, ToTensor).
"""

from typing import Any, Dict, List, Sequence, Tuple, Union
import random
import cv2
import numpy as np
import torch


class Compose:
    """Compose multiple transforms together."""

    def __init__(self, transforms: Sequence[Any]):
        self.transforms = transforms

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


class Resize:
    """Resize image and adjust polygon coordinates."""

    def __init__(self, size: Union[int, Tuple[int, int], List[int]]):
        if isinstance(size, int):
            self.target_w = size
            self.target_h = size
        else:
            self.target_w = size[0]
            self.target_h = size[1]

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data["image"]
        h, w = img.shape[:2]

        if (w, h) == (self.target_w, self.target_h):
            return data

        scale_x = self.target_w / float(w)
        scale_y = self.target_h / float(h)

        data["image"] = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        if "polygons" in data:
            new_polygons = []
            for poly in data["polygons"]:
                p = poly.copy()
                p[:, 0] = p[:, 0] * scale_x
                p[:, 1] = p[:, 1] * scale_y
                new_polygons.append(p)
            data["polygons"] = new_polygons

        return data


class RandomRotate:
    """Randomly rotate image and polygon coordinates around center."""

    def __init__(self, max_angle: float = 10.0, prob: float = 0.5):
        self.max_angle = max_angle
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.prob:
            return data

        img = data["image"]
        h, w = img.shape[:2]
        angle = random.uniform(-self.max_angle, self.max_angle)
        center = (w / 2.0, h / 2.0)

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        data["image"] = cv2.warpAffine(img, rot_mat, (w, h), borderValue=(240, 240, 240))

        if "polygons" in data:
            new_polygons = []
            for poly in data["polygons"]:
                # Homogeneous coordinates (N, 3)
                ones = np.ones((len(poly), 1), dtype=np.float32)
                pts = np.hstack([poly, ones])
                rotated_pts = np.dot(rot_mat, pts.T).T
                # Clip to image boundaries
                rotated_pts[:, 0] = np.clip(rotated_pts[:, 0], 0, w)
                rotated_pts[:, 1] = np.clip(rotated_pts[:, 1], 0, h)
                new_polygons.append(rotated_pts.astype(np.float32))
            data["polygons"] = new_polygons

        return data


class RandomPerspective:
    """Random 4-corner perspective distortion for realistic camera tilt angles."""

    def __init__(self, distortion_scale: float = 0.15, prob: float = 0.3):
        self.distortion_scale = distortion_scale
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.prob:
            return data

        img = data["image"]
        h, w = img.shape[:2]

        dx = int(w * self.distortion_scale)
        dy = int(h * self.distortion_scale)

        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([
            [random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), h - random.randint(0, dy)],
            [random.randint(0, dx), h - random.randint(0, dy)],
        ])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        data["image"] = cv2.warpPerspective(img, matrix, (w, h), borderValue=(240, 240, 240))

        if "polygons" in data:
            new_polygons = []
            for poly in data["polygons"]:
                pts = poly.reshape(-1, 1, 2).astype(np.float32)
                warped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
                warped[:, 0] = np.clip(warped[:, 0], 0, w)
                warped[:, 1] = np.clip(warped[:, 1], 0, h)
                new_polygons.append(warped)
            data["polygons"] = new_polygons

        return data


class ColorJitter:
    """Random brightness and contrast jitter."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, prob: float = 0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.prob = prob

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.prob:
            return data

        img = data["image"].astype(np.float32)
        if self.brightness > 0:
            alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * alpha

        if self.contrast > 0:
            beta = random.uniform(-self.contrast * 128, self.contrast * 128)
            img = img + beta

        data["image"] = np.clip(img, 0, 255).astype(np.uint8)
        return data


class NormalizeImage:
    """Normalize RGB image using ImageNet mean and std."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data["image"].astype(np.float32) / 255.0
        data["image"] = (img - self.mean) / self.std
        return data


class ToTensor:
    """Convert numpy HWC image to PyTorch CHW FloatTensor."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data["image"]
        if img.ndim == 2:
            img = img[:, :, None]
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        data["image"] = torch.from_numpy(img).float()
        return data
