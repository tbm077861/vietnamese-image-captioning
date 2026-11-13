"""Image loading and preprocessing shared by training, feature extraction and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Some dataset files miss their last few bytes; decode them anyway instead of failing mid-run.
# scripts/verify_data.py lists such files as "truncated".
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(path: str | Path) -> Image.Image:
    """Open an image as RGB with its EXIF orientation applied (phone photos end up upright)."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Resize the whole image to a square (no cropping) and normalise it: (3, size, size)."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std
