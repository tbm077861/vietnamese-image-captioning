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


def to_rgb(image: Image.Image) -> Image.Image:
    """Return an upright RGB image.

    The EXIF orientation is applied (phone photos end up upright) and transparent areas are
    shown on white, as an image viewer would, instead of exposing the hidden colour under them.
    """
    image = ImageOps.exif_transpose(image)
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return to_rgb(image)


def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Resize the whole image to a square (no cropping) and normalise it: (3, size, size)."""
    image = to_rgb(image).resize((image_size, image_size), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std
