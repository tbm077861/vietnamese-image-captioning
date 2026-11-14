"""Pretrained CNN image encoder producing a grid of region features."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

# name -> (constructor, pretrained weights, feature channels)
BACKBONES = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2, 2048),
}


class ImageEncoder(nn.Module):
    """ResNet without its pooling and classification head.

    Output: (batch, regions, channels); a 224 px input gives 7 x 7 = 49 regions.
    A frozen encoder always stays in eval mode so its BatchNorm statistics never change.
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True, trainable: bool = False):
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"Unknown encoder {name!r}; available: {sorted(BACKBONES)}")
        constructor, weights, feature_dim = BACKBONES[name]
        backbone = constructor(weights=weights if pretrained else None)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.name = name
        self.feature_dim = feature_dim
        self.trainable = trainable
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        if not trainable:
            self.eval()

    def train(self, mode: bool = True) -> ImageEncoder:
        return super().train(mode and self.trainable)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            feature_map = self.body(images)
        return feature_map.flatten(2).transpose(1, 2)
