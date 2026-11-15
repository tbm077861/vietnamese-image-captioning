"""Pretrained image encoders producing a grid of region features.

Three families share one wrapper: ResNet (torchvision), and the vision towers of CLIP and SigLIP
(``transformers``). Licences and sources of the weights are recorded in docs/MODELS.md; the
Hugging Face revisions are pinned so that every download fetches the files that were checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class EncoderSpec:
    family: str  # "resnet", "clip" or "siglip"
    feature_dim: int  # channels of one region feature
    normalization: str  # key of vic.data.image.NORMALIZATION
    source: object  # torchvision weights, or a Hugging Face repository id
    revision: str | None = None  # pinned commit of the Hugging Face repository
    native_size: int = 224  # image size the weights were trained with
    patch_size: int | None = None  # None: convolutional, 32 px per region


ENCODERS = {
    "resnet50": EncoderSpec(
        "resnet", 2048, "imagenet", (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2)
    ),
    "resnet101": EncoderSpec(
        "resnet", 2048, "imagenet", (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V2)
    ),
    "clip_vit_b16": EncoderSpec(
        "clip",
        768,
        "clip",
        "openai/clip-vit-base-patch16",
        revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
        patch_size=16,
    ),
    "siglip_b16": EncoderSpec(
        "siglip",
        768,
        "siglip",
        "google/siglip-base-patch16-224",
        revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        patch_size=16,
    ),
}


def encoder_spec(name: str) -> EncoderSpec:
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder {name!r}; available: {sorted(ENCODERS)}")
    return ENCODERS[name]


def _build_body(spec: EncoderSpec, pretrained: bool) -> nn.Module:
    if spec.family == "resnet":
        constructor, weights = spec.source
        backbone = constructor(weights=weights if pretrained else None)
        return nn.Sequential(*list(backbone.children())[:-2])

    # Imported here so that ResNet-only use does not pay for importing transformers.
    from transformers import (
        CLIPVisionConfig,
        CLIPVisionModel,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from transformers.utils import logging as transformers_logging

    # Loading a vision tower from a full checkpoint reports the unused text weights; that is
    # expected, so keep only real errors.
    transformers_logging.set_verbosity_error()
    model_class, config_class = {
        "clip": (CLIPVisionModel, CLIPVisionConfig),
        "siglip": (SiglipVisionModel, SiglipVisionConfig),
    }[spec.family]
    if pretrained:
        return model_class.from_pretrained(spec.source, revision=spec.revision)
    config = config_class(
        hidden_size=spec.feature_dim, patch_size=spec.patch_size, image_size=spec.native_size
    )
    return model_class(config)  # random weights with the real architecture (tests)


class ImageEncoder(nn.Module):
    """A pretrained backbone without its pooling / classification head.

    Output: (batch, regions, channels).

    - ResNet: the last feature map; a 224 px input gives 7 x 7 = 49 regions of 2048 channels.
    - CLIP: patch tokens of the penultimate layer, without the class token. The last layer is
      trained only through the class token, so the layer before it keeps more local detail.
    - SigLIP: patch tokens of the last layer (the model has no class token).

    A ViT-B/16 at 224 px gives 14 x 14 = 196 regions of 768 channels. A frozen encoder always
    stays in eval mode so its normalisation statistics never change.
    """

    def __init__(self, name: str = "resnet50", pretrained: bool = True, trainable: bool = False):
        super().__init__()
        self.spec = encoder_spec(name)
        self.body = _build_body(self.spec, pretrained)
        self.name = name
        self.feature_dim = self.spec.feature_dim
        self.trainable = trainable
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        if not trainable:
            self.eval()

    def train(self, mode: bool = True) -> ImageEncoder:
        return super().train(mode and self.trainable)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            if self.spec.family == "resnet":
                return self.body(images).flatten(2).transpose(1, 2)
            # Position embeddings are interpolated for sizes other than the native one.
            resized = images.shape[-1] != self.spec.native_size
            if self.spec.family == "clip":
                output = self.body(
                    pixel_values=images, output_hidden_states=True, interpolate_pos_encoding=resized
                )
                return output.hidden_states[-2][:, 1:]
            output = self.body(pixel_values=images, interpolate_pos_encoding=resized)
            return output.last_hidden_state
