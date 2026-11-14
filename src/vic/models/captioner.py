"""Encoder + decoder wrapper and the model factory."""

from __future__ import annotations

import torch
from torch import nn

from vic.models.decoder import DECODERS
from vic.models.encoder import BACKBONES, ImageEncoder


class Captioner(nn.Module):
    """A decoder with an optional image encoder.

    Training on cached features needs no encoder (``encoder=None``); pass ``images`` instead of
    ``features`` to run the encoder on the fly (fine-tuning, inference on new images).
    """

    def __init__(self, decoder: nn.Module, encoder: ImageEncoder | None = None) -> None:
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("This model was built without an image encoder")
        return self.encoder(images)

    def forward(
        self,
        input_ids: torch.Tensor,
        features: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            features = self.encode(images)
        return self.decoder(features, input_ids)


def build_model(
    model_config: dict, vocab_size: int, pad_id: int, with_encoder: bool, pretrained: bool = True
) -> Captioner:
    encoder_config = model_config["encoder"]
    decoder_config = dict(model_config["decoder"])
    decoder_type = decoder_config.pop("type")
    if decoder_type not in DECODERS:
        raise ValueError(f"Unknown decoder {decoder_type!r}; available: {sorted(DECODERS)}")

    feature_dim = BACKBONES[encoder_config["name"]][2]
    decoder = DECODERS[decoder_type](
        vocab_size=vocab_size, feature_dim=feature_dim, pad_id=pad_id, **decoder_config
    )
    encoder = None
    if with_encoder:
        encoder = ImageEncoder(
            name=encoder_config["name"],
            pretrained=pretrained,
            trainable=bool(encoder_config.get("trainable", False)),
        )
    return Captioner(decoder, encoder)
