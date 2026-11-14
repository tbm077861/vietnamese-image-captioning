"""Checkpoint saving / loading.

A checkpoint is self-contained for inference: it stores the experiment config, the vocabulary
and the decoder weights. Encoder weights are stored only when the encoder was trained;
a frozen encoder is rebuilt from the pretrained weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from vic.models import Captioner, build_model
from vic.text.vocab import Vocabulary


def save_checkpoint(
    path: str | Path,
    *,
    model: Captioner,
    config: dict,
    vocab: Vocabulary,
    progress: dict,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder_trained = model.encoder is not None and model.encoder.trainable
    payload = {
        "config": config,
        "vocab": vocab.to_list(),
        "progress": progress,
        "decoder": model.decoder.state_dict(),
        "encoder": model.encoder.state_dict() if encoder_trained else None,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    # Write to a temporary file first so an interrupted save never corrupts the checkpoint.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} not found")
    return torch.load(path, map_location=map_location, weights_only=True)


def model_from_checkpoint(
    checkpoint: dict, with_encoder: bool, device: torch.device
) -> tuple[Captioner, Vocabulary, dict]:
    """Rebuild the model for evaluation or inference."""
    config = checkpoint["config"]
    vocab = Vocabulary(checkpoint["vocab"])
    model = build_model(config["model"], len(vocab), vocab.pad_id, with_encoder=with_encoder)
    model.decoder.load_state_dict(checkpoint["decoder"])
    if with_encoder and checkpoint.get("encoder") is not None:
        model.encoder.load_state_dict(checkpoint["encoder"])
    return model.to(device).eval(), vocab, config
