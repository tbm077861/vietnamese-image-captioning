"""Caption a single image with a trained checkpoint (used by the CLI and, later, the demo)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from vic.data.image import load_image, preprocess_image
from vic.engine.checkpoint import load_checkpoint, model_from_checkpoint
from vic.models import beam_search, greedy_decode
from vic.text.tokenizer import detokenize
from vic.utils import get_device


class CaptionPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = get_device(device)
        checkpoint = load_checkpoint(checkpoint_path)
        self.model, self.vocab, self.config = model_from_checkpoint(
            checkpoint, with_encoder=True, device=self.device
        )
        self.image_size = int(self.config["model"]["encoder"].get("image_size", 224))
        eval_config = self.config.get("eval", {})
        self.default_beam_size = int(eval_config.get("beam_size", 3))
        self.length_penalty = float(eval_config.get("length_penalty", 0.7))
        self.max_length = int(self.config["text"].get("max_length", 30))

    @torch.no_grad()
    def caption(self, image: Image.Image | str | Path, beam_size: int | None = None) -> str:
        """Return a Vietnamese caption for a PIL image or an image path."""
        if not isinstance(image, Image.Image):
            image = load_image(image)
        beam_size = self.default_beam_size if beam_size is None else beam_size
        vocab = self.vocab
        suppress_ids = (vocab.pad_id, vocab.bos_id, vocab.unk_id)
        pixels = preprocess_image(image, self.image_size).unsqueeze(0).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
            features = self.model.encode(pixels)
            if beam_size <= 1:
                token_ids = greedy_decode(
                    self.model.decoder, features, vocab.bos_id, vocab.eos_id, self.max_length,
                    suppress_ids,
                )[0]  # fmt: skip
            else:
                token_ids = beam_search(
                    self.model.decoder, features, vocab.bos_id, vocab.eos_id, beam_size,
                    self.max_length, self.length_penalty, suppress_ids,
                )  # fmt: skip
        return detokenize(vocab.decode(token_ids))
