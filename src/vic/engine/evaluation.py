"""Caption generation over a dataset split and metric computation.

Metrics are computed on syllables: word-level tokens are split back ("sân_bóng" -> "sân bóng")
and references come from the normalised captions. Scores therefore do not depend on the word
segmenter and stay comparable between models using different tokenisation levels.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from vic.metrics import compute_metrics
from vic.models import Captioner, beam_search, greedy_decode
from vic.text.tokenizer import to_syllables
from vic.text.vocab import Vocabulary


def build_references(records: list[dict]) -> dict[str, list[list[str]]]:
    return {
        record["image_id"]: [caption["normalized"].split() for caption in record["captions"]]
        for record in records
    }


@torch.no_grad()
def generate_captions(
    model: Captioner,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    use_features: bool,
    beam_size: int = 1,
    max_length: int = 30,
    length_penalty: float = 0.7,
    use_amp: bool = True,
) -> dict[str, list[str]]:
    """Return generated tokens per image id (greedy when ``beam_size`` <= 1)."""
    model.eval()
    suppress_ids = (vocab.pad_id, vocab.bos_id, vocab.unk_id)
    hypotheses: dict[str, list[str]] = {}
    for visuals, image_ids in loader:
        visuals = visuals.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            features = visuals if use_features else model.encode(visuals)
            if beam_size <= 1:
                sequences = greedy_decode(
                    model.decoder, features, vocab.bos_id, vocab.eos_id, max_length, suppress_ids
                )
            else:
                sequences = [
                    beam_search(
                        model.decoder,
                        features[index : index + 1],
                        vocab.bos_id,
                        vocab.eos_id,
                        beam_size,
                        max_length,
                        length_penalty,
                        suppress_ids,
                    )
                    for index in range(features.size(0))
                ]
        for image_id, token_ids in zip(image_ids, sequences, strict=True):
            hypotheses[image_id] = vocab.decode(token_ids)
    return hypotheses


def score_captions(
    hypotheses: dict[str, list[str]], references: dict[str, list[list[str]]]
) -> dict[str, float]:
    return compute_metrics(
        references, {image_id: to_syllables(tokens) for image_id, tokens in hypotheses.items()}
    )


def format_scores(scores: dict[str, float]) -> str:
    """One line for logs: BLEU / METEOR as percentages, CIDEr on its native 0-10 scale."""
    parts = []
    for name, value in scores.items():
        parts.append(f"{name} {value * 100:.2f}" if name != "CIDEr" else f"CIDEr {value:.4f}")
    return " | ".join(parts)
