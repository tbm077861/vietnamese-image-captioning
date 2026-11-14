"""Caption generation over a dataset split and metric computation.

Metrics are computed on syllables: word-level tokens are split back ("sân_bóng" -> "sân bóng")
and references come from the normalised captions. Scores therefore do not depend on the word
segmenter and stay comparable between models using different tokenisation levels.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vic.config import apply_overrides
from vic.engine.checkpoint import load_checkpoint, model_from_checkpoint
from vic.engine.data import load_experiment_data, make_image_loader
from vic.metrics import compute_metrics
from vic.models import Captioner, beam_search, greedy_decode
from vic.text.tokenizer import detokenize, to_syllables
from vic.text.vocab import Vocabulary
from vic.utils import get_device, write_json


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


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str = "test",
    beam_size: int | None = None,
    output: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Caption every image of ``split`` with a checkpoint, score the captions, save the result.

    The result (scores and all generated captions) is written to ``output``, by default
    ``<run>/eval_<split>_beam<k>.json`` next to the checkpoint directory, and returned.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    config = apply_overrides(checkpoint["config"], overrides)
    eval_config = config.get("eval", {})
    if beam_size is None:
        beam_size = int(eval_config.get("beam_size", 3))

    device = get_device(config.get("device", "auto"))
    data = load_experiment_data(config, vocab=Vocabulary(checkpoint["vocab"]))
    records = data.records[split]
    if not records:
        raise ValueError(f"The dataset has no {split!r} split")
    model, vocab, _ = model_from_checkpoint(checkpoint, not data.use_features, device)
    loader = make_image_loader(data, split, int(eval_config.get("batch_size", 128)))
    hypotheses = generate_captions(
        model,
        loader,
        vocab,
        device,
        data.use_features,
        beam_size=beam_size,
        max_length=int(config["text"].get("max_length", 30)),
        length_penalty=float(eval_config.get("length_penalty", 0.7)),
        use_amp=device.type == "cuda",
    )
    scores = score_captions(hypotheses, build_references(records))

    if output is None:
        run_dir = Path(checkpoint_path).resolve().parent.parent
        output = run_dir / f"eval_{split}_beam{beam_size}.json"
    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": config["dataset"]["name"],
        "split": split,
        "num_images": len(records),
        "beam_size": beam_size,
        "training_progress": checkpoint["progress"],
        "scores": scores,
        "predictions": {key: detokenize(tokens) for key, tokens in hypotheses.items()},
    }
    write_json(result, output, indent=1)
    return {**result, "output": str(output)}


def format_scores(scores: dict[str, float]) -> str:
    """One line for logs: BLEU / METEOR as percentages, CIDEr on its native 0-10 scale."""
    parts = []
    for name, value in scores.items():
        parts.append(f"{name} {value * 100:.2f}" if name != "CIDEr" else f"CIDEr {value:.4f}")
    return " | ".join(parts)
