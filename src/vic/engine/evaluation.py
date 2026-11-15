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
from vic.engine.protocol import evaluation_filename
from vic.metrics import caption_statistics, compute_metrics
from vic.models import Captioner, beam_search, greedy_decode
from vic.text.tokenizer import detokenize, to_syllables
from vic.text.vocab import Vocabulary
from vic.utils import get_device, write_json

TEST_LOCK_MESSAGE = (
    "The test split is locked: it is evaluated once, for the final model. Choose models, "
    "hyper-parameters and decoding on the validation split; pass --final (final=True) only for "
    "the final report."
)


def decoding_settings(eval_config: dict, **overrides) -> dict:
    """Decoding arguments of ``generate_captions`` from the ``eval`` section of a config."""
    settings = {
        "beam_size": int(eval_config.get("beam_size", 3)),
        "length_penalty": float(eval_config.get("length_penalty", 0.7)),
        "no_repeat_ngram_size": int(eval_config.get("no_repeat_ngram_size", 0)),
        "beam_stop": str(eval_config.get("beam_stop", "first")),
    }
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return settings


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
    no_repeat_ngram_size: int = 0,
    beam_stop: str = "first",
) -> dict[str, list[str]]:
    """Return generated tokens per image id (greedy when ``beam_size`` <= 1).

    ``no_repeat_ngram_size`` and ``beam_stop`` are described in ``vic.models.decoding``.
    """
    model.eval()
    suppress_ids = (vocab.pad_id, vocab.bos_id, vocab.unk_id)
    hypotheses: dict[str, list[str]] = {}
    for visuals, image_ids in loader:
        visuals = visuals.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            features = visuals if use_features else model.encode(visuals)
            if beam_size <= 1:
                sequences = greedy_decode(
                    model.decoder, features, vocab.bos_id, vocab.eos_id, max_length,
                    suppress_ids, no_repeat_ngram_size,
                )  # fmt: skip
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
                        no_repeat_ngram_size,
                        beam_stop,
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
    final: bool = False,
    decoding: dict | None = None,
) -> dict:
    """Caption every image of ``split`` with a checkpoint, score the captions, save the result.

    The result (scores, naturalness diagnostics and all generated captions) is written to
    ``output``, by default ``<run>/<evaluation_filename(split, decoding)>`` next to the
    checkpoint directory, and returned. ``decoding`` overrides the ``eval`` settings of the
    checkpoint config (see ``decoding_settings``); ``beam_size`` overrides both. The test split
    is locked: it is refused unless ``final`` is set, because every model, hyper-parameter and
    decoding choice must be made on the validation split.
    """
    if split == "test" and not final:
        raise PermissionError(TEST_LOCK_MESSAGE)
    checkpoint = load_checkpoint(checkpoint_path)
    config = apply_overrides(checkpoint["config"], overrides)
    eval_config = config.get("eval", {})
    requested = dict(decoding or {})
    if beam_size is not None:
        requested["beam_size"] = beam_size
    decoding = decoding_settings(eval_config, **requested)
    beam_size = decoding["beam_size"]

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
        max_length=int(config["text"].get("max_length", 30)),
        use_amp=device.type == "cuda",
        **decoding,
    )
    references = build_references(records)
    scores = score_captions(hypotheses, references)
    diagnostics = caption_statistics(to_syllables(tokens) for tokens in hypotheses.values())
    reference_diagnostics = caption_statistics(
        caption for captions in references.values() for caption in captions
    )

    if output is None:
        run_dir = Path(checkpoint_path).resolve().parent.parent
        output = run_dir / evaluation_filename(split, decoding)
    result = {
        "checkpoint": str(checkpoint_path),
        "dataset": config["dataset"]["name"],
        "split": split,
        "num_images": len(records),
        "beam_size": beam_size,
        "decoding": decoding,
        "training_progress": checkpoint["progress"],
        "scores": scores,
        "diagnostics": diagnostics,
        "reference_diagnostics": reference_diagnostics,
        "predictions": {key: detokenize(tokens) for key, tokens in hypotheses.items()},
    }
    write_json(result, output, indent=1)
    return {**result, "output": str(output)}


def study_decoding(
    checkpoint_path: str | Path,
    settings: list[dict],
    output: str | Path | None = None,
    overrides: list[str] | None = None,
) -> list[dict]:
    """Score several decoding settings of one checkpoint on the validation split.

    Each setting is a dict with any of ``beam_size``, ``length_penalty``, ``beam_stop`` and
    ``no_repeat_ngram_size``; missing keys come from the checkpoint config. Decoding is chosen
    on validation only, so there is no split argument. Rows (settings, scores, diagnostics)
    are written to ``output``, by default ``<run>/decoding_study.json``, after every setting.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    config = apply_overrides(checkpoint["config"], overrides)
    eval_config = config.get("eval", {})
    device = get_device(config.get("device", "auto"))
    data = load_experiment_data(config, vocab=Vocabulary(checkpoint["vocab"]))
    model, vocab, _ = model_from_checkpoint(checkpoint, not data.use_features, device)
    loader = make_image_loader(data, "val", int(eval_config.get("batch_size", 128)))
    references = build_references(data.records["val"])
    if output is None:
        output = Path(checkpoint_path).resolve().parent.parent / "decoding_study.json"

    rows = []
    for setting in settings:
        decoding = decoding_settings(eval_config, **setting)
        hypotheses = generate_captions(
            model,
            loader,
            vocab,
            device,
            data.use_features,
            max_length=int(config["text"].get("max_length", 30)),
            use_amp=device.type == "cuda",
            **decoding,
        )
        rows.append(
            {
                **decoding,
                "scores": score_captions(hypotheses, references),
                "diagnostics": caption_statistics(
                    to_syllables(tokens) for tokens in hypotheses.values()
                ),
            }
        )
        write_json({"checkpoint": str(checkpoint_path), "split": "val", "rows": rows}, output, 1)
    return rows


def format_scores(scores: dict[str, float]) -> str:
    """One line for logs: BLEU / METEOR as percentages, CIDEr on its native 0-10 scale."""
    parts = []
    for name, value in scores.items():
        parts.append(f"{name} {value * 100:.2f}" if name != "CIDEr" else f"CIDEr {value:.4f}")
    return " | ".join(parts)
