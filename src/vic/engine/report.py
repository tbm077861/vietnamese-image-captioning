"""Summarise a training run as Markdown: learning curve, scores, diagnostics, sample captions.

Reads metrics.jsonl and every eval_<split>_beam<k>.json written by evaluate.py. Sample images
are drawn with a fixed seed, so the same call always shows the same examples.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

from vic.config import data_paths, load_yaml
from vic.metrics import cider_d
from vic.utils import read_json

SCORE_COLUMNS = ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR", "CIDEr")


def _number(value: float, percent: bool) -> str:
    return f"{value * 100:.2f}" if percent else f"{value:.4f}"


def learning_curve(run_dir: Path) -> list[str]:
    """Per-epoch table from metrics.jsonl; the train loss is the mean of the logged windows."""
    train_losses: dict[int, list[float]] = {}
    validations = []
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["type"] == "train":
            train_losses.setdefault(record["epoch"], []).append(record["loss"])
        elif record["type"] == "val":
            validations.append(record)
    scored = [record for record in validations if "CIDEr" in record]
    best_epoch = max(scored, key=lambda record: record["CIDEr"])["epoch"] if scored else None

    lines = [
        "| Epoch | Loss train | Loss val | BLEU-4 val | CIDEr val | |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for record in validations:
        epoch = record["epoch"]
        train_loss = statistics.fmean(train_losses[epoch]) if epoch in train_losses else None
        lines.append(
            f"| {epoch} | {train_loss:.3f} | {record['val_loss']:.3f} | "
            f"{_number(record.get('BLEU-4', 0.0), True)} | "
            f"{_number(record.get('CIDEr', 0.0), False)} | "
            f"{'tốt nhất' if epoch == best_epoch else ''} |"
        )
    return lines


def load_evaluations(run_dir: Path) -> list[dict]:
    evaluations = [read_json(path) for path in sorted(run_dir.glob("eval_*_beam*.json"))]
    order = {"val": 0, "test": 1, "train": 2}
    return sorted(evaluations, key=lambda item: (order[item["split"]], item["beam_size"]))


def score_table(evaluations: list[dict]) -> list[str]:
    lines = [
        "| Tập | Giải mã | " + " | ".join(SCORE_COLUMNS) + " |",
        "|---|---|" + "---:|" * len(SCORE_COLUMNS),
    ]
    for item in evaluations:
        decoding = "greedy" if item["beam_size"] <= 1 else f"beam {item['beam_size']}"
        cells = [_number(item["scores"][name], name != "CIDEr") for name in SCORE_COLUMNS]
        lines.append(f"| {item['split']} | {decoding} | " + " | ".join(cells) + " |")
    return lines


def diagnostics(evaluation: dict, references: dict[str, list[str]]) -> list[str]:
    """Length, diversity and repetition statistics of the generated captions."""
    predictions = evaluation["predictions"]
    hypotheses = [caption.split() for caption in predictions.values()]
    reference_tokens = [ref.split() for key in predictions for ref in references[key]]
    generated_words = {word for tokens in hypotheses for word in tokens}
    reference_words = {word for tokens in reference_tokens for word in tokens}
    repeated = sum(
        any(a == b for a, b in zip(tokens, tokens[1:], strict=False)) for tokens in hypotheses
    )
    repeated_share = 100 * repeated / len(hypotheses)
    most_common = Counter(predictions.values()).most_common(3)
    lines = [
        f"- Số ảnh: {len(hypotheses)}.",
        f"- Độ dài trung bình (âm tiết): câu sinh ra "
        f"{statistics.fmean(len(t) for t in hypotheses):.1f}, câu tham chiếu "
        f"{statistics.fmean(len(t) for t in reference_tokens):.1f}.",
        f"- Số câu sinh ra khác nhau: {len(set(predictions.values()))} trên {len(hypotheses)}.",
        f"- Số âm tiết khác nhau được dùng: {len(generated_words)}, trong khi câu tham chiếu "
        f"dùng {len(reference_words)}.",
        f"- Câu có một âm tiết lặp liền nhau: {repeated} ({repeated_share:.1f}%).",
        "- Ba câu xuất hiện nhiều nhất:",
    ]
    lines += [f'  - {count} lần: "{caption}"' for caption, count in most_common]
    return lines


def per_image_cider(evaluation: dict, references: dict[str, list[str]]) -> dict[str, float]:
    image_ids = sorted(evaluation["predictions"])
    refs = [[ref.split() for ref in references[key]] for key in image_ids]
    hyps = [evaluation["predictions"][key].split() for key in image_ids]
    _, scores = cider_d(refs, hyps)
    return dict(zip(image_ids, scores, strict=True))


def example_block(
    image_id: str, evaluation: dict, raw: dict, cider: float, max_references: int = 1
) -> list[str]:
    """One example: the generated caption and the first ``max_references`` reference captions.

    Reference captions are dataset text, so published reports quote as few as needed.
    """
    lines = [
        f"**Ảnh `{image_id}`** (CIDEr {cider:.2f})",
        "",
        f"- Sinh ra: *{evaluation['predictions'][image_id]}*",
    ]
    lines += [f"- Tham chiếu: {caption}" for caption in raw[image_id][:max_references]]
    return [*lines, ""]


def build_report(
    run_dir: str | Path,
    samples: int = 10,
    sample_split: str = "test",
    analysis_split: str = "val",
    beam_size: int = 3,
    worst: int = 0,
    seed: int = 42,
    max_references: int = 1,
) -> str:
    """Return the Markdown report of a run.

    ``samples`` random images of ``sample_split`` are shown with their generated caption and
    their first ``max_references`` reference captions; diagnostics (and optionally the
    ``worst`` lowest-CIDEr images) use ``analysis_split``. Both use the evaluation decoded
    with ``beam_size``. Scores always use every reference caption.
    """
    run_dir = Path(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    images = read_json(data_paths(config).captions)["images"]
    references = {r["image_id"]: [c["normalized"] for c in r["captions"]] for r in images}
    raw = {r["image_id"]: [c["raw"] for c in r["captions"]] for r in images}
    evaluations = load_evaluations(run_dir)
    by_key = {(item["split"], item["beam_size"]): item for item in evaluations}

    lines = [f"# Kết quả: {config['run_name']}", "", "## Đường cong học", ""]
    lines += learning_curve(run_dir)
    lines += ["", "## Điểm số", ""] + score_table(evaluations)

    analysis = by_key.get((analysis_split, beam_size))
    if analysis is not None:
        lines += ["", f"## Chẩn đoán trên {analysis_split} (beam {beam_size})", ""]
        lines += diagnostics(analysis, references)
        if worst > 0:
            scores = per_image_cider(analysis, references)
            lines += ["", f"### {worst} ảnh {analysis_split} có CIDEr thấp nhất", ""]
            for image_id in sorted(scores, key=scores.get)[:worst]:
                lines += example_block(image_id, analysis, raw, scores[image_id], max_references)

    sampled = by_key.get((sample_split, beam_size))
    if sampled is not None and samples > 0:
        scores = per_image_cider(sampled, references)
        chosen = random.Random(seed).sample(sorted(sampled["predictions"]), samples)
        lines += [
            "",
            f"## {samples} ảnh {sample_split} chọn ngẫu nhiên (seed {seed}, beam {beam_size})",
            "",
        ]
        for image_id in chosen:
            lines += example_block(image_id, sampled, raw, scores[image_id], max_references)

    return "\n".join(lines) + "\n"
