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
from vic.engine.protocol import decoding_label, is_protocol_decoding
from vic.metrics import caption_statistics, cider_d
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
        if not is_protocol_decoding(item.get("decoding")):
            decoding = decoding_label(item["decoding"])
        cells = [_number(item["scores"][name], name != "CIDEr") for name in SCORE_COLUMNS]
        lines.append(f"| {item['split']} | {decoding} | " + " | ".join(cells) + " |")
    return lines


def diagnostics(evaluation: dict, references: dict[str, list[str]]) -> list[str]:
    """Length, diversity and repetition statistics of the generated captions."""
    predictions = evaluation["predictions"]
    generated = caption_statistics(caption.split() for caption in predictions.values())
    human = caption_statistics(ref.split() for key in predictions for ref in references[key])
    most_common = Counter(predictions.values()).most_common(3)
    lines = [
        f"- Số ảnh: {generated['num_captions']}.",
        f"- Độ dài trung bình (âm tiết): câu sinh ra {generated['mean_length']:.1f}, câu tham "
        f"chiếu {human['mean_length']:.1f}.",
        f"- Số câu sinh ra khác nhau: {generated['distinct_captions']} trên "
        f"{generated['num_captions']}.",
        f"- Số âm tiết khác nhau được dùng: {generated['distinct_tokens']}, trong khi câu tham "
        f"chiếu dùng {human['distinct_tokens']}.",
        f"- Câu có cụm 2 âm tiết lặp lại: {100 * generated['repeated_bigram_rate']:.1f}% "
        f"(câu tham chiếu: {100 * human['repeated_bigram_rate']:.1f}%).",
        f"- Câu có cụm 3 âm tiết lặp lại: {100 * generated['repeated_trigram_rate']:.1f}% "
        f"(câu tham chiếu: {100 * human['repeated_trigram_rate']:.1f}%).",
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
    by_key = {  # analyses and examples use the comparison decoding
        (item["split"], item["beam_size"]): item
        for item in evaluations
        if is_protocol_decoding(item.get("decoding"))
    }

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


def build_comparison(
    run_dirs: list[str | Path],
    split: str = "val",
    beam_size: int = 3,
    samples: int = 50,
    seed: int = 42,
    max_references: int = 1,
) -> str:
    """Captions of several runs for the same fixed sample of images, side by side.

    The sample depends only on ``seed`` and the split, so every comparison shows the same
    images. Meant for reading by eye (is a higher-scoring model still natural?).
    """
    run_dirs = [Path(run_dir) for run_dir in run_dirs]
    config = load_yaml(run_dirs[0] / "config.yaml")
    images = read_json(data_paths(config).captions)["images"]
    raw = {r["image_id"]: [c["raw"] for c in r["captions"]] for r in images}
    evaluations = [read_json(run / f"eval_{split}_beam{beam_size}.json") for run in run_dirs]
    image_ids = sorted(evaluations[0]["predictions"])
    for run_dir, evaluation in zip(run_dirs, evaluations, strict=True):
        if sorted(evaluation["predictions"]) != image_ids:
            raise ValueError(f"{run_dir} was evaluated on a different set of images")
    chosen = random.Random(seed).sample(image_ids, min(samples, len(image_ids)))

    names = ", ".join(f"`{run.name}`" for run in run_dirs)
    lines = [
        f"# So sánh caption: {names}",
        "",
        f"{len(chosen)} ảnh {split} cố định (seed {seed}), giải mã beam {beam_size}.",
        "",
    ]
    for image_id in chosen:
        lines += [f"**Ảnh `{image_id}`**", ""]
        for run_dir, evaluation in zip(run_dirs, evaluations, strict=True):
            lines.append(f"- `{run_dir.name}`: *{evaluation['predictions'][image_id]}*")
        lines += [f"- Tham chiếu: {caption}" for caption in raw[image_id][:max_references]]
        lines.append("")
    return "\n".join(lines)
