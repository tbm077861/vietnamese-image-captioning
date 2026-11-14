"""Turn raw annotations into tokenised captions, fixed splits and a vocabulary."""

from __future__ import annotations

import random
import statistics

from vic.text.tokenizer import CaptionTokenizer
from vic.text.vocab import Vocabulary


def assign_missing_splits(
    records: list[dict], val_ratio: float, test_ratio: float, seed: int
) -> list[dict]:
    """Carve a val and / or test split out of train when the dataset does not ship one.

    Splitting is by image (all captions of an image stay together) and depends only on the
    seed and the image ids, not on the order of the annotation file.
    """
    existing = {record["split"] for record in records}
    train_ids = sorted(record["image_id"] for record in records if record["split"] == "train")
    random.Random(seed).shuffle(train_ids)
    num_test = round(len(train_ids) * test_ratio) if "test" not in existing else 0
    num_val = round(len(train_ids) * val_ratio) if "val" not in existing else 0
    reassigned = dict.fromkeys(train_ids[:num_test], "test")
    reassigned.update(dict.fromkeys(train_ids[num_test : num_test + num_val], "val"))
    return [
        {**record, "split": reassigned.get(record["image_id"], record["split"])}
        for record in records
    ]


def tokenize_records(records: list[dict], tokenizer: CaptionTokenizer) -> list[dict]:
    """Attach the normalised text and tokens to every caption; empty captions are dropped."""
    prepared = []
    for record in records:
        captions = []
        for raw in record["captions"]:
            tokens = tokenizer.tokenize(raw)
            if tokens:
                captions.append(
                    {"raw": raw, "normalized": tokenizer.normalize(raw), "tokens": tokens}
                )
        if captions:
            prepared.append({**record, "captions": captions})
    return prepared


def build_vocabulary(
    records: list[dict], min_freq: int = 1, max_size: int | None = None
) -> Vocabulary:
    """Build the vocabulary from the training split only."""
    train_tokens = (
        caption["tokens"]
        for record in records
        if record["split"] == "train"
        for caption in record["captions"]
    )
    return Vocabulary.build(train_tokens, min_freq=min_freq, max_size=max_size)


def caption_statistics(records: list[dict], vocab: Vocabulary, max_length: int) -> dict:
    """Per-split counts, caption lengths (in tokens) and out-of-vocabulary rates."""
    stats = {}
    for split in ("train", "val", "test"):
        lengths, unknown, total = [], 0, 0
        images = [record for record in records if record["split"] == split]
        for record in images:
            for caption in record["captions"]:
                lengths.append(len(caption["tokens"]))
                total += len(caption["tokens"])
                unknown += sum(token not in vocab.stoi for token in caption["tokens"])
        if not lengths:
            continue
        ordered = sorted(lengths)
        stats[split] = {
            "images": len(images),
            "captions": len(lengths),
            "mean_length": round(statistics.fmean(lengths), 2),
            "p95_length": ordered[int(0.95 * (len(ordered) - 1))],
            "max_length": ordered[-1],
            "truncated_ratio": round(sum(n > max_length for n in lengths) / len(lengths), 4),
            "oov_ratio": round(unknown / max(1, total), 4),
        }
    return stats
