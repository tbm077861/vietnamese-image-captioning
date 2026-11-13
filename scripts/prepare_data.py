"""Normalise and tokenise captions, fix the splits and build the vocabulary.

Usage:
    python scripts/prepare_data.py --config configs/baseline_lstm.yaml

Reads the unified annotations written by download_data.py and writes captions.json, vocab.json
and stats.json under data/processed/<dataset>/<text setting>/.
"""

from __future__ import annotations

import argparse

from vic.config import data_paths, load_config
from vic.data.annotations import load_annotations
from vic.data.prepare import (
    assign_missing_splits,
    build_vocabulary,
    caption_statistics,
    tokenize_records,
)
from vic.text.tokenizer import CaptionTokenizer
from vic.utils import setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. text.level=syllable")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config, args.overrides)
    paths = data_paths(config)
    text = config["text"]
    seed = int(config.get("seed", 42))
    split_config = config["dataset"].get("splits", {})

    annotations = load_annotations(paths.annotations)
    records = assign_missing_splits(
        annotations["images"],
        val_ratio=float(split_config.get("val_ratio", 0.1)),
        test_ratio=float(split_config.get("test_ratio", 0.1)),
        seed=seed,
    )
    tokenizer = CaptionTokenizer(
        level=text.get("level", "word"),
        lowercase=bool(text.get("lowercase", True)),
        strip_punctuation=bool(text.get("strip_punctuation", True)),
    )
    logger.info(
        f"Tokenising {sum(len(r['captions']) for r in records)} captions ({tokenizer.level})"
    )
    records = tokenize_records(records, tokenizer)
    vocab = build_vocabulary(records, int(text.get("min_freq", 1)), text.get("max_vocab_size"))
    stats = {
        "dataset": annotations["dataset"],
        "tokenizer": tokenizer.to_config(),
        "split_seed": seed,
        "min_freq": int(text.get("min_freq", 1)),
        "vocab_size": len(vocab),
        "splits": caption_statistics(records, vocab, int(text.get("max_length", 30))),
    }

    write_json(
        {"dataset": annotations["dataset"], "tokenizer": tokenizer.to_config(), "images": records},
        paths.captions,
    )
    vocab.save(paths.vocab)
    write_json(stats, paths.processed_dir / "stats.json", indent=2)

    logger.info(f"Vocabulary: {len(vocab)} tokens (min_freq={stats['min_freq']})")
    for split, values in stats["splits"].items():
        logger.info(f"{split:>5}: {values}")
    logger.info(f"Wrote {paths.captions} and {paths.vocab}")


if __name__ == "__main__":
    main()
