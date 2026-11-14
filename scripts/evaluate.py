"""Evaluate a checkpoint on a dataset split: BLEU-1..4, CIDEr (primary), METEOR (reference).

Usage:
    python scripts/evaluate.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --split test
    python scripts/evaluate.py --checkpoint <ckpt> --split val test --beam-size 1 3 5

Several splits and beam sizes may be given; every combination is evaluated. Scores and the
generated captions are written next to the run: <run>/eval_<split>_beam<k>.json.
"""

from __future__ import annotations

import argparse

from vic.engine.evaluation import evaluate_checkpoint, format_scores
from vic.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test"])
    parser.add_argument(
        "--beam-size", nargs="+", type=int, default=[None], help="1 = greedy; default: config"
    )
    parser.add_argument("--output", default=None, help="result file (single evaluation only)")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data_root=...")
    args = parser.parse_args()
    if args.output and len(args.split) * len(args.beam_size) > 1:
        parser.error("--output can only be used with one split and one beam size")

    logger = setup_logging()
    for split in args.split:
        for beam_size in args.beam_size:
            result = evaluate_checkpoint(
                args.checkpoint, split, beam_size, args.output, args.overrides
            )
            logger.info(
                f"{result['dataset']} / {split} / {result['num_images']} images / "
                f"beam {result['beam_size']} | {format_scores(result['scores'])}"
            )
            logger.info(f"Wrote {result['output']}")
    logger.info("BLEU and CIDEr are the primary metrics; METEOR is exact-match only (reference).")


if __name__ == "__main__":
    main()
