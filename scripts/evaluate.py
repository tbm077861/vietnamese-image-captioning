"""Evaluate a checkpoint on a dataset split: BLEU-1..4, CIDEr (primary), METEOR (reference).

Usage:
    python scripts/evaluate.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --split test

Scores and the generated captions are written next to the run:
<run>/eval_<split>_beam<k>.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vic.config import _parse_overrides, _set_by_path
from vic.engine.checkpoint import load_checkpoint, model_from_checkpoint
from vic.engine.data import load_experiment_data, make_image_loader
from vic.engine.evaluation import build_references, format_scores, generate_captions, score_captions
from vic.text.tokenizer import detokenize
from vic.text.vocab import Vocabulary
from vic.utils import get_device, setup_logging, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--beam-size", type=int, default=None, help="default: eval.beam_size")
    parser.add_argument("--output", default=None, help="result file (default: inside the run)")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data_root=...")
    args = parser.parse_args()

    logger = setup_logging()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    for key, value in _parse_overrides(args.overrides):
        _set_by_path(config, key, value)
    eval_config = config.get("eval", {})
    beam_size = (
        args.beam_size if args.beam_size is not None else int(eval_config.get("beam_size", 3))
    )

    device = get_device(config.get("device", "auto"))
    data = load_experiment_data(config, vocab=Vocabulary(checkpoint["vocab"]))
    model, vocab, _ = model_from_checkpoint(checkpoint, not data.use_features, device)
    records = data.records[args.split]
    if not records:
        raise SystemExit(f"The dataset has no {args.split!r} split")

    loader = make_image_loader(data, args.split, int(eval_config.get("batch_size", 128)))
    hypotheses = generate_captions(
        model, loader, vocab, device, data.use_features,
        beam_size=beam_size,
        max_length=int(config["text"].get("max_length", 30)),
        length_penalty=float(eval_config.get("length_penalty", 0.7)),
        use_amp=device.type == "cuda",
    )  # fmt: skip
    scores = score_captions(hypotheses, build_references(records))

    output = args.output or (
        Path(args.checkpoint).resolve().parent.parent / f"eval_{args.split}_beam{beam_size}.json"
    )
    write_json(
        {
            "checkpoint": str(args.checkpoint),
            "dataset": config["dataset"]["name"],
            "split": args.split,
            "num_images": len(records),
            "beam_size": beam_size,
            "training_progress": checkpoint["progress"],
            "scores": scores,
            "predictions": {key: detokenize(tokens) for key, tokens in hypotheses.items()},
        },
        output,
        indent=1,
    )
    logger.info(
        f"{config['dataset']['name']} / {args.split} / {len(records)} images / beam {beam_size}"
    )
    logger.info(format_scores(scores))
    logger.info("BLEU and CIDEr are the primary metrics; METEOR is exact-match only (reference).")
    logger.info(f"Wrote {output}")


if __name__ == "__main__":
    main()
