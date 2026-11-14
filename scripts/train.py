"""Train a captioning model.

Usage:
    python scripts/train.py --config configs/baseline_lstm.yaml
    python scripts/train.py --config configs/baseline_lstm.yaml --resume auto
    python scripts/train.py --config configs/baseline_lstm.yaml --max-steps 300 run_name=smoke

Outputs go to <output_dir>/<run_name>/: config.yaml, train.log, metrics.jsonl, tensorboard/,
checkpoints/last.pt, checkpoints/best.pt, summary.json. A run directory that already holds
checkpoints is never overwritten silently: use --resume auto, another run_name, or --overwrite.
"""

from __future__ import annotations

import argparse

from vic.config import load_config
from vic.engine.trainer import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument(
        "--resume",
        default=None,
        help="checkpoint to continue from, or 'auto' for <run>/checkpoints/last.pt if it exists",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N steps (smoke)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="discard the checkpoints and logs of a previous run with the same run_name",
    )
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. train.batch_size=64")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    run_training(config, resume=args.resume, max_steps=args.max_steps, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
