"""Show the captions of several runs for the same fixed sample of images, side by side.

Usage:
    python scripts/compare_runs.py --runs outputs/baseline_lstm outputs/attention_lstm
    python scripts/compare_runs.py --runs outputs/a outputs/b --samples 50 --output compare.md

Reads <run>/eval_<split>_beam<k>.json (run evaluate.py first). The sample is fixed by --seed,
so the same images are read every time; the default is 50 validation images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vic.engine.report import build_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="+", required=True, help="run directories")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-references", type=int, default=1)
    parser.add_argument("--output", default=None, help="write Markdown here (default: stdout)")
    args = parser.parse_args()

    text = build_comparison(
        args.runs, args.split, args.beam_size, args.samples, args.seed, args.max_references
    )
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(text)


if __name__ == "__main__":
    main()
