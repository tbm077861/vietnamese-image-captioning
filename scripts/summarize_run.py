"""Summarise a training run as Markdown: learning curve, scores, diagnostics, sample captions.

Usage:
    python scripts/summarize_run.py --run outputs/baseline_lstm --output report.md

Run evaluate.py first for every split / beam size that should appear in the score table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vic.engine.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run directory, e.g. outputs/baseline_lstm")
    parser.add_argument("--output", default=None, help="write Markdown here (default: stdout)")
    parser.add_argument("--samples", type=int, default=10, help="number of random examples")
    parser.add_argument("--sample-split", default="test")
    parser.add_argument("--analysis-split", default="val", help="split used for diagnostics")
    parser.add_argument("--beam-size", type=int, default=3, help="decoding shown in examples")
    parser.add_argument("--worst", type=int, default=0, help="also list the N lowest-CIDEr images")
    parser.add_argument("--seed", type=int, default=42, help="seed of the random examples")
    parser.add_argument(
        "--max-references",
        type=int,
        default=1,
        help="reference captions quoted per example (dataset text: keep it small when publishing)",
    )
    args = parser.parse_args()

    report = build_report(
        args.run,
        samples=args.samples,
        sample_split=args.sample_split,
        analysis_split=args.analysis_split,
        beam_size=args.beam_size,
        worst=args.worst,
        seed=args.seed,
        max_references=args.max_references,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(report)


if __name__ == "__main__":
    main()
