"""Regenerate the experiment table in docs/EXPERIMENTS.md from the run directories.

Usage:
    python scripts/update_experiments.py
    python scripts/update_experiments.py --outputs outputs --doc docs/EXPERIMENTS.md

Only validation results are read (summary.json, eval_val_beam<k>.json). Text outside the table
markers is kept. Run directories named smoke*, debug* or tmp* are left out.
"""

from __future__ import annotations

import argparse

from vic.engine.experiments import DEFAULT_EXCLUDE, update_experiments_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", default="outputs", help="directory holding the runs")
    parser.add_argument("--doc", default="docs/EXPERIMENTS.md")
    parser.add_argument("--beam-size", type=int, default=3, help="decoding of the main columns")
    parser.add_argument("--exclude", nargs="*", default=list(DEFAULT_EXCLUDE), help="name patterns")
    args = parser.parse_args()

    rows = update_experiments_doc(args.outputs, args.doc, args.beam_size, tuple(args.exclude))
    print(f"Wrote {args.doc} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
