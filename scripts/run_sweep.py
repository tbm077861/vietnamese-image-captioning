"""Train and evaluate a list of experiment configs one after another, unattended.

Usage:
    python scripts/run_sweep.py configs/a.yaml configs/b.yaml
    python scripts/run_sweep.py configs/sweeps/                    # every *.yaml of a directory
    python scripts/run_sweep.py configs/baseline_lstm.yaml --seeds 42 43 44

For every config: train, evaluate best.pt on the validation split (greedy and beam 3 by
default), then refresh docs/EXPERIMENTS.md. Finished runs are skipped and unfinished ones resume,
so the same command can simply be run again after an interruption. A failing run is recorded in
the table and does not stop the others. With --seeds, the config's own seed keeps its run_name
and the other seeds run as <run_name>_seed<N>. The test split is never evaluated here.
"""

from __future__ import annotations

import argparse
import sys

from vic.engine.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", help="experiment YAML files or directories of them")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None, help="run every config per seed"
    )
    parser.add_argument("--beam-size", nargs="+", type=int, default=[1, 3], help="1 = greedy")
    parser.add_argument("--doc", default="docs/EXPERIMENTS.md", help="experiment table to refresh")
    parser.add_argument(
        "--set", nargs="+", default=None, metavar="KEY=VALUE", help="overrides for every config"
    )
    args = parser.parse_args()

    results = run_sweep(args.configs, args.seeds, args.beam_size, args.set, args.doc)
    sys.exit(1 if any(entry["status"] == "failed" for entry in results) else 0)


if __name__ == "__main__":
    main()
