"""Run a list of experiment configs one after another, unattended, one process per run.

For every config: train (finished runs are skipped, unfinished ones resume from ``last.pt``),
evaluate ``best.pt`` on the validation split with each requested beam size under the comparison
decoding and once with the chosen decoding (``vic.engine.protocol``). Missing evaluations of
finished runs are added. The test split is never touched.

Every run happens in its own child process (``python -I -X utf8 -m vic.engine.sweep ...``): the
memory of a run goes back to the system before the next one starts, and a crash or a killed
process costs that run only. The parent plans the runs, starts the children one at a time and
collects their results; it never imports PyTorch.

A child refuses to start (exit code 3) unless its results can be tied to one commit: it must be
started inside the git checkout its code comes from, and that checkout must be clean
(``vic.utils.worktree_problem``). The sweep stops at the first refusal. A run that fails is
recorded and the sweep moves on; calling the sweep again retries it. The experiment table is
refreshed once, after the last run, so the working tree stays clean while runs are in progress.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from vic.config import load_config, save_yaml
from vic.engine.experiments import run_status, update_experiments_doc
from vic.engine.protocol import CHOSEN_DECODING, PROTOCOL_DECODING, evaluation_filename
from vic.utils import git_commit, read_json, setup_logging, worktree_problem, write_json

REFUSED_EXIT_CODE = 3


@dataclass(frozen=True)
class PlannedRun:
    """One run of a sweep: the config file, the seed and the resulting config."""

    config_path: Path
    seed: int
    config: dict

    @property
    def name(self) -> str:
        return self.config["run_name"]


def expand_config_paths(paths: Sequence[str | Path]) -> list[Path]:
    """Files are kept in the given order; a directory contributes its ``*.yaml`` files, sorted."""
    expanded: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            expanded += sorted(path.glob("*.yaml"))
        elif path.exists():
            expanded.append(path)
        else:
            raise FileNotFoundError(f"Config {path} not found")
    return expanded


def plan_runs(
    paths: Sequence[str | Path],
    seeds: Sequence[int] | None = None,
    overrides: list[str] | None = None,
) -> list[PlannedRun]:
    """Load every config; with ``seeds``, one run per seed.

    The run whose seed equals the config's own seed keeps the config's ``run_name``; the others
    are named ``<run_name>_seed<N>``, which is how the experiment table groups seeds.
    """
    runs = []
    for path in expand_config_paths(paths):
        base = load_config(path, overrides)
        if overrides:
            base["overrides"] = list(overrides)  # recorded in config.yaml and summary.json
        own_seed = int(base.get("seed", 42))
        for seed in seeds or [own_seed]:
            config = copy.deepcopy(base)
            if int(seed) != own_seed:
                config["seed"] = int(seed)
                config["run_name"] = f"{base['run_name']}_seed{int(seed)}"
            runs.append(PlannedRun(path, int(seed), config))
    names = [run.name for run in runs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Several configs share a run_name: {duplicates}")
    return runs


def sweep_configs(
    paths: Sequence[str | Path],
    seeds: Sequence[int] | None = None,
    overrides: list[str] | None = None,
) -> list[dict]:
    """The configs of ``plan_runs``."""
    return [run.config for run in plan_runs(paths, seeds, overrides)]


def child_command(
    run: PlannedRun,
    beam_sizes: Sequence[int],
    overrides: list[str] | None,
    result_path: str | Path,
) -> list[str]:
    """Command line of the child process that trains and evaluates ``run``."""
    command = [
        sys.executable, "-I", "-X", "utf8", "-m", "vic.engine.sweep",
        "--config", str(run.config_path), "--seed", str(run.seed),
        "--result", str(result_path), "--beam-size", *map(str, beam_sizes),
    ]  # fmt: skip
    if overrides:
        command += ["--set", *overrides]
    return command


def _launch_process(command: list[str]) -> int:
    """Run a child with this process's console and output; return its exit code."""
    return subprocess.run(command, check=False).returncode


def run_sweep(
    paths: Sequence[str | Path],
    seeds: Sequence[int] | None = None,
    beam_sizes: Sequence[int] = (1, 3),
    overrides: list[str] | None = None,
    doc_path: str | Path | None = "docs/EXPERIMENTS.md",
    launch: Callable[[list[str]], int] | None = None,
) -> list[dict]:
    """Run every config in a child process; return one entry per started run.

    Each entry has the run name, its status ("completed", "skipped", "failed" or "refused"), its
    validation CIDEr per decoding and the error, if any. ``launch`` runs a child command and
    returns its exit code (tests replace it).
    """
    runs = plan_runs(paths, seeds, overrides)
    launch = launch or _launch_process
    table_beam = 3 if 3 in beam_sizes else max(beam_sizes)  # main columns of the table
    results: list[dict] = []
    logger = setup_logging()
    with tempfile.TemporaryDirectory(prefix="vic_sweep_") as scratch:
        for position, run in enumerate(runs, start=1):
            logger.info(f"=== [{position}/{len(runs)}] {run.name} ===")
            result_path = Path(scratch) / f"run{position}.json"
            code = launch(child_command(run, beam_sizes, overrides, result_path))
            logger = setup_logging()  # the child may have written to the same console
            entry = {"run": run.name, "status": "failed", "cider": {}, "error": None}
            if result_path.exists():
                entry.update(read_json(result_path))
            else:
                entry["error"] = f"the child process ended with exit code {code} and no result"
            results.append(entry)
            if entry["status"] == "refused":
                logger.error(f"Sweep stopped, {run.name} refused to start: {entry['error']}")
                break

    started = [entry for entry in results if entry["status"] != "refused"]
    if doc_path is not None and started:
        output_dir = runs[len(results) - 1].config.get("output_dir", "outputs")
        try:
            update_experiments_doc(output_dir, doc_path, table_beam)
        except Exception as error:
            logger.error(f"Could not update {doc_path}: {error}")

    logger.info("=== Sweep finished ===")
    for entry in results:
        cider = ", ".join(f"{k}: {v:.4f}" for k, v in entry["cider"].items())
        logger.info(f"{entry['run']}: {entry['status']} {cider}{entry['error'] or ''}")
    if len(results) < len(runs):
        logger.info(f"{len(runs) - len(results)} run(s) not started")
    return results


# --- child process ------------------------------------------------------------------------------


def _train_and_evaluate(config: dict, beam_sizes: Sequence[int], overrides: list[str] | None):
    from vic.engine.evaluation import evaluate_checkpoint
    from vic.engine.trainer import run_training

    run_dir = Path(config.get("output_dir", "outputs")) / config["run_name"]
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    trained = run_status(summary, config) != "completed"
    if trained:
        run_training(config, resume="auto")
    best = run_dir / "checkpoints" / "best.pt"
    decodings = {f"beam {k}": {**PROTOCOL_DECODING, "beam_size": k} for k in beam_sizes}
    decodings["chosen"] = dict(CHOSEN_DECODING)
    scores = {}
    for label, decoding in decodings.items():
        output = run_dir / evaluation_filename("val", decoding)
        if trained or not output.exists():
            result = evaluate_checkpoint(best, "val", overrides=overrides, decoding=decoding)
            scores[label] = result["scores"]["CIDEr"]
        else:
            scores[label] = read_json(output)["scores"]["CIDEr"]
    return {"status": "completed" if trained else "skipped", "cider": scores}


def run_one(config: dict, beam_sizes: Sequence[int], overrides: list[str] | None) -> dict:
    """Train (unless finished) and evaluate one run in this process; failures are returned."""
    entry = {"status": "failed", "cider": {}, "error": None}
    try:
        entry.update(_train_and_evaluate(config, beam_sizes, overrides))
    except Exception as error:
        entry["error"] = f"{type(error).__name__}: {error}"
        setup_logging().error(f"Run {config['run_name']} failed:\n{traceback.format_exc()}")
        _record_failure(config, entry["error"])
    return entry


def _record_failure(config: dict, error: str) -> None:
    """Make sure a failed run shows up in the experiment table."""
    run_dir = Path(config.get("output_dir", "outputs")) / config["run_name"]
    if not (run_dir / "config.yaml").exists():
        save_yaml(config, run_dir / "config.yaml")
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    if summary.get("status") == "failed":
        return  # the trainer already recorded it
    if summary.get("status") == "completed":
        # Training is fine and must not be repeated; only the evaluation broke.
        summary["evaluation_error"] = error[:500]
        write_json(summary, summary_path, indent=2)
        return
    summary.update(
        {
            "run_name": config["run_name"],
            "status": "failed",
            "error": error[:500],
            "seed": int(config.get("seed", 42)),
            "git_commit": summary.get("git_commit") or git_commit(),
        }
    )
    write_json(summary, summary_path, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    """Child process: one run of a sweep. Writes its result to ``--result``; see the module doc."""
    parser = argparse.ArgumentParser(description="Train and evaluate one run of a sweep")
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result", required=True, help="JSON file that receives the result")
    parser.add_argument("--beam-size", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--set", nargs="+", default=None, metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    problem = worktree_problem()  # before anything heavy is loaded
    if problem is not None:
        error = f"refused to run: {problem}"
        write_json({"status": "refused", "cider": {}, "error": error}, args.result)
        print(error, file=sys.stderr)
        return REFUSED_EXIT_CODE
    (run,) = plan_runs([args.config], [args.seed], args.set)
    entry = run_one(run.config, args.beam_size, args.set)
    write_json(entry, args.result)
    return 1 if entry["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
