"""Run a list of experiment configs one after another, unattended.

For every config: train (finished runs are skipped, unfinished ones resume from ``last.pt``),
evaluate ``best.pt`` on the validation split with each requested beam size under the comparison
decoding and once with the chosen decoding (``vic.engine.protocol``), then refresh the
experiment table. Missing evaluations of finished runs are added. A run that fails is recorded
and the sweep moves on to the next one; calling the sweep again retries it. The test split is
never touched.
"""

from __future__ import annotations

import copy
import gc
import traceback
from collections.abc import Sequence
from pathlib import Path

import torch

from vic.config import load_config, save_yaml
from vic.engine.evaluation import evaluate_checkpoint
from vic.engine.experiments import run_status, update_experiments_doc
from vic.engine.protocol import CHOSEN_DECODING, PROTOCOL_DECODING, evaluation_filename
from vic.engine.trainer import run_training
from vic.utils import git_commit, read_json, setup_logging, write_json


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


def sweep_configs(
    paths: Sequence[str | Path],
    seeds: Sequence[int] | None = None,
    overrides: list[str] | None = None,
) -> list[dict]:
    """Load every config; with ``seeds``, one copy per seed.

    The copy whose seed equals the config's own seed keeps the config's ``run_name``; the others
    are named ``<run_name>_seed<N>``, which is how the experiment table groups seeds.
    """
    configs = []
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
            configs.append(config)
    names = [config["run_name"] for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Several configs share a run_name: {duplicates}")
    return configs


def _run_one(config: dict, beam_sizes: Sequence[int], overrides: list[str] | None) -> dict:
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


def run_sweep(
    paths: Sequence[str | Path],
    seeds: Sequence[int] | None = None,
    beam_sizes: Sequence[int] = (1, 3),
    overrides: list[str] | None = None,
    doc_path: str | Path | None = "docs/EXPERIMENTS.md",
) -> list[dict]:
    """Run every config; return one entry per run with its status and validation CIDEr."""
    configs = sweep_configs(paths, seeds, overrides)
    table_beam = 3 if 3 in beam_sizes else max(beam_sizes)  # main columns of the table
    results = []
    for position, config in enumerate(configs, start=1):
        name = config["run_name"]
        logger = setup_logging()
        logger.info(f"=== [{position}/{len(configs)}] {name} ===")
        entry = {"run": name, "status": "failed", "cider": {}, "error": None}
        try:
            entry.update(_run_one(config, beam_sizes, overrides))
        except Exception as error:  # one broken run must not stop the sweep
            entry["error"] = f"{type(error).__name__}: {error}"
            logger = setup_logging()
            logger.error(f"Run {name} failed:\n{traceback.format_exc()}")
            _record_failure(config, entry["error"])
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(entry)
        if doc_path is not None:
            try:
                update_experiments_doc(config.get("output_dir", "outputs"), doc_path, table_beam)
            except Exception as error:
                setup_logging().error(f"Could not update {doc_path}: {error}")

    logger = setup_logging()
    logger.info("=== Sweep finished ===")
    for entry in results:
        cider = ", ".join(f"{k}: {v:.4f}" for k, v in entry["cider"].items())
        logger.info(f"{entry['run']}: {entry['status']} {cider}{entry['error'] or ''}")
    return results


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
