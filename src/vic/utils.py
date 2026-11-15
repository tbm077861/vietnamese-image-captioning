"""Seeding, device selection and logging helpers."""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set both ways: several runs may share one process (scripts/run_sweep.py).
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False


def git_commit() -> str | None:
    """Short hash of the checked-out commit, with "-dirty" when tracked files are modified.

    Documentation is ignored (the experiment table is rewritten after every run of a sweep).
    Returns None outside a git checkout or when git is unavailable.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)*.md"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{commit}-dirty" if status else commit or None


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available (run scripts/check_env.py)")
    return torch.device(name)


def close_logging() -> logging.Logger:
    """Detach and close the project logger's handlers (releases the log file on Windows)."""
    logger = logging.getLogger("vic")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    return logger


def setup_logging(log_file: str | Path | None = None) -> logging.Logger:
    """Log to the console and optionally to a UTF-8 file."""
    logger = close_logging()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class JsonlWriter:
    """Append one JSON object per line; safe to reopen when a run is resumed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data, path: str | Path, indent: int | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
