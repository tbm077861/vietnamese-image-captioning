"""Seeding, device selection, git checks and logging helpers.

NumPy and PyTorch are imported inside the functions that need them, so that light processes
(the parent of a sweep, the experiment table) can use this module without loading them.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    import numpy as np
    import torch

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


def _git_toplevel(directory: Path) -> Path | None:
    try:
        output = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(output).resolve() if output else None


def worktree_problem(
    code_dir: str | Path | None = None, work_dir: str | Path | None = None
) -> str | None:
    """Why a run must not start from here, or None when it may.

    A run may start only when its results can be tied to one commit: ``work_dir`` (default: the
    working directory, where configs are read) must lie in the git checkout that ``code_dir``
    (default: this package, i.e. the code that will run) comes from, and that checkout must have
    no modified, staged or untracked file. Ignored files (data, outputs) do not count.
    """
    code_dir = Path(code_dir) if code_dir is not None else Path(__file__).resolve().parent
    work_dir = Path(work_dir) if work_dir is not None else Path.cwd()
    code_root = _git_toplevel(code_dir)
    if code_root is None:
        return f"the code in {code_dir} is not in a git checkout (or git is unavailable)"
    if _git_toplevel(work_dir) != code_root:
        return (
            f"the working directory {work_dir} is not in the checkout the code comes from "
            f"({code_root}); start the run from that checkout"
        )
    try:
        lines = subprocess.run(
            ["git", "-C", str(code_root), "status", "--porcelain", "--untracked-files=normal"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        ).stdout.splitlines()  # fmt: skip
    except (OSError, subprocess.CalledProcessError) as error:
        return f"git status failed in {code_root}: {error}"
    if lines:
        shown = "; ".join(line.strip() for line in lines[:10])
        more = f" and {len(lines) - 10} more" if len(lines) > 10 else ""
        return f"the working tree of {code_root} is not clean ({shown}{more}); commit first"
    return None


def get_device(name: str = "auto") -> torch.device:
    import torch

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
