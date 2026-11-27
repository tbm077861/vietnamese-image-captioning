"""Sweep process isolation: the clean-checkout rule and the real child process."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import vic
from vic.engine.sweep import REFUSED_EXIT_CODE, PlannedRun, child_command
from vic.utils import read_json, worktree_problem

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, global_config: Path, *args: str) -> None:
    # Isolated from the machine's git configuration (hooks, signing, ignore rules).
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": str(global_config)}
    identity = ["-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
    subprocess.run(["git", "-C", str(repo), *identity, *args], check=True, capture_output=True,
                   env=env)  # fmt: skip


@needs_git
def test_worktree_problem_requires_a_clean_checkout_of_the_running_code(tmp_path):
    repo = tmp_path / "repo"
    code = repo / "src" / "pkg"
    code.mkdir(parents=True)
    (repo / ".gitignore").write_text("/outputs/\n", encoding="utf-8")
    (code / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "gitconfig"
    config.write_text("", encoding="utf-8")
    _git(repo, config, "init", "-q")
    _git(repo, config, "add", ".")
    _git(repo, config, "commit", "-q", "-m", "start")
    assert worktree_problem(code, repo) is None
    (repo / "outputs").mkdir()
    (repo / "outputs" / "summary.json").write_text("{}", encoding="utf-8")
    assert worktree_problem(code, code) is None  # ignored files and subdirectories are fine

    (repo / "draft.py").write_text("x = 1\n", encoding="utf-8")
    assert "draft.py" in worktree_problem(code, repo)  # untracked
    (repo / "draft.py").unlink()
    (code / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert "module.py" in worktree_problem(code, repo)  # modified
    _git(repo, config, "add", ".")
    assert "module.py" in worktree_problem(code, repo)  # staged
    _git(repo, config, "commit", "-q", "-m", "change")
    assert worktree_problem(code, repo) is None

    elsewhere = tmp_path / "elsewhere"  # e.g. another checkout while the package comes from repo
    elsewhere.mkdir()
    assert "not in the checkout the code comes from" in worktree_problem(code, elsewhere)
    assert "not in a git checkout" in worktree_problem(elsewhere, elsewhere)


@needs_git
def test_child_process_refuses_to_run_outside_the_checkout_of_its_code(tmp_path):
    result = tmp_path / "result.json"
    run = PlannedRun(tmp_path / "unused.yaml", 42, {"run_name": "unused"})
    completed = subprocess.run(
        child_command(run, [3], ["eval.batch_size=8"], result),
        cwd=tmp_path, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, check=False,
    )  # fmt: skip
    assert completed.returncode == REFUSED_EXIT_CODE, completed.stderr
    entry = read_json(result)
    assert entry["status"] == "refused" and "refused to run" in entry["error"]
    assert "refused to run" in completed.stderr
    assert not (tmp_path / "outputs").exists()


def test_sweep_parent_does_not_load_pytorch():
    source = Path(vic.__file__).resolve().parents[1]  # this checkout's src directory
    code = (
        "import sys, vic.engine.sweep, vic.engine.experiments; "
        "print(sorted(name for name in ('numpy', 'torch') if name in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True, text=True, timeout=120, check=True,
    )  # fmt: skip
    assert completed.stdout.strip() == "[]"
