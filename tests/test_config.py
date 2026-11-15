"""Tests for experiment config loading: inheritance and overrides."""

import pytest

from vic.config import config_differences, data_paths, load_config, save_yaml


def test_base_config_is_inherited_and_merged_key_by_key(tmp_path):
    base = tmp_path / "base.yaml"
    save_yaml(
        {
            "seed": 1,
            "run_name": "base",
            "notes": "only about the base run",
            "dataset": {"name": "toy"},
            "model": {
                "encoder": {"name": "resnet50", "image_size": 224},
                "decoder": {"type": "lstm"},
            },
            "train": {"lr": 0.001, "epochs": 30},
        },
        base,
    )
    child = tmp_path / "child.yaml"
    save_yaml(
        {
            "base_config": str(base),
            "run_name": "child",
            "model": {"encoder": {"name": "siglip_b16"}},
            "train": {"lr": 0.0002},
        },
        child,
    )
    grandchild = tmp_path / "grandchild.yaml"
    save_yaml({"base_config": str(child), "run_name": "grandchild", "seed": 7}, grandchild)

    config = load_config(child, ["train.epochs=5"])
    assert config["run_name"] == "child" and config["seed"] == 1 and "base_config" not in config
    assert config["model"]["encoder"] == {"name": "siglip_b16", "image_size": 224}
    assert config["model"]["decoder"] == {"type": "lstm"}
    assert config["train"] == {"lr": 0.0002, "epochs": 5}
    assert "notes" not in config  # notes belong to one experiment
    assert data_paths(config).features_dir.name == "siglip_b16_224"

    deeper = load_config(grandchild)
    assert deeper["run_name"] == "grandchild" and deeper["seed"] == 7
    assert deeper["train"]["lr"] == 0.0002 and deeper["model"]["encoder"]["name"] == "siglip_b16"


def test_circular_base_config_is_rejected(tmp_path):
    first, second = tmp_path / "a.yaml", tmp_path / "b.yaml"
    save_yaml({"base_config": str(second), "run_name": "a"}, first)
    save_yaml({"base_config": str(first), "run_name": "b"}, second)
    with pytest.raises(ValueError, match="Circular"):
        load_config(first)


def test_config_differences_lists_changed_leaves():
    old = {"data": {"features_in_memory": True}, "train": {"lr": 0.1}, "overrides": None}
    new = {
        "data": {"features_in_memory": False},
        "train": {"lr": 0.1, "epochs": 3},
        "overrides": ["x=1"],
    }
    assert config_differences(old, new) == [
        "data.features_in_memory: True -> False",
        "train.epochs: None -> 3",
    ]
    assert config_differences(old, old) == []


def test_every_experiment_config_loads_and_builds(monkeypatch):
    from pathlib import Path

    from vic.engine.sweep import sweep_configs
    from vic.models.captioner import build_model

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)  # base_config and dataset_config paths are relative to the repo
    paths = sorted(Path("configs").glob("*.yaml")) + sorted(Path("configs/sweeps").glob("*.yaml"))
    configs = sweep_configs(paths)  # also rejects duplicate run names
    assert len(configs) == len(paths)
    for config in configs:
        model = build_model(config["model"], vocab_size=50, pad_id=0, with_encoder=False)
        assert sum(p.numel() for p in model.parameters()) > 0, config["run_name"]
    # The prepared Step 5 sweep stays within the approved budget of 15 runs.
    assert len(sweep_configs(["configs/sweeps"])) <= 15
