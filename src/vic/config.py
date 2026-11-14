"""YAML configuration loading and the data directory layout.

An experiment file points at a dataset file through ``dataset_config``; the dataset settings are
loaded into ``cfg["dataset"]``. Switching dataset therefore means changing that one line.
All relative paths are resolved against the working directory (run scripts from the repo root).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def _set_by_path(config: dict, dotted_key: str, value: Any) -> None:
    node = config
    *parents, leaf = dotted_key.split(".")
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise KeyError(f"Cannot override {dotted_key!r}: {key!r} is not a mapping")
    node[leaf] = value


def _parse_overrides(overrides: list[str] | None) -> list[tuple[str, Any]]:
    parsed = []
    for item in overrides or []:
        key, separator, raw_value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")
        parsed.append((key.strip(), yaml.safe_load(raw_value)))
    return parsed


def apply_overrides(config: dict, overrides: list[str] | None) -> dict:
    """Apply ``key.subkey=value`` overrides to an already loaded config, in place."""
    for key, value in _parse_overrides(overrides):
        _set_by_path(config, key, value)
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load an experiment config, merge its dataset config and apply ``key=value`` overrides."""
    config = copy.deepcopy(load_yaml(path))
    parsed = _parse_overrides(overrides)
    for key, value in parsed:
        if not key.startswith("dataset."):
            _set_by_path(config, key, value)
    if "dataset_config" in config:
        config["dataset"] = load_yaml(config["dataset_config"])
    for key, value in parsed:
        if key.startswith("dataset."):
            _set_by_path(config, key, value)
    if "dataset" not in config:
        raise KeyError(f"{path} must define 'dataset_config' or an inline 'dataset' section")
    return config


def load_any_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Accept an experiment file or a bare dataset file; always return ``{"dataset": ...}``.

    The data scripts (download, verify) only need the dataset section, so they take either.
    """
    config = load_yaml(path)
    if "dataset_config" in config or "dataset" in config:
        return load_config(path, overrides)
    for key, value in _parse_overrides(overrides):
        _set_by_path(config, key.removeprefix("dataset."), value)
    return {"dataset": config}


@dataclass(frozen=True)
class DataPaths:
    raw_dir: Path  # downloaded files, exactly as published
    annotations: Path  # unified annotation file written by the download step
    processed_dir: Path  # tokenised captions and vocabulary for one text setting
    captions: Path
    vocab: Path
    features_dir: Path  # cached encoder features for one encoder setting


def data_paths(config: dict) -> DataPaths:
    dataset = config["dataset"]
    root = Path(config.get("data_root", "data"))
    name = dataset["name"]
    text = config.get("text", {})
    text_tag = f"{text.get('level', 'word')}_minfreq{text.get('min_freq', 1)}"
    encoder = config.get("model", {}).get("encoder", {})
    encoder_tag = f"{encoder.get('name', 'resnet50')}_{encoder.get('image_size', 224)}"
    processed_dir = root / "processed" / name / text_tag
    return DataPaths(
        raw_dir=root / "raw" / name,
        annotations=root / "processed" / name / "annotations.json",
        processed_dir=processed_dir,
        captions=processed_dir / "captions.json",
        vocab=processed_dir / "vocab.json",
        features_dir=root / "features" / name / encoder_tag,
    )
