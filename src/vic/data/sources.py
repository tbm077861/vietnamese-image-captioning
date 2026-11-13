"""Dataset adapters: convert a published dataset layout to the unified annotation records.

To add a dataset, write its YAML under configs/datasets/ and, if its annotation layout is new,
register one more adapter here. Nothing else in the code base is dataset-specific.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from vic.data.annotations import make_record
from vic.utils import read_json

Adapter = Callable[[Path, dict], list[dict]]
ADAPTERS: dict[str, Adapter] = {}

# Split names used by datasets, mapped to ours.
SPLIT_ALIASES = {"train": "train", "val": "val", "dev": "val", "valid": "val", "test": "test"}


def register(name: str) -> Callable[[Adapter], Adapter]:
    def decorator(function: Adapter) -> Adapter:
        ADAPTERS[name] = function
        return function

    return decorator


def load_records(raw_dir: Path, dataset_config: dict) -> list[dict]:
    layout = dataset_config["format"]
    if layout not in ADAPTERS:
        raise ValueError(f"Unknown dataset format {layout!r}; available: {sorted(ADAPTERS)}")
    records = ADAPTERS[layout](Path(raw_dir), dataset_config)
    excluded = {str(image_id) for image_id in dataset_config.get("exclude_images", [])}
    return [record for record in records if record["image_id"] not in excluded]


def _annotation_files(dataset_config: dict) -> list[tuple[str, str]]:
    return [
        (SPLIT_ALIASES[split], file_name)
        for split, file_name in dataset_config["annotation_files"].items()
    ]


@register("filename_captions")
def load_filename_captions(raw_dir: Path, dataset_config: dict) -> list[dict]:
    """One JSON per split: {"<image file>": {"captions": ["...", ...]}, ...} (UIT-OpenViIC)."""
    image_dir = dataset_config.get("image_dir", "images")
    records = []
    for split, file_name in _annotation_files(dataset_config):
        for image_file, entry in read_json(raw_dir / file_name).items():
            records.append(
                make_record(
                    Path(image_file).stem, f"{image_dir}/{image_file}", split, entry["captions"]
                )
            )
    return records


@register("coco_captions")
def load_coco_captions(raw_dir: Path, dataset_config: dict) -> list[dict]:
    """One COCO-style JSON per split: {"images": [{id, file_name}], "annotations": [...]}.

    Used by UIT-ViIC and KTVIC, among others.
    """
    image_dir = dataset_config.get("image_dir", "images")
    records = []
    for split, file_name in _annotation_files(dataset_config):
        data = read_json(raw_dir / file_name)
        captions: dict[str, list[str]] = {}
        for annotation in data["annotations"]:
            captions.setdefault(str(annotation["image_id"]), []).append(annotation["caption"])
        for image in data["images"]:
            image_id = str(image["id"])
            if image_id in captions:
                records.append(
                    make_record(
                        image_id, f"{image_dir}/{image['file_name']}", split, captions[image_id]
                    )
                )
    return records
