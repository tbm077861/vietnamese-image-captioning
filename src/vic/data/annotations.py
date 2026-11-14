"""The unified annotation format shared by every dataset.

Each dataset adapter converts the published files into one JSON document:

    {
      "dataset": "<name>",
      "images": [
        {"image_id": "42", "file_name": "images/42.jpg", "split": "train",
         "captions": ["raw caption", ...]},
        ...
      ]
    }

``file_name`` is relative to the dataset's raw directory. ``split`` is one of train / val / test.
Everything downstream (verification, preprocessing, training) reads only this format.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from vic.utils import read_json, write_json

SPLITS = ("train", "val", "test")


def make_record(image_id, file_name: str, split: str, captions: list[str]) -> dict:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r} for image {image_id}")
    return {
        "image_id": str(image_id),
        "file_name": str(file_name).replace("\\", "/"),
        "split": split,
        "captions": list(captions),
    }


def save_annotations(path: str | Path, dataset_name: str, records: list[dict]) -> None:
    image_ids = [record["image_id"] for record in records]
    if len(set(image_ids)) != len(image_ids):
        duplicates = [key for key, count in Counter(image_ids).items() if count > 1]
        raise ValueError(f"Duplicate image ids, e.g. {duplicates[:3]}")
    write_json({"dataset": dataset_name, "images": records}, path)


def load_annotations(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/download_data.py first")
    return read_json(path)


def split_summary(records: list[dict]) -> dict[str, dict[str, int]]:
    """Number of images and captions per split."""
    summary = {}
    for split in SPLITS:
        in_split = [record for record in records if record["split"] == split]
        if in_split:
            summary[split] = {
                "images": len(in_split),
                "captions": sum(len(record["captions"]) for record in in_split),
            }
    return summary
