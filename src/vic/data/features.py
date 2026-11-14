"""On-disk cache of encoder features (half precision).

Layout of a cache directory:

    features.npy   float16 array of shape (num_images, regions, channels)
    index.json     {"image_ids": [...], "encoder": ..., "image_size": ..., "shape": [...]}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vic.utils import read_json, write_json

FEATURES_FILE = "features.npy"
INDEX_FILE = "index.json"


def create_feature_cache(directory: str | Path, shape: tuple[int, int, int]) -> np.memmap:
    """Allocate the feature file; rows are written in place by the extraction script."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(
        directory / FEATURES_FILE, mode="w+", dtype=np.float16, shape=shape
    )


def write_feature_index(directory: str | Path, image_ids: list[str], **metadata) -> None:
    write_json({"image_ids": list(image_ids), **metadata}, Path(directory) / INDEX_FILE)


class FeatureStore:
    """Read cached features by image id.

    The file is opened lazily (so the store can be sent to DataLoader workers) and is either
    memory-mapped or, with ``in_memory=True``, loaded fully into RAM.
    """

    def __init__(self, directory: str | Path, in_memory: bool = False) -> None:
        self.directory = Path(directory)
        index_path = self.directory / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"No feature cache in {self.directory} - run scripts/extract_features.py first"
            )
        self.metadata = read_json(index_path)
        self.image_ids: list[str] = self.metadata["image_ids"]
        self._rows = {image_id: row for row, image_id in enumerate(self.image_ids)}
        self.in_memory = in_memory
        self._array: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __contains__(self, image_id: str) -> bool:
        return image_id in self._rows

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            path = self.directory / FEATURES_FILE
            self._array = np.load(path) if self.in_memory else np.load(path, mmap_mode="r")
        return self._array

    def __getitem__(self, image_id: str) -> torch.Tensor:
        return torch.from_numpy(np.array(self.array[self._rows[image_id]]))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_array"] = None  # reopen in the worker process instead of pickling the data
        return state
