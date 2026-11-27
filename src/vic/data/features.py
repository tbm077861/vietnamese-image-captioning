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
FEATURE_ACCESS = ("mmap", "read")


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

    The file is opened lazily (so the store can be sent to DataLoader workers). Three ways to
    read it, all returning the same values:

    - ``in_memory=True``: the whole array is loaded into RAM;
    - ``access="mmap"`` (default): the file is memory-mapped, and the rows read so far stay in
      the working set of the process (a full epoch maps the whole training split);
    - ``access="read"``: each row is read with a plain file read. The operating system caches
      the file in memory it can reclaim at any time (counted as available), so the process stays
      small. Meant for machines short of RAM.
    """

    def __init__(
        self, directory: str | Path, in_memory: bool = False, access: str = "mmap"
    ) -> None:
        if access not in FEATURE_ACCESS:
            raise ValueError(f"Unknown feature access {access!r}; expected one of {FEATURE_ACCESS}")
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
        self.access = access
        self._array: np.ndarray | None = None
        self._handle = None  # open file for access="read"
        self._layout: tuple[int, tuple[int, ...], int, np.dtype] | None = None

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

    def _read_row(self, row: int) -> np.ndarray:
        if self._handle is None:
            handle = open(self.directory / FEATURES_FILE, "rb")  # noqa: SIM115 - kept open
            major, _ = np.lib.format.read_magic(handle)
            if major == 1:
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
            if fortran_order or len(shape) != 3:
                handle.close()
                raise ValueError(f"Unexpected layout of {self.directory / FEATURES_FILE}")
            row_shape = tuple(shape[1:])
            row_bytes = int(np.prod(row_shape)) * dtype.itemsize
            self._handle, self._layout = handle, (handle.tell(), row_shape, row_bytes, dtype)
        offset, row_shape, row_bytes, dtype = self._layout
        self._handle.seek(offset + row * row_bytes)
        buffer = np.empty(row_bytes, dtype=np.uint8)  # not zero-filled, unlike a bytearray
        if self._handle.readinto(buffer) != row_bytes:
            raise EOFError(f"{self.directory / FEATURES_FILE} ends before row {row}")
        return buffer.view(dtype).reshape(row_shape)

    def __getitem__(self, image_id: str) -> torch.Tensor:
        row = self._rows[image_id]
        if self.access == "read" and not self.in_memory:
            return torch.from_numpy(self._read_row(row))
        return torch.from_numpy(np.array(self.array[row]))

    def close(self) -> None:
        """Release the open file (``access="read"``); it is reopened on the next read."""
        if self._handle is not None:
            self._handle.close()
        self._handle, self._layout = None, None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Reopen in the worker process instead of pickling the data or the file handle.
        state.update(_array=None, _handle=None, _layout=None)
        return state
