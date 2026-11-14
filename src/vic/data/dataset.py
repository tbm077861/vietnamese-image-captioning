"""Datasets, collation and deterministic batching.

A "visual store" maps an image id to a tensor: either cached encoder features
(``FeatureStore``) or a preprocessed image (``ImageStore``). The datasets do not care which.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from vic.data.image import preprocess_image
from vic.text.vocab import Vocabulary


class ImageStore:
    """Load and preprocess images from the raw dataset directory."""

    def __init__(self, raw_dir: str | Path, records: list[dict], image_size: int = 224) -> None:
        self.raw_dir = Path(raw_dir)
        self.image_size = image_size
        self._files = {record["image_id"]: record["file_name"] for record in records}

    def __getitem__(self, image_id: str) -> torch.Tensor:
        with Image.open(self.raw_dir / self._files[image_id]) as image:
            return preprocess_image(image, self.image_size)


class CaptionPairDataset(Dataset):
    """One sample per (image, caption) pair, for training and validation loss."""

    def __init__(
        self, records: list[dict], vocab: Vocabulary, visual_store, max_length: int = 30
    ) -> None:
        self.visual_store = visual_store
        self.samples: list[tuple[str, list[int]]] = []
        for record in records:
            for caption in record["captions"]:
                tokens = caption["tokens"][:max_length]
                if tokens:
                    self.samples.append((record["image_id"], vocab.encode(tokens)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id, token_ids = self.samples[index]
        return self.visual_store[image_id], torch.tensor(token_ids, dtype=torch.long)


class ImageDataset(Dataset):
    """One sample per image, for caption generation and feature extraction."""

    def __init__(self, records: list[dict], visual_store) -> None:
        self.image_ids = [record["image_id"] for record in records]
        self.visual_store = visual_store

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id = self.image_ids[index]
        return self.visual_store[image_id], image_id


def collate_pairs(batch: list[tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> dict:
    """Stack visuals and pad captions; inputs and targets are the caption shifted by one."""
    visuals, captions = zip(*batch, strict=True)
    padded = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=pad_id)
    return {
        "visual": torch.stack(visuals),
        "input_ids": padded[:, :-1],
        "target_ids": padded[:, 1:],
    }


def collate_images(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, list[str]]:
    visuals, image_ids = zip(*batch, strict=True)
    return torch.stack(visuals), list(image_ids)


class EpochBatchSampler(Sampler[list[int]]):
    """Batches whose order depends only on (seed, epoch).

    This makes every epoch reproducible and lets a resumed run continue mid-epoch: call
    ``set_epoch(epoch, start_batch=k)`` to skip the first ``k`` batches without loading them.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = epoch
        self.start_batch = start_batch

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return max(0, self.batches_per_epoch - self.start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            order = list(range(self.num_samples))
        for batch_index in range(self.start_batch, self.batches_per_epoch):
            start = batch_index * self.batch_size
            yield order[start : start + self.batch_size]
