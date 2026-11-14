"""Build the datasets and loaders an experiment needs from its config."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from torch.utils.data import DataLoader

from vic.config import data_paths
from vic.data.annotations import SPLITS
from vic.data.dataset import (
    CaptionPairDataset,
    EpochBatchSampler,
    ImageDataset,
    ImageStore,
    collate_images,
    collate_pairs,
)
from vic.data.features import FeatureStore
from vic.text.vocab import Vocabulary
from vic.utils import read_json


@dataclass
class ExperimentData:
    vocab: Vocabulary
    records: dict[str, list[dict]]  # split -> image records with tokenised captions
    store: FeatureStore | ImageStore
    use_features: bool


def load_experiment_data(config: dict, vocab: Vocabulary | None = None) -> ExperimentData:
    """Load captions, vocabulary and the visual store (cached features or raw images)."""
    paths = data_paths(config)
    if not paths.captions.exists():
        raise FileNotFoundError(f"{paths.captions} not found - run scripts/prepare_data.py first")
    images = read_json(paths.captions)["images"]
    vocab = vocab or Vocabulary.load(paths.vocab)
    records = {split: [r for r in images if r["split"] == split] for split in SPLITS}

    data_config = config.get("data", {})
    use_features = bool(data_config.get("use_cached_features", True))
    if use_features:
        store = FeatureStore(paths.features_dir, bool(data_config.get("features_in_memory", False)))
        missing = [r["image_id"] for r in images if r["image_id"] not in store]
        if missing:
            raise RuntimeError(
                f"{len(missing)} images have no cached features (e.g. {missing[:3]}) - "
                "re-run scripts/extract_features.py"
            )
    else:
        store = ImageStore(paths.raw_dir, images, config["model"]["encoder"].get("image_size", 224))
    return ExperimentData(vocab, records, store, use_features)


def make_pair_loader(
    data: ExperimentData,
    split: str,
    batch_size: int,
    max_length: int,
    seed: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, EpochBatchSampler]:
    dataset = CaptionPairDataset(data.records[split], data.vocab, data.store, max_length)
    sampler = EpochBatchSampler(len(dataset), batch_size, seed=seed, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate_pairs, pad_id=data.vocab.pad_id),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return loader, sampler


def make_image_loader(
    data: ExperimentData,
    split: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        ImageDataset(data.records[split], data.store),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_images,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
