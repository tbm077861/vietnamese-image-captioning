"""Tests for split assignment, caption preparation, datasets, collation and batching."""

import torch

from vic.data.annotations import make_record
from vic.data.dataset import CaptionPairDataset, EpochBatchSampler, ImageDataset, collate_pairs
from vic.data.prepare import (
    assign_missing_splits,
    build_vocabulary,
    caption_statistics,
    tokenize_records,
)
from vic.text import CaptionTokenizer


def _records(count=20, split="train"):
    return [
        make_record(str(index), f"images/{index}.jpg", split, [f"Ảnh số {index}.", "Một con mèo!"])
        for index in range(count)
    ]


def test_missing_splits_are_carved_from_train_deterministically():
    records = _records(20)
    first = assign_missing_splits(records, val_ratio=0.1, test_ratio=0.2, seed=42)
    counts = {split: sum(r["split"] == split for r in first) for split in ("train", "val", "test")}
    assert counts == {"train": 14, "val": 2, "test": 4}
    # Same seed -> same assignment, whatever the order of the annotation file.
    second = assign_missing_splits(list(reversed(records)), 0.1, 0.2, seed=42)
    assert {r["image_id"]: r["split"] for r in first} == {r["image_id"]: r["split"] for r in second}
    other = assign_missing_splits(records, 0.1, 0.2, seed=7)
    assert {r["image_id"]: r["split"] for r in first} != {r["image_id"]: r["split"] for r in other}


def test_existing_splits_are_left_untouched():
    records = (
        _records(10)
        + _records(3, "val")[:0]
        + [
            make_record("v", "images/v.jpg", "val", ["a"]),
            make_record("t", "images/t.jpg", "test", ["b"]),
        ]
    )
    assigned = assign_missing_splits(records, val_ratio=0.5, test_ratio=0.5, seed=1)
    assert assigned == records


def test_tokenize_records_and_vocabulary_use_only_train():
    records = [
        make_record("1", "a.jpg", "train", ["Một con mèo.", "   "]),
        make_record("2", "b.jpg", "test", ["Một con voi"]),
        make_record("3", "c.jpg", "train", ["..."]),
    ]
    prepared = tokenize_records(records, CaptionTokenizer(level="syllable"))
    assert [r["image_id"] for r in prepared] == ["1", "2"]  # image 3 has no usable caption
    assert prepared[0]["captions"] == [
        {"raw": "Một con mèo.", "normalized": "một con mèo", "tokens": ["một", "con", "mèo"]}
    ]
    vocab = build_vocabulary(prepared)
    assert "mèo" in vocab.stoi and "voi" not in vocab.stoi

    stats = caption_statistics(prepared, vocab, max_length=2)
    assert stats["train"]["captions"] == 1 and stats["train"]["truncated_ratio"] == 1.0
    assert stats["test"]["oov_ratio"] == round(1 / 3, 4)
    assert "val" not in stats


class _FakeStore:
    def __getitem__(self, image_id):
        return torch.full((2, 4), float(image_id))


def _prepared():
    tokenizer = CaptionTokenizer(level="syllable")
    records = [
        make_record("1", "a.jpg", "train", ["một con mèo đang ngủ", "con mèo"]),
        make_record("2", "b.jpg", "train", ["hai con chó"]),
    ]
    prepared = tokenize_records(records, tokenizer)
    return prepared, build_vocabulary(prepared)


def test_pair_dataset_and_collate():
    prepared, vocab = _prepared()
    dataset = CaptionPairDataset(prepared, vocab, _FakeStore(), max_length=3)
    assert len(dataset) == 3
    visual, ids = dataset[0]
    assert visual.shape == (2, 4) and float(visual[0, 0]) == 1.0
    # Truncated to three tokens plus the start / end markers.
    assert ids.tolist() == [vocab.bos_id, *vocab.encode(["một", "con", "mèo"], False), vocab.eos_id]

    batch = collate_pairs([dataset[0], dataset[1]], pad_id=vocab.pad_id)
    assert batch["visual"].shape == (2, 2, 4)
    assert batch["input_ids"].shape == batch["target_ids"].shape == (2, 4)
    assert batch["input_ids"][:, 0].tolist() == [vocab.bos_id, vocab.bos_id]
    assert batch["target_ids"][1].tolist()[-2:] == [vocab.eos_id, vocab.pad_id]
    # Targets are the inputs shifted left by one position.
    assert batch["input_ids"][0, 1:].tolist() == batch["target_ids"][0, :-1].tolist()

    images = ImageDataset(prepared, _FakeStore())
    assert len(images) == 2 and images[1][1] == "2"


def test_epoch_batch_sampler_is_reproducible_and_resumable():
    sampler = EpochBatchSampler(num_samples=10, batch_size=4, seed=3)
    assert sampler.batches_per_epoch == 3
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    assert sorted(index for batch in epoch0 for index in batch) == list(range(10))
    sampler.set_epoch(1)
    assert list(sampler) != epoch0
    sampler.set_epoch(0)
    assert list(sampler) == epoch0
    # Resuming after one batch yields exactly the remaining batches.
    sampler.set_epoch(0, start_batch=1)
    assert list(sampler) == epoch0[1:] and len(sampler) == 2

    ordered = EpochBatchSampler(5, 2, shuffle=False, drop_last=True)
    assert list(ordered) == [[0, 1], [2, 3]]
