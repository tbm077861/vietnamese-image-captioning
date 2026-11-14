"""End-to-end training tests on a tiny synthetic dataset (CPU, cached features)."""

import copy
import json

import numpy as np
import pytest
import torch

from vic.config import data_paths
from vic.data.annotations import make_record
from vic.data.features import FeatureStore, create_feature_cache, write_feature_index
from vic.data.prepare import build_vocabulary, tokenize_records
from vic.engine.checkpoint import load_checkpoint, model_from_checkpoint
from vic.engine.data import load_experiment_data, make_image_loader
from vic.engine.evaluation import (
    build_references,
    evaluate_checkpoint,
    generate_captions,
    score_captions,
)
from vic.engine.report import build_report
from vic.engine.trainer import run_training
from vic.text import CaptionTokenizer
from vic.utils import read_json, write_json

SUBJECTS = ["con mèo", "con chó", "cậu bé", "cô gái"]
ACTIONS = ["đang ngủ", "đang chạy", "đang ăn"]


def _make_config(tmp_path, **train_overrides):
    train = {
        "epochs": 4,
        "batch_size": 8,
        "lr": 0.01,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        "warmup_steps": 0,
        "scheduler": "none",
        "amp": False,
        "num_workers": 0,
        "log_every": 2,
        "save_every_steps": 0,
        "tensorboard": False,
        "deterministic": True,
    }
    train.update(train_overrides)
    return {
        "seed": 0,
        "run_name": "tiny",
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "data_root": str(tmp_path / "data"),
        "dataset": {"name": "toy"},
        "text": {"level": "syllable", "min_freq": 1, "max_length": 10},
        "data": {"use_cached_features": True, "features_in_memory": True},
        "model": {
            "encoder": {"name": "resnet50", "image_size": 224, "trainable": False},
            "decoder": {
                "type": "lstm", "embed_dim": 32, "hidden_dim": 48, "num_layers": 1, "dropout": 0.0,
            },
        },
        "eval": {"beam_size": 2, "batch_size": 16, "length_penalty": 0.7},
        "train": train,
    }  # fmt: skip


@pytest.fixture
def toy_config(tmp_path):
    """24 images whose caption is fully determined by a feature pattern."""
    config = _make_config(tmp_path)
    paths = data_paths(config)
    generator = np.random.default_rng(0)
    prototypes = generator.normal(size=(len(SUBJECTS) * len(ACTIONS), 2048)).astype(np.float32)

    records, features = [], []
    for index in range(24):
        kind = index % len(prototypes)
        caption = f"{SUBJECTS[kind % len(SUBJECTS)]} {ACTIONS[kind // len(SUBJECTS)]}"
        split = "train" if index < 12 else ("val" if index < 18 else "test")
        records.append(make_record(str(index), f"images/{index}.jpg", split, [caption, caption]))
        features.append(np.tile(prototypes[kind], (4, 1)))

    prepared = tokenize_records(records, CaptionTokenizer(level="syllable"))
    write_json({"dataset": "toy", "images": prepared}, paths.captions)
    build_vocabulary(prepared).save(paths.vocab)
    cache = create_feature_cache(paths.features_dir, (24, 4, 2048))
    cache[:] = np.stack(features).astype(np.float16)
    cache.flush()
    del cache
    write_feature_index(paths.features_dir, [r["image_id"] for r in records], encoder="resnet50")
    return config


def test_feature_store_round_trip(toy_config):
    store = FeatureStore(data_paths(toy_config).features_dir)
    assert len(store) == 24 and "5" in store and "99" not in store
    feature = store["5"]
    assert feature.shape == (4, 2048) and feature.dtype == torch.float16


def test_training_learns_checkpoints_and_evaluates(toy_config):
    config = copy.deepcopy(toy_config)
    config["train"]["epochs"] = 30
    summary = run_training(config)
    assert summary["last_loss"] < 0.5 * summary["first_loss"]
    assert summary["epochs_completed"] == 30 and summary["best_score"] > 0

    run_dir = data_paths(config).raw_dir.parents[2] / "outputs" / "tiny"
    assert (run_dir / "config.yaml").exists() and (run_dir / "summary.json").exists()
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"type": "val"' in line and "CIDEr" in line for line in lines)

    # The best checkpoint is self-contained: rebuild the model and caption the test split.
    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.pt")
    device = torch.device("cpu")
    model, vocab, saved_config = model_from_checkpoint(checkpoint, False, device)
    data = load_experiment_data(saved_config, vocab=vocab)
    loader = make_image_loader(data, "test", batch_size=4)
    for beam_size in (1, 2):
        hypotheses = generate_captions(
            model, loader, vocab, device, True, beam_size=beam_size, use_amp=False
        )
        scores = score_captions(hypotheses, build_references(data.records["test"]))
        assert scores["BLEU-4"] > 0.9  # the mapping is deterministic, so it is learnable
    assert "<unk>" not in {token for tokens in hypotheses.values() for token in tokens}


def test_resume_reproduces_an_uninterrupted_run(toy_config):
    straight = copy.deepcopy(toy_config)
    straight["run_name"] = "straight"
    run_training(straight)

    resumed = copy.deepcopy(toy_config)
    resumed["run_name"] = "resumed"
    first = run_training(resumed, max_steps=5)  # stops in the middle of epoch 2 (3 steps/epoch)
    assert first["steps_this_run"] == 5 and first["epochs_completed"] == 1
    second = run_training(resumed, resume="auto")
    assert second["global_step"] == 12 and second["steps_this_run"] == 7

    outputs = data_paths(toy_config).raw_dir.parents[2] / "outputs"
    weights_a = load_checkpoint(outputs / "straight" / "checkpoints" / "last.pt")["decoder"]
    weights_b = load_checkpoint(outputs / "resumed" / "checkpoints" / "last.pt")["decoder"]
    for name, tensor in weights_a.items():
        assert torch.allclose(tensor, weights_b[name], atol=1e-6), name
    assert read_json(outputs / "resumed" / "summary.json")["epochs_completed"] == 4


def test_evaluate_checkpoint_and_report(toy_config):
    config = copy.deepcopy(toy_config)
    config["train"]["epochs"] = 8
    run_training(config)
    run_dir = data_paths(config).raw_dir.parents[2] / "outputs" / "tiny"
    best = run_dir / "checkpoints" / "best.pt"
    for split in ("val", "test"):
        result = evaluate_checkpoint(best, split, beam_size=2)
        saved = read_json(result["output"])
        assert result["num_images"] == 6 and saved["scores"] == result["scores"]
        assert len(saved["predictions"]) == 6 and saved["split"] == split
    assert (run_dir / "eval_test_beam2.json").exists()

    report = build_report(run_dir, samples=3, beam_size=2, worst=2, seed=1)
    assert "## Đường cong học" in report and "tốt nhất" in report
    assert "| val | beam 2 |" in report and "| test | beam 2 |" in report
    assert report.count("**Ảnh `") == 3 + 2  # three random test images, two worst val images
    # By default one reference caption is quoted per example; scores still use all of them.
    assert report.count("- Sinh ra:") == 5 and report.count("- Tham chiếu:") == 5
    both = build_report(run_dir, samples=3, beam_size=2, worst=2, seed=1, max_references=2)
    assert both.count("- Tham chiếu:") == 10
    # The same seed always selects the same examples.
    assert report == build_report(run_dir, samples=3, beam_size=2, worst=2, seed=1)


def test_fresh_run_never_overwrites_checkpoints_silently(toy_config):
    config = copy.deepcopy(toy_config)
    run_training(config, max_steps=2)
    with pytest.raises(RuntimeError, match="already contains checkpoints"):
        run_training(config, max_steps=2)
    run_dir = data_paths(config).raw_dir.parents[2] / "outputs" / "tiny"
    summary = run_training(config, max_steps=3, overwrite=True)
    assert summary["global_step"] == 3
    lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["step"] for line in lines] == [2]  # the old log is gone


def test_resume_rejects_a_different_vocabulary(toy_config):
    config = copy.deepcopy(toy_config)
    run_training(config, max_steps=2)
    paths = data_paths(config)
    vocab = read_json(paths.vocab)
    write_json([*vocab, "mới"], paths.vocab)
    with pytest.raises(RuntimeError, match="vocabulary"):
        run_training(config, resume="auto")
