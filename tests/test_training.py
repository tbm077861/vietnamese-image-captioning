"""End-to-end training tests on a tiny synthetic dataset (CPU, cached features)."""

import copy
import json

import numpy as np
import pytest
import torch

from vic.config import data_paths, save_yaml
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
from vic.engine.experiments import (
    BEGIN_MARKER,
    END_MARKER,
    collect_runs,
    experiments_table,
    update_experiments_doc,
)
from vic.engine.report import build_comparison, build_report
from vic.engine.sweep import run_sweep
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
    with pytest.raises(PermissionError, match="locked"):
        evaluate_checkpoint(best, "test", beam_size=2)
    assert not (run_dir / "eval_test_beam2.json").exists()
    for split in ("val", "test"):
        result = evaluate_checkpoint(best, split, beam_size=2, final=split == "test")
        saved = read_json(result["output"])
        assert saved["diagnostics"]["num_captions"] == 6
        assert saved["reference_diagnostics"]["num_captions"] == 12
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

    blocked = evaluate_checkpoint(best, "val", decoding={"no_repeat_ngram_size": 2})
    assert blocked["output"].endswith("eval_val_beam2_norepeat2.json")
    assert blocked["decoding"]["no_repeat_ngram_size"] == 2 and blocked["beam_size"] == 2
    assert read_json(run_dir / "eval_val_beam2.json")["decoding"]["no_repeat_ngram_size"] == 0
    with_blocking = build_report(run_dir, samples=3, beam_size=2, worst=2, seed=1)
    assert "| val | beam 2, LP 0,7, dừng first, chặn lặp 2-gram |" in with_blocking
    assert with_blocking.count("**Ảnh `") == 5


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


def _outputs(config):
    return data_paths(config).raw_dir.parents[2] / "outputs"


def test_summary_records_status_commit_and_time(toy_config):
    config = copy.deepcopy(toy_config)
    partial = run_training(config, max_steps=2)
    assert partial["status"] == "interrupted"
    summary = run_training(config, resume="auto")
    assert summary["status"] == "completed" and summary["seed"] == 0
    assert summary["parameter_count"] > 0 and summary["best_epoch"] >= 1
    assert summary["train_seconds"] >= partial["train_seconds"] > 0
    assert "git_commit" in summary and summary["init_from"] is None
    assert summary["overrides"] is None
    assert [item["epoch"] for item in summary["resume_history"]] == [0]
    assert summary["resume_history"][0]["config_changes"] == []


def test_resume_records_config_changes(toy_config):
    config = copy.deepcopy(toy_config)
    run_training(config, max_steps=4)  # stops in epoch 2
    config["data"]["features_in_memory"] = False
    config["overrides"] = ["data.features_in_memory=false"]
    summary = run_training(config, resume="auto")
    assert summary["status"] == "completed" and summary["overrides"] == config["overrides"]
    (resume,) = summary["resume_history"]
    assert resume["epoch"] == 1 and resume["global_step"] == 4
    assert resume["config_changes"] == ["data.features_in_memory: True -> False"]
    rows = {row["run"]: row for row in collect_runs(_outputs(config), beam_size=2)}
    assert rows["tiny"]["resume_history"] == summary["resume_history"]
    table = "\n".join(experiments_table(list(rows.values()), beam_size=2))
    assert (
        "ngắt rồi resume từ epoch 1" in table and "override: data.features_in_memory=false" in table
    )


def test_failed_run_writes_a_summary(toy_config):
    config = copy.deepcopy(toy_config)
    config["train"]["lr"] = float("nan")  # the first loss is finite, the second is not
    with pytest.raises(RuntimeError, match="Non-finite"):
        run_training(config)
    summary = read_json(_outputs(config) / "tiny" / "summary.json")
    assert summary["status"] == "failed" and "Non-finite" in summary["error"]


def test_init_from_copies_weights_but_not_progress(toy_config):
    source = copy.deepcopy(toy_config)
    source["run_name"] = "source"
    run_training(source)
    best = _outputs(source) / "source" / "checkpoints" / "best.pt"

    target = copy.deepcopy(toy_config)
    target["run_name"] = "target"
    target["train"]["lr"] = 0.0  # weights must stay exactly those of the source
    summary = run_training(target, init_from=str(best), max_steps=1)
    assert summary["global_step"] == 1 and summary["init_from"] == str(best)
    started = load_checkpoint(_outputs(target) / "target" / "checkpoints" / "last.pt")
    assert started["progress"]["epoch"] == 0 and started["progress"]["best_score"] is None
    assert started["config"]["train"]["init_from"] == str(best)
    for name, tensor in load_checkpoint(best)["decoder"].items():
        assert torch.equal(tensor, started["decoder"][name]), name


def test_sweep_runs_seeds_skips_finished_and_survives_failures(toy_config, tmp_path):
    good = tmp_path / "good.yaml"
    save_yaml(toy_config, good)
    broken_config = copy.deepcopy(toy_config)
    broken_config["run_name"] = "broken"
    broken_config["model"]["decoder"]["type"] = "does_not_exist"
    broken = tmp_path / "broken.yaml"
    save_yaml(broken_config, broken)
    doc = tmp_path / "EXPERIMENTS.md"
    lines = ["# Title", "", BEGIN_MARKER, "old", END_MARKER, "", "Keep me.", ""]
    doc.write_text("\n".join(lines), encoding="utf-8")

    results = run_sweep(
        [broken, good], seeds=[0, 1], beam_sizes=(1, 2), overrides=["eval.batch_size=8"],
        doc_path=doc,
    )  # fmt: skip
    by_run = {entry["run"]: entry for entry in results}
    assert set(by_run) == {"broken", "broken_seed1", "tiny", "tiny_seed1"}
    assert by_run["broken"]["status"] == "failed" and "does_not_exist" in by_run["broken"]["error"]
    assert by_run["tiny"]["status"] == by_run["tiny_seed1"]["status"] == "completed"
    outputs = _outputs(toy_config)
    assert (outputs / "tiny_seed1" / "eval_val_beam2.json").exists()
    assert (outputs / "tiny_seed1" / "eval_val_beam3_norepeat3.json").exists()
    assert set(by_run["tiny"]["cider"]) == {"beam 1", "beam 2", "chosen"}
    assert read_json(outputs / "tiny" / "summary.json")["overrides"] == ["eval.batch_size=8"]
    assert not list(outputs.glob("*/eval_test_*.json"))  # a sweep never touches the test split

    text = doc.read_text(encoding="utf-8")
    assert "Keep me." in text and "old" not in text
    assert "`tiny_seed1`" in text and "**lỗi**" in text and "### Độ nhiễu giữa các seed" in text
    rows = {row["run"]: row for row in collect_runs(outputs, beam_size=2)}
    assert rows["tiny"]["status"] == "completed" and rows["tiny"]["cider"] is not None
    assert rows["tiny"]["diagnostics"]["num_captions"] == 6 and rows["tiny"]["parameters"] > 0
    assert rows["tiny"]["cider_chosen"] is not None and "giải mã chốt" in text
    assert rows["broken"]["status"] == "failed" and rows["broken"]["cider"] is None

    again = run_sweep([good], seeds=[0, 1], beam_sizes=(1, 2), doc_path=None)
    assert [entry["status"] for entry in again] == ["skipped", "skipped"]
    assert again[0]["cider"] == by_run["tiny"]["cider"]


def test_experiments_doc_is_created_and_comparison_uses_fixed_images(toy_config, tmp_path):
    first = copy.deepcopy(toy_config)
    second = copy.deepcopy(toy_config)
    second["run_name"] = "other"
    outputs = _outputs(toy_config)
    for config in (first, second):
        run_training(config)
        evaluate_checkpoint(outputs / config["run_name"] / "checkpoints" / "best.pt", "val", 2)
    doc = tmp_path / "docs" / "EXPERIMENTS.md"
    rows = update_experiments_doc(outputs, doc, beam_size=2)
    assert len(rows) == 2 and "`other`" in doc.read_text(encoding="utf-8")

    comparison = build_comparison([outputs / "tiny", outputs / "other"], beam_size=2, samples=4)
    assert comparison.count("**Ảnh `") == 4 and comparison.count("- `other`:") == 4
    assert comparison == build_comparison(
        [outputs / "tiny", outputs / "other"], beam_size=2, samples=4
    )


def test_attention_decoder_trains_end_to_end(toy_config):
    config = copy.deepcopy(toy_config)
    config["train"]["epochs"] = 30
    config["model"]["decoder"] = {
        "type": "lstm_attention",
        "embed_dim": 32,
        "hidden_dim": 48,
        "attention_dim": 40,
        "dropout": 0.0,
    }
    summary = run_training(config)
    assert summary["last_loss"] < 0.5 * summary["first_loss"] and summary["best_score"] > 0
    best = _outputs(config) / "tiny" / "checkpoints" / "best.pt"
    result = evaluate_checkpoint(best, "val", beam_size=2, overrides=["eval.beam_stop=best"])
    assert result["decoding"]["beam_stop"] == "best" and result["scores"]["BLEU-1"] > 0.5


def test_transformer_decoder_trains_end_to_end(toy_config):
    config = copy.deepcopy(toy_config)
    config["train"].update({"epochs": 40, "lr": 0.003, "label_smoothing": 0.1})
    config["model"]["decoder"] = {
        "type": "transformer",
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 64,
        "dropout": 0.0,
    }
    summary = run_training(config)
    assert summary["last_loss"] < 0.6 * summary["first_loss"] and summary["best_score"] > 0
    best = _outputs(config) / "tiny" / "checkpoints" / "best.pt"
    result = evaluate_checkpoint(best, "val", beam_size=2)
    assert result["scores"]["BLEU-1"] > 0.5
