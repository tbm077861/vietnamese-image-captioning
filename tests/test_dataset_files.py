"""Tests for the dataset layer: config loading, adapters, download helpers, verification."""

import json
import logging
import zipfile

import pytest
from PIL import Image

from vic.config import data_paths, load_any_config, load_config, save_yaml
from vic.data.annotations import load_annotations, make_record, save_annotations, split_summary
from vic.data.download import download_file, extract_zip, resolve_url
from vic.data.sources import load_records
from vic.data.verify import verify_dataset

LOGGER = logging.getLogger("test")


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_filename_captions_adapter_maps_dev_to_val_and_excludes(tmp_path):
    _write_json(tmp_path / "train.json", {"001.jpg": {"captions": ["một", "hai"]}})
    _write_json(tmp_path / "dev.json", {"002.jpg": {"captions": ["ba"]}})
    _write_json(tmp_path / "test.json", {"003.jpg": {"captions": ["bốn"]}})
    dataset = {
        "format": "filename_captions",
        "image_dir": "images",
        "annotation_files": {"train": "train.json", "dev": "dev.json", "test": "test.json"},
    }
    records = load_records(tmp_path, dataset)
    assert [(r["image_id"], r["split"], r["file_name"]) for r in records] == [
        ("001", "train", "images/001.jpg"),
        ("002", "val", "images/002.jpg"),
        ("003", "test", "images/003.jpg"),
    ]
    assert records[0]["captions"] == ["một", "hai"]
    assert len(load_records(tmp_path, {**dataset, "exclude_images": ["002"]})) == 2


def test_coco_captions_adapter_groups_captions_by_image(tmp_path):
    _write_json(
        tmp_path / "captions_train.json",
        {
            "images": [{"id": 7, "file_name": "a.jpg"}, {"id": 8, "file_name": "b.jpg"}],
            "annotations": [
                {"image_id": 7, "caption": "một"},
                {"image_id": 7, "caption": "hai"},
            ],
        },
    )
    dataset = {"format": "coco_captions", "annotation_files": {"train": "captions_train.json"}}
    records = load_records(tmp_path, dataset)
    assert len(records) == 1  # images without captions are skipped
    assert records[0] == make_record("7", "images/a.jpg", "train", ["một", "hai"])


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_records(tmp_path, {"format": "nope"})


def test_annotations_round_trip_and_duplicate_ids(tmp_path):
    records = [
        make_record("1", "images/1.jpg", "train", ["a", "b"]),
        make_record("2", "images/2.jpg", "test", ["c"]),
    ]
    path = tmp_path / "annotations.json"
    save_annotations(path, "toy", records)
    loaded = load_annotations(path)
    assert loaded["dataset"] == "toy" and loaded["images"] == records
    assert split_summary(records) == {
        "train": {"images": 1, "captions": 2},
        "test": {"images": 1, "captions": 1},
    }
    with pytest.raises(ValueError):
        save_annotations(path, "toy", [records[0], records[0]])
    with pytest.raises(ValueError):
        make_record("3", "x.jpg", "dev", [])


def test_config_merges_dataset_and_applies_overrides(tmp_path):
    dataset_file = tmp_path / "toy.yaml"
    save_yaml({"name": "toy", "format": "coco_captions"}, dataset_file)
    experiment_file = tmp_path / "experiment.yaml"
    save_yaml(
        {
            "seed": 1,
            "dataset_config": str(dataset_file),
            "data_root": str(tmp_path / "data"),
            "text": {"level": "syllable", "min_freq": 2},
            "model": {"encoder": {"name": "resnet50", "image_size": 224}},
        },
        experiment_file,
    )
    config = load_config(experiment_file, ["seed=7", "text.min_freq=3", "dataset.format=other"])
    assert config["seed"] == 7
    assert config["dataset"] == {"name": "toy", "format": "other"}
    paths = data_paths(config)
    assert paths.raw_dir == tmp_path / "data" / "raw" / "toy"
    assert paths.captions.parent.name == "syllable_minfreq3"
    assert paths.features_dir.name == "resnet50_224"
    # The data scripts accept the bare dataset file too.
    assert load_any_config(dataset_file)["dataset"]["name"] == "toy"
    with pytest.raises(ValueError):
        load_config(experiment_file, ["not-an-override"])


def test_download_skips_existing_file_and_finishes_complete_partial(tmp_path):
    done = tmp_path / "done.bin"
    done.write_bytes(b"12345")
    # No network access happens in either case: the URL is not even resolvable.
    assert download_file("http://invalid.invalid/x", done, 5, LOGGER) == done
    target = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(b"abc")
    assert download_file("http://invalid.invalid/x", target, 3, LOGGER).read_bytes() == b"abc"
    assert "id=abc" in resolve_url({"name": "f", "gdrive_id": "abc"})
    with pytest.raises(ValueError):
        resolve_url({"name": "f"})


def test_extract_zip_rejects_paths_outside_the_target(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as bundle:
        bundle.writestr("images/a.txt", "ok")
    extract_zip(good, tmp_path / "out", LOGGER)
    assert (tmp_path / "out" / "images" / "a.txt").read_text() == "ok"

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(RuntimeError):
        extract_zip(bad, tmp_path / "out2", LOGGER)
    assert not (tmp_path / "escape.txt").exists()


def test_verify_reports_missing_and_corrupt_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (32, 24), "red").save(image_dir / "ok.jpg")
    (image_dir / "broken.jpg").write_bytes(b"not an image")
    Image.new("RGB", (8, 8)).save(image_dir / "unused.jpg")
    records = [
        make_record("ok", "images/ok.jpg", "train", ["một con mèo", "một con mèo"]),
        make_record("broken", "images/broken.jpg", "train", ["hai con chó"]),
        make_record("gone", "images/gone.jpg", "test", ["  "]),
    ]
    report = verify_dataset(tmp_path, records, image_dir="images", workers=2)
    assert report["ok"] is False
    assert report["missing_images"] == ["gone"]
    assert [item["image_id"] for item in report["corrupt_images"]] == ["broken"]
    assert report["images_without_captions"] == ["gone"]
    assert report["empty_captions"] == 1
    assert report["duplicate_captions_within_image"] == 1
    assert report["image_width"]["max"] == 32 and report["image_height"]["max"] == 24
    assert report["image_files_without_annotation"] == 1
    assert report["caption_length_syllables"]["max"] == 3

    clean = verify_dataset(tmp_path, records[:1], image_dir="images", workers=1)
    assert clean["ok"] is True


def test_verify_keeps_images_with_a_cut_off_end(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    path = image_dir / "cut.jpg"
    Image.effect_noise((96, 96), 64).convert("RGB").save(path, quality=95)
    path.write_bytes(path.read_bytes()[:-40])
    records = [make_record("cut", "images/cut.jpg", "train", ["một bức ảnh"])]
    report = verify_dataset(tmp_path, records, image_dir="images", workers=1)
    assert report["truncated_images"] == ["cut"]
    assert report["corrupt_images"] == [] and report["ok"] is True
    assert report["image_width"]["max"] == 96
