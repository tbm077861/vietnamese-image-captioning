"""Download a dataset and convert its annotations to the unified format.

Usage:
    python scripts/download_data.py --config configs/datasets/uit_openviic.yaml

Files go to data/raw/<dataset>/ (never committed). Downloads resume when interrupted; archives
are extracted once and then deleted unless --keep-archives is given. The unified annotation
file is written to data/processed/<dataset>/annotations.json.
"""

from __future__ import annotations

import argparse

from vic.config import data_paths, load_any_config
from vic.data.annotations import save_annotations, split_summary
from vic.data.download import download_file, extract_zip, extraction_marker, resolve_url
from vic.data.sources import load_records
from vic.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="dataset (or experiment) YAML file")
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep zip files after extraction"
    )
    parser.add_argument("--skip-download", action="store_true", help="only rebuild annotations")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_any_config(args.config)
    dataset = config["dataset"]
    paths = data_paths(config)
    logger.info(f"Dataset: {dataset.get('title', dataset['name'])} | license: {dataset['license']}")
    logger.info(f"Source : {dataset['homepage']}")

    if not args.skip_download:
        for file_spec in dataset.get("files", []):
            name = file_spec["name"]
            marker = extraction_marker(paths.raw_dir, name)
            if file_spec.get("extract") and marker.exists():
                logger.info(f"{name}: already extracted")
                continue
            archive = download_file(
                resolve_url(file_spec), paths.raw_dir / name, file_spec.get("bytes"), logger
            )
            if file_spec.get("extract"):
                extract_zip(archive, paths.raw_dir, logger)
                marker.write_text("ok", encoding="utf-8")
                if not args.keep_archives:
                    archive.unlink()

    records = load_records(paths.raw_dir, dataset)
    save_annotations(paths.annotations, dataset["name"], records)
    for split, counts in split_summary(records).items():
        logger.info(f"{split:>5}: {counts['images']} images, {counts['captions']} captions")
    logger.info(f"Wrote {paths.annotations}")
    logger.info("Next: python scripts/verify_data.py --config " + args.config)


if __name__ == "__main__":
    main()
