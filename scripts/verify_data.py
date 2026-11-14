"""Check a downloaded dataset: every image readable, captions present, summary statistics.

Usage:
    python scripts/verify_data.py --config configs/datasets/uit_openviic.yaml

Writes data/processed/<dataset>/verify_report.json and exits with code 1 when images are
missing or unreadable, or when an image has no caption.
"""

from __future__ import annotations

import argparse

from vic.config import data_paths, load_any_config
from vic.data.annotations import load_annotations
from vic.data.verify import verify_dataset
from vic.utils import setup_logging, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="dataset (or experiment) YAML file")
    parser.add_argument("--skip-images", action="store_true", help="check captions only")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    logger = setup_logging()
    config = load_any_config(args.config)
    dataset = config["dataset"]
    paths = data_paths(config)
    records = load_annotations(paths.annotations)["images"]
    logger.info(f"Verifying {dataset['name']}: {len(records)} images")
    report = verify_dataset(
        paths.raw_dir,
        records,
        image_dir=dataset.get("image_dir"),
        check_images=not args.skip_images,
        workers=args.workers,
    )
    report_path = paths.annotations.with_name("verify_report.json")
    write_json(report, report_path, indent=2)

    for split, counts in report["splits"].items():
        logger.info(f"{split:>5}: {counts['images']} images, {counts['captions']} captions")
    logger.info(f"Captions per image : {report['captions_per_image']}")
    logger.info(f"Caption length     : {report['caption_length_syllables']} (syllables)")
    logger.info(
        f"Empty captions: {report['empty_captions']} | duplicates within an image: "
        f"{report['duplicate_captions_within_image']} | images without captions: "
        f"{len(report['images_without_captions'])}"
    )
    if not args.skip_images:
        logger.info(f"Image width  : {report['image_width']}")
        logger.info(f"Image height : {report['image_height']}")
        logger.info(f"Modes: {report['image_modes']} | formats: {report['image_formats']}")
        logger.info(
            f"EXIF-rotated: {report['images_with_exif_rotation']} | files without annotation: "
            f"{report.get('image_files_without_annotation', 'n/a')}"
        )
        logger.info(
            f"Missing images: {len(report['missing_images'])} | "
            f"unreadable images: {len(report['corrupt_images'])} | "
            f"truncated but readable: {len(report['truncated_images'])} "
            f"{report['truncated_images'][:10]}"
        )
        for item in report["corrupt_images"][:10]:
            logger.info(f"  unreadable {item['image_id']}: {item['error']}")
    logger.info(f"Result: {'OK' if report['ok'] else 'PROBLEMS FOUND'} -> {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
