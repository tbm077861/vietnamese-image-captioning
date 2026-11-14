"""Dataset integrity checks and descriptive statistics."""

from __future__ import annotations

import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

from vic.data.annotations import split_summary

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
EXIF_ORIENTATION_TAG = 0x0112


def _describe(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": round(statistics.fmean(ordered), 2),
        "median": ordered[len(ordered) // 2],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def inspect_image(path: Path) -> dict:
    """Fully decode one image and describe it; failures are reported, never raised."""
    if not path.exists():
        return {"status": "missing"}
    try:
        with Image.open(path) as image:
            image.load()
            return {
                "status": "ok",
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
                "rotated": image.getexif().get(EXIF_ORIENTATION_TAG, 1) not in (0, 1),
            }
    except Exception as error:  # noqa: BLE001 - any decoding failure is reported
        return {"status": "corrupt", "error": f"{type(error).__name__}: {error}"[:200]}


def inspect_images(paths: list[Path], workers: int = 8) -> tuple[list[dict], list[bool]]:
    """Inspect all files strictly, then retry failures while tolerating a cut-off end.

    Files whose last bytes are missing still decode (the final rows are filled in), which is
    also how the training loader reads them; they are kept and flagged as truncated.
    Returns (results, truncated flags).
    """
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    truncated = [False] * len(paths)
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(inspect_image, paths))
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        for index, result in enumerate(results):
            if result["status"] == "corrupt":
                retried = inspect_image(paths[index])
                if retried["status"] == "ok":
                    results[index] = retried
                    truncated[index] = True
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous
    return results, truncated


def verify_dataset(
    raw_dir: str | Path,
    records: list[dict],
    image_dir: str | None = None,
    check_images: bool = True,
    workers: int = 8,
) -> dict:
    """Check captions and images; ``report["ok"]`` is False when something must be fixed."""
    raw_dir = Path(raw_dir)
    captions_per_image = [len(record["captions"]) for record in records]
    lengths = [len(caption.split()) for record in records for caption in record["captions"]]
    empty_captions = sum(
        not caption.strip() for record in records for caption in record["captions"]
    )
    duplicate_captions = sum(
        len(record["captions"]) - len(set(record["captions"])) for record in records
    )
    without_captions = [
        record["image_id"] for record in records if not any(c.strip() for c in record["captions"])
    ]

    report = {
        "num_images": len(records),
        "num_captions": len(lengths),
        "splits": split_summary(records),
        "captions_per_image": _describe(captions_per_image),
        "caption_length_syllables": _describe(lengths),
        "empty_captions": empty_captions,
        "duplicate_captions_within_image": duplicate_captions,
        "images_without_captions": without_captions,
    }

    missing: list[str] = []
    corrupt: list[dict] = []
    truncated: list[str] = []
    if check_images:
        paths = [raw_dir / record["file_name"] for record in records]
        results, truncated_flags = inspect_images(paths, workers)
        truncated = [
            r["image_id"] for r, flag in zip(records, truncated_flags, strict=True) if flag
        ]
        good = [result for result in results if result["status"] == "ok"]
        for record, result in zip(records, results, strict=True):
            if result["status"] == "missing":
                missing.append(record["image_id"])
            elif result["status"] == "corrupt":
                corrupt.append({"image_id": record["image_id"], "error": result["error"]})
        report["image_width"] = _describe([result["width"] for result in good])
        report["image_height"] = _describe([result["height"] for result in good])
        report["image_modes"] = dict(Counter(result["mode"] for result in good))
        report["image_formats"] = dict(Counter(str(result["format"]) for result in good))
        report["images_with_exif_rotation"] = sum(result["rotated"] for result in good)
        if image_dir is not None and (raw_dir / image_dir).is_dir():
            referenced = {Path(record["file_name"]).name for record in records}
            on_disk = [
                path.name
                for path in (raw_dir / image_dir).iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            ]
            report["image_files_without_annotation"] = sum(
                name not in referenced for name in on_disk
            )
    report["missing_images"] = missing
    report["corrupt_images"] = corrupt
    report["truncated_images"] = truncated  # readable with a cut-off end: a warning only
    report["ok"] = not (missing or corrupt or without_captions)
    return report
