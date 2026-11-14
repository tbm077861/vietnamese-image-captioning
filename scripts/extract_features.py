"""Run the frozen image encoder once and cache its features (float16) on disk.

Usage:
    python scripts/extract_features.py --config configs/baseline_lstm.yaml [--force]

Features for every image of the dataset go to data/features/<dataset>/<encoder>_<size>/.
Training with data.use_cached_features=true reads them instead of running the encoder.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from vic.config import data_paths, load_config
from vic.data.annotations import load_annotations
from vic.data.dataset import ImageDataset, ImageStore, collate_images
from vic.data.features import INDEX_FILE, create_feature_cache, write_feature_index
from vic.models import ImageEncoder
from vic.utils import get_device, read_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="recompute an existing cache")
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. model.encoder.name=..."
    )
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config, args.overrides)
    paths = data_paths(config)
    encoder_config = config["model"]["encoder"]
    image_size = int(encoder_config.get("image_size", 224))
    records = load_annotations(paths.annotations)["images"]
    image_ids = [record["image_id"] for record in records]

    index_path = paths.features_dir / INDEX_FILE
    if index_path.exists() and not args.force:
        if read_json(index_path)["image_ids"] == image_ids:
            logger.info(f"Feature cache already complete: {paths.features_dir} (use --force)")
            return
        logger.info("Existing cache does not match the annotations; recomputing")
    index_path.unlink(missing_ok=True)  # the index marks a finished cache

    device = get_device(config.get("device", "auto"))
    use_amp = device.type == "cuda"
    encoder = ImageEncoder(encoder_config["name"], pretrained=True, trainable=False).to(device)
    loader = DataLoader(
        ImageDataset(records, ImageStore(paths.raw_dir, records, image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_images,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    cache = None
    row = 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, _ in loader:
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                features = encoder(images.to(device, non_blocking=True))
            features = features.to(torch.float16).cpu().numpy()
            if cache is None:
                shape = (len(records), *features.shape[1:])
                cache = create_feature_cache(paths.features_dir, shape)
                logger.info(f"Extracting {shape} float16 features with {encoder.name} on {device}")
            cache[row : row + len(features)] = features
            row += len(features)
            if (row // args.batch_size) % 20 == 0:
                logger.info(f"{row}/{len(records)} images")
    cache.flush()
    write_feature_index(
        paths.features_dir,
        image_ids,
        encoder=encoder.name,
        image_size=image_size,
        shape=list(cache.shape),
        dtype="float16",
    )
    size_gb = cache.nbytes / 1024**3
    elapsed = time.perf_counter() - start
    logger.info(f"Done: {row} images, {size_gb:.2f} GB, {elapsed:.0f} s -> {paths.features_dir}")


if __name__ == "__main__":
    main()
