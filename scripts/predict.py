"""Generate a Vietnamese caption for any image.

Usage:
    python scripts/predict.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt --image a.jpg
    python scripts/predict.py --checkpoint <ckpt> --image a.jpg b.png --beam-size 5
"""

from __future__ import annotations

import argparse
import sys

from vic.engine.inference import CaptionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, nargs="+", help="one or more image files")
    parser.add_argument("--beam-size", type=int, default=None, help="1 = greedy decoding")
    parser.add_argument("--device", default="auto", help="auto, cuda or cpu")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    predictor = CaptionPredictor(args.checkpoint, device=args.device)
    for image_path in args.image:
        print(f"{image_path}: {predictor.caption(image_path, beam_size=args.beam_size)}")


if __name__ == "__main__":
    main()
