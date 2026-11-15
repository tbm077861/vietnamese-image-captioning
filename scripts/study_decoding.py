"""Compare decoding settings of one checkpoint on the validation split.

Usage:
    python scripts/study_decoding.py --checkpoint outputs/baseline_lstm/checkpoints/best.pt
    python scripts/study_decoding.py --checkpoint <ckpt> --beam-size 1 3 5 --no-repeat 0 3

Every combination of --beam-size, --beam-stop, --length-penalty and --no-repeat is decoded and
scored (greedy ignores the length penalty and the stopping rule, so it runs once per --no-repeat
value). Rows go to <run>/decoding_study.json and a Markdown table is printed. Decoding is always
chosen on validation, never on the test split.
"""

from __future__ import annotations

import argparse
import itertools
import sys

from vic.engine.evaluation import study_decoding

HEADER = (
    "| Beam | Dừng | Length penalty | Chặn lặp | CIDEr | BLEU-4 | Độ dài TB | Câu khác nhau "
    "| Âm tiết khác nhau | Lặp 3-gram (%) |"
)


def format_row(row: dict) -> str:
    greedy = row["beam_size"] <= 1
    stats = row["diagnostics"]
    return (
        f"| {'greedy' if greedy else row['beam_size']} | {'—' if greedy else row['beam_stop']} "
        f"| {'—' if greedy else row['length_penalty']} | {row['no_repeat_ngram_size'] or '—'} "
        f"| {row['scores']['CIDEr']:.4f} | {row['scores']['BLEU-4'] * 100:.2f} "
        f"| {stats['mean_length']:.1f} | {stats['distinct_captions']} "
        f"| {stats['distinct_tokens']} | {100 * stats['repeated_trigram_rate']:.1f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--beam-size", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--length-penalty", nargs="+", type=float, default=[0.0, 0.7, 1.0, 1.3])
    parser.add_argument("--beam-stop", nargs="+", default=["first", "best"])
    parser.add_argument("--no-repeat", nargs="+", type=int, default=[0], help="n-gram size; 0 off")
    parser.add_argument("--output", default=None)
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. data_root=...")
    args = parser.parse_args()

    settings = []
    for no_repeat in args.no_repeat:
        if 1 in args.beam_size:
            settings.append({"beam_size": 1, "no_repeat_ngram_size": no_repeat})
        wide = [beam for beam in args.beam_size if beam > 1]
        for beam, stop, penalty in itertools.product(wide, args.beam_stop, args.length_penalty):
            settings.append(
                {
                    "beam_size": beam,
                    "beam_stop": stop,
                    "length_penalty": penalty,
                    "no_repeat_ngram_size": no_repeat,
                }
            )
    rows = study_decoding(args.checkpoint, settings, args.output, args.overrides)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(HEADER)
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
