"""Decoding settings shared by evaluation, sweeps and the experiment table.

Kept free of heavy imports so that the experiment table can be rebuilt without PyTorch.
"""

from __future__ import annotations

PROTOCOL_DECODING = {"length_penalty": 0.7, "beam_stop": "first", "no_repeat_ngram_size": 0}
"""Decoding of the model comparison (columns "beam 3" / "greedy"), unchanged since stage 5 began.

Model selection uses CIDEr with beam 3 under this decoding.
"""

CHOSEN_DECODING = {
    "beam_size": 3,
    "length_penalty": 0.7,
    "beam_stop": "first",
    "no_repeat_ngram_size": 3,
}
"""Decoding chosen on the validation split in step 2 for the final model (docs/EXPERIMENTS.md)."""


def _settings(decoding: dict) -> tuple[int, float, str, int]:
    return (
        int(decoding.get("beam_size", 3)),
        float(decoding.get("length_penalty", 0.7)),
        str(decoding.get("beam_stop", "first")),
        int(decoding.get("no_repeat_ngram_size", 0)),
    )


def evaluation_filename(split: str, decoding: dict) -> str:
    """``eval_<split>_beam<k>[_lp<x>][_stop<rule>][_norepeat<n>].json``.

    Protocol settings add no suffix, so comparison results keep their historical names and an
    evaluation with another decoding never overwrites them. Greedy ignores the beam options.
    """
    beam, penalty, stop, no_repeat = _settings(decoding)
    name = f"eval_{split}_beam{beam}"
    if beam > 1 and penalty != PROTOCOL_DECODING["length_penalty"]:
        name += f"_lp{penalty:g}"
    if beam > 1 and stop != PROTOCOL_DECODING["beam_stop"]:
        name += f"_stop{stop}"
    if no_repeat:
        name += f"_norepeat{no_repeat}"
    return name + ".json"


def is_protocol_decoding(decoding: dict | None) -> bool:
    """True for the comparison decoding; files written before decoding was recorded count too."""
    if decoding is None:
        return True
    return evaluation_filename("x", decoding) == f"eval_x_beam{_settings(decoding)[0]}.json"


def decoding_label(decoding: dict) -> str:
    """Short Vietnamese description, e.g. "beam 3, LP 0,7, dừng first, chặn lặp 3-gram"."""
    beam, penalty, stop, no_repeat = _settings(decoding)
    parts = ["greedy"] if beam <= 1 else [f"beam {beam}", f"LP {penalty:g}".replace(".", ","),
                                          f"dừng {stop}"]  # fmt: skip
    parts.append(f"chặn lặp {no_repeat}-gram" if no_repeat else "không chặn lặp")
    return ", ".join(parts)
