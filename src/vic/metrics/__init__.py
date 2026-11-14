"""Caption metrics: BLEU-1..4 and CIDEr-D (primary), exact-match METEOR (reference only)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from vic.metrics.bleu import corpus_bleu
from vic.metrics.cider import cider_d
from vic.metrics.meteor import meteor_exact

__all__ = ["compute_metrics", "corpus_bleu", "cider_d", "meteor_exact"]


def compute_metrics(
    references: Mapping[str, Sequence[Sequence[str]]],
    hypotheses: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Score tokenised hypotheses against tokenised references, keyed by image id.

    BLEU and METEOR are in [0, 1]; CIDEr is on its usual 0-10 scale.
    """
    missing = set(hypotheses) - set(references)
    if missing:
        raise KeyError(f"{len(missing)} hypotheses have no references, e.g. {sorted(missing)[:3]}")
    image_ids = sorted(hypotheses)
    refs = [references[image_id] for image_id in image_ids]
    hyps = [hypotheses[image_id] for image_id in image_ids]

    bleu = corpus_bleu(refs, hyps, max_n=4)
    cider, _ = cider_d(refs, hyps)
    scores = {f"BLEU-{order + 1}": value for order, value in enumerate(bleu)}
    scores["CIDEr"] = cider
    scores["METEOR"] = meteor_exact(refs, hyps)
    return scores
