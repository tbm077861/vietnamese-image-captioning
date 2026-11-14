"""CIDEr-D (Vedantam et al., 2015), following the reference COCO caption implementation.

TF-IDF weighted n-gram cosine similarity (n = 1..4) between a candidate and its references,
with count clipping and a Gaussian length penalty, scaled by 10. Document frequencies come from
the reference captions of the evaluated set, as in the standard COCO evaluation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from vic.metrics.ngrams import ngram_counts


def _to_vector(counts: Counter, doc_freq: Counter, log_num_docs: float, max_n: int):
    vector: list[dict] = [{} for _ in range(max_n)]
    norm = [0.0] * max_n
    length = 0
    for ngram, term_freq in counts.items():
        idf = log_num_docs - math.log(max(1.0, doc_freq.get(ngram, 0)))
        order = len(ngram) - 1
        weight = term_freq * idf
        vector[order][ngram] = weight
        norm[order] += weight * weight
        # The reference implementation measures length in bigrams; kept for comparable scores.
        if order == 1:
            length += term_freq
    return vector, [math.sqrt(value) for value in norm], length


def _similarity(hyp, ref, max_n: int, sigma: float) -> list[float]:
    hyp_vector, hyp_norm, hyp_length = hyp
    ref_vector, ref_norm, ref_length = ref
    length_penalty = math.exp(-((hyp_length - ref_length) ** 2) / (2 * sigma**2))
    values = [0.0] * max_n
    for order in range(max_n):
        for ngram, weight in hyp_vector[order].items():
            ref_weight = ref_vector[order].get(ngram, 0.0)
            values[order] += min(weight, ref_weight) * ref_weight
        if hyp_norm[order] != 0 and ref_norm[order] != 0:
            values[order] /= hyp_norm[order] * ref_norm[order]
        values[order] *= length_penalty
    return values


def cider_d(
    references: Sequence[Sequence[Sequence[str]]],
    hypotheses: Sequence[Sequence[str]],
    max_n: int = 4,
    sigma: float = 6.0,
) -> tuple[float, list[float]]:
    """Return (corpus score, per-image scores). Scores range from 0 to 10."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        return 0.0, []

    ref_counts = [[ngram_counts(ref, max_n) for ref in refs] for refs in references]
    doc_freq: Counter = Counter()
    for refs in ref_counts:
        for ngram in {ngram for ref in refs for ngram in ref}:
            doc_freq[ngram] += 1
    log_num_docs = math.log(len(references))

    scores = []
    for refs, hyp in zip(ref_counts, hypotheses, strict=True):
        hyp_vector = _to_vector(ngram_counts(hyp, max_n), doc_freq, log_num_docs, max_n)
        per_order = [0.0] * max_n
        for ref in refs:
            ref_vector = _to_vector(ref, doc_freq, log_num_docs, max_n)
            for order, value in enumerate(_similarity(hyp_vector, ref_vector, max_n, sigma)):
                per_order[order] += value
        scores.append(10.0 * sum(per_order) / max_n / len(refs))
    return sum(scores) / len(scores), scores
