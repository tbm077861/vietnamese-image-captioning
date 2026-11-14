"""Corpus-level BLEU-1..N with multiple references (Papineni et al., 2002).

Matches the COCO caption evaluation: clipped n-gram precision against all references, brevity
penalty from the closest reference length, no smoothing.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from vic.metrics.ngrams import ngram_counts


def corpus_bleu(
    references: Sequence[Sequence[Sequence[str]]],
    hypotheses: Sequence[Sequence[str]],
    max_n: int = 4,
) -> list[float]:
    """Return [BLEU-1, ..., BLEU-max_n], each in [0, 1]."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    matched = [0] * max_n
    total = [0] * max_n
    hyp_length = 0
    ref_length = 0
    for refs, hyp in zip(references, hypotheses, strict=True):
        hyp_length += len(hyp)
        ref_length += min((len(ref) for ref in refs), key=lambda n: (abs(n - len(hyp)), n))

        max_ref_counts: Counter = Counter()
        for ref in refs:
            for ngram, count in ngram_counts(ref, max_n).items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)
        for ngram, count in ngram_counts(hyp, max_n).items():
            order = len(ngram) - 1
            total[order] += count
            matched[order] += min(count, max_ref_counts[ngram])

    if hyp_length == 0:
        return [0.0] * max_n
    brevity_penalty = 1.0 if hyp_length >= ref_length else math.exp(1.0 - ref_length / hyp_length)

    scores = []
    log_precision_sum = 0.0
    zero_precision = False
    for order in range(max_n):
        if matched[order] == 0 or total[order] == 0:
            zero_precision = True
        if zero_precision:
            scores.append(0.0)
            continue
        log_precision_sum += math.log(matched[order] / total[order])
        scores.append(brevity_penalty * math.exp(log_precision_sum / (order + 1)))
    return scores
