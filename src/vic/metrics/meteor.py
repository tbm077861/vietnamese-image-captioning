"""METEOR with exact matching only - a reference metric, not a headline number.

The full METEOR also matches stems, synonyms and paraphrases, which needs language resources
that do not exist for Vietnamese. This version aligns identical tokens only, so it rewards
unigram precision / recall and penalises fragmented word order, nothing more. Parameters follow
METEOR 1.5 for English (alpha 0.85, beta 0.2, gamma 0.6). The corpus score is the mean over
images of the best score against any reference.
"""

from __future__ import annotations

from collections.abc import Sequence

ALPHA = 0.85
BETA = 0.2
GAMMA = 0.6


def _align(hyp: Sequence[str], ref: Sequence[str]) -> list[tuple[int, int]]:
    """Align identical tokens one-to-one, taking the longest common runs first.

    Greedy on run length: always yields the maximum number of matches and, in practice, few
    chunks (the reference tool searches for the minimum).
    """
    hyp_free = [True] * len(hyp)
    ref_free = [True] * len(ref)
    pairs: list[tuple[int, int]] = []
    while True:
        best_length, best_hyp, best_ref = 0, 0, 0
        for i in range(len(hyp)):
            for j in range(len(ref)):
                length = 0
                while (
                    i + length < len(hyp)
                    and j + length < len(ref)
                    and hyp_free[i + length]
                    and ref_free[j + length]
                    and hyp[i + length] == ref[j + length]
                ):
                    length += 1
                if length > best_length:
                    best_length, best_hyp, best_ref = length, i, j
        if best_length == 0:
            return sorted(pairs)
        for offset in range(best_length):
            hyp_free[best_hyp + offset] = False
            ref_free[best_ref + offset] = False
            pairs.append((best_hyp + offset, best_ref + offset))


def meteor_exact_sentence(hyp: Sequence[str], ref: Sequence[str]) -> float:
    if not hyp or not ref:
        return 0.0
    pairs = _align(hyp, ref)
    matches = len(pairs)
    if matches == 0:
        return 0.0
    chunks = 1 + sum(
        1
        for (prev_hyp, prev_ref), (cur_hyp, cur_ref) in zip(pairs, pairs[1:], strict=False)
        if cur_hyp != prev_hyp + 1 or cur_ref != prev_ref + 1
    )
    precision = matches / len(hyp)
    recall = matches / len(ref)
    f_mean = precision * recall / (ALPHA * precision + (1 - ALPHA) * recall)
    identical = chunks == 1 and matches == len(hyp) == len(ref)
    fragmentation = 0.0 if identical else chunks / matches
    return f_mean * (1 - GAMMA * fragmentation**BETA)


def meteor_exact(
    references: Sequence[Sequence[Sequence[str]]], hypotheses: Sequence[Sequence[str]]
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        return 0.0
    scores = [
        max((meteor_exact_sentence(hyp, ref) for ref in refs), default=0.0)
        for refs, hyp in zip(references, hypotheses, strict=True)
    ]
    return sum(scores) / len(scores)
