from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def ngram_counts(tokens: Sequence[str], max_n: int = 4) -> Counter:
    """Count all n-grams of order 1..max_n; keys are tuples of tokens."""
    counts: Counter = Counter()
    for n in range(1, max_n + 1):
        for start in range(len(tokens) - n + 1):
            counts[tuple(tokens[start : start + n])] += 1
    return counts
