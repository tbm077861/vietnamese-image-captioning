"""Naturalness guards: length, diversity and repetition statistics of a set of captions.

These numbers are not quality scores. They are watched next to BLEU / CIDEr so that a model
which gains score by producing fewer, more generic or more repetitive sentences is noticed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def has_repeated_ngram(tokens: Sequence[str], n: int) -> bool:
    """True when some n-gram occurs more than once in the caption."""
    seen = set()
    for start in range(len(tokens) - n + 1):
        ngram = tuple(tokens[start : start + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def caption_statistics(captions: Iterable[Sequence[str]]) -> dict[str, float]:
    """Statistics over tokenised captions (syllables everywhere in this project).

    ``repeated_bigram_rate`` and ``repeated_trigram_rate`` are the shares of captions in which a
    2-gram / 3-gram occurs more than once. Human captions repeat some 2-grams legitimately
    ("một người ... một người"), so compare the rates with those of the reference captions
    rather than with zero.
    """
    captions = [list(tokens) for tokens in captions]
    count = len(captions)
    if count == 0:
        raise ValueError("caption_statistics needs at least one caption")
    return {
        "num_captions": count,
        "distinct_captions": len({tuple(tokens) for tokens in captions}),
        "distinct_tokens": len({token for tokens in captions for token in tokens}),
        "mean_length": sum(len(tokens) for tokens in captions) / count,
        "repeated_bigram_rate": sum(has_repeated_ngram(t, 2) for t in captions) / count,
        "repeated_trigram_rate": sum(has_repeated_ngram(t, 3) for t in captions) / count,
    }
