"""Vietnamese caption normalisation.

Steps: Unicode NFC, character fixes, lowercase, tone-mark placement, punctuation, whitespace.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that look like Vietnamese letters but are different code points.
_CHAR_FIXES = str.maketrans(
    {
        "Ð": "Đ",  # LATIN CAPITAL LETTER ETH -> D WITH STROKE
        "ð": "đ",  # LATIN SMALL LETTER ETH -> d with stroke
        "​": "",  # zero-width space
        "﻿": "",  # byte order mark
        " ": " ",  # no-break space
        "_": " ",  # the underscore is reserved for joining syllables of one word
    }
)

# Tone marks on the "oa", "oe", "uy" rhymes can be written in two styles ("hoà" / "hòa").
# Both are valid, so captions typed by different people mix them and split the vocabulary.
# We keep a single style: the mark goes on the first vowel when the rhyme ends the syllable.
_TONE_STYLE = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}  # fmt: skip
# "qu" is a consonant cluster, so "quý" already carries its mark on the only vowel.
_TONE_STYLE_RE = re.compile(r"(?<!q)(" + "|".join(_TONE_STYLE) + r")(?!\w)")

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tone_placement(text: str) -> str:
    """Rewrite new-style tone placement ("hoà", "thuỷ") to the old style ("hòa", "thủy").

    Expects lowercase NFC text.
    """
    return _TONE_STYLE_RE.sub(lambda match: _TONE_STYLE[match.group(1)], text)


def normalize_caption(text: str, lowercase: bool = True, strip_punctuation: bool = True) -> str:
    """Return a cleaned caption with single spaces between syllables."""
    text = unicodedata.normalize("NFC", text).translate(_CHAR_FIXES)
    if lowercase:
        text = text.lower()
        text = normalize_tone_placement(text)
    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
