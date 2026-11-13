"""Caption tokenisation at syllable or word level.

Vietnamese words can span several space-separated syllables ("sân bóng"). At word level the
syllables of one word are joined with an underscore ("sân_bóng"), the usual convention.
"""

from __future__ import annotations

from collections.abc import Iterable

from vic.text.normalize import normalize_caption

LEVELS = ("syllable", "word")


class CaptionTokenizer:
    def __init__(
        self, level: str = "word", lowercase: bool = True, strip_punctuation: bool = True
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"Unknown tokenisation level {level!r}; expected one of {LEVELS}")
        self.level = level
        self.lowercase = lowercase
        self.strip_punctuation = strip_punctuation
        self._segment = None

    def normalize(self, text: str) -> str:
        return normalize_caption(text, self.lowercase, self.strip_punctuation)

    def tokenize(self, text: str) -> list[str]:
        text = self.normalize(text)
        if not text:
            return []
        if self.level == "syllable":
            return text.split()
        return [word.replace(" ", "_") for word in self._word_segment(text)]

    def _word_segment(self, text: str) -> list[str]:
        if self._segment is None:
            try:
                from underthesea import word_tokenize
            except ImportError as error:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "Word-level tokenisation needs the 'underthesea' package "
                    "(pip install -r requirements.txt) or set text.level to 'syllable'."
                ) from error
            self._segment = word_tokenize
        return self._segment(text)

    def to_config(self) -> dict:
        return {
            "level": self.level,
            "lowercase": self.lowercase,
            "strip_punctuation": self.strip_punctuation,
        }


def to_syllables(tokens: Iterable[str]) -> list[str]:
    """Split word-level tokens back into syllables ("sân_bóng" -> "sân", "bóng")."""
    return " ".join(tokens).replace("_", " ").split()


def detokenize(tokens: Iterable[str]) -> str:
    """Turn tokens into a readable sentence."""
    return " ".join(to_syllables(tokens))
