"""Token <-> id mapping with special tokens."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class Vocabulary:
    def __init__(self, tokens: Sequence[str]) -> None:
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError(f"The vocabulary must start with {SPECIAL_TOKENS}")
        if len(set(tokens)) != len(tokens):
            raise ValueError("The vocabulary contains duplicate tokens")
        self.itos = list(tokens)
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(
        cls,
        token_lists: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> Vocabulary:
        """Build from tokenised captions; ordering is deterministic (frequency, then token)."""
        counts = Counter(token for tokens in token_lists for token in tokens)
        for special in SPECIAL_TOKENS:
            counts.pop(special, None)
        kept = [token for token, count in counts.items() if count >= min_freq]
        kept.sort(key=lambda token: (-counts[token], token))
        if max_size is not None:
            kept = kept[: max(0, max_size - len(SPECIAL_TOKENS))]
        return cls([*SPECIAL_TOKENS, *kept])

    def encode(self, tokens: Sequence[str], add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in tokens]
        return [self.bos_id, *ids, self.eos_id] if add_special else ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        """Convert ids to tokens, stopping at the first end-of-sentence id."""
        tokens = []
        for index in ids:
            index = int(index)
            if index == self.eos_id:
                if not skip_special:
                    tokens.append(EOS_TOKEN)
                break
            if skip_special and index in (self.pad_id, self.bos_id):
                continue
            tokens.append(self.itos[index])
        return tokens

    def to_list(self) -> list[str]:
        return list(self.itos)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.itos, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Vocabulary:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
