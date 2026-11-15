"""Greedy and beam-search decoding for any decoder implementing the shared interface."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _suppress(logits: torch.Tensor, suppress_ids: Sequence[int]) -> torch.Tensor:
    """Make the given token ids (e.g. <unk>, <pad>, <bos>) impossible to generate."""
    if suppress_ids:
        logits = logits.float()
        logits[:, list(suppress_ids)] = float("-inf")
    return logits


def _banned_tokens(sequence: Sequence[int], ngram_size: int) -> list[int]:
    """Tokens that would complete an n-gram already present in ``sequence``."""
    if ngram_size <= 0 or len(sequence) < ngram_size:
        return []
    prefix = list(sequence[len(sequence) - ngram_size + 1 :]) if ngram_size > 1 else []
    return [
        sequence[start + ngram_size - 1]
        for start in range(len(sequence) - ngram_size + 1)
        if list(sequence[start : start + ngram_size - 1]) == prefix
    ]


@torch.no_grad()
def greedy_decode(
    decoder: nn.Module,
    features: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_length: int = 30,
    suppress_ids: Sequence[int] = (),
    no_repeat_ngram_size: int = 0,
) -> list[list[int]]:
    """Decode a whole batch; returns token ids without the start / end markers.

    ``no_repeat_ngram_size`` > 0 forbids generating an n-gram of that size twice in a caption.
    """
    batch = features.size(0)
    state = decoder.init_state(features)
    tokens = torch.full((batch,), bos_id, dtype=torch.long, device=features.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=features.device)
    histories: list[list[int]] = [[] for _ in range(batch)]
    steps = []
    for _ in range(max_length):
        logits, state = decoder.step(tokens, state)
        logits = _suppress(logits, suppress_ids)
        if no_repeat_ngram_size > 0:
            logits = logits.float()
            for row, history in enumerate(histories):
                banned = _banned_tokens(history, no_repeat_ngram_size)
                if banned:
                    logits[row, banned] = float("-inf")
        tokens = logits.argmax(dim=-1)
        steps.append(tokens)
        if no_repeat_ngram_size > 0:
            for history, token in zip(histories, tokens.tolist(), strict=True):
                history.append(token)
        finished |= tokens == eos_id
        if bool(finished.all()):
            break
    sequences = torch.stack(steps, dim=1).tolist()
    return [seq[: seq.index(eos_id)] if eos_id in seq else seq for seq in sequences]


BEAM_STOPS = ("first", "best")


@torch.no_grad()
def beam_search(
    decoder: nn.Module,
    features: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 3,
    max_length: int = 30,
    length_penalty: float = 0.7,
    suppress_ids: Sequence[int] = (),
    no_repeat_ngram_size: int = 0,
    stop: str = "first",
) -> list[int]:
    """Decode one image (features of shape (1, regions, channels)) with beam search.

    Hypotheses are ranked by ``log_prob / length ** length_penalty``; 0 disables normalisation.

    ``stop`` selects when the search ends:

    - "first": a finished hypothesis leaves the beam, which shrinks, and the search ends once
      ``beam_size`` hypotheses have finished. Fast, but the first hypotheses to finish are the
      short ones, so longer captions with a better normalised score are never explored.
    - "best": the beam keeps ``beam_size`` live hypotheses and the search ends only when no live
      hypothesis can still beat the best finished one (its score can only fall, and its length
      can at most reach ``max_length``).

    ``no_repeat_ngram_size`` > 0 forbids generating an n-gram of that size twice in a caption.
    """
    if features.size(0) != 1:
        raise ValueError("beam_search decodes one image at a time")
    if stop not in BEAM_STOPS:
        raise ValueError(f"Unknown beam stop {stop!r}; available: {BEAM_STOPS}")
    device = features.device
    state = decoder.init_state(features)
    tokens = torch.full((1,), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    sequences: list[list[int]] = [[]]
    completed: list[tuple[float, list[int]]] = []
    # "best" looks at twice the beam so that finished hypotheses do not empty it.
    width = beam_size if stop == "first" else 2 * beam_size

    def normalised(score: float, length: int) -> float:
        return score / (max(length, 1) ** length_penalty)

    for _ in range(max_length):
        logits, state = decoder.step(tokens, state)
        log_probs = torch.log_softmax(_suppress(logits.float(), suppress_ids), dim=-1)
        if no_repeat_ngram_size > 0:
            for row, sequence in enumerate(sequences):
                banned = _banned_tokens(sequence, no_repeat_ngram_size)
                if banned:
                    log_probs[row, banned] = float("-inf")
        vocab_size = log_probs.size(-1)
        candidates = (scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_index = candidates.topk(min(width, candidates.numel()))
        beam_index = top_index // vocab_size
        token_index = top_index % vocab_size

        keep = []
        new_sequences = []
        for position, (beam, token, score) in enumerate(
            zip(beam_index.tolist(), token_index.tolist(), top_scores.tolist(), strict=True)
        ):
            if score == float("-inf"):
                continue
            if token == eos_id:
                if position < beam_size:  # an end ranked below the beam is not a real candidate
                    completed.append((normalised(score, len(sequences[beam]) + 1), sequences[beam]))
            elif len(keep) < beam_size:
                keep.append(position)
                new_sequences.append([*sequences[beam], token])
        if not keep:
            break
        keep_index = torch.tensor(keep, dtype=torch.long, device=device)
        state = decoder.reorder_state(state, beam_index[keep_index])
        tokens = token_index[keep_index]
        scores = top_scores[keep_index]
        sequences = new_sequences
        if stop == "first":
            if len(completed) >= beam_size:
                break
        elif completed:
            # Upper bound of what the best live hypothesis can still reach.
            bound = float(scores.max()) / ((max_length + 1) ** length_penalty)
            if max(score for score, _ in completed) >= bound:
                break
    else:
        # Length limit reached: unfinished beams compete with the completed ones.
        completed.extend(
            (normalised(score, len(seq)), seq)
            for score, seq in zip(scores.tolist(), sequences, strict=True)
        )

    if not completed:
        completed = [
            (normalised(score, len(seq)), seq)
            for score, seq in zip(scores.tolist(), sequences, strict=True)
        ]
    return max(completed, key=lambda item: item[0])[1]
