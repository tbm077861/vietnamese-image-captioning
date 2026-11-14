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


@torch.no_grad()
def greedy_decode(
    decoder: nn.Module,
    features: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_length: int = 30,
    suppress_ids: Sequence[int] = (),
) -> list[list[int]]:
    """Decode a whole batch; returns token ids without the start / end markers."""
    batch = features.size(0)
    state = decoder.init_state(features)
    tokens = torch.full((batch,), bos_id, dtype=torch.long, device=features.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=features.device)
    steps = []
    for _ in range(max_length):
        logits, state = decoder.step(tokens, state)
        tokens = _suppress(logits, suppress_ids).argmax(dim=-1)
        steps.append(tokens)
        finished |= tokens == eos_id
        if bool(finished.all()):
            break
    sequences = torch.stack(steps, dim=1).tolist()
    return [seq[: seq.index(eos_id)] if eos_id in seq else seq for seq in sequences]


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
) -> list[int]:
    """Decode one image (features of shape (1, regions, channels)) with beam search.

    Hypotheses are ranked by ``log_prob / length ** length_penalty``; 0 disables normalisation.
    """
    if features.size(0) != 1:
        raise ValueError("beam_search decodes one image at a time")
    device = features.device
    state = decoder.init_state(features)
    tokens = torch.full((1,), bos_id, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    sequences: list[list[int]] = [[]]
    completed: list[tuple[float, list[int]]] = []

    def normalised(score: float, length: int) -> float:
        return score / (max(length, 1) ** length_penalty)

    for _ in range(max_length):
        logits, state = decoder.step(tokens, state)
        log_probs = torch.log_softmax(_suppress(logits.float(), suppress_ids), dim=-1)
        vocab_size = log_probs.size(-1)
        candidates = (scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_index = candidates.topk(min(beam_size, candidates.numel()))
        beam_index = top_index // vocab_size
        token_index = top_index % vocab_size

        keep = []
        new_sequences = []
        for position, (beam, token, score) in enumerate(
            zip(beam_index.tolist(), token_index.tolist(), top_scores.tolist(), strict=True)
        ):
            if token == eos_id:
                completed.append((normalised(score, len(sequences[beam]) + 1), sequences[beam]))
            else:
                keep.append(position)
                new_sequences.append([*sequences[beam], token])
        if not keep or len(completed) >= beam_size:
            break
        keep_index = torch.tensor(keep, dtype=torch.long, device=device)
        state = decoder.reorder_state(state, beam_index[keep_index])
        tokens = token_index[keep_index]
        scores = top_scores[keep_index]
        sequences = new_sequences
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
