"""Caption decoders.

Every decoder exposes the same interface so that training and decoding code is shared:

- ``forward(features, input_ids) -> logits``: teacher-forced training, (batch, time, vocab).
- ``init_state(features) -> state``: prepare step-by-step decoding.
- ``step(tokens, state) -> (logits, state)``: one decoding step, logits are (batch, vocab).
- ``reorder_state(state, index) -> state``: select / duplicate batch entries (beam search).

``features`` is the encoder grid (batch, regions, channels), possibly in half precision.
"""

from __future__ import annotations

import torch
from torch import nn

State = tuple[torch.Tensor, torch.Tensor]


class LSTMDecoder(nn.Module):
    """Show-and-Tell style baseline: the mean image feature initialises the LSTM state."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.5,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.init_h = nn.Linear(feature_dim, num_layers * hidden_dim)
        self.init_c = nn.Linear(feature_dim, num_layers * hidden_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, features: torch.Tensor) -> State:
        pooled = features.float().mean(dim=1) if features.dim() == 3 else features.float()
        pooled = self.dropout(pooled)
        batch = pooled.size(0)

        def to_state(layer: nn.Linear) -> torch.Tensor:
            state = torch.tanh(layer(pooled)).view(batch, self.num_layers, self.hidden_dim)
            return state.transpose(0, 1).contiguous()

        return to_state(self.init_h), to_state(self.init_c)

    def forward(self, features: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        state = self.init_state(features)
        embedded = self.dropout(self.embedding(input_ids))
        hidden, _ = self.lstm(embedded, state)
        return self.output(self.dropout(hidden))

    def step(self, tokens: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        hidden, state = self.lstm(self.embedding(tokens).unsqueeze(1), state)
        return self.output(hidden.squeeze(1)), state

    @staticmethod
    def reorder_state(state: State, index: torch.Tensor) -> State:
        hidden, cell = state
        return hidden.index_select(1, index), cell.index_select(1, index)


DECODERS = {"lstm": LSTMDecoder}
