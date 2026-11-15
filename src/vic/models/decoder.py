"""Caption decoders.

Every decoder exposes the same interface so that training and decoding code is shared:

- ``forward(features, input_ids) -> logits``: teacher-forced training, (batch, time, vocab).
- ``init_state(features) -> state``: prepare step-by-step decoding.
- ``step(tokens, state) -> (logits, state)``: one decoding step, logits are (batch, vocab).
- ``reorder_state(state, index) -> state``: select / duplicate batch entries (beam search).

``features`` is the encoder grid (batch, regions, channels), possibly in half precision.
Every decoder accepts ``feature_norm=True`` to apply a LayerNorm to the incoming features;
use it for encoders whose features are not on a unit scale (ViT residual streams).
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
        feature_norm: bool = False,
    ) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim) if feature_norm else nn.Identity()
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
        features = self.feature_norm(features.float())
        pooled = features.mean(dim=1) if features.dim() == 3 else features
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


AttentionState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class AttentionLSTMDecoder(nn.Module):
    """LSTM with soft attention over the region grid (Show, Attend and Tell style).

    At every step the decoder weighs the regions with
    ``alpha = softmax(w . tanh(W_f f + W_h h))`` and feeds ``[embedding; context]`` to an
    ``nn.LSTMCell``, where the context is the alpha-weighted sum of the projected regions.
    The state is ``(h, c, regions, region_keys)``: the projected regions and their attention
    keys ``W_f f`` are computed once per image and carried along.
    """

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.5,
        pad_id: int = 0,
        feature_norm: bool = False,
    ) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(feature_dim) if feature_norm else nn.Identity()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.project = nn.Linear(feature_dim, attention_dim)
        self.init_h = nn.Linear(attention_dim, hidden_dim)
        self.init_c = nn.Linear(attention_dim, hidden_dim)
        self.attend_regions = nn.Linear(attention_dim, attention_dim)
        self.attend_hidden = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attend_score = nn.Linear(attention_dim, 1, bias=False)
        self.cell = nn.LSTMCell(embed_dim + attention_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, features: torch.Tensor) -> AttentionState:
        if features.dim() == 2:  # a pooled vector is a grid with a single region
            features = features.unsqueeze(1)
        features = self.feature_norm(features.float())
        regions = self.dropout(torch.relu(self.project(features)))
        pooled = regions.mean(dim=1)
        hidden = torch.tanh(self.init_h(pooled))
        cell = torch.tanh(self.init_c(pooled))
        return hidden, cell, regions, self.attend_regions(regions)

    def attention(self, hidden: torch.Tensor, region_keys: torch.Tensor) -> torch.Tensor:
        """Attention weights over the regions, (batch, regions)."""
        energy = torch.tanh(region_keys + self.attend_hidden(hidden).unsqueeze(1))
        return torch.softmax(self.attend_score(energy).squeeze(-1), dim=-1)

    def _advance(
        self, embedded: torch.Tensor, state: AttentionState
    ) -> tuple[torch.Tensor, AttentionState]:
        hidden, cell, regions, region_keys = state
        alpha = self.attention(hidden, region_keys)
        context = torch.bmm(alpha.unsqueeze(1), regions).squeeze(1)
        hidden, cell = self.cell(torch.cat([embedded, context], dim=-1), (hidden, cell))
        return self.output(self.dropout(hidden)), (hidden, cell, regions, region_keys)

    def forward(self, features: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        state = self.init_state(features)
        embedded = self.dropout(self.embedding(input_ids))
        logits = []
        for position in range(input_ids.size(1)):
            step_logits, state = self._advance(embedded[:, position], state)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)

    def step(
        self, tokens: torch.Tensor, state: AttentionState
    ) -> tuple[torch.Tensor, AttentionState]:
        return self._advance(self.dropout(self.embedding(tokens)), state)

    @staticmethod
    def reorder_state(state: AttentionState, index: torch.Tensor) -> AttentionState:
        return tuple(tensor.index_select(0, index) for tensor in state)


DECODERS = {"lstm": LSTMDecoder, "lstm_attention": AttentionLSTMDecoder}
