"""Inference-only PyTorch architecture matching the 2022 classifier."""

import torch
from torch import nn


class ViralLSTMClassifier(nn.Module):
    """Classify one padded nucleotide-token sequence into six read classes."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_id: int,
        hidden_dim: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_id,
        )
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_dim, output_size)
        self.pad_id = pad_id

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        recurrent_output, _ = self.rnn(embedded)
        last_token_index = (token_ids != self.pad_id).sum(dim=-1) - 1
        batch_index = torch.arange(token_ids.shape[0], device=token_ids.device)
        hidden_state = recurrent_output[batch_index, last_token_index, :]
        return self.out_layer(hidden_state)
