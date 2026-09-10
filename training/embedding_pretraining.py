"""Unsupervised CBOW-style pretraining of token embeddings.

Named `SkipGramEmbeddingModel` in the original 2022 code, but the objective
it implements is Continuous Bag of Words (CBOW): a token is predicted from
the sum of its surrounding context embeddings within a fixed window, not the
reverse (skip-gram predicts context from a center token). This module keeps
the CBOW name to match the actual objective.
"""

from __future__ import annotations

import torch
from torch import nn


class CBOWEmbeddingModel(nn.Module):
    """Predict each token from the sum of its left/right context embeddings."""

    def __init__(self, vocab_size: int, embedding_dim: int, pad_id: int, window_size: int) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.window_size = window_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.out_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)  # [B, L, E]
        context_sums = []
        for position in range(token_ids.shape[1]):
            left = max(position - self.window_size, 0)
            right = min(position + self.window_size + 1, token_ids.shape[1])
            context = torch.cat(
                [embedded[:, left:position], embedded[:, position + 1 : right]], dim=1
            )
            context_sums.append(context.sum(dim=1))
        context_sums = torch.stack(context_sums, dim=1)  # [B, L, E]
        return self.out_layer(context_sums)  # [B, L, V]


def train_cbow_embedding(
    model: CBOWEmbeddingModel,
    train_loader,
    device: torch.device,
    lr: float = 1e-3,
    num_epochs: int = 3,
) -> list[float]:
    """Train the CBOW objective; returns per-epoch mean training loss."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_id)

    epoch_losses = []
    for _ in range(num_epochs):
        total_loss = 0.0
        for token_ids, _ in train_loader:
            token_ids = token_ids.to(device)
            logits = model(token_ids).permute(0, 2, 1)  # [B, V, L] for CrossEntropyLoss
            loss = loss_fn(logits, token_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(train_loader))
    return epoch_losses
