"""Supervised training of the six-class LSTM classifier.

Builds on the same `ViralLSTMClassifier` used for inference (see
`src/viral_classifier/model.py`) rather than a separate training-only
architecture, so the trained and deployed models are guaranteed to match.
Optionally seeds the embedding layer with CBOW-pretrained weights (see
`embedding_pretraining.py`) and freezes it, matching the configuration
actually shipped in `artifacts/classifier.pth` (see `training/README.md`
for why that configuration's reported accuracy is disputed).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from viral_classifier.model import ViralLSTMClassifier


def build_classifier(
    vocab_size: int,
    pad_id: int,
    num_classes: int,
    embedding_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 1,
    pretrained_embedding: torch.Tensor | None = None,
    freeze_embedding: bool = True,
) -> ViralLSTMClassifier:
    model = ViralLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        pad_id=pad_id,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_size=num_classes,
    )
    if pretrained_embedding is not None:
        with torch.no_grad():
            model.embedding.weight.copy_(pretrained_embedding)
        model.embedding.weight.requires_grad_(not freeze_embedding)
    return model


def train_classifier(
    model: ViralLSTMClassifier,
    train_loader,
    valid_loader,
    train_labels: list[int],
    device: torch.device,
    lr: float = 1e-3,
    num_epochs: int = 5,
) -> dict[str, list[float]]:
    """Class-weighted cross-entropy training, matching the reported imbalance handling."""
    model.to(device)
    class_weight = np.bincount(train_labels).sum() / (
        len(set(train_labels)) * np.bincount(train_labels)
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_accuracy": [], "valid_accuracy": []}
    for _ in range(num_epochs):
        model.train()
        history["train_accuracy"].append(
            _run_epoch(model, train_loader, device, loss_fn, optimizer)
        )
        model.eval()
        with torch.no_grad():
            history["valid_accuracy"].append(_run_epoch(model, valid_loader, device))
    return history


def _run_epoch(model, loader, device, loss_fn=None, optimizer=None) -> float:
    correct, total = 0, 0
    for token_ids, labels in loader:
        token_ids, labels = token_ids.to(device), labels.to(device)
        logits = model(token_ids)
        if optimizer is not None:
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += labels.shape[0]
    return correct / total
