"""Fit the PCA projection used for embedding visualization.

Reproduces how `artifacts/pca.pkl`, `artifacts/reference_embeddings.npy`, and
`artifacts/reference_labels.pkl` were derived: sample reads per class, sum
each read's token embeddings from the trained classifier, and fit a 3D PCA
on the pooled result. The shipped artifacts were fit on 100,000 sampled
reads per class (600,000 total).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from tokenizers import Tokenizer

from viral_classifier.model import ViralLSTMClassifier


def summed_embeddings(
    sequences: list[str], tokenizer: Tokenizer, model: ViralLSTMClassifier, device: torch.device
) -> np.ndarray:
    """Sum each sequence's token embeddings into one vector per read."""
    encodings = tokenizer.encode_batch(sequences)
    token_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(encoding.ids) for encoding in encodings],
        batch_first=True,
        padding_value=tokenizer.padding["pad_id"],
    ).to(device)
    with torch.inference_mode():
        embedded = model.embedding(token_ids).sum(dim=1)
    return embedded.cpu().numpy()


def fit_projection(
    embeddings: np.ndarray, labels: list[str]
) -> tuple[PCA, np.ndarray, list[str]]:
    """Fit a 3-component PCA on pooled embeddings; returns the fitted transform."""
    pca = PCA(n_components=3)
    projected = pca.fit_transform(embeddings)
    return pca, projected, labels
