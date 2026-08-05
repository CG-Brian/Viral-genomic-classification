"""Artifact loading and prediction for the restored viral-read classifier."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from typing import Any
import warnings

import numpy as np
import torch
from sklearn.exceptions import InconsistentVersionWarning
from tokenizers import Tokenizer

from viral_classifier.config import ArtifactPaths, ModelConfig
from viral_classifier.model import ViralLSTMClassifier
from viral_classifier.validation import normalize_sequence


@dataclass(frozen=True)
class Prediction:
    sequence: str
    tokens: tuple[str, ...]
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float


@dataclass(frozen=True)
class ProjectionData:
    points: np.ndarray
    labels: tuple[str, ...]
    query_point: np.ndarray


class ViralReadPredictor:
    """Own the immutable tokenizer, model, and PCA objects used for inference."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: ViralLSTMClassifier,
        labels: tuple[str, ...],
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        reference_embeddings: np.ndarray,
        reference_labels: tuple[str, ...],
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.labels = labels
        self.pca_mean = pca_mean
        self.pca_components = pca_components
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels

    @classmethod
    def load_default(
        cls,
        paths: ArtifactPaths | None = None,
        config: ModelConfig | None = None,
    ) -> ViralReadPredictor:
        paths = paths or ArtifactPaths()
        config = config or ModelConfig()
        missing = [str(path) for path in paths.required() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

        with paths.labels.open(encoding="utf-8") as label_file:
            label_to_index = json.load(label_file)
        labels = tuple(
            label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])
        )
        if len(labels) != config.output_size:
            raise ValueError(
                f"Expected {config.output_size} labels, found {len(labels)} in {paths.labels}"
            )

        tokenizer = Tokenizer.from_file(str(paths.tokenizer))
        padding = tokenizer.padding
        if padding is None:
            raise ValueError(f"Tokenizer has no padding configuration: {paths.tokenizer}")

        model = ViralLSTMClassifier(
            vocab_size=tokenizer.get_vocab_size(),
            embedding_dim=config.embedding_dim,
            pad_id=padding["pad_id"],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            output_size=config.output_size,
        )
        state_dict = torch.load(paths.classifier, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        with paths.pca.open("rb") as pca_file:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                historical_pca: Any = pickle.load(pca_file)
        pca_mean = np.asarray(historical_pca.mean_, dtype=np.float32)
        pca_components = np.asarray(historical_pca.components_, dtype=np.float32)
        if pca_mean.shape != (config.embedding_dim,) or pca_components.shape != (
            3,
            config.embedding_dim,
        ):
            raise ValueError(f"Unexpected PCA dimensions in {paths.pca}")

        reference_embeddings = np.load(paths.reference_embeddings, mmap_mode="r")
        with paths.reference_labels.open("rb") as labels_file:
            reference_labels = tuple(pickle.load(labels_file))
        if reference_embeddings.shape != (len(reference_labels), 3):
            raise ValueError(
                "Reference projection and label counts do not match: "
                f"{paths.reference_embeddings}, {paths.reference_labels}"
            )

        return cls(
            tokenizer=tokenizer,
            model=model,
            labels=labels,
            pca_mean=pca_mean,
            pca_components=pca_components,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
        )

    def predict(self, raw: str) -> Prediction:
        sequence = normalize_sequence(raw)
        encoding = self.tokenizer.encode(sequence)
        token_ids = torch.tensor([encoding.ids], dtype=torch.long)

        with torch.inference_mode():
            logits = self.model(token_ids).squeeze(0)
            scores = torch.softmax(logits, dim=-1).cpu().numpy()

        probabilities = {
            label: float(score) for label, score in zip(self.labels, scores, strict=True)
        }
        predicted_class = max(probabilities, key=probabilities.__getitem__)
        return Prediction(
            sequence=sequence,
            tokens=tuple(encoding.tokens),
            probabilities=probabilities,
            predicted_class=predicted_class,
            confidence=probabilities[predicted_class],
        )

    def project(self, raw: str) -> np.ndarray:
        sequence = normalize_sequence(raw)
        encoding = self.tokenizer.encode(sequence)
        token_ids = torch.tensor([encoding.ids], dtype=torch.long)
        with torch.inference_mode():
            embedding = self.model.embedding(token_ids).sum(dim=1).cpu().numpy()
        projected = (embedding - self.pca_mean) @ self.pca_components.T
        return projected[0]

    def projection_data(self, raw: str, seed: int = 2022) -> ProjectionData:
        """Sample reference points reproducibly and project one submitted read."""
        generator = np.random.default_rng(seed)
        sampled_indices: list[np.ndarray] = []
        label_array = np.asarray(self.reference_labels)
        for label in self.labels:
            indices = np.flatnonzero(label_array == label)
            count = min(50, len(indices))
            sampled_indices.append(generator.choice(indices, size=count, replace=False))

        selected = np.concatenate(sampled_indices)
        points = np.asarray(self.reference_embeddings[selected], dtype=np.float32)
        labels = tuple(label_array[selected].tolist())
        return ProjectionData(
            points=points,
            labels=labels,
            query_point=self.project(raw),
        )
