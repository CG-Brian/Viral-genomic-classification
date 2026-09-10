"""Repository-relative configuration for the model artifacts."""

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArtifactPaths:
    classifier: Path = REPOSITORY_ROOT / "artifacts" / "classifier.pth"
    tokenizer: Path = REPOSITORY_ROOT / "artifacts" / "tokenizer.json"
    labels: Path = REPOSITORY_ROOT / "artifacts" / "labels.json"
    pca: Path = REPOSITORY_ROOT / "artifacts" / "pca.pkl"
    reference_embeddings: Path = (
        REPOSITORY_ROOT / "artifacts" / "reference_embeddings.npy"
    )
    reference_labels: Path = REPOSITORY_ROOT / "artifacts" / "reference_labels.pkl"

    def required(self) -> tuple[Path, ...]:
        return (
            self.classifier,
            self.tokenizer,
            self.labels,
            self.pca,
            self.reference_embeddings,
            self.reference_labels,
        )


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 1
    output_size: int = 6
