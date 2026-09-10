"""Sequence loading, splitting, and batching for training.

Expects a directory laid out as `<root>/<class_name>/*` where each file holds
one nucleotide read per line (FASTA header lines starting with `>` are
skipped). This mirrors the PACIFIC-derived corpus layout used for the
original training runs.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_sequences(base_path: str, exclude_human_for_embedding: bool = False) -> tuple[list[str], list[str]]:
    """Read every sequence file under `base_path`, one class per subdirectory."""
    sequences: list[str] = []
    labels: list[str] = []
    for class_name in os.listdir(base_path):
        if exclude_human_for_embedding and class_name == "Human":
            continue
        class_dir = f"{base_path}/{class_name}"
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            with open(f"{class_dir}/{file_name}") as handle:
                for line in handle:
                    if line.startswith(">"):
                        continue
                    sequences.append(line)
                    labels.append(class_name)
    return sequences, labels


def get_three_way_split(
    sequences: list[str], labels: list[str], seed: int | None = None
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Stratified 80/10/10 train/validation/test split."""
    train_seq, holdout_seq, train_label, holdout_label = train_test_split(
        sequences, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    valid_seq, test_seq, valid_label, test_label = train_test_split(
        holdout_seq, holdout_label, test_size=0.5, stratify=holdout_label, random_state=seed
    )
    return train_seq, valid_seq, test_seq, train_label, valid_label, test_label


class SequenceDataset(Dataset):
    """Tokenizes reads on access and encodes class labels as integer ids."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[str],
        tokenizer_file: str,
        label_to_index: dict[str, int] | None = None,
    ) -> None:
        self.sequences = sequences
        self.label_to_index = label_to_index or {
            label: index for index, label in enumerate(sorted(set(labels)))
        }
        self.label_ids = [self.label_to_index[label] for label in labels]
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.tokenizer.enable_padding()

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        encoded = self.tokenizer.encode(self.sequences[index].strip())
        return torch.LongTensor(encoded.ids), self.label_ids[index]

    def collate(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        padded = pad_sequence(
            sequences, batch_first=True, padding_value=self.tokenizer.padding["pad_id"]
        )
        return padded, torch.LongTensor(labels)


def class_weights(labels: list[str], label_to_index: dict[str, int]) -> np.ndarray:
    """Inverse-frequency class weights, matching the reported imbalance handling."""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.array(sorted(label_to_index, key=label_to_index.get))
    return compute_class_weight("balanced", classes=classes, y=np.asarray(labels))
