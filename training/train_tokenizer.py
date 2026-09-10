"""Train the BPE nucleotide tokenizer.

Matches the methodology behind `artifacts/tokenizer.json`: a Hugging Face
`tokenizers` BPE model with a 24,000-token vocabulary, trained directly on
the nucleotide read corpus (no k-mer pre-splitting — BPE discovers
variable-length subsequences on its own).
"""

from __future__ import annotations

from tokenizers import Tokenizer, models, trainers

VOCAB_SIZE = 24_000


def train_tokenizer(sequences: list[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    """Fit a BPE tokenizer over raw nucleotide sequences."""
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(show_progress=True, vocab_size=vocab_size)
    tokenizer.train_from_iterator(sequences, trainer=trainer)
    tokenizer.enable_padding()
    return tokenizer


if __name__ == "__main__":
    import sys

    from training.dataset import load_sequences

    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m training.train_tokenizer <reads_dir> <output.json>")

    reads_dir, output_path = sys.argv[1], sys.argv[2]
    sequences, _ = load_sequences(reads_dir)
    tokenizer = train_tokenizer(sequences)
    tokenizer.save(output_path)
