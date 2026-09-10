"""Validation for DNA-formatted reads accepted by the classifier."""

MIN_SEQUENCE_LENGTH = 20
MAX_SEQUENCE_LENGTH = 1_000
SUPPORTED_NUCLEOTIDES = frozenset("ACGT")


class SequenceValidationError(ValueError):
    """Raised when a sequence cannot be processed by the classifier."""


def normalize_sequence(raw: str) -> str:
    """Normalize a DNA read and reject inputs outside the model contract."""
    if not isinstance(raw, str):
        raise SequenceValidationError("Sequence must be text.")

    sequence = "".join(raw.split()).upper()
    if not MIN_SEQUENCE_LENGTH <= len(sequence) <= MAX_SEQUENCE_LENGTH:
        raise SequenceValidationError(
            f"Enter a sequence between {MIN_SEQUENCE_LENGTH} and "
            f"{MAX_SEQUENCE_LENGTH:,} nucleotides."
        )

    invalid = sorted(set(sequence) - SUPPORTED_NUCLEOTIDES)
    if invalid:
        raise SequenceValidationError(
            f"Unsupported nucleotide symbols: {', '.join(invalid)}. "
            "This model accepts DNA-formatted A/C/G/T reads only."
        )

    return sequence
