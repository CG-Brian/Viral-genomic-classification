import pytest

from viral_classifier.validation import SequenceValidationError, normalize_sequence


def test_normalizes_case_and_whitespace():
    assert normalize_sequence("acgt\n acgt\tacgt acgt acgt") == "ACGTACGTACGTACGTACGT"


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "ACGT", "A" * 1001, "A" * 19, "AUGC" * 5, "ACNT" * 5],
)
def test_rejects_unsupported_sequences(raw):
    with pytest.raises(SequenceValidationError):
        normalize_sequence(raw)


def test_accepts_boundaries():
    assert normalize_sequence("A" * 20) == "A" * 20
    assert normalize_sequence("T" * 1000) == "T" * 1000


def test_rejects_non_text_input():
    with pytest.raises(SequenceValidationError, match="text"):
        normalize_sequence(None)  # type: ignore[arg-type]
