import numpy as np
import pytest

EXPECTED_LABELS = [
    "Coronaviridae",
    "Human",
    "Influenza",
    "Metapneumovirus",
    "Rhinovirus",
    "Sars_Cov-2",
]
SEQUENCE = "ACGT" * 38


def test_predictor_returns_ordered_probabilities(predictor):
    result = predictor.predict(SEQUENCE)

    assert len(result.probabilities) == 6
    assert list(result.probabilities) == EXPECTED_LABELS
    assert np.isfinite(list(result.probabilities.values())).all()
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.predicted_class in result.probabilities
    assert result.confidence == result.probabilities[result.predicted_class]
    assert result.tokens


def test_predictor_is_deterministic(predictor):
    first = predictor.predict(SEQUENCE)
    second = predictor.predict(SEQUENCE)

    assert first.probabilities == second.probabilities
    assert first.tokens == second.tokens


def test_projection_is_three_dimensional(predictor):
    point = predictor.project(SEQUENCE)
    assert point.shape == (3,)
    assert np.isfinite(point).all()
