import pytest

from viral_classifier.predictor import ViralReadPredictor


@pytest.fixture(scope="session")
def predictor():
    return ViralReadPredictor.load_default()


@pytest.fixture(scope="session")
def prediction(predictor):
    return predictor.predict("ACGT" * 38)
