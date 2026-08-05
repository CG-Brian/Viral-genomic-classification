import numpy as np

from viral_classifier.visualization import probability_figure, projection_figure


def test_probability_figure_has_six_bars(prediction):
    figure = probability_figure(prediction)

    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 6


def test_projection_sampling_is_deterministic(predictor):
    first = predictor.projection_data("ACGT" * 38, seed=2022)
    second = predictor.projection_data("ACGT" * 38, seed=2022)

    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.query_point, second.query_point)
    assert first.labels == second.labels


def test_projection_figure_marks_query(predictor):
    data = predictor.projection_data("ACGT" * 38, seed=2022)
    figure = projection_figure(data)

    assert figure.data[-1].name == "Submitted read"
    assert figure.data[-1].marker.symbol == "diamond"
