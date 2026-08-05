"""Plotly figures for model output and historical embedding projections."""

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from viral_classifier.predictor import Prediction, ProjectionData


def probability_figure(prediction: Prediction) -> go.Figure:
    """Render model class scores without implying clinical calibration."""
    labels = list(prediction.probabilities)
    scores = list(prediction.probabilities.values())
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=scores,
                marker_color=[
                    "#2563eb" if label == prediction.predicted_class else "#94a3b8"
                    for label in labels
                ],
                hovertemplate="%{x}<br>Model score: %{y:.2%}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title="Six-class model output",
        xaxis_title="Read class",
        yaxis_title="Model score",
        yaxis_range=[0, 1],
        showlegend=False,
        margin=dict(l=40, r=20, t=55, b=90),
    )
    return figure


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def projection_figure(data: ProjectionData) -> go.Figure:
    """Render the historical PCA projection with a distinct submitted point."""
    label_array = np.asarray(data.labels)
    figure = go.Figure()
    for label in _ordered_unique(data.labels):
        points = data.points[label_array == label]
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=label,
                marker=dict(size=3, opacity=0.55),
                hovertemplate=f"{label}<extra></extra>",
            )
        )

    query = data.query_point
    figure.add_trace(
        go.Scatter3d(
            x=[query[0]],
            y=[query[1]],
            z=[query[2]],
            mode="markers",
            name="Submitted read",
            marker=dict(size=8, color="black", symbol="diamond"),
            hovertemplate="Submitted read<extra></extra>",
        )
    )
    figure.update_layout(
        title="Historical embedding projection (PCA)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        margin=dict(l=0, r=0, t=55, b=0),
    )
    return figure
