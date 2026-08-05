"""Gradio entry point for the restored viral-read classifier demo."""

from __future__ import annotations

import logging

import gradio as gr

from viral_classifier.predictor import ViralReadPredictor
from viral_classifier.validation import SequenceValidationError
from viral_classifier.visualization import probability_figure, projection_figure


LOGGER = logging.getLogger(__name__)
PREDICTOR = ViralReadPredictor.load_default()


def run_prediction(raw_sequence: str):
    """Return display-ready outputs while keeping classification independent of PCA."""
    try:
        prediction = PREDICTOR.predict(raw_sequence)
    except SequenceValidationError as exc:
        return (
            "No prediction",
            "Model score unavailable",
            "",
            None,
            None,
            f"Input error: {exc}",
        )

    class_output = f"### Predicted read class\n{prediction.predicted_class}"
    confidence_output = f"### Model score\n{prediction.confidence:.1%}"
    token_output = " | ".join(prediction.tokens)
    score_plot = probability_figure(prediction)

    try:
        projection = PREDICTOR.projection_data(prediction.sequence)
        pca_plot = projection_figure(projection)
        status = (
            f"Processed {len(prediction.sequence)} nucleotides as "
            f"{len(prediction.tokens)} BPE tokens."
        )
    except Exception as exc:  # PCA is optional; preserve a valid classification.
        LOGGER.exception("Could not build the historical PCA projection", exc_info=exc)
        pca_plot = None
        status = (
            f"Processed {len(prediction.sequence)} nucleotides. "
            "Classification succeeded, but the historical PCA projection is unavailable."
        )

    return class_output, confidence_output, token_output, score_plot, pca_plot, status


PAGE_CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
.project-kicker { color: #475569; letter-spacing: .08em; text-transform: uppercase; }
.result-card { border: 1px solid #dbeafe; border-radius: 14px; padding: 4px 16px; }
"""


with gr.Blocks(title="Viral Read Classifier", analytics_enabled=False) as demo:
    gr.Markdown(
        """
<div class="project-kicker">2022 project · restored inference demo</div>

# Viral genomic read classification

Explore a historical BPE–LSTM model trained to assign synthetic respiratory
RNA-seq reads to one of six classes. Enter a **DNA-formatted read using A/C/G/T**.
The model accepts 20–1,000 nucleotides; its original training reads were 150 nt.
"""
    )

    with gr.Row():
        with gr.Column(scale=3):
            sequence_input = gr.Textbox(
                label="Nucleotide sequence",
                placeholder="ACGT…",
                lines=7,
                max_lines=12,
            )
            submit_button = gr.Button("Classify read", variant="primary")
            status_output = gr.Markdown("Enter a sequence to inspect the model output.")
        with gr.Column(scale=2, elem_classes="result-card"):
            class_output = gr.Markdown("No prediction")
            confidence_output = gr.Markdown("Model score unavailable")

    probability_output = gr.Plot(label="Class scores")
    token_output = gr.Code(label="BPE tokens", interactive=False)
    projection_output = gr.Plot(label="Embedding projection")

    gr.Markdown(
        """
### Interpretation and limitations

- Scores are six-class model outputs for an individual read, not patient-level
  infection probabilities.
- PCA proximity visualizes the historical learned representation and does not
  establish phylogenetic relatedness.
- The original evaluation used synthetic PACIFIC-derived reads and has not been
  reproduced during this inference-only restoration.
- **Research demonstration only — not a clinical diagnostic tool.**
"""
    )

    outputs = [
        class_output,
        confidence_output,
        token_output,
        probability_output,
        projection_output,
        status_output,
    ]
    submit_button.click(run_prediction, inputs=sequence_input, outputs=outputs)
    sequence_input.submit(run_prediction, inputs=sequence_input, outputs=outputs)


if __name__ == "__main__":
    demo.launch(css=PAGE_CSS)
