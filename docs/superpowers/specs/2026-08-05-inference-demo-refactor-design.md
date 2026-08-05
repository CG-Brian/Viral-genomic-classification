# Viral Classifier Inference Demo Refactor

## Objective

Restore the 2022 viral genomic read classifier as a reproducible Bio AI inference demo. Preserve the original experiments as historical records while separating them from maintained runtime code.

This phase does not retrain the model or claim that the original evaluation has been reproduced.

## Scope

### Included

- Load the existing tokenizer, classifier weights, labels, PCA model, and reference embeddings.
- Replace the Dash interface with a Gradio interface suitable for local use and Hugging Face Spaces.
- Package maintained inference code under `src/viral_classifier`.
- Validate nucleotide inputs and expose clear prediction errors.
- Display the predicted class, class probabilities, BPE tokens, and PCA projection.
- Preserve the 2022 notebooks, training utilities, and result images under `archive/2022`.
- Add smoke and unit tests for artifact loading, validation, prediction, and application import.
- Document that the application is a research demonstration rather than a clinical diagnostic tool.

### Excluded

- Retraining or fine-tuning the classifier.
- Reproducing the PACIFIC dataset locally.
- Re-estimating the reported accuracy or F1 score.
- Building a TypeScript frontend or a separate HTTP API.
- Deleting historical artifacts before determining which classifier weight is canonical.
- Creating Git commits.

## Target Structure

```text
Viral-genomic-classification/
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── viral_classifier/
│       ├── __init__.py
│       ├── config.py
│       ├── model.py
│       ├── predictor.py
│       ├── validation.py
│       └── visualization.py
├── artifacts/
│   ├── classifier.pth
│   ├── tokenizer.json
│   ├── labels.json
│   ├── pca.pkl
│   ├── reference_embeddings.npy
│   └── reference_labels.pkl
├── examples/
│   └── sequences.json
├── tests/
│   ├── test_validation.py
│   ├── test_predictor.py
│   └── test_app_smoke.py
└── archive/
    └── 2022/
        ├── README.md
        ├── dash_app.py
        ├── notebooks/
        ├── training/
        └── results/
```

## Component Boundaries

### Application entry point

The root `app.py` assembles Gradio components and converts predictor results into UI outputs. It contains no model architecture, tokenization, or PCA implementation.

### Configuration

`config.py` resolves artifact paths relative to the repository rather than the caller's current working directory. It defines the classifier architecture required to load the historical state dictionary.

### Model

`model.py` contains only the PyTorch classifier definition needed for inference. Historical CBOW and training-only model definitions remain in the archive.

### Predictor

`predictor.py` owns artifact loading, tokenization, tensor construction, inference mode, class ordering, and probability conversion. Classification and embedding extraction are explicit methods rather than behavior inferred from tensor rank.

### Validation

`validation.py` normalizes whitespace and case, verifies the nucleotide alphabet, and enforces documented minimum and maximum lengths. Validation failures are returned as user-facing errors.

The initial accepted alphabet is `A`, `C`, `G`, and `T`. `U` and ambiguous bases such as `N` are rejected with an explanatory message because the historical tokenizer and training data use DNA-formatted reads.

### Visualization

`visualization.py` creates probability and PCA figures from typed predictor outputs. PCA proximity is described as a projection of the learned representation, not evidence of phylogenetic relatedness.

## Data Flow

```text
raw sequence
  -> normalization and validation
  -> BPE tokenization
  -> padded token tensor
  -> LSTM classifier in evaluation/inference mode
  -> six-class softmax probabilities
  -> predicted class and confidence
```

For the optional projection:

```text
token embeddings
  -> sum across non-padding tokens
  -> historical PCA transform
  -> plot with sampled reference points
```

## Historical Material

The original notebooks keep their saved outputs and are renamed in execution order. They are not imported by maintained application code.

`archive/2022/README.md` documents:

- the original Colab and Kaggle environment;
- the external PACIFIC data dependency;
- which notebook generated each artifact;
- that the notebooks are historical experimental records rather than a currently verified training pipeline;
- that the maintained application uses the saved artifacts for inference only.

The old Dash application is retained as `archive/2022/dash_app.py`.

## Artifact Policy

The two existing classifier state dictionaries are retained until both are loaded against their expected architecture and compared with the application configuration. The model currently referenced by the deployed application becomes `artifacts/classifier.pth`. The other file remains in the archive with its original name and documented uncertainty.

Artifacts are loaded once per application process and reused for predictions. The implementation must not deserialize user-provided files.

## User Interface

The Gradio page contains:

- a short project and historical context statement;
- a nucleotide sequence input;
- selectable example reads when trustworthy examples are available;
- predicted class and confidence;
- a six-class probability chart;
- the BPE token sequence;
- a PCA projection with a clear interpretive caveat;
- model limitations and a non-clinical-use notice.

If trustworthy labeled example reads are not available in the repository, the example selector is omitted rather than populated with fabricated samples.

## Error Handling

- Missing or incompatible artifacts fail application startup with the exact artifact path and cause.
- Invalid user input produces a concise validation message without running the model.
- Empty inputs and sequences outside the supported length range are rejected.
- PCA failure does not invalidate a successful classification; the prediction is returned with a visualization warning.
- Internal exception details are logged but not presented as biological conclusions.

## Verification

Tests cover:

- normalization and nucleotide validation;
- rejection of empty, invalid, and unsupported-length reads;
- loading the canonical artifacts;
- returning six finite probabilities whose sum is approximately one;
- preserving label order from `labels.json`;
- deterministic predictions for a fixed input;
- application import without starting a server;
- construction of both probability and PCA figures.

A final manual smoke test runs the Gradio application locally and performs one prediction through the UI.

## Documentation Claims

The README describes the reported 99.5% result as a historical held-out score from a random split of synthetic PACIFIC-derived reads. It does not describe that value as external, clinical, or novel-virus performance. It also distinguishes class posterior output from patient-level infection probability.

## Deployment

The root `app.py` and dependency files support Hugging Face Spaces. Deployment credentials and remote publishing are outside this local refactor and require separate user authorization.
