# Viral Classifier Inference Demo Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the historical viral-read classifier as a tested Gradio inference demo while preserving the original 2022 experiments in an archive.

**Architecture:** A root Gradio entry point calls a focused `viral_classifier` package for validation, artifact loading, inference, and plotting. Historical Dash, training, and notebook material is isolated under `archive/2022` and is not imported at runtime.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Tokenizers, NumPy, pandas, scikit-learn, Plotly, Gradio, pytest

## Global Constraints

- Use the existing trained weights; do not retrain or fine-tune.
- Accept DNA-formatted reads containing only `A`, `C`, `G`, and `T`, after removing whitespace and normalizing case.
- Accept normalized sequences from 20 through 1,000 nucleotides.
- Treat PCA output only as a projection of learned representations, not phylogenetic evidence.
- Preserve notebook outputs and historical result images.
- Do not fabricate labeled example sequences.
- Do not create Git commits.
- Do not publish or deploy remotely without separate user authorization.

---

## File Map

- `app.py`: constructs the Gradio interface and maps a sequence to display outputs.
- `src/viral_classifier/config.py`: immutable repository-relative artifact paths and model dimensions.
- `src/viral_classifier/model.py`: inference-only LSTM classifier definition.
- `src/viral_classifier/validation.py`: sequence normalization and validation.
- `src/viral_classifier/predictor.py`: artifact loading and typed prediction operations.
- `src/viral_classifier/visualization.py`: Plotly probability and PCA figures.
- `archive/2022/`: original Dash app, notebooks, training code, ambiguous weights, and result images.
- `artifacts/`: only assets used by the maintained inference application.
- `tests/`: unit and smoke tests.

### Task 1: Archive historical experiments and identify the canonical artifacts

**Files:**
- Create: `archive/2022/README.md`
- Move: `ML/RNN_Training.ipynb` to `archive/2022/notebooks/03_lstm_training.ipynb`
- Move: `ML/Skipgram_training.ipynb` to `archive/2022/notebooks/02_cbow_training.ipynb`
- Move: `ML/tokenizer.ipynb` to `archive/2022/notebooks/01_tokenizer_training.ipynb`
- Move: `ML/vector_load.ipynb` to `archive/2022/notebooks/04_embedding_projection.ipynb`
- Move: `ML/assets/RNN_training_record_Aug_3.ipynb` to `archive/2022/notebooks/rnn_training_record.ipynb`
- Move: `ML/train.py` and `ML/dataloader.py` to `archive/2022/training/`
- Move: `app.py` to `archive/2022/dash_app.py`
- Move: historical result PNG files to `archive/2022/results/`
- Create: `artifacts/`

**Interfaces:**
- Consumes: the existing repository and Git history.
- Produces: a documented archive and a verified mapping from existing assets to maintained artifact names.

- [ ] **Step 1: Inspect both state dictionaries without modifying them**

Run a read-only Python probe that lists state-dictionary keys and tensor shapes for `ML/assets/model.pth` and `ML/assets/sgrnn_emb_ftrue.pth`. Compare the shapes with the current `app.py` dimensions: embedding 256, hidden 512, one LSTM layer, six outputs.

Expected canonical result: `ML/assets/sgrnn_emb_ftrue.pth`, because it is the file referenced by the final Dash application. Record any conflicting evidence before moving either weight.

- [ ] **Step 2: Move historical source material into the archive**

Preserve notebook JSON and saved outputs byte-for-byte. Move rather than rewrite the notebooks. Preserve the original Dash app as historical source.

- [ ] **Step 3: Populate maintained artifacts**

Move the verified canonical files to:

```text
artifacts/classifier.pth
artifacts/tokenizer.json
artifacts/labels.json
artifacts/pca.pkl
artifacts/reference_embeddings.npy
artifacts/reference_labels.pkl
```

Move the non-canonical state dictionary to `archive/2022/artifacts/` using its original filename.

- [ ] **Step 4: Write the archive README**

Document the notebook-to-artifact flow, Colab/Kaggle paths, PACIFIC dependency, historical status, and the fact that current inference does not import the archived code.

- [ ] **Step 5: Verify preservation**

Run `git status --short`, list both new trees with `rg --files`, and confirm that every original notebook, PNG, and state dictionary has exactly one destination.

### Task 2: Add sequence validation with tests

**Files:**
- Create: `src/viral_classifier/__init__.py`
- Create: `src/viral_classifier/validation.py`
- Create: `tests/test_validation.py`
- Create: `pyproject.toml`

**Interfaces:**
- Produces: `normalize_sequence(raw: str) -> str` and `SequenceValidationError(ValueError)`.

- [ ] **Step 1: Write failing validation tests**

```python
import pytest

from viral_classifier.validation import SequenceValidationError, normalize_sequence


def test_normalizes_case_and_whitespace():
    assert normalize_sequence("acgt\n acgt\tacgt acgt acgt") == "ACGTACGTACGTACGTACGT"


@pytest.mark.parametrize("raw", ["", "   ", "ACGT", "A" * 1001, "A" * 19, "AUGC", "ACNT" ])
def test_rejects_unsupported_sequences(raw):
    with pytest.raises(SequenceValidationError):
        normalize_sequence(raw)
```

- [ ] **Step 2: Run the test and verify failure**

Run: `PYTHONPATH=src pytest tests/test_validation.py -v`

Expected: collection fails because `viral_classifier.validation` does not exist.

- [ ] **Step 3: Implement minimal validation**

```python
class SequenceValidationError(ValueError):
    pass


def normalize_sequence(raw: str) -> str:
    if not isinstance(raw, str):
        raise SequenceValidationError("Sequence must be text.")
    sequence = "".join(raw.split()).upper()
    if not 20 <= len(sequence) <= 1000:
        raise SequenceValidationError("Enter a sequence between 20 and 1,000 nucleotides.")
    invalid = sorted(set(sequence) - set("ACGT"))
    if invalid:
        raise SequenceValidationError(f"Unsupported nucleotide symbols: {', '.join(invalid)}")
    return sequence
```

- [ ] **Step 4: Run validation tests**

Run: `PYTHONPATH=src pytest tests/test_validation.py -v`

Expected: all tests pass.

### Task 3: Package and verify model inference

**Files:**
- Create: `src/viral_classifier/config.py`
- Create: `src/viral_classifier/model.py`
- Create: `src/viral_classifier/predictor.py`
- Create: `tests/test_predictor.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: normalized DNA strings and the six canonical artifact paths.
- Produces: `Prediction`, `ViralReadPredictor.predict(raw: str) -> Prediction`, and `ViralReadPredictor.project(raw: str) -> numpy.ndarray`.

- [ ] **Step 1: Write failing artifact and prediction tests**

```python
import numpy as np
import pytest

from viral_classifier.predictor import ViralReadPredictor


def test_predictor_returns_ordered_probabilities():
    predictor = ViralReadPredictor.load_default()
    result = predictor.predict("ACGT" * 38)
    assert len(result.probabilities) == 6
    assert list(result.probabilities) == [
        "Coronaviridae", "Human", "Influenza",
        "Metapneumovirus", "Rhinovirus", "Sars_Cov-2",
    ]
    assert np.isfinite(list(result.probabilities.values())).all()
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.predicted_class in result.probabilities
    assert result.tokens


def test_predictor_is_deterministic():
    predictor = ViralReadPredictor.load_default()
    first = predictor.predict("ACGT" * 38)
    second = predictor.predict("ACGT" * 38)
    assert first.probabilities == second.probabilities
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src pytest tests/test_predictor.py -v`

Expected: collection fails because `viral_classifier.predictor` does not exist.

- [ ] **Step 3: Implement repository-relative configuration**

Create a frozen `ArtifactPaths` dataclass and a `ModelConfig` dataclass. Resolve the repository as `Path(__file__).resolve().parents[2]`; never rely on the process working directory.

- [ ] **Step 4: Implement the inference-only model**

Port `RnnModelForClassification` with an embedding layer, one batch-first LSTM, padding-aware last-token selection, and a six-output linear layer. Keep constructor dimensions explicit and compatible with the verified state dictionary.

- [ ] **Step 5: Implement typed prediction**

```python
@dataclass(frozen=True)
class Prediction:
    sequence: str
    tokens: tuple[str, ...]
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float
```

`load_default()` loads JSON labels in integer order, constructs the tokenizer and classifier once, loads the state dictionary on CPU, and calls `model.eval()`. `predict()` validates the sequence, encodes it, executes under `torch.inference_mode()`, and applies `torch.softmax(logits, dim=-1)`.

- [ ] **Step 6: Run predictor tests**

Run: `PYTHONPATH=src pytest tests/test_predictor.py -v`

Expected: all tests pass and both historical weights have an explicitly documented disposition.

### Task 4: Isolate visualization behavior

**Files:**
- Create: `src/viral_classifier/visualization.py`
- Create: `tests/test_visualization.py`

**Interfaces:**
- Consumes: `Prediction`, the predictor embedding vector, historical PCA, and reference projection arrays.
- Produces: `ProjectionData`, `ViralReadPredictor.projection_data(raw: str, seed: int = 2022) -> ProjectionData`, `probability_figure(prediction: Prediction) -> plotly.graph_objects.Figure`, and `projection_figure(data: ProjectionData) -> plotly.graph_objects.Figure`.

- [ ] **Step 1: Write failing figure tests**

```python
import numpy as np

from viral_classifier.visualization import probability_figure


def test_probability_figure_has_six_bars(prediction):
    figure = probability_figure(prediction)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 6


def test_projection_sampling_is_deterministic(predictor):
    first = predictor.projection_data("ACGT" * 38, seed=2022)
    second = predictor.projection_data("ACGT" * 38, seed=2022)
    np.testing.assert_array_equal(first.points, second.points)
```

Add shared `predictor` and `prediction` session fixtures to `tests/conftest.py` so the model artifacts load once for the visualization and predictor tests.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src pytest tests/test_visualization.py -v`

Expected: failure because visualization functions are undefined.

- [ ] **Step 3: Implement probability plotting**

Build one Plotly bar trace in label order, format confidence as a percentage, and do not imply calibrated clinical probability.

- [ ] **Step 4: Implement deterministic PCA plotting**

Define `ProjectionData` as a frozen dataclass containing `points: numpy.ndarray`, `labels: tuple[str, ...]`, and `query_point: numpy.ndarray`. Use `numpy.random.default_rng(seed)` and sample at most 50 reference points per class without replacement. Mark the submitted read as a distinct black diamond. Catch PCA-only exceptions at the application boundary so classification remains available.

- [ ] **Step 5: Run visualization tests**

Run: `PYTHONPATH=src pytest tests/test_visualization.py -v`

Expected: all visualization tests pass.

### Task 5: Replace Dash with a Gradio application

**Files:**
- Create: `app.py`
- Create: `tests/test_app_smoke.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Delete after archival verification: `Procfile`

**Interfaces:**
- Consumes: `ViralReadPredictor`, validation errors, and visualization functions.
- Produces: importable `demo: gradio.Blocks` and `run_prediction(raw: str)` callback.

- [ ] **Step 1: Write a failing import smoke test**

```python
def test_app_exposes_demo():
    import app
    assert app.demo is not None
    assert callable(app.run_prediction)
```

- [ ] **Step 2: Run the smoke test and verify failure**

Run: `PYTHONPATH=src pytest tests/test_app_smoke.py -v`

Expected: failure because the archived Dash app left no root application.

- [ ] **Step 3: Implement the Gradio interface**

Construct a `gr.Blocks` page with project context, a sequence textbox, submit button, predicted class, confidence, token display, probability plot, PCA plot, validation status, limitations, and a non-clinical-use notice. Do not include examples unless a trustworthy labeled source is found in the preserved data.

- [ ] **Step 4: Implement callback error boundaries**

Validation errors return a readable status and empty result components. PCA errors return the valid classification plus a projection warning. Artifact failures remain startup errors with explicit paths.

- [ ] **Step 5: Update dependencies**

Define compatible Python and package ranges in `pyproject.toml` and retain a concise `requirements.txt` for Spaces. Remove the PyTorch 1.11 wheel-index override and the unused Dash/Gunicorn dependencies.

- [ ] **Step 6: Run application tests**

Run: `PYTHONPATH=src pytest tests/test_app_smoke.py tests/test_validation.py tests/test_predictor.py tests/test_visualization.py -v`

Expected: all tests pass without starting a network server.

### Task 6: Replace the one-line README with a research-demo README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: verified runtime commands, canonical artifact selection, and test results.
- Produces: a self-contained portfolio landing page.

- [ ] **Step 1: Document the project accurately**

Include the historical context, task definition, six classes, BPE-to-LSTM architecture, repository layout, installation, local launch, test command, artifact provenance, known limitations, and PACIFIC citation.

- [ ] **Step 2: State historical results with their evaluation boundary**

Use wording equivalent to:

> The original 2022 experiment reported 99.5% accuracy on a random held-out split of synthetic PACIFIC-derived reads. This repository does not treat that score as evidence of clinical performance, novel-virus detection, or generalization to unseen genome assemblies.

- [ ] **Step 3: Document archive status**

Explain that notebooks retain their original outputs but depend on external Colab/Kaggle data and are not part of the maintained inference runtime.

- [ ] **Step 4: Verify all documented commands**

Run every installation-independent command shown in the README and correct any mismatch before completion.

### Task 7: Final verification and manual smoke test

**Files:**
- Modify only files implicated by failing verification.

**Interfaces:**
- Consumes: the complete refactored project.
- Produces: evidence that the local inference demo is operational.

- [ ] **Step 1: Run static compilation**

Run: `python -m compileall -q app.py src tests`

Expected: exit code 0.

- [ ] **Step 2: Run the complete test suite**

Run: `PYTHONPATH=src pytest -v`

Expected: all tests pass.

- [ ] **Step 3: Start the application locally**

Run: `PYTHONPATH=src python app.py`

Expected: Gradio reports a local URL and no artifact-loading exception.

- [ ] **Step 4: Exercise one UI prediction**

Submit a 152-nt `ACGT` sequence, verify that a class, confidence, six probabilities, tokens, and PCA figure appear, then stop the local server.

- [ ] **Step 5: Review the working tree without committing**

Run: `git status --short` and `git diff --check`.

Expected: only intentional refactor changes and no whitespace errors. Do not stage or commit files.
