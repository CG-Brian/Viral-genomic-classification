# Training methodology

This documents how `artifacts/` was produced and provides a reference
implementation of each stage. **It is not a one-command reproduction**: the
PACIFIC-derived read corpus (~8M reads, not distributed with this
repository) and the original Colab/Kaggle GPU environment are both required
to actually run it end to end. Evidence for every claim below is either the
original project report or the original notebooks/code (recovered from git
history); where they disagree, that is stated explicitly rather than
resolved by guessing.

## Dataset

Reads are sourced from the synthetic PACIFIC training corpus (ART-simulated
short reads). Six classes, with substantial imbalance:

| Class | Reads |
|---|---:|
| Human (control) | 3,531,439 |
| Rhinovirus (A/B/C) | 1,357,334 |
| Influenza (H1N1, H2N2, H3N2, H5N1, H7N9, H9N2, B) | 1,123,125 |
| SARS-CoV-2 | 866,496 |
| Coronaviridae (other alpha/beta/gamma) | 646,548 |
| Metapneumovirus | 532,881 |

`dataset.py` loads this layout and computes inverse-frequency class weights
(`sklearn.utils.class_weight.compute_class_weight`) to counteract the
imbalance in the classifier's loss, per the original report.

## 1. BPE tokenizer

Trained with Hugging Face `tokenizers` (`models.BPE`, vocab size 24,000) over
the full read corpus, character-initialized on `A`/`C`/`G`/`T`. This is
`artifacts/tokenizer.json`. Not reproduced as a module here — the recipe is
~10 lines against `tokenizers.trainers.BpeTrainer` and doesn't warrant a
separate file; see the original notebook preserved in git history at
`git show 3c923fd:archive/2022/notebooks/01_tokenizer_training.ipynb`.

## 2. CBOW embedding pretraining

`embedding_pretraining.py`. An embedding layer plus a linear decoder is
trained, unsupervised, to predict each token from the sum of its window-2
left/right context embeddings (embedding_dim=256). The original code named
this class `SkipGramEmbeddingModel`, but the objective — predicting a center
token from summed context — is CBOW, not skip-gram; this repository's
version is renamed accordingly.

## 3. LSTM classifier

`classifier_training.py`. Reuses the same `ViralLSTMClassifier` defined in
`src/viral_classifier/model.py` (one LSTM layer, 512 hidden units, six
outputs) rather than a separate training-only class, and optionally loads
the CBOW-pretrained embedding with `freeze_embedding=True/False`.

## What was actually shipped, and the accuracy discrepancy

The production Dash app (`git show 3c923fd:archive/2022/dash_app.py`)
hardcoded `sgrnn_emb_ftrue.pth` as its weight file — "skipgram/CBOW-RNN,
embedding **f**reeze=**true**" — which is today's `artifacts/classifier.pth`.
The embedding-visualization notebook derived `reference_embeddings.npy`/
`pca.pkl` from that same file's embedding layer, so the frozen-CBOW
configuration is confirmed as what this repository ships.

However, the original project report ran three configurations and reported:

| Embedding configuration | Weighted F1 |
|---|---|
| CBOW-pretrained, **frozen** (the shipped configuration) | ≈ 0.90 |
| CBOW-pretrained, fine-tuned (unfrozen) | ≈ 0.74 (overfit) |
| No pretraining — embedding trained jointly with the LSTM from scratch | > 0.995 |

The report's Results section attributes the headline **99.5% accuracy /
F1 > 0.995** to the *no-pretraining* run — not the frozen-CBOW run that was
actually deployed. Its own Discussion section then contradicts this,
crediting CBOW pretraining for the >0.99 score and describing the
no-pretraining run as "poor." No saved metrics survive in the recovered
notebooks (outputs were stripped before commit) to settle which claim is
correct. This repository does not resolve that contradiction; the main
[README](../README.md#evaluation-results) reports both numbers and states
which configuration is actually deployed.

## 4. Embedding / PCA analysis

`embedding_analysis.py`. Sums each read's token embeddings, pools 100,000
sampled reads per class (600,000 total — matching
`artifacts/reference_embeddings.npy`), and fits a 3-component PCA. The
demo (`src/viral_classifier/predictor.py`) reuses this fitted PCA's
`mean_`/`components_` at inference time; it does not refit PCA.
