# 2022 experiment archive

This directory preserves the original viral-read classification experiments and
Dash application created in 2022. These files are historical records; the
maintained inference demo does not import them.

## Original environment

The notebooks were run in Google Colab and Kaggle with GPU acceleration. They
refer to external paths such as `../input/pacific-sra/trainingdata`, so they are
not a self-contained or currently verified training pipeline. The input data
came from the synthetic training data published with PACIFIC.

## Experiment flow

1. `notebooks/01_tokenizer_training.ipynb` trained the 24,000-token BPE
   tokenizer now stored at `../../artifacts/tokenizer.json`.
2. `notebooks/02_cbow_training.ipynb` experimented with CBOW-style pretraining
   for nucleotide-token embeddings.
3. `notebooks/03_lstm_training.ipynb` trained and evaluated the six-class LSTM
   classifier.
4. `notebooks/04_embedding_projection.ipynb` produced the saved PCA transform
   and reference projections used by the demo.

`notebooks/rnn_training_record.ipynb` and `results/` retain saved outputs from
the original experiments. The reported scores have not been reproduced as part
of the current inference-only restoration.

## Historical application and source

- `dash_app.py` is the original Dash interface.
- `Procfile` is the original Heroku/Gunicorn process declaration.
- `training/` contains the original model, data-loading, training, and inference
  utilities.
- `artifacts/model.pth` is the alternative classifier weight previously stored
  as `ML/assets/model.pth`.

The final 2022 Dash application referenced `sgrnn_emb_ftrue.pth`; that state
dictionary is therefore the canonical historical deployment weight and is now
stored as `../../artifacts/classifier.pth`. Both state dictionaries use a
24,000 × 256 embedding, a one-layer 512-unit LSTM, and six output classes.
