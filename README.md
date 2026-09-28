# Trial conversion model

Predicts, from a trial's first 3 days of behaviour, whether the user will convert to a paid plan at trial end. One row per completed trial, pulled from the `ml.trial_snapshot_latest` table.

All logic lives in the `trial_conversion_model` package under `src/`. The scripts in `scripts/` are thin entry points, and the notebook only imports from the package.

## Setup

1. Install dependencies:

   ```
   uv sync
   ```

2. Copy `.env.example` to `.env` and fill in the database credentials. The code reads `.env` itself, so no shell setup is needed.

## Run

Pull the training extract into `data/01_raw_data/trials_raw.csv` (about 1,500 rows):

```
uv run scripts/load_data.py
```

If the snapshot already exists it is read instead of the database. Delete it to force a fresh pull.

Build the feature table, train the model, and save the model and its metrics:

```
uv run scripts/train.py
```

This writes:

- `data/03_preprocessed_data/trials_features.csv`, the cleaned data with the derived feature columns
- `models/model.pkl`, the fitted XGBoost classifier
- `models/metrics.json`, macro F1, AUC, and the confusion matrix on the held-out split

Score the saved model on the held-out split and print the metrics:

```
uv run scripts/evaluate.py
```

Expected: macro F1 about 0.79 and AUC about 0.87 on 379 held-out rows.

## Tests

Plain Python scripts, no test runner needed:

```
for t in tests/test_*.py; do uv run python "$t"; done
```

Each file prints `ok`.

## Layout

```
src/trial_conversion_model/
  data.py       load the snapshot or query the database, then clean
  features.py   one function per derived feature, plus encoding
  train.py      train/test split and model fitting
  predict.py    column alignment and scoring
  evaluate.py   metrics
scripts/        load_data.py, train.py, evaluate.py
tests/          one test file per module
notebooks/      exploration, imports from the package
data/           snapshots and feature tables (gitignored)
models/         model.pkl and metrics.json (gitignored)
```
