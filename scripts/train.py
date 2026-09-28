"""Thin entry point: build the feature table, train the model, save both plus metrics.

    uv run scripts/train.py

Production runs a script, never a notebook. All the real logic lives in
src/trial_conversion_model/, so this stays a few lines.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from trial_conversion_model.data import load_data
from trial_conversion_model.evaluate import evaluate_held_out
from trial_conversion_model.features import add_features
from trial_conversion_model.train import train_model

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "03_preprocessed_data" / "trials_features.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def main() -> None:
    df = load_data()

    features = add_features(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_PATH, index=False)
    print(f"Feature table saved to {PROCESSED_PATH}")

    model = train_model(df)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    metrics = evaluate_held_out(model, df)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}  AUC: {metrics['auc']:.3f}")


if __name__ == "__main__":
    main()
