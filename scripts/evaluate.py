"""Thin entry point: evaluate the saved model.

    uv run scripts/evaluate.py

Like scripts/train.py, this just wires things together. The metric logic would
live in src/my_ml_project/ so it stays testable.
"""

from __future__ import annotations
from pathlib import Path

import joblib

from trial_conversion_model.data import load_data
from trial_conversion_model.evaluate import evaluate_held_out

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit("No model found. Run `uv run scripts/train.py` first.")

    model = joblib.load(MODEL_PATH)
    df = load_data()
    metrics = evaluate_held_out(model, df)

    print(f"Held-out rows: {metrics['held_out_rows']}")
    print(f"Macro F1:      {metrics['macro_f1']:.3f}")
    print(f"AUC:           {metrics['auc']:.3f}")
    print("Confusion matrix (rows: actual 0/1, cols: predicted 0/1):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")


if __name__ == "__main__":
    main()