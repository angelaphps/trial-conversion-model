"""Evaluation logic.

`evaluate_model` turns true labels and predicted probabilities into the metrics
we report. Macro F1 is the decision metric; AUC and the confusion matrix are
there for context. `evaluate_held_out` runs that on the held-out split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from trial_conversion_model.features import build_features
from trial_conversion_model.predict import predict
from trial_conversion_model.train import split_data

THRESHOLD = 0.5


def evaluate_model(y: pd.Series, probs: pd.Series) -> dict:
    """Return macro F1, AUC, and the confusion matrix for one set of predictions."""
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "auc": float(roc_auc_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }


def evaluate_held_out(model, df: pd.DataFrame) -> dict:
    """Score the model on the held-out rows of df and return the metrics."""
    _, test_df = split_data(df)
    X, y = build_features(test_df)
    probs = predict(model, X)
    metrics = evaluate_model(y, probs)
    metrics["held_out_rows"] = len(y)
    return metrics
