"""Inference logic.

Turns a fitted model and some rows into predictions.
"""

from __future__ import annotations

import pandas as pd


def align_columns(model, X: pd.DataFrame) -> pd.DataFrame:
    """Match X's columns to the ones the model was trained on.

    One-hot encoding only creates columns for categories present in the rows
    being scored, so a batch with no US trials would have no `country_US`
    column. Missing columns are added as 0 and the order is set to match.
    """
    return X.reindex(columns=model.feature_names_in_, fill_value=0)


def predict(model, X: pd.DataFrame) -> pd.Series:
    """Return the predicted probability of conversion for each row in X."""
    X = align_columns(model, X)
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=X.index, name="conversion_prob")
