"""Training logic (the real work, kept out of the thin script).

`train_model` takes cleaned data in and returns a fitted model. Keeping it as a
function, not a script, means scripts, notebooks, tests, and pipelines can all
call it without copying code.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from trial_conversion_model.features import TARGET, build_features

TEST_SIZE = 0.25
RANDOM_STATE = 42
MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and held-out test rows.

    Stratified on the target with a fixed seed, so training and evaluation
    always agree on which rows are held out.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )
    return train_df, test_df


def train_model(df: pd.DataFrame) -> XGBClassifier:
    """Fit and return a model on the training rows."""
    train_df, _ = split_data(df)
    X, y = build_features(train_df)
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(X, y)
    return model
