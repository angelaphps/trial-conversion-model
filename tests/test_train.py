"""Plain tests for train.py. Run with: uv run python tests/test_train.py"""

import numpy as np
import pandas as pd

from trial_conversion_model.features import build_features
from trial_conversion_model.train import split_data, train_model


def sample(n: int = 40) -> pd.DataFrame:
    # small synthetic frame: converters use the app more, so there is
    # something for the model to learn
    rng = np.random.default_rng(0)
    converted = np.array([0, 1] * (n // 2))
    base = 2 + 6 * converted
    return pd.DataFrame(
        {
            "trial_id": range(n),
            "country": rng.choice(["EU", "US", "India", "Rest"], n),
            "device_type": rng.choice(["iOS", "Android"], n),
            "sessions_day1": rng.poisson(base),
            "sessions_day2": rng.poisson(base),
            "sessions_day3": rng.poisson(base),
            "listen_sessions_3d": rng.poisson(base),
            "total_minutes_3d": rng.poisson(base * 10),
            "converted": converted,
        }
    )


def test_split_sizes():
    train_df, test_df = split_data(sample())
    assert len(train_df) == 30
    assert len(test_df) == 10


def test_split_is_stratified():
    train_df, test_df = split_data(sample())
    assert train_df["converted"].mean() == 0.5
    assert test_df["converted"].mean() == 0.5


def test_split_is_repeatable():
    _, first = split_data(sample())
    _, second = split_data(sample())
    assert first["trial_id"].tolist() == second["trial_id"].tolist()


def test_train_model_returns_fitted_model():
    df = sample()
    model = train_model(df)
    _, test_df = split_data(df)
    X, _ = build_features(test_df)
    probs = model.predict_proba(X)[:, 1]
    assert len(probs) == len(test_df)
    assert ((probs >= 0) & (probs <= 1)).all()


if __name__ == "__main__":
    test_split_sizes()
    test_split_is_stratified()
    test_split_is_repeatable()
    test_train_model_returns_fitted_model()
    print("ok")
