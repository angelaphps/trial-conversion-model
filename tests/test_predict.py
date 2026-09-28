"""Plain tests for predict.py. Run with: uv run python tests/test_predict.py"""

import pandas as pd

from trial_conversion_model.features import add_features, encode_features
from trial_conversion_model.predict import align_columns, predict
from trial_conversion_model.train import train_model
from test_train import sample


def test_align_adds_missing_columns_as_zero():
    model = train_model(sample())
    # only EU and iOS rows, so encoding yields no US/India/Rest/Android columns
    rows = sample().query("country == 'EU' and device_type == 'iOS'")
    X = encode_features(add_features(rows))
    aligned = align_columns(model, X)
    assert list(aligned.columns) == list(model.feature_names_in_)
    assert (aligned["country_US"] == 0).all()
    assert (aligned["device_type_Android"] == 0).all()


def test_predict_returns_probabilities():
    df = sample()
    model = train_model(df)
    X = encode_features(add_features(df))
    probs = predict(model, X)
    assert isinstance(probs, pd.Series)
    assert list(probs.index) == list(X.index)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_predict_works_with_missing_categories():
    model = train_model(sample())
    rows = sample().query("country == 'EU' and device_type == 'iOS'")
    X = encode_features(add_features(rows))
    probs = predict(model, X)
    assert len(probs) == len(rows)


if __name__ == "__main__":
    test_align_adds_missing_columns_as_zero()
    test_predict_returns_probabilities()
    test_predict_works_with_missing_categories()
    print("ok")
