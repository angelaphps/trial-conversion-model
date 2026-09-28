"""Plain tests for evaluate.py. Run with: uv run python tests/test_evaluate.py"""

import pandas as pd

from trial_conversion_model.evaluate import evaluate_model


def test_perfect_predictions():
    y = pd.Series([0, 0, 1, 1])
    probs = pd.Series([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_model(y, probs)
    assert metrics["macro_f1"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_is_applied():
    # 0.5 exactly counts as a positive prediction
    y = pd.Series([0, 1])
    probs = pd.Series([0.5, 0.5])
    metrics = evaluate_model(y, probs)
    assert metrics["confusion_matrix"] == [[0, 1], [0, 1]]


if __name__ == "__main__":
    test_perfect_predictions()
    test_threshold_is_applied()
    print("ok")
