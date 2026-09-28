"""Plain tests for data.py. Run with: uv run python tests/test_data.py"""

import pandas as pd

from trial_conversion_model.data import clean_data


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": [1, 1, 2],
            "snapshot_date": ["2026-09-24"] * 3,
            "trial_started_at": ["2026-08-01"] * 3,
        }
    )


def test_dates_are_parsed():
    df = clean_data(sample())
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["trial_started_at"])


def test_duplicate_trials_are_dropped():
    df = clean_data(sample())
    assert df["trial_id"].tolist() == [1, 2]


def test_input_is_not_modified():
    raw = sample()
    clean_data(raw)
    assert not pd.api.types.is_datetime64_any_dtype(raw["snapshot_date"])
    assert len(raw) == 3


if __name__ == "__main__":
    test_dates_are_parsed()
    test_duplicate_trials_are_dropped()
    test_input_is_not_modified()
    print("ok")
