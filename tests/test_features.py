"""Plain tests for features.py. Run with: uv run python tests/test_features.py"""

import pandas as pd

from trial_conversion_model.features import (
    add_active_days_3d,
    add_avg_session_minutes,
    add_day1_share,
    add_listen_share,
    add_sessions_3d,
    build_features,
)


def sample() -> pd.DataFrame:
    # row 0: normal trial; row 1: no sessions at all; row 2: everything on day 1
    return pd.DataFrame(
        {
            "trial_id": [1, 2, 3],
            "country": ["EU", "US", "EU"],
            "device_type": ["iOS", "Android", "iOS"],
            "sessions_day1": [4, 0, 5],
            "sessions_day2": [0, 0, 0],
            "sessions_day3": [2, 0, 0],
            "listen_sessions_3d": [3, 0, 5],
            "total_minutes_3d": [60, 0, 50],
            "converted": [0, 0, 1],
        }
    )


def test_sessions_3d():
    df = add_sessions_3d(sample())
    assert df["sessions_3d"].tolist() == [6, 0, 5]


def test_active_days_3d():
    df = add_active_days_3d(sample())
    assert df["active_days_3d"].tolist() == [2, 0, 1]


def test_day1_share():
    df = add_day1_share(add_sessions_3d(sample()))
    assert df["day1_share"].round(4).tolist() == [0.6667, 0.0, 1.0]


def test_listen_share():
    df = add_listen_share(add_sessions_3d(sample()))
    assert df["listen_share"].tolist() == [0.5, 0.0, 1.0]


def test_avg_session_minutes():
    df = add_avg_session_minutes(add_sessions_3d(sample()))
    assert df["avg_session_minutes"].tolist() == [10.0, 0.0, 10.0]


def test_build_features():
    X, y = build_features(sample())
    assert len(X) == 3
    assert y.tolist() == [0, 0, 1]
    assert "converted" not in X.columns
    assert "country_EU" in X.columns and "device_type_iOS" in X.columns
    assert X.isna().sum().sum() == 0


if __name__ == "__main__":
    test_sessions_3d()
    test_active_days_3d()
    test_day1_share()
    test_listen_share()
    test_avg_session_minutes()
    test_build_features()
    print("ok")
