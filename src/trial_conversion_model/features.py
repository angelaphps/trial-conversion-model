"""Feature engineering.

Keep feature logic here as plain, testable functions, so your notebook and your
training script share one source of truth. Each feature is its own function
that takes a DataFrame and returns one with the new column added.
"""

from __future__ import annotations

import pandas as pd

SESSION_COLS = ["sessions_day1", "sessions_day2", "sessions_day3"]
CATEGORICAL = ["country", "device_type"]
FEATURES = [
    "sessions_3d",
    "active_days_3d",
    "day1_share",
    "listen_share",
    "avg_session_minutes",
    "total_minutes_3d",
    *CATEGORICAL,
]
TARGET = "converted"


def add_sessions_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Total sessions across the first 3 days."""
    df["sessions_3d"] = df[SESSION_COLS].sum(axis=1)
    return df


def add_active_days_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Number of the first 3 days with at least one session."""
    df["active_days_3d"] = (df[SESSION_COLS] > 0).sum(axis=1)
    return df


def add_day1_share(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the 3-day sessions that happened on day 1.

    Trials with no sessions divide by zero. Zero engagement is real
    information, so those are filled with 0 rather than dropped.
    """
    df["day1_share"] = (df["sessions_day1"] / df["sessions_3d"]).fillna(0)
    return df


def add_listen_share(df: pd.DataFrame) -> pd.DataFrame:
    """Share of the 3-day sessions that were listening sessions. Zero-filled as above."""
    df["listen_share"] = (df["listen_sessions_3d"] / df["sessions_3d"]).fillna(0)
    return df


def add_avg_session_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Average minutes per session over the first 3 days. Zero-filled as above."""
    df["avg_session_minutes"] = (df["total_minutes_3d"] / df["sessions_3d"]).fillna(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived columns. Order matters: the shares need sessions_3d first."""
    df = df.copy()
    df = add_sessions_3d(df)
    df = add_active_days_3d(df)
    df = add_day1_share(df)
    df = add_listen_share(df)
    df = add_avg_session_minutes(df)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns and one-hot encode the categorical ones."""
    return pd.get_dummies(df[FEATURES], columns=CATEGORICAL)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a cleaned DataFrame into a feature matrix X and a target y."""
    df = add_features(df)
    X = encode_features(df)
    y = df[TARGET]
    return X, y
