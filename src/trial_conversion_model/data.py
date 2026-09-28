"""Data loading and cleaning.

`load_data` fetches the raw data (from a local snapshot if present, otherwise
from the database) and then cleans it.
"""

from __future__ import annotations

from os import getenv
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "01_raw_data" / "trials_raw.csv"
QUERY = "SELECT * FROM ml.trial_snapshot_latest"


def get_data() -> pd.DataFrame:
    """Return the raw data, reading the local snapshot if it exists.

    If there is no snapshot, query the database and save one so the
    database is only hit once.
    """
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)

    engine = create_engine(
        f"postgresql+psycopg://{getenv('DB_USER')}:{getenv('DB_PASSWORD')}"
        f"@{getenv('DB_HOST')}:{getenv('DB_PORT', '5432')}/{getenv('DB_NAME')}"
    )
    df = pd.read_sql(QUERY, engine)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw data."""
    df = df.copy()

    # dates load as strings from the CSV snapshot; make them real dates
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["trial_started_at"] = pd.to_datetime(df["trial_started_at"])

    # one row per trial
    df = df.drop_duplicates(subset="trial_id")

    return df


def load_data() -> pd.DataFrame:
    """
    Load and return the cleaned dataset.
    """
    df = get_data()
    df = clean_data(df)
    return df
