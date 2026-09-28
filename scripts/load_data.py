"""Thin entry point: pull the training extract into data/01_raw_data/.

    uv run scripts/load_data.py

Queries the database and saves a CSV snapshot. If the snapshot already exists
it is read instead, so delete it to force a fresh pull.
"""

from __future__ import annotations

from trial_conversion_model.data import RAW_PATH, get_data


def main() -> None:
    df = get_data()
    print(f"{len(df)} rows at {RAW_PATH}")


if __name__ == "__main__":
    main()
