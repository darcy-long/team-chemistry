"""
Download Statcast pitch-by-pitch data, one parquet per season, into data/raw/.
deg.py reads these files. Seasons already downloaded are skipped.

Needs internet, and each season takes a while.

Usage:
    python src/pull_statcast.py
"""

from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

cache.enable()

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]  # no 2020


def pull_season(year):
    out_path = RAW_DIR / f"statcast_{year}.parquet"
    if out_path.exists():
        print(f"{year}: already downloaded, skipping")
        return pd.read_parquet(out_path)

    print(f"{year}: downloading...")
    df = statcast(start_dt=f"{year}-03-15", end_dt=f"{year}-11-15")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{year}: saved {len(df):,} rows")
    return df


if __name__ == "__main__":
    for year in SEASONS:
        pull_season(year)
