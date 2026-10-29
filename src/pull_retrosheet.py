"""
Download Retrosheet play-by-play (plays.csv) for each season into data/raw/retrosheet/.
This is the source for the baserunning indicator and the standings.
Seasons already downloaded are skipped.

Usage:
    python src/pull_retrosheet.py
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "retrosheet"
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]  # no 2020

# All-Star game rows use fake team codes, so they get dropped
ALL_STAR_CODES = {"ALS", "NLS"}


def pull_season(year, force=False):
    cache = RAW_DIR / f"{year}plays.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    url = f"https://www.retrosheet.org/downloads/plays/{year}plays.zip"
    print(f"{year}: downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(csv_name), low_memory=False)

    df = df[~df["batteam"].isin(ALL_STAR_CODES) & ~df["pitteam"].isin(ALL_STAR_CODES)]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"{year}: saved {len(df):,} rows")
    return df


if __name__ == "__main__":
    for year in SEASONS:
        pull_season(year)
