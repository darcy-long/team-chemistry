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
ALL_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]  # no 2020

# Retrosheet posts a season's play-by-play after the season ends, sometimes months later.
# A season listed here has no file yet (2026 returned 404 on 2026-11-05), so every script that
# loops over SEASONS skips it. As soon as the file has been downloaded it is used automatically;
# once Retrosheet has it, take the year out of this set.
NOT_PUBLISHED_YET = {2026}
SEASONS = [y for y in ALL_SEASONS if y not in NOT_PUBLISHED_YET or (RAW_DIR / f"{y}plays.parquet").exists()]

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
    have, missing = [], []
    for year in ALL_SEASONS:
        try:
            have.append((year, len(pull_season(year))))
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                print(f"{year}: not on Retrosheet yet, skipping")
                missing.append(year)
            else:
                raise
    print(f"\nDone. {len(have)} seasons of play-by-play are in {RAW_DIR}:")
    for year, n in have:
        print(f"  {year}: {n:,} plays")
    if missing:
        print(f"Not on Retrosheet yet: {missing}")
