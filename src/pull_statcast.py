"""
Download Statcast pitch-by-pitch data, one parquet per season, into data/raw/.
deg.py and play_pitch_join.py read these files. Seasons already downloaded are skipped.

Uses pybaseball, which caches its requests, so an interrupted run is cheaper to repeat.
Needs internet, and each season takes a while.

The window is March 15 to November 15 (or today, if earlier), so a season still in progress is
only downloaded through today. Re-run with --force once it has finished.

Usage:
    python src/pull_statcast.py                # every season in SEASONS not already downloaded
    python src/pull_statcast.py 2026           # one season
    python src/pull_statcast.py 2026 --force   # download it again (e.g. the season has since ended)
"""

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

cache.enable()

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]  # no 2020


def pull_season(year, force=False):
    out_path = RAW_DIR / f"statcast_{year}.parquet"
    if out_path.exists() and not force:
        print(f"{year}: already downloaded, skipping")
        return pd.read_parquet(out_path)

    end = min(date(year, 11, 15), date.today())
    print(f"{year}: downloading March 15 to {end}...")
    df = statcast(start_dt=f"{year}-03-15", end_dt=end.isoformat())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{year}: saved {len(df):,} rows through {df['game_date'].max()}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download Statcast pitch data via pybaseball.")
    ap.add_argument("years", nargs="*", type=int, help="seasons to download (default: all in SEASONS)")
    ap.add_argument("--force", action="store_true", help="download again even if the season already exists")
    args = ap.parse_args()
    sizes = {year: len(pull_season(year, force=args.force)) for year in args.years or SEASONS}
    print(f"\nDone. Statcast pitch data is in {RAW_DIR}:")
    for year, n in sizes.items():
        print(f"  statcast_{year}.parquet: {n:,} pitches")
