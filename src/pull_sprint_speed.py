"""
Download Statcast Sprint Speed for each season into data/raw/sprint_speed/.
Used as the runner-speed control for Extra Bases Taken (a fast runner should
be expected to take more extra bases). Seasons already downloaded are skipped.

Join it to Retrosheet plays with the MLBAM ID from player_crosswalk.py
(sprint speed's player_id is the MLBAM ID).

Usage:
    python src/pull_sprint_speed.py
"""

from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sprint_speed"
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]  # no 2020


def pull_season(year, min_opp=5, force=False):
    """Sprint speed for one season. min_opp = minimum sprint opportunities to be listed."""
    cache = CACHE_DIR / f"{year}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    from pybaseball import statcast_sprint_speed

    print(f"{year}: downloading sprint speed...")
    df = statcast_sprint_speed(year, min_opp=min_opp)
    df["season"] = year

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def pull_all_seasons(seasons=SEASONS):
    return pd.concat([pull_season(y) for y in seasons], ignore_index=True)


if __name__ == "__main__":
    all_speed = pull_all_seasons()
    print(f"{len(all_speed):,} player-seasons")
