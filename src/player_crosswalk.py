"""
Crosswalk between Retrosheet player IDs and MLBAM (Statcast) player IDs.
Retrosheet and Statcast name players differently, so this is needed any time
the two sources are joined at the player level (e.g. adding Statcast sprint
speed to Retrosheet baserunning plays).

Source: Chadwick Bureau Register (github.com/chadwickbureau/register), which is
split across 16 files, people-0.csv through people-f.csv.

Usage:
    python src/player_crosswalk.py
"""

import io
from pathlib import Path

import pandas as pd
import requests

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "register_crosswalk.parquet"
URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv"


def get_crosswalk(force=False):
    """Returns a table with columns key_retro and key_mlbam (players with both IDs only)."""
    if CACHE_PATH.exists() and not force:
        return pd.read_parquet(CACHE_PATH)

    frames = []
    for suffix in "0123456789abcdef":
        resp = requests.get(URL.format(suffix), timeout=60)
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text), usecols=["key_retro", "key_mlbam"], low_memory=False))

    xwalk = pd.concat(frames, ignore_index=True).dropna(subset=["key_retro", "key_mlbam"])
    xwalk["key_mlbam"] = xwalk["key_mlbam"].astype(int)
    xwalk = xwalk.drop_duplicates(subset="key_retro")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    xwalk.to_parquet(CACHE_PATH, index=False)
    return xwalk


if __name__ == "__main__":
    xwalk = get_crosswalk()
    print(f"{len(xwalk):,} player ID pairs")
    print(xwalk.head())
