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
NAMES_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "register_names.parquet"
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


def get_names(force=False):
    """Returns a table with columns key_retro and name ("First Last") for every player in the
    register that has a Retrosheet ID, whether or not he has an MLBAM ID."""
    if NAMES_CACHE_PATH.exists() and not force:
        return pd.read_parquet(NAMES_CACHE_PATH)

    frames = []
    for suffix in "0123456789abcdef":
        resp = requests.get(URL.format(suffix), timeout=60)
        resp.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(resp.text), usecols=["key_retro", "name_first", "name_last"],
                                  low_memory=False))

    names = pd.concat(frames, ignore_index=True).dropna(subset=["key_retro"])
    names["name"] = (names["name_first"].fillna("") + " " + names["name_last"].fillna("")).str.strip()
    names = names[["key_retro", "name"]].drop_duplicates(subset="key_retro")

    NAMES_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    names.to_parquet(NAMES_CACHE_PATH, index=False)
    return names


if __name__ == "__main__":
    xwalk = get_crosswalk()
    names = get_names()
    print(f"Done. Saved to {CACHE_PATH.parent}:")
    print(f"  {CACHE_PATH.name}: {len(xwalk):,} players with both a Retrosheet and an MLBAM ID")
    print(f"  {NAMES_CACHE_PATH.name}: {len(names):,} players with a Retrosheet ID and a name")
