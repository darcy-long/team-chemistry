"""
Check that Retrosheet and Statcast team codes line up after the mapping in
team_codes.py. For each season, prints any team code that appears in one source
but has no match in the other. Prints nothing but "OK" per season when they agree.

Needs data/raw/retrosheet/ and data/raw/statcast_YYYY.parquet.

Usage:
    python src/check_team_codes.py
"""

from pathlib import Path

import pandas as pd

from pull_retrosheet import SEASONS
from team_codes import RETROSHEET_TO_STATCAST

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    for year in SEASONS:
        retro = pd.read_parquet(RAW_DIR / "retrosheet" / f"{year}plays.parquet", columns=["batteam", "gametype"])
        retro = retro[retro["gametype"] == "regular"]
        retro_codes = set(retro["batteam"].map(RETROSHEET_TO_STATCAST).fillna(retro["batteam"]))

        sc = pd.read_parquet(RAW_DIR / f"statcast_{year}.parquet", columns=["home_team", "away_team", "game_type"])
        sc = sc[sc["game_type"] == "R"]
        statcast_codes = set(sc["home_team"]) | set(sc["away_team"])

        only_retro = sorted(retro_codes - statcast_codes)
        only_statcast = sorted(statcast_codes - retro_codes)
        if only_retro or only_statcast:
            print(f"{year}: only in Retrosheet {only_retro}, only in Statcast {only_statcast}")
        else:
            print(f"{year}: OK ({len(retro_codes)} teams)")


if __name__ == "__main__":
    main()
