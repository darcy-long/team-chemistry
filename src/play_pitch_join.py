"""
Play + pitch join: attach Statcast pitch data to every Retrosheet play.

One row per Retrosheet play (all original Retrosheet columns kept), plus:
    game_pk, at_bat_number   Statcast game and at-bat this play was matched to
    sc_status                ok | no_statcast_pa | mismatch  (see below)
    sc_<field>               Statcast fields from the LAST pitch of the plate appearance:
                             launch_speed, launch_angle, hit_distance_sc, hc_x, hc_y,
                             estimated_ba_using_speedangle, pitch_type, release_speed,
                             events, description, bb_type
    pitch_types, pitch_speeds, pitch_descriptions, pitch_zones
                             lists with every pitch of the plate appearance, in order

How plays are matched: game_pk comes from game_crosswalk.py. Within a game, each
Retrosheet plate appearance is matched to the next Statcast at-bat with the same inning,
half-inning, batter and pitcher (Retrosheet player IDs go through player_crosswalk.py);
see align_game. Counting plate appearances doesn't work because Statcast numbers
unfinished plate appearances (e.g. a caught stealing for the 3rd out) that Retrosheet doesn't.
Plays that are not plate appearances (steals, pickoffs, wild pitches) take the at-bat of the
plate appearance they happened in. Every match is checked again against inning, half, batter
and pitcher. A play that fails is marked sc_status = "mismatch" and gets NO Statcast data, so
wrong pitches are never attached.
    no_statcast_pa: no Statcast at-bat was found for the play

Input:  data/raw/retrosheet/YYYYplays.parquet, data/raw/statcast_YYYY.parquet,
        data/processed/game_crosswalk.csv
Output: data/processed/plays_statcast/YYYY.parquet

Usage:
    python src/play_pitch_join.py            # all seasons
    python src/play_pitch_join.py 2023       # one season
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from game_crosswalk import get_game_crosswalk
from player_crosswalk import get_crosswalk
from pull_retrosheet import SEASONS, pull_season

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "plays_statcast"

# Statcast fields taken from the last pitch of the plate appearance
LAST_PITCH_FIELDS = [
    "launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y",
    "estimated_ba_using_speedangle", "pitch_type", "release_speed",
    "events", "description", "bb_type",
]
# Statcast fields kept as a list over every pitch: column name -> Statcast column
PITCH_LIST_FIELDS = {
    "pitch_types": "pitch_type",
    "pitch_speeds": "release_speed",
    "pitch_descriptions": "description",
    "pitch_zones": "zone",
}
SC_COLUMNS = (
    ["game_pk", "at_bat_number", "pitch_number", "inning", "inning_topbot", "batter", "pitcher", "zone"]
    + LAST_PITCH_FIELDS
)


def statcast_plate_appearances(year):
    """One row per (game_pk, at_bat_number): last-pitch fields + lists of every pitch."""
    sc = pd.read_parquet(RAW_DIR / f"statcast_{year}.parquet", columns=sorted(set(SC_COLUMNS)))
    sc = sc.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    grouped = sc.groupby(["game_pk", "at_bat_number"], sort=False)

    # the true last pitch (groupby.last() would skip nulls and mix pitches)
    last = grouped.tail(1).set_index(["game_pk", "at_bat_number"])
    out = last[["inning", "inning_topbot", "batter", "pitcher"] + LAST_PITCH_FIELDS].copy()
    out = out.rename(columns={f: f"sc_{f}" for f in LAST_PITCH_FIELDS})
    out = out.rename(columns={"inning": "sc_inning", "inning_topbot": "sc_half",
                              "batter": "sc_batter", "pitcher": "sc_pitcher"})
    lists = grouped[list(set(PITCH_LIST_FIELDS.values()))].agg(list)
    lists = lists.rename(columns={col: name for name, col in PITCH_LIST_FIELDS.items()})
    return out.join(lists).reset_index()


LOOKAHEAD = 6  # how many Statcast at-bats to skip ahead when looking for the next match


def align_game(ret, sc):
    """
    Match one game's Retrosheet rows to Statcast at-bat numbers.

    ret: DataFrame of the game's plays in order (inning, top_bot, batter_id, pitcher_id, pa)
    sc:  DataFrame of the game's Statcast at-bats in order (inning, half, batter, pitcher, at_bat_number)
    Returns an array with a Statcast at_bat_number (or -1) for each Retrosheet row.

    Counting plate appearances isn't enough: Statcast gives an at-bat number to a
    plate appearance that never finishes (e.g. caught stealing for the 3rd out), and
    Retrosheet doesn't, so every later count is off by one. Instead, walk both lists
    together and match each Retrosheet plate appearance to the next Statcast at-bat
    with the same inning, half, batter and pitcher.
    """
    sc_key = list(zip(sc["inning"], sc["half"], sc["batter"], sc["pitcher"]))
    sc_ab = sc["at_bat_number"].to_numpy()
    ret_key = list(zip(ret["inning"], ret["top_bot"], ret["batter_id"], ret["pitcher_id"]))
    is_pa = ret["pa"].to_numpy() == 1

    result = np.full(len(ret), -1)
    pos = 0  # next Statcast at-bat to look at
    pa_pos = np.full(len(ret), -1)  # index into sc for each plate appearance
    for i in np.flatnonzero(is_pa):
        for k in range(pos, min(pos + LOOKAHEAD, len(sc_key))):
            if sc_key[k] == ret_key[i]:
                pa_pos[i] = k
                pos = k + 1
                break

    # Plays that aren't plate appearances (steals, pickoffs, wild pitches) belong to the
    # plate appearance that finishes next, or to the unfinished one just before it
    # (inning-ending caught stealing / pickoff).
    next_pa = len(sc_key)
    for i in range(len(ret) - 1, -1, -1):
        if is_pa[i]:
            next_pa = pa_pos[i]
            result[i] = sc_ab[next_pa] if next_pa >= 0 else -1
        elif next_pa >= 0:
            for k in (next_pa, next_pa - 1):
                if 0 <= k < len(sc_key) and sc_key[k][:3] == ret_key[i][:3]:
                    result[i] = sc_ab[k]
                    break
    return result


def join_season(year):
    crosswalk = get_game_crosswalk()
    crosswalk = crosswalk[crosswalk["season"] == year][["gid", "game_pk"]]
    players = get_crosswalk().set_index("key_retro")["key_mlbam"]

    plays = pull_season(year).merge(crosswalk, on="gid", how="left")
    plays["_batter_id"] = plays["batter"].map(players)
    plays["_pitcher_id"] = plays["pitcher"].map(players)

    sc_pa = statcast_plate_appearances(year)
    sc_pa["half"] = sc_pa["sc_half"].map({"Top": 0, "Bot": 1})
    sc_games = {pk: g for pk, g in sc_pa.rename(columns={
        "sc_inning": "inning", "sc_batter": "batter", "sc_pitcher": "pitcher"}).groupby("game_pk")}

    # Retrosheet plays are in game order within each gid, so row order is kept
    at_bat = np.full(len(plays), -1)
    for gid, idx in plays.groupby("gid").indices.items():
        pk = plays["game_pk"].iat[idx[0]]
        if pd.isna(pk) or pk not in sc_games:
            continue
        g = plays.iloc[idx].rename(columns={"_batter_id": "batter_id", "_pitcher_id": "pitcher_id"})
        at_bat[idx] = align_game(g, sc_games[pk])
    plays["at_bat_number"] = at_bat

    joined = plays.merge(sc_pa.drop(columns="half"), on=["game_pk", "at_bat_number"], how="left")
    joined = joined.drop(columns=["_batter_id", "_pitcher_id"])

    # Final check on every matched play: inning, half, batter and pitcher all agree
    sc_half = joined["sc_half"].map({"Top": 0, "Bot": 1})
    has_pa = joined["sc_inning"].notna()
    ok = (
        has_pa
        & (joined["inning"] == joined["sc_inning"])
        & (joined["top_bot"] == sc_half)
        & (joined["batter"].map(players) == joined["sc_batter"])
        & (joined["pitcher"].map(players) == joined["sc_pitcher"])
    )
    joined["sc_status"] = np.where(ok, "ok", np.where(has_pa, "mismatch", "no_statcast_pa"))

    # Never keep Statcast data on a play that failed the check
    sc_cols = [c for c in joined.columns if c.startswith("sc_") and c != "sc_status"] + list(PITCH_LIST_FIELDS)
    joined.loc[~ok, sc_cols] = np.nan
    return joined.drop(columns=["sc_inning", "sc_half", "sc_batter", "sc_pitcher"])


def main(years):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        df = join_season(year)
        counts = df["sc_status"].value_counts()
        pa = df[df["pa"] == 1]["sc_status"].value_counts(normalize=True)
        print(f"{year}: {len(df):,} plays | " + ", ".join(f"{k} {v:,}" for k, v in counts.items())
              + f" | plate appearances ok {pa.get('ok', 0):.2%}")
        df.to_parquet(OUT_DIR / f"{year}.parquet", index=False)
    print(f"\nDone. Saved {len(years)} season(s) to {OUT_DIR}")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or SEASONS)
