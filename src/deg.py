"""
Defensive Effort Gap (DEG)

Idea: a team that keeps trying should turn balls in play into outs just as well
when the game is a blowout as when it is close. If a team stops trying once the
game is out of reach, its conversion should drop.

For each team-season and each score state (close, ahead, behind):
    expected outs   = sum of (1 - xBA) over balls in play
    actual outs     = number of balls in play that were outs
    conversion gap  = (actual - expected) / balls in play

    DEG-Ahead  = gap(ahead)  - gap(close)
    DEG-Behind = gap(behind) - gap(close)

Score states (7th inning or later only, from the fielding team's view):
    close  : margin within 2 runs
    ahead  : leading by 5+ runs
    behind : trailing by 5+ runs

Input: Statcast pitch-level data, one parquet per season (data/raw/statcast_YYYY.parquet).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]  # no 2020
BLOWOUT_MARGIN = 5
CLOSE_MARGIN = 2
MIN_INNING = 7

OUT_EVENTS = {
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "fielders_choice_out", "triple_play", "sac_fly", "sac_fly_double_play",
}
BUNT_EVENTS = {"sac_bunt", "sac_bunt_double_play"}
BUNT_REGEX = r"\bbunt(?:s|ed|ing)?\b"


def fill_sac_fly_xba(df):
    """Statcast leaves xBA blank on sac flies, so fill it in with the league
    average xBA for balls hit with a similar exit speed and launch angle."""
    df = df.copy()
    is_sf = df["events"].isin(["sac_fly", "sac_fly_double_play"])
    if not is_sf.any():
        return df

    is_bunt = df["events"].isin(BUNT_EVENTS) | df["des"].str.contains(BUNT_REGEX, case=False, na=False)
    known = df[
        df["estimated_ba_using_speedangle"].notna()
        & df["launch_speed"].notna()
        & df["launch_angle"].notna()
        & (df["events"] != "home_run")
        & ~is_bunt
    ].copy()
    known["speed_bin"] = known["launch_speed"].round()
    known["angle_bin"] = known["launch_angle"].round()
    grid = (
        known.groupby(["speed_bin", "angle_bin"])["estimated_ba_using_speedangle"]
        .mean()
        .rename("grid_xba")
        .reset_index()
    )
    # if a sac fly falls in an empty bin, use the average for fly balls / line drives
    fallback = known.loc[known["bb_type"].isin(["fly_ball", "line_drive"]), "estimated_ba_using_speedangle"].mean()

    sf = df.loc[is_sf, ["launch_speed", "launch_angle"]].copy()
    sf["speed_bin"] = sf["launch_speed"].round()
    sf["angle_bin"] = sf["launch_angle"].round()
    sf["row_id"] = sf.index
    sf = sf.merge(grid, on=["speed_bin", "angle_bin"], how="left").set_index("row_id")

    df.loc[is_sf, "estimated_ba_using_speedangle"] = sf["grid_xba"].fillna(fallback).values
    return df


def get_balls_in_play(df):
    """Keep regular-season balls in play that a fielder could actually make a play on.
    Home runs and bunts are dropped. Sac flies count as outs."""
    df = df[df["game_type"] == "R"]
    df = fill_sac_fly_xba(df)
    bip = df[df["estimated_ba_using_speedangle"].notna()].copy()

    is_bunt = bip["events"].isin(BUNT_EVENTS) | bip["des"].str.contains(BUNT_REGEX, case=False, na=False)
    bip = bip[~is_bunt & (bip["events"] != "home_run")].copy()

    # the team in the field is the home team when the visitors are batting (top of the inning)
    bip["fielding_team"] = np.where(bip["inning_topbot"] == "Top", bip["home_team"], bip["away_team"])
    bip["fielding_margin"] = bip["fld_score"] - bip["bat_score"]
    bip["is_out"] = bip["events"].isin(OUT_EVENTS).astype(int)
    bip["expected_out"] = 1.0 - bip["estimated_ba_using_speedangle"]
    return bip


def add_score_state(bip, blowout_margin=BLOWOUT_MARGIN):
    """Label each ball in play as close / ahead / behind / other."""
    bip = bip.copy()
    margin = bip["fielding_margin"]
    late = bip["inning"] >= MIN_INNING

    bip["score_state"] = "other"
    bip.loc[late & (margin.abs() <= CLOSE_MARGIN), "score_state"] = "close"
    bip.loc[late & (margin >= blowout_margin), "score_state"] = "ahead"
    bip.loc[late & (margin <= -blowout_margin), "score_state"] = "behind"
    return bip


def compute_deg(bip):
    """One row per team-season with the conversion gap in each state and the DEG values."""
    bip = bip[bip["score_state"].isin(["close", "ahead", "behind"])]

    grouped = (
        bip.groupby(["season", "fielding_team", "score_state"])
        .agg(bip=("is_out", "count"), actual=("is_out", "sum"), expected=("expected_out", "sum"))
        .unstack("score_state")
        .fillna(0)
    )
    grouped.columns = [f"{stat}_{state}" for stat, state in grouped.columns]
    # a team can have no balls in play in a state, so make sure every column exists
    for stat in ["bip", "actual", "expected"]:
        for state in ["close", "ahead", "behind"]:
            if f"{stat}_{state}" not in grouped.columns:
                grouped[f"{stat}_{state}"] = 0
    out = grouped.reset_index()

    for s in ["close", "ahead", "behind"]:
        out[f"gap_{s}"] = (out[f"actual_{s}"] - out[f"expected_{s}"]) / out[f"bip_{s}"]

    out["deg_ahead_rate"] = out["gap_ahead"] - out["gap_close"]
    out["deg_behind_rate"] = out["gap_behind"] - out["gap_close"]
    out["deg_diff_rate"] = out["deg_behind_rate"] - out["deg_ahead_rate"]
    return out


def run(seasons=SEASONS):
    """Compute DEG for every season and save to data/processed/deg.csv."""
    results = []
    for year in seasons:
        print(f"Processing {year}...")
        raw = pd.read_parquet(RAW_DIR / f"statcast_{year}.parquet")
        raw["season"] = year
        bip = get_balls_in_play(raw)
        bip = add_score_state(bip)
        results.append(compute_deg(bip))

    deg = pd.concat(results, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    deg.to_csv(OUT_DIR / "deg.csv", index=False)
    print(f"Saved {len(deg)} team-seasons to {OUT_DIR / 'deg.csv'}")
    return deg


if __name__ == "__main__":
    run()
