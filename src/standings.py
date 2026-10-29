"""
Team standings from Retrosheet play-by-play, one row per team-season.

Adds a few columns used to compare the Hustle Index to winning:
    pythag_W            : wins expected from runs scored and allowed
                          W_exp = games * R^1.83 / (R^1.83 + RA^1.83)
    win_outperformance  : actual wins - pythag_W (positive = won more than the run totals say)
    one_run_pct         : win% in one-run games

Regular season only. A game's final score is the sum of the runs scored on each
play by that team's batters.

Input:  data/raw/retrosheet/YYYYplays.parquet  (pull_retrosheet.py)
Output: data/processed/standings.csv

Usage:
    python src/standings.py
"""

from pathlib import Path

import pandas as pd

from pull_retrosheet import SEASONS, pull_season
from team_codes import RETROSHEET_TO_STATCAST

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "standings.csv"
PYTHAG_EXPONENT = 1.83


def game_results(plays):
    """One row per game with each side's team and final runs."""
    plays = plays[plays["gametype"] == "regular"]
    visitors = plays[plays["vis_home"] == 0].groupby("gid").agg(vis_team=("batteam", "first"), vis_runs=("runs", "sum"))
    home = plays[plays["vis_home"] == 1].groupby("gid").agg(home_team=("batteam", "first"), home_runs=("runs", "sum"))
    return visitors.join(home).reset_index()


def team_games(games):
    """Turn each game into two rows, one from each team's point of view."""
    sides = [("vis_team", "vis_runs", "home_runs"), ("home_team", "home_runs", "vis_runs")]
    rows = []
    for team_col, scored, allowed in sides:
        rows.append(pd.DataFrame({
            "team": games[team_col].map(RETROSHEET_TO_STATCAST).fillna(games[team_col]),
            "runs_scored": games[scored],
            "runs_allowed": games[allowed],
        }))
    return pd.concat(rows, ignore_index=True)


def season_standings(plays, year):
    tg = team_games(game_results(plays))
    margin = tg["runs_scored"] - tg["runs_allowed"]
    tg["win"] = (margin > 0).astype(int)
    tg["loss"] = (margin < 0).astype(int)
    tg["one_run_win"] = tg["win"] * (margin.abs() == 1)
    tg["one_run_loss"] = tg["loss"] * (margin.abs() == 1)

    st = tg.groupby("team").agg(
        W=("win", "sum"), L=("loss", "sum"),
        R=("runs_scored", "sum"), RA=("runs_allowed", "sum"),
        one_run_W=("one_run_win", "sum"), one_run_L=("one_run_loss", "sum"),
    ).reset_index()

    st["season"] = year
    st["games"] = st["W"] + st["L"]
    st["win_pct"] = st["W"] / st["games"]
    st["run_diff"] = st["R"] - st["RA"]
    st["pythag_W"] = st["games"] * st["R"] ** PYTHAG_EXPONENT / (st["R"] ** PYTHAG_EXPONENT + st["RA"] ** PYTHAG_EXPONENT)
    st["win_outperformance"] = st["W"] - st["pythag_W"]
    st["one_run_games"] = st["one_run_W"] + st["one_run_L"]
    st["one_run_pct"] = st["one_run_W"] / st["one_run_games"]
    return st


def compute(seasons=SEASONS):
    return pd.concat([season_standings(pull_season(y), y) for y in seasons], ignore_index=True)


if __name__ == "__main__":
    standings = compute()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    standings.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(standings)} team-seasons to {OUT_PATH}")
