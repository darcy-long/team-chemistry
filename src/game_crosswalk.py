"""
Game crosswalk: Retrosheet game ID (gid) <-> Statcast game_pk.

Retrosheet gid looks like BOS202303300 (home team + date + game number, 0 unless
a doubleheader). Statcast uses game_pk. Games are matched on date + home team +
away team. For doubleheaders, the final score is used to tell the two games
apart. If both games share the identical final score, the home starting pitcher
breaks the tie (each game has a different starter; game_pk order is not reliable).

Statcast spring training / exhibition games have no Retrosheet match and are dropped.

Output: data/processed/game_crosswalk.csv
        one row per Retrosheet game: season, gid, game_pk, date, home, away (Statcast
        team codes), home/away final score, match_method

Usage:
    python src/game_crosswalk.py
"""

from pathlib import Path

import pandas as pd

from player_crosswalk import get_crosswalk
from pull_retrosheet import SEASONS, pull_season
from team_codes import to_statcast_code

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def retrosheet_games(year):
    """One row per Retrosheet game with Statcast-style team codes, final score, and home starter."""
    plays = pull_season(year)
    plays = plays.assign(
        visruns=plays["runs"].where(plays["top_bot"] == 0, 0),
        homeruns=plays["runs"].where(plays["top_bot"] == 1, 0),
    )
    games = plays.groupby("gid").agg(
        date=("date", "first"),
        gametype=("gametype", "first"),
        away_score=("visruns", "sum"),
        home_score=("homeruns", "sum"),
    )
    # home team = the team batting in the bottom half; away = batting in the top
    home = plays[plays["top_bot"] == 1].groupby("gid")["batteam"].first()
    away = plays[plays["top_bot"] == 0].groupby("gid")["batteam"].first()
    games["home"] = home.map(to_statcast_code)
    games["away"] = away.map(to_statcast_code)

    # Home starting pitcher (pitching in top of 1st) in MLBAM ID format
    players = get_crosswalk().set_index("key_retro")["key_mlbam"]
    games["home_sp"] = plays[plays["top_bot"] == 0].groupby("gid")["pitcher"].first().map(players)

    games = games.reset_index()
    games["date"] = pd.to_datetime(games["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
    games["game_num"] = games["gid"].str[-1].astype(int)
    return games


def statcast_games(year):
    """One row per Statcast regular/postseason game with final score and home starter."""
    sc = pd.read_parquet(
        RAW_DIR / f"statcast_{year}.parquet",
        columns=[
            "game_pk", "game_date", "game_type", "home_team", "away_team",
            "post_home_score", "post_away_score", "inning", "inning_topbot",
            "pitcher", "at_bat_number", "pitch_number",
        ],
    )
    sc = sc[~sc["game_type"].isin(["S", "E"])]  # drop spring training / exhibition

    # Home starting pitcher (pitching in top of 1st)
    inn1 = sc[sc["inning"] == 1].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    h_sp = inn1[inn1["inning_topbot"] == "Top"].groupby("game_pk")["pitcher"].first()

    games = sc.groupby("game_pk").agg(
        date=("game_date", "first"),
        home=("home_team", "first"),
        away=("away_team", "first"),
        home_score=("post_home_score", "max"),
        away_score=("post_away_score", "max"),
    )
    games["home_sp"] = h_sp
    games = games.reset_index()
    games["date"] = pd.to_datetime(games["date"]).dt.strftime("%Y-%m-%d")
    return games


def match_season(year):
    ret = retrosheet_games(year)
    sc = statcast_games(year)
    keys = ["date", "home", "away"]

    ret["n_ret"] = ret.groupby(keys)["gid"].transform("size")
    sc["n_sc"] = sc.groupby(keys)["game_pk"].transform("size")

    # Single game that day for these teams on both sides: direct match
    single = ret[ret["n_ret"] == 1].merge(sc[sc["n_sc"] == 1], on=keys, suffixes=("", "_sc"))
    single["match_method"] = "date_teams"

    # Doubleheaders (or any day with more than one game): match on final score first.
    # When final scores don't separate them (same score in both games), match on
    # starting pitcher, which uniquely identifies each game.
    ret_multi = ret[ret["n_ret"] > 1].copy()
    sc_multi = sc[sc["n_sc"] > 1].copy()
    score_keys = keys + ["home_score", "away_score"]
    ret_multi["score_dups"] = ret_multi.groupby(score_keys)["gid"].transform("size")
    sc_multi["score_dups"] = sc_multi.groupby(score_keys)["game_pk"].transform("size")
    by_score = ret_multi[ret_multi["score_dups"] == 1].merge(
        sc_multi[sc_multi["score_dups"] == 1], on=score_keys, suffixes=("", "_sc")
    )
    by_score["match_method"] = "doubleheader_score"

    rest_ret = ret_multi[~ret_multi["gid"].isin(by_score["gid"])].copy()
    rest_sc = sc_multi[~sc_multi["game_pk"].isin(by_score["game_pk"])].copy()
    by_sp = rest_ret.merge(rest_sc, on=keys + ["home_sp"], suffixes=("", "_sc"))
    by_sp["match_method"] = "doubleheader_starter"

    # Fall back to game_num / game_pk order only if starter match didn't resolve it
    rest_ret2 = rest_ret[~rest_ret["gid"].isin(by_sp["gid"])].copy()
    rest_sc2 = rest_sc[~rest_sc["game_pk"].isin(by_sp["game_pk"])].copy()
    if len(rest_ret2) > 0:
        rest_ret2["order"] = rest_ret2.groupby(keys)["game_num"].rank(method="first")
        rest_sc2["order"] = rest_sc2.groupby(keys)["game_pk"].rank(method="first")
        by_order = rest_ret2.merge(rest_sc2, on=keys + ["order"], suffixes=("", "_sc"))
        by_order["match_method"] = "doubleheader_order"
        multi = pd.concat([by_score, by_sp, by_order], ignore_index=True)
    else:
        multi = pd.concat([by_score, by_sp], ignore_index=True)

    # score-matched rows share their score by construction, so fill Statcast's columns
    multi["home_score_sc"] = multi["home_score_sc"].fillna(multi["home_score"])
    multi["away_score_sc"] = multi["away_score_sc"].fillna(multi["away_score"])

    out = pd.concat([single, multi], ignore_index=True)
    out["season"] = year
    out = out.rename(columns={"home_score_sc": "sc_home_score", "away_score_sc": "sc_away_score"})
    return ret, sc, out[
        ["season", "gid", "game_pk", "date", "home", "away", "home_score", "away_score",
         "sc_home_score", "sc_away_score", "match_method"]
    ]


def build_crosswalk():
    frames = []
    for year in SEASONS:
        ret, sc, matched = match_season(year)
        n_unmatched_ret = len(ret) - len(matched)
        matched_pks = set(matched["game_pk"])
        n_unmatched_sc = (~sc["game_pk"].isin(matched_pks)).sum()
        mism = (matched["home_score"] != matched["sc_home_score"]) | (matched["away_score"] != matched["sc_away_score"])
        print(f"{year}: {len(matched)} matched | unmatched Retrosheet {n_unmatched_ret}, "
              f"unmatched Statcast {n_unmatched_sc} | score mismatches {mism.sum()}")
        frames.append(matched)
    return pd.concat(frames, ignore_index=True)


def get_game_crosswalk():
    """Load the saved crosswalk, building it if needed."""
    path = OUT_DIR / "game_crosswalk.csv"
    if path.exists():
        return pd.read_csv(path)
    df = build_crosswalk()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = build_crosswalk()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "game_crosswalk.csv", index=False)
    print(f"\nDone. Saved {len(df):,} games to {OUT_DIR / 'game_crosswalk.csv'}")
    print("How they were matched:")
    print(df["match_method"].value_counts().to_string())
    n_bad = int(((df["home_score"] != df["sc_home_score"]) | (df["away_score"] != df["sc_away_score"])).sum())
    print(f"Games where the two sources' final scores differ: {n_bad}")
