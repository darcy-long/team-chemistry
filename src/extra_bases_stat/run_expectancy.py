"""
Run expectancy (RE24): how many runs a team is expected to score from a base-out state to the
end of the half-inning, built separately for each season from every Retrosheet play.

    base_state = 1 if a runner is on 1st + 2 if on 2nd + 4 if on 3rd      (0 = bases empty ... 7 = loaded)
    RE(season, base_state, outs) = average runs scored from that play to the end of the half-inning

Rules:
    - Half-innings that did not finish (walk-offs, rain) are left out: their runs-to-end are cut short.
    - "NP" rows (substitutions, no play) are left out: nothing happens on them.
    - With 3 outs the expectancy is 0 (lookup_re handles that).

The run value of a play is RE(after) + runs scored on the play - RE(before). Over a season those add
up to about zero, which is a check that the table and the play data agree.

Output: data/processed/extra_bases_stat/run_expectancy.csv  (season, base_state, outs, re, n)

Usage (from the repo root):
    python src/extra_bases_stat/run_expectancy.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for the shared modules

from pull_retrosheet import SEASONS

REPO = Path(__file__).resolve().parents[2]
PLAYS_DIR = REPO / "data" / "processed" / "plays_statcast"
OUT_DIR = REPO / "data" / "processed" / "extra_bases_stat"
RE_PATH = OUT_DIR / "run_expectancy.csv"

PLAY_COLUMNS = [
    "gid", "event", "inning", "top_bot", "outs_pre", "outs_post",
    "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post", "br3_post", "runs",
]


def base_state(on1, on2, on3):
    """0-7 from three True/False arrays (or Series): 1st = 1, 2nd = 2, 3rd = 4."""
    return (np.asarray(on1, dtype=int) + 2 * np.asarray(on2, dtype=int) + 4 * np.asarray(on3, dtype=int))


def state_from_runner_columns(df, suffix):
    """Base state from the br1/br2/br3 runner-ID columns ('pre' or 'post')."""
    return base_state(df[f"br1_{suffix}"].notna(), df[f"br2_{suffix}"].notna(), df[f"br3_{suffix}"].notna())


def load_all_plays():
    frames = []
    for year in SEASONS:
        df = pd.read_parquet(PLAYS_DIR / f"{year}.parquet", columns=PLAY_COLUMNS)
        df["season"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_re_table(plays):
    """Per-season RE24. plays needs season, gid, inning, top_bot, event, outs_pre, outs_post, runs and the
    br1/2/3_pre columns, in the order the plays happened. Returns season, base_state, outs, re, n."""
    p = plays[plays["event"] != "NP"].reset_index(drop=True)
    half = p.groupby(["season", "gid", "inning", "top_bot"], sort=False)
    # runs from this play to the end of the half-inning, this play included
    p["runs_to_end"] = half["runs"].transform("sum") - half["runs"].cumsum() + p["runs"]
    finished = half["outs_post"].transform("last") == 3
    p = p[finished & (p["outs_pre"] < 3)].copy()
    p["base_state"] = state_from_runner_columns(p, "pre")
    table = (p.groupby(["season", "base_state", "outs_pre"])["runs_to_end"]
             .agg(re="mean", n="count").reset_index().rename(columns={"outs_pre": "outs"}))
    return table


def lookup_re(table, season, state, outs):
    """RE for arrays of season, base state and outs (3 or more outs = 0). NaN where the table has no cell."""
    idx = pd.MultiIndex.from_frame(table[["season", "base_state", "outs"]])
    key = pd.MultiIndex.from_arrays([np.asarray(season), np.asarray(state), np.minimum(np.asarray(outs), 2)])
    pos = idx.get_indexer(key)
    values = np.where(pos >= 0, table["re"].to_numpy()[np.maximum(pos, 0)], np.nan)
    return np.where(np.asarray(outs) >= 3, 0.0, values)


def add_play_run_value(plays, table):
    """Add run_value = RE(after) + runs - RE(before) to every play."""
    plays = plays.copy()
    before = lookup_re(table, plays["season"], state_from_runner_columns(plays, "pre"), plays["outs_pre"])
    after = lookup_re(table, plays["season"], state_from_runner_columns(plays, "post"), plays["outs_post"])
    plays["run_value"] = after + plays["runs"] - before
    return plays


def get_re_table(rebuild=False):
    """The saved table, building (and saving) it from the plays if it is missing."""
    if RE_PATH.exists() and not rebuild:
        return pd.read_csv(RE_PATH)
    table = build_re_table(load_all_plays())
    RE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RE_PATH, index=False)
    return table


if __name__ == "__main__":
    plays = load_all_plays()
    table = build_re_table(plays)
    RE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RE_PATH, index=False)
    print(f"Saved {len(table)} cells to {RE_PATH}\n")

    empty0 = table[(table["base_state"] == 0) & (table["outs"] == 0)].set_index("season")["re"]
    print("RE, bases empty, 0 outs, by season:")
    print(empty0.round(3).to_string(), "\n")

    # Check: play run values add up to about zero within each season (finished innings only)
    valued = add_play_run_value(plays[(plays["event"] != "NP") & (plays["outs_pre"] < 3)], table)
    print("Mean play run value by season (should be about 0):")
    print(valued.groupby("season")["run_value"].mean().round(5).to_string())
