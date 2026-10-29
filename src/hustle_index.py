"""
Hustle Index

Combines hustle indicators into one score per team-season.

Each indicator is turned into a z-score within its season (0 = league average,
1 = one standard deviation better), then the z-scores are averaged using the
weights below. That average is z-scored again so the final index is also
"standard deviations from the league average" for the season.

To add an indicator:
    1. Save a CSV in data/processed/ with one row per team-season
       (a team column, a "season" column, and a score column).
    2. Add one entry to INDICATORS below.
    That's it. The z-scores, composite, leaderboard and correlations all pick it up.

Usage:
    python src/hustle_index.py          # leaderboard for the latest season
    python src/hustle_index.py 2023     # leaderboard for a specific season
"""

import sys

from pathlib import Path

import pandas as pd

from team_codes import RETROSHEET_TO_STATCAST

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# INDICATORS: add new ones here.
#   file      : CSV in data/processed/
#   team_col  : column with the team code (Retrosheet or Statcast codes both work)
#   score_col : column with the raw indicator value
#   weight    : relative weight in the composite (1.0 = same as the others)
#   higher_is_better : set to False if a LOWER raw value means more hustle
# ---------------------------------------------------------------------------
INDICATORS = {
    "extra_bases": {
        "file": "extra_bases.csv",
        "team_col": "team",
        "score_col": "raw_score",
        "weight": 1.0,
        "higher_is_better": True,
    },
    "deg_behind": {
        "file": "deg.csv",
        "team_col": "fielding_team",
        "score_col": "deg_behind_rate",
        "weight": 1.0,
        "higher_is_better": True,
    },
}


def zscore(x):
    """Standardize a column (mean 0, std 1). Returns 0 if every value is the same."""
    std = x.std(ddof=1)
    if std == 0 or pd.isna(std):
        return x * 0.0
    return (x - x.mean()) / std


def to_statcast_code(code):
    """Retrosheet and Statcast use different team codes (e.g. CHN vs CHC)."""
    return RETROSHEET_TO_STATCAST.get(code, code)


def load_indicator(name):
    """Load one indicator and add its within-season z-score."""
    indicator = INDICATORS[name]
    df = pd.read_csv(DATA_DIR / indicator["file"])

    df = df.rename(columns={indicator["team_col"]: "team", indicator["score_col"]: f"{name}_raw"})
    df["team"] = df["team"].map(to_statcast_code)
    df[f"{name}_z"] = df.groupby("season")[f"{name}_raw"].transform(zscore)
    if not indicator.get("higher_is_better", True):
        df[f"{name}_z"] = -df[f"{name}_z"]
    return df[["team", "season", f"{name}_raw", f"{name}_z"]]


def build_hustle_index(indicators=None, weights=None):
    """Weighted average of the indicator z-scores, re-standardized within each season."""
    indicators = indicators or list(INDICATORS)
    weights = weights or {n: INDICATORS[n]["weight"] for n in indicators}

    merged = load_indicator(indicators[0])
    for name in indicators[1:]:
        merged = merged.merge(load_indicator(name), on=["team", "season"], how="inner")

    total = sum(weights[n] for n in indicators)
    merged["composite"] = sum(weights[n] * merged[f"{n}_z"] for n in indicators) / total
    merged["hustle_z"] = merged.groupby("season")["composite"].transform(zscore)
    merged["hustle_index"] = merged["hustle_z"].round(2)
    merged["season_rank"] = merged.groupby("season")["hustle_z"].rank(ascending=False, method="min").astype(int)

    # add wins and losses so the index can be compared to winning
    standings = pd.read_csv(DATA_DIR / "standings.csv")
    standings["team"] = standings["team"].map(to_statcast_code)
    merged = merged.merge(standings, on=["team", "season"], how="left")

    return merged.drop(columns="composite").sort_values(["season", "season_rank"]).reset_index(drop=True)


def correlate_with_winning(df):
    """Correlation of each score with a few team outcomes."""
    outcomes = ["win_pct", "run_diff", "win_outperformance", "one_run_pct"]
    scores = ["hustle_z"] + [c for c in df.columns if c.endswith("_z") and c != "hustle_z"]
    return df[scores + outcomes].corr().loc[scores, outcomes].round(3)


def print_leaderboard(df, season):
    """Print the hustle ranking for one season, with a column for every indicator."""
    s = df[df["season"] == season]
    names = [c[:-2] for c in df.columns if c.endswith("_z") and c != "hustle_z"]

    print(f"\n{season} Hustle Index (z-score, 0 = league average)")
    print(f"{'Rank':<6}{'Team':<7}{'Hustle':>8}" + "".join(f"{n:>14}" for n in names) + f"{'W-L':>9}")
    for _, r in s.iterrows():
        cols = "".join(f"{r[n + '_z']:>+14.2f}" for n in names)
        print(f"{r['season_rank']:<6}{r['team']:<7}{r['hustle_z']:>+8.2f}{cols}{int(r['W']):>5}-{int(r['L'])}")


if __name__ == "__main__":
    index = build_hustle_index()
    index.to_csv(DATA_DIR / "hustle_index.csv", index=False)
    print(f"Saved {len(index)} team-seasons to {DATA_DIR / 'hustle_index.csv'}")

    season = int(sys.argv[1]) if len(sys.argv) > 1 else int(index["season"].max())
    print_leaderboard(index, season)
    print("\nCorrelation with winning:")
    print(correlate_with_winning(index))
