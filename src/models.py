"""
Expected-rate model for the Extra Bases Taken indicator.

For each baserunning opportunity, the runner either took the extra base (1) or
didn't (0). We ask: how often does a runner take it in that same situation
(say, runner on 1st, single, 2 outs)? A team's score is how much more (or less)
often its runners took the extra base than expected.

Small situations are noisy, so each situation's rate is pulled toward the league
average ("shrinkage"):
    expected = (n * situation_rate + m * league_rate) / (n + m)
n is the number of opportunities in the situation and m is the shrinkage
strength. Then:
    residual   = took_it - expected                 (per opportunity)
    team score = mean residual per team-season

Steps, in order: compute_shrunk_expected_rate -> compute_residuals -> aggregate_team_season
"""

import pandas as pd


def compute_shrunk_expected_rate(df, outcome_col, bucket_cols, m=20.0):
    """Add an expected_rate column: the shrunk rate for each row's situation.

    outcome_col : 0/1 column (1 = took the extra base)
    bucket_cols : columns that define a situation, e.g. ["opportunity_type", "outs_pre"]
    m           : higher = trust the league average more, lower = trust the situation more
                  (20 is a starting point, check it against your bucket sizes)
    """
    league_rate = df[outcome_col].mean()

    buckets = df.groupby(bucket_cols)[outcome_col].agg(n="count", rate="mean").reset_index()
    buckets["expected_rate"] = (buckets["n"] * buckets["rate"] + m * league_rate) / (buckets["n"] + m)

    return df.merge(buckets[bucket_cols + ["expected_rate"]], on=bucket_cols, how="left")


def compute_residuals(df, outcome_col, expected_col="expected_rate"):
    """Add a residual column: what happened minus what was expected."""
    df = df.copy()
    df["residual"] = df[outcome_col] - df[expected_col]
    return df


def aggregate_team_season(df, team_col, season_col, min_opportunities=50):
    """Average the residuals into one raw_score per team-season.
    Team-seasons with fewer than min_opportunities chances are dropped (too noisy)."""
    agg = (
        df.groupby([team_col, season_col])["residual"]
        .agg(raw_score="mean", n_opportunities="count")
        .reset_index()
    )
    dropped = agg[agg["n_opportunities"] < min_opportunities]
    if len(dropped):
        print(f"Dropping {len(dropped)} team-seasons with under {min_opportunities} opportunities")
    return agg[agg["n_opportunities"] >= min_opportunities].reset_index(drop=True)
