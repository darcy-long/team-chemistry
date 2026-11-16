"""Tests for src/models.py using small made-up data."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import aggregate_team_season, compute_residuals, compute_shrunk_expected_rate


def make_opps():
    return pd.DataFrame({
        "team": ["A"] * 4 + ["B"] * 4,
        "season": [2023] * 8,
        "situation": ["x", "x", "y", "y", "x", "x", "y", "y"],
        "took_it": [1, 1, 0, 0, 0, 0, 1, 0],
    })


def test_shrinkage_pulls_toward_league_average():
    out = compute_shrunk_expected_rate(make_opps(), "took_it", ["situation"], m=20)
    league = make_opps()["took_it"].mean()
    # situation x has a raw rate of 0.5 but only 4 chances, so it should sit close to the league rate
    x_rate = out.loc[out["situation"] == "x", "expected_rate"].iloc[0]
    assert abs(x_rate - league) < abs(0.5 - league) or x_rate == league


def test_residuals_sum_to_zero_with_no_shrinkage():
    out = compute_shrunk_expected_rate(make_opps(), "took_it", ["situation"], m=0)
    out = compute_residuals(out, "took_it")
    assert abs(out["residual"].sum()) < 1e-12


def test_small_team_seasons_are_dropped():
    out = compute_shrunk_expected_rate(make_opps(), "took_it", ["situation"])
    out = compute_residuals(out, "took_it")
    assert aggregate_team_season(out, "team", "season", min_opportunities=5).empty
    assert len(aggregate_team_season(out, "team", "season", min_opportunities=4)) == 2


def test_league_group_col_shrinks_toward_each_groups_own_rate():
    # group A goes 20 of 21 times, group B 1 of 21; each has one single-play "small" situation
    df = pd.DataFrame({
        "g": ["A"] * 21 + ["B"] * 21,
        "s": ["big"] * 20 + ["small"] + ["big"] * 20 + ["small"],
        "y": [1] * 20 + [0] + [0] * 20 + [1],
    })
    pooled = compute_shrunk_expected_rate(df, "y", ["g", "s"], m=20)
    grouped = compute_shrunk_expected_rate(df, "y", ["g", "s"], m=20, league_group_col="g")

    def small_a(d):
        return d[(d.g == "A") & (d.s == "small")]["expected_rate"].iloc[0]

    # one play that stayed, pulled toward the pooled rate (21/42) or A's own rate (20/21)
    assert small_a(pooled) == pytest.approx((0 + 20 * 0.5) / 21)
    assert small_a(grouped) == pytest.approx((0 + 20 * (20 / 21)) / 21)
