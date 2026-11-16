"""Tests for src/extra_bases_stat/smooth_model.py (needs scikit-learn: run with the repo's .venv)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "extra_bases_stat"))

from smooth_model import FEATURE_GROUPS, bucket_odds, feature_columns, make_features, smooth_odds, three_way_log_loss


def synthetic(n=6000, seed=0):
    """One kind of chance whose odds of going rise smoothly with throw distance, over 3 seasons."""
    rng = np.random.default_rng(seed)
    dist = rng.uniform(100, 300, n)
    p_go = 1 / (1 + np.exp(-(dist - 200) / 25))
    went = rng.random(n) < p_go
    out = went & (rng.random(n) < 0.1)
    return pd.DataFrame({
        "opportunity_type": "2nd_on_single", "season": rng.choice([2016, 2017, 2018], n),
        "throw_dist": dist, "outs_pre": rng.integers(0, 3, n), "hittype": rng.choice(["G", "L", "F"], n),
        "took_extra_base": went.astype(int), "thrown_out": out.astype(int),
        "throw_dist_bin": pd.qcut(dist, 3, labels=False),
    }), p_go


def test_smooth_odds_are_valid_probabilities_and_add_up():
    opps, _ = synthetic()
    odds = smooth_odds(opps, groups=("base",), features=opps[feature_columns(("base",))])
    assert odds.notna().all().all()
    assert odds["p_attempt"].between(0, 1).all() and odds["p_success"].between(0, 1).all()
    assert (odds["p_safe"] + odds["p_out"]).between(0, 1).all()
    assert np.allclose(odds["p_safe"] + odds["p_out"], odds["p_attempt"])


def test_smooth_beats_three_buckets_on_a_smooth_truth_and_follows_it():
    opps, p_go = synthetic()
    smooth = smooth_odds(opps, groups=("base",), features=opps[feature_columns(("base",))])
    buckets = bucket_odds(opps, ["hittype", "outs_pre", "opportunity_type", "throw_dist_bin"])
    assert (three_way_log_loss(opps, smooth["p_safe"], smooth["p_out"])
            < three_way_log_loss(opps, buckets["p_safe"], buckets["p_out"]))
    assert np.corrcoef(smooth["p_attempt"], p_go)[0, 1] > 0.95


def test_odds_never_fall_as_the_throw_gets_longer():
    opps, _ = synthetic()
    odds = smooth_odds(opps, groups=("base",), features=opps[feature_columns(("base",))])
    order = np.argsort(opps["throw_dist"].to_numpy())
    # the monotonic constraint holds within a fixed set of other features; on average the trend must rise
    smoothed = pd.Series(odds["p_attempt"].to_numpy()[order]).rolling(500).mean().dropna().to_numpy()
    assert (np.diff(smoothed) > -0.01).all()


def test_each_season_is_predicted_without_its_own_data():
    # a chance's odds must not depend on its own result: flipping the outcomes of one season leaves that season's
    # odds exactly the same, because they come from the other seasons
    opps, _ = synthetic()
    cols = opps[feature_columns(("base",))]
    first = smooth_odds(opps, groups=("base",), features=cols)
    flipped = opps.copy()
    mask = flipped["season"] == 2017
    flipped.loc[mask, "took_extra_base"] = 1 - flipped.loc[mask, "took_extra_base"]
    flipped.loc[mask, "thrown_out"] = 0
    second = smooth_odds(flipped, groups=("base",), features=cols)
    assert np.allclose(first.loc[mask, "p_attempt"], second.loc[mask, "p_attempt"])


def test_bucket_odds_use_only_other_seasons_and_fall_back_to_the_kind_rate():
    opps = pd.DataFrame({
        "opportunity_type": "k", "season": [1, 1, 2, 2, 3],
        "b": ["x", "x", "x", "y", "z"],   # season 3 is the only one with bucket z
        "took_extra_base": [1, 1, 0, 1, 1], "thrown_out": [0, 0, 0, 0, 0],
    })
    odds = bucket_odds(opps, ["b"])
    assert odds["p_attempt"].iloc[4] == pytest.approx(0.75)   # z unseen elsewhere: kind rate of seasons 1-2 (3 of 4)
    assert odds["p_attempt"].iloc[2] == pytest.approx(1.0)    # season 2, bucket x: only season 1 and 3 count -> 2 of 2


def test_three_way_log_loss_is_the_log_of_the_probability_of_what_happened():
    opps = pd.DataFrame({"took_extra_base": [1, 1, 0], "thrown_out": [0, 1, 0]})   # safe, out, held
    loss = three_way_log_loss(opps, [0.5, 0.2, 0.3], [0.1, 0.25, 0.2])
    assert loss == pytest.approx(-(np.log(0.5) + np.log(0.25) + np.log(0.5)) / 3)


def test_make_features_has_every_group_column():
    opps = pd.DataFrame({
        "sc_hc_x": [125.42, 100.0], "sc_hc_y": [198.27 - 100, 120.0], "throw_dist": [90.0, 200.0], "outs_pre": [0, 2],
        "hittype": ["G", None], "fielder_pos": ["CF", None], "sc_launch_speed": [95.0, np.nan],
        "sc_launch_angle": [10.0, np.nan], "hang_time": [3.0, np.nan], "sc_hit_distance_sc": [250, pd.NA],
        "br1_pre": ["a", None], "br2_pre": [None, None], "br3_pre": [None, "c"], "bathand": ["L", "R"],
        "season": [2019, 2021], "site": ["BOS07", "NYC16"],
    })
    f = make_features(opps)
    assert set(feature_columns(FEATURE_GROUPS)) <= set(f.columns)
    assert f["bat_left"].tolist() == [1.0, 0.0] and f["on1"].tolist() == [1.0, 0.0] and f["on3"].tolist() == [0.0, 1.0]
    assert f["angle"].iloc[0] == pytest.approx(0)       # straight up the middle
    assert f["home_dist"].iloc[0] == pytest.approx(250)  # 100 pixels at 2.5 ft each
    assert np.isnan(f["hit_distance"].iloc[1])
