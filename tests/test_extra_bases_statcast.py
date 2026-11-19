"""Tests for src/extra_bases_stat/ (throw-distance geometry, fielding position, hang time, and the six opportunity types)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "extra_bases_stat"))

from extra_bases_statcast import add_batted_ball_location, aggregate_by_type


def make(opp_type, hc_x, hc_y):
    return pd.DataFrame({
        "opportunity_type": [opp_type] * 2, "sc_status": ["ok"] * 2,
        "sc_hc_x": [hc_x, hc_x], "sc_hc_y": [hc_y, hc_y],
    })


def test_ball_at_home_plate_is_zero_feet_from_home():
    out = add_batted_ball_location(make("2nd_on_single", 125.42, 198.27), n_bins=1)
    assert out["throw_dist"].iloc[0] == pytest.approx(0)


def test_ball_straight_back_300_feet_is_300_feet_from_home():
    # 120 pixels toward center field = 300 feet
    out = add_batted_ball_location(make("1st_on_double", 125.42, 198.27 - 120), n_bins=1)
    assert out["throw_dist"].iloc[0] == pytest.approx(300)
    assert out["spray"].iloc[0] == "center"


def test_throw_target_for_first_on_single_is_third_base():
    # a ball fielded right at third base (63.64 ft toward left field and out) is 0 ft from the throw
    px = 63.64 / 2.5
    out = add_batted_ball_location(make("1st_on_single", 125.42 - px, 198.27 - px), n_bins=1)
    assert out["throw_dist"].iloc[0] == pytest.approx(0, abs=0.1)
    assert out["spray"].iloc[0] == "left"


def test_rows_without_a_statcast_match_are_dropped():
    df = make("2nd_on_single", 125.42, 150)
    df.loc[0, "sc_status"] = "mismatch"
    assert len(add_batted_ball_location(df, n_bins=1)) == 1


def test_hang_time_orders_popups_above_line_drives_and_flags_low_balls():
    from extra_bases_statcast import add_hang_time
    df = pd.DataFrame({
        "sc_launch_speed": [95.0, 70.0, 90.0, 85.0, None],
        "sc_launch_angle": [15.0, 60.0, 35.0, 3.0, 20.0],  # line drive, popup, fly, grounder, no data
    })
    out = add_hang_time(df, n_bins=3)
    assert out["hang_time"].iloc[1] > out["hang_time"].iloc[2] > out["hang_time"].iloc[0]
    assert out["hang_bin"].tolist()[3:] == [-1, -2]  # low ball, then missing launch data
    assert out["hang_bin"].iloc[1] > out["hang_bin"].iloc[0]


def test_throw_targets_for_new_types():
    # batter_for_2: a ball fielded at second base is 0 ft from the throw; sac fly: ball at home is 0 ft
    out = add_batted_ball_location(make("batter_for_2", 125.42, 198.27 - (127.28 / 2.5)), n_bins=1)
    assert out["throw_dist"].iloc[0] == pytest.approx(0, abs=0.1)
    out = add_batted_ball_location(make("3rd_on_fly", 125.42, 198.27), n_bins=1)
    assert out["throw_dist"].iloc[0] == pytest.approx(0)


def test_fielder_position_from_loc():
    from extra_bases_statcast import add_fielder_position
    df = pd.DataFrame({"loc": ["7D", "78XD+", "8S", "89", "9L", "56", "34+", "6M", None]})
    assert add_fielder_position(df)["fielder_pos"].tolist() == \
        ["LF", "LC", "CF", "RC", "RF", "IF", "IF", "IF", "Unknown"]


def test_outcome_rewards_attempt_or_result():
    from extra_bases_statcast import add_outcome
    df = pd.DataFrame({
        "took_extra_base": [1, 1, 0],   # safe, thrown out, held
        "thrown_out": [0, 1, 0],
    })
    assert add_outcome(df, reward_hustle=True)["outcome"].tolist() == [1, 1, 0]
    assert add_outcome(df, reward_hustle=False)["outcome"].tolist() == [1, -1, 0]
    assert add_outcome(df, reward_hustle=False, out_penalty=2.5)["outcome"].tolist() == [1, -2.5, 0]


def test_runner_totals_sum_residuals_and_split_by_kind(monkeypatch):
    import extra_bases_statcast as eb
    monkeypatch.setattr(eb, "runner_names", lambda: pd.Series({"a01": "Pat Runner", "b02": "Sam Slow"}))
    opps = pd.DataFrame({
        "runner_id": ["a01", "a01", "a01", "b02"],
        "key_mlbam": [1, 1, 1, 2],
        "season": [2023, 2023, 2023, 2023],
        "team": ["BOS", "BOS", "NYY", "NYY"],
        "opportunity_type": ["1st_on_single", "1st_on_single", "3rd_on_fly", "1st_on_single"],
        "took_extra_base": [1, 1, 1, 0],
        "thrown_out": [0, 1, 0, 0],
        "outcome": [1.0, -1.0, 1.0, 0.0],
        "expected_rate": [0.5, 0.5, 0.8, 0.4],
        "residual": [0.5, -1.5, 0.2, -0.4],
    })
    out = eb.aggregate_runners(opps).set_index("name")
    assert out.loc["Pat Runner", "net_bases_above_expected"] == pytest.approx(-0.8)
    assert out.loc["Pat Runner", "n_opportunities"] == 3
    assert (out.loc["Pat Runner", "n_attempts"], out.loc["Pat Runner", "n_thrown_out"],
            out.loc["Pat Runner", "n_safe"]) == (3, 1, 2)
    assert out.loc["Pat Runner", "team"] == "BOS"  # most chances
    assert out.loc["Pat Runner", "above_1st_on_single"] == pytest.approx(-1.0)
    assert out.loc["Pat Runner", "above_3rd_on_fly"] == pytest.approx(0.2)
    pat = out.loc["Pat Runner"]
    assert pat["actual"] - pat["expected"] == pytest.approx(pat["net_bases_above_expected"])
    career = eb.aggregate_runners(opps, by_season=False)
    assert "season" not in career.columns and len(career) == 2


def test_regular_season_only_drops_playoff_games():
    from extra_bases_statcast import regular_season_only
    df = pd.DataFrame({"gametype": ["regular", "wildcard", "divisionseries", "lcs", "worldseries", "regular"],
                       "x": range(6)})
    assert regular_season_only(df)["x"].tolist() == [0, 5]


def test_drop_kinds_leaves_out_whole_kinds():
    from extra_bases_statcast import DEFAULT_EXCLUDED_KINDS, drop_kinds
    df = pd.DataFrame({"opportunity_type": ["batter_for_3", "batter_for_2", "1st_on_single", "batter_for_3"], "x": range(4)})
    assert drop_kinds(df, ("batter_for_3",))["x"].tolist() == [1, 2]
    assert drop_kinds(df, DEFAULT_EXCLUDED_KINDS)["x"].tolist() == [0, 1, 2, 3]  # all kinds are counted by default


def test_hit_coordinates_are_not_mirrored():
    """Left of the field must be negative x (third base is at -x in TARGET_BASE_XY). Distance from home would
    look the same if x were flipped, so check the side directly with Retrosheet's fielding zone."""
    import pytest
    from extra_bases_statcast import FEET_PER_PIXEL, HOME_X, PLAYS_DIR
    path = PLAYS_DIR / "2023.parquet"
    if not path.exists():
        pytest.skip("plays_statcast/2023.parquet not built")
    d = pd.read_parquet(path, columns=["loc", "sc_hc_x", "sc_status"])
    d = d[d["sc_status"] == "ok"].dropna(subset=["loc", "sc_hc_x"])
    d["x"] = (d["sc_hc_x"] - HOME_X) * FEET_PER_PIXEL
    zone = d["loc"].str.extract(r"^(\d+)", expand=False)
    assert d.loc[zone == "7", "x"].mean() < -100      # left field
    assert d.loc[zone == "9", "x"].mean() > 100       # right field
    assert d.loc[zone == "5", "x"].mean() < 0         # third baseman
    assert d.loc[zone == "3", "x"].mean() > 0         # first baseman


def test_held_out_expected_falls_back_to_the_kind_average_for_unseen_situations():
    from extra_bases_statcast import leave_one_season_out_expected
    df = pd.DataFrame({
        "season":           [2020, 2020, 2020, 2021, 2021, 2021],
        "opportunity_type": ["a",  "a",  "b",  "a",  "a",  "b"],
        "hittype":          ["G",  "G",  "G",  "G",  "L",  "G"],   # 2021's (a, L) play has no match in 2020
        "took_extra_base":  [1,    0,    1,    1,    1,    0],
    }, index=[100, 101, 102, 5, 6, 7])                              # a non-default index, as in real use
    out = leave_one_season_out_expected(df, ["opportunity_type", "hittype"], m=0)
    assert not out.isna().any()
    assert out.loc[5] == 0.5     # (a, G) in 2020: one go, one hold
    assert out.loc[6] == 0.5     # unseen (a, L): falls back to kind a's 2020 average
    assert out.loc[7] == 1.0     # (b, G) in 2020
    assert out.loc[[100, 101]].tolist() == [1.0, 1.0] and out.loc[102] == 0.0   # 2020 predicted from 2021


def test_team_total_is_the_sum_of_residuals_not_the_raw_score_times_chances():
    # one team: 10 batter_for_2 chances averaging -0.1, 2 3rd_on_fly chances averaging +0.9
    rows = [("BOS", 2024, "batter_for_2", -0.1)] * 10 + [("BOS", 2024, "3rd_on_fly", 0.9)] * 2
    opps = pd.DataFrame(rows, columns=["team", "season", "opportunity_type", "residual"])
    out = aggregate_by_type(opps, min_opportunities=1).iloc[0]
    assert out["net_bases_above_expected"] == pytest.approx(-1.0 + 1.8)
    assert out["per_chance"] == pytest.approx(0.8 / 12)
    assert out["raw_score"] == pytest.approx((-0.1 + 0.9) / 2)    # kinds weigh the same: not the per-chance rate
    runs = aggregate_by_type(opps, min_opportunities=1, runs=True).iloc[0]
    assert runs["runs_above_expected"] == pytest.approx(0.8)
