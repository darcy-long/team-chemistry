"""Tests for src/deg.py using small made-up data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from deg import add_score_state, compute_deg, get_balls_in_play


def make_raw():
    return pd.DataFrame({
        "game_type": ["R"] * 6,
        "inning": [8, 8, 8, 8, 3, 9],
        "inning_topbot": ["Top"] * 6,
        "home_team": ["NYY"] * 6,
        "away_team": ["BOS"] * 6,
        "fld_score": [3, 9, 0, 4, 3, 1],
        "bat_score": [2, 2, 8, 4, 2, 1],
        "events": ["field_out", "single", "home_run", "sac_bunt", "field_out", "sac_fly"],
        "des": ["groundout", "single", "homer", "sacrifice bunt", "groundout", "sac fly"],
        "estimated_ba_using_speedangle": [0.2, 0.6, 0.9, 0.1, 0.3, np.nan],
        "launch_speed": [90.0, 100.0, 105.0, 40.0, 85.0, 95.0],
        "launch_angle": [10.0, 12.0, 30.0, -10.0, 5.0, 28.0],
        "bb_type": ["ground_ball", "line_drive", "fly_ball", "ground_ball", "ground_ball", "fly_ball"],
    })


def test_balls_in_play_drops_home_runs_and_bunts():
    bip = get_balls_in_play(make_raw())
    assert "home_run" not in set(bip["events"])
    assert "sac_bunt" not in set(bip["events"])
    assert len(bip) == 4  # out, single, out (inning 3), sac fly


def test_sac_fly_gets_an_xba_and_counts_as_out():
    bip = get_balls_in_play(make_raw())
    sf = bip[bip["events"] == "sac_fly"].iloc[0]
    assert not np.isnan(sf["estimated_ba_using_speedangle"])
    assert sf["is_out"] == 1


def test_fielding_team_is_home_team_in_top_of_inning():
    bip = get_balls_in_play(make_raw())
    assert (bip["fielding_team"] == "NYY").all()


def test_score_states():
    bip = add_score_state(get_balls_in_play(make_raw()))
    states = dict(zip(bip["events"] + bip["inning"].astype(str), bip["score_state"]))
    assert states["field_out8"] == "close"   # up 1 in the 8th
    assert states["single8"] == "ahead"      # up 7 in the 8th
    assert states["field_out3"] == "other"   # too early in the game


def test_compute_deg_math():
    # 2 balls in play when close (1 out, 1 hit), 1 when behind (1 out)
    bip = pd.DataFrame({
        "season": [2023] * 3,
        "fielding_team": ["NYY"] * 3,
        "score_state": ["close", "close", "behind"],
        "is_out": [1, 0, 1],
        "expected_out": [0.5, 0.5, 0.5],
    })
    row = compute_deg(bip).iloc[0]
    assert row["gap_close"] == 0.0               # (1 - 1.0) / 2
    assert row["gap_behind"] == 0.5              # (1 - 0.5) / 1
    assert row["deg_behind_rate"] == 0.5
