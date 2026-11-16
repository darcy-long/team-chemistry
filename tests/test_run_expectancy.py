"""Tests for src/extra_bases_stat/run_expectancy.py (per-season RE24 and play run values)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "extra_bases_stat"))

from run_expectancy import add_play_run_value, base_state, build_re_table, lookup_re

COLS = ["season", "gid", "inning", "top_bot", "event", "outs_pre", "outs_post",
        "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post", "br3_post", "runs"]


def play(gid, top_bot, event, outs_pre, outs_post, pre=(None, None, None), post=(None, None, None), runs=0, inning=1):
    return (2019, gid, inning, top_bot, event, outs_pre, outs_post, *pre, *post, runs)


def plays(rows):
    return pd.DataFrame(rows, columns=COLS).replace({None: np.nan})


# One finished half-inning: single, home run (2 runs), then three strikeouts.
# And one unfinished one (a walk-off style cut-off), plus an NP row that must be ignored.
ROWS = [
    play("G1", 0, "S8", 0, 0, post=("a", None, None)),
    play("G1", 0, "NP", 0, 0, pre=("a", None, None), post=("a", None, None)),
    play("G1", 0, "HR", 0, 0, pre=("a", None, None), runs=2),
    play("G1", 0, "K", 0, 1),
    play("G1", 0, "K", 1, 2),
    play("G1", 0, "K", 2, 3),
    play("G1", 1, "S8", 0, 0, post=("z", None, None), runs=5),   # never finished: left out
]


def test_base_state_codes():
    assert base_state([False, True, False, True], [False, False, True, True], [False, False, False, True]).tolist() == [0, 1, 2, 7]


def test_re_table_averages_runs_to_the_end_of_the_inning():
    table = build_re_table(plays(ROWS)).set_index(["base_state", "outs"])["re"]
    assert table[(0, 0)] == pytest.approx(1.0)   # single's inning scored 2, the first K's scored 0
    assert table[(1, 0)] == pytest.approx(2.0)   # HR with a man on
    assert table[(0, 1)] == pytest.approx(0.0)
    assert table[(0, 2)] == pytest.approx(0.0)


def test_unfinished_half_innings_are_left_out():
    table = build_re_table(plays(ROWS))
    assert table["n"].sum() == 5   # 5 real plays in the finished half-inning, nothing from the other one or the NP


def test_lookup_zero_at_three_outs_and_nan_for_missing_cells():
    table = build_re_table(plays(ROWS))
    got = lookup_re(table, [2019, 2019, 2019, 2018], [1, 0, 5, 0], [0, 3, 0, 0])
    assert got[0] == pytest.approx(2.0)
    assert got[1] == 0.0
    assert np.isnan(got[2])   # state 5 never happened
    assert np.isnan(got[3])   # wrong season


def test_play_run_value_is_re_after_plus_runs_minus_re_before():
    df = plays(ROWS)
    table = build_re_table(df)
    out = add_play_run_value(df.iloc[[0, 2]], table)
    # single: RE(runner on 1st, 0 out) - RE(empty, 0 out) = 2 - 1
    assert out["run_value"].iloc[0] == pytest.approx(1.0)
    # HR: RE(empty, 0 out) + 2 runs - RE(runner on 1st, 0 out) = 1 + 2 - 2
    assert out["run_value"].iloc[1] == pytest.approx(1.0)
