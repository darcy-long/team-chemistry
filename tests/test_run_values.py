"""Tests for src/extra_bases_stat/run_values.py (run value of safe / held / out for each chance)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "extra_bases_stat"))

from run_values import NEEDED_COLUMNS, add_run_values, apply_flat_values, check_construction


def fake_re(state, outs):
    """A made-up run expectancy with a different value for every cell, so any mix-up shows."""
    if outs >= 3:
        return 0.0
    runners = (state & 1) + ((state >> 1) & 1) + ((state >> 2) & 1)
    return (2 - outs) * 0.30 + 0.50 * runners + 0.01 * state


RE = pd.DataFrame([(2019, s, o, fake_re(s, o), 100) for s in range(8) for o in range(3)],
                  columns=["season", "base_state", "outs", "re", "n"])


def st(on1=0, on2=0, on3=0):
    return on1 + 2 * on2 + 4 * on3


def opp(kind, pre=(None, None, None), post=(None, None, None), scored=(None, None, None, None),
        outs=(0, 0), runs=0, took=0, thrown=0, event=""):
    return dict(season=2019, opportunity_type=kind, event=event, batter="b", br1_pre=pre[0], br2_pre=pre[1], br3_pre=pre[2],
                br1_post=post[0], br2_post=post[1], br3_post=post[2], run_b=scored[0], run1=scored[1],
                run2=scored[2], run3=scored[3], outs_pre=outs[0], outs_post=outs[1], runs=runs,
                took_extra_base=took, thrown_out=thrown)


def run(*rows):
    df = pd.DataFrame(list(rows)).replace({None: np.nan})
    return add_run_values(df[NEEDED_COLUMNS], RE)


def test_batter_for_2_held_safe_out():
    held = run(opp("batter_for_2", post=("b", None, None)))
    safe = run(opp("batter_for_2", post=(None, "b", None), took=1))
    out = run(opp("batter_for_2", outs=(0, 1), took=1, thrown=1))
    first, second, gone = fake_re(st(on1=1), 0), fake_re(st(on2=1), 0), fake_re(0, 1)
    assert held["rv_safe"][0] == pytest.approx(second - first)
    assert held["rv_out"][0] == pytest.approx(gone - first)
    assert held["rv_actual"][0] == 0
    assert safe["rv_actual"][0] == pytest.approx(second - first)
    assert out["rv_actual"][0] == pytest.approx(gone - first)
    assert safe["rv_gap"][0] == pytest.approx(0) and out["rv_gap"][0] == pytest.approx(0)


def test_sac_fly_safe_is_a_run_and_a_third_out_scores_nothing():
    # runner on 3rd, 1 out, fly ball: the batter is out. Held = 2 outs, runner on 3rd.
    r = run(opp("3rd_on_fly", pre=(None, None, "r"), scored=(None, None, None, "r"), outs=(1, 2), runs=1, took=1))
    held = fake_re(st(on3=1), 2)
    assert r["rv_safe"][0] == pytest.approx(fake_re(0, 2) + 1 - held)
    assert r["rv_out"][0] == pytest.approx(0 - held)   # runner tagged out for the 3rd out: no run
    assert r["rv_gap"][0] == pytest.approx(0)


def test_throw_advances_of_runners_behind_are_not_credited_to_the_runner():
    # runner on 1st goes to 3rd on a single and the batter takes 2nd on the throw
    r = run(opp("1st_on_single", pre=("a", None, None), post=(None, "b", "a"), took=1))
    # held: runner at 2nd, batter at 1st. safe: runner at 3rd, batter still at 1st (not 2nd)
    assert r["rv_safe"][0] == pytest.approx(fake_re(st(1, 0, 1), 0) - fake_re(st(1, 1, 0), 0))
    assert r["rv_actual"][0] == pytest.approx(r["rv_safe"][0])
    assert r["rv_gap"][0] == pytest.approx(0)   # with the batter as he really ended up, the real play is reproduced


def test_runner_ahead_on_the_base_blocks_the_extra_base():
    # runners on 1st and 2nd, single: the runner on 2nd stops at 3rd, so the runner on 1st can't go to 3rd
    r = run(opp("1st_on_single", pre=("a", "c", None), post=("b", "a", "c")))
    assert bool(r["safe_blocked"][0])
    assert r["rv_safe"][0] == pytest.approx(0)


def test_third_out_at_home_counts_runs_ahead_of_the_runner_only():
    # 2 outs, runners on 2nd and 3rd, single: the runner on 3rd scores, then the runner from 2nd is thrown out at home
    r = run(opp("2nd_on_single", pre=(None, "a", "c"), post=("b", None, None), scored=(None, None, None, "c"),
                outs=(2, 3), runs=1, took=1, thrown=1, event="S8.3-H;2XH(82)"))
    held = fake_re(st(on1=1, on3=1), 2) + 1   # runner stays at 3rd, run still scored
    assert r["rv_out"][0] == pytest.approx(1 - held)   # the run counts (it scored first), no more outs
    assert r["rv_actual"][0] == pytest.approx(r["rv_out"][0])
    assert r["rv_gap"][0] == pytest.approx(0)


def test_walkoff_has_no_run_value():
    # the game ended on the hit: the batter is never put on a base and no out was made
    r = run(opp("1st_on_double", pre=("a", None, None), scored=(None, "a", None, None), outs=(2, 2), runs=1, took=1))
    assert not r["rv_ok"][0]
    assert np.isnan(r["rv_safe"][0]) and np.isnan(r["rv_actual"][0])


def test_check_construction_counts_mismatches_and_unknown_kinds_fail():
    r = run(opp("batter_for_2", post=("b", None, None)), opp("batter_for_2", post=(None, "b", None), took=1))
    summary = check_construction(r)
    assert summary.loc["batter_for_2", "chances"] == 2 and summary.loc["batter_for_2", "mismatch"] == 0
    with pytest.raises(ValueError):
        run(opp("steal_of_second", post=("b", None, None)))


def test_inning_ending_play_clears_the_bases_so_only_the_text_says_who_is_out():
    # 2 outs, runner on 1st thrown out at 3rd to end the inning; the batter is not on any base afterwards
    # but was not out. Without the text, the batter would be counted as a second out and the chance dropped.
    r = run(opp("1st_on_single", pre=("a", None, None), outs=(2, 3), took=1, thrown=1, event="S8/L8S.1X3(85)"))
    assert r["rv_ok"][0]
    assert r["rv_actual"][0] == pytest.approx(r["rv_out"][0])
    assert r["rv_gap"][0] == pytest.approx(0)
    # held = runner at 2nd, batter at 1st, 2 outs; out = inning over
    assert r["rv_out"][0] == pytest.approx(0 - fake_re(st(on1=1, on2=1), 2))


def test_batter_caught_on_a_fly_ball_is_out_even_when_the_inning_ends():
    # 1 out, runner on 3rd tagging up is thrown out at home: batter out (catch) + runner out = 3 outs
    r = run(opp("3rd_on_fly", pre=(None, None, "r"), outs=(1, 3), took=1, thrown=1, event="8/F8.3XH(82)"))
    assert r["rv_ok"][0]
    assert r["rv_gap"][0] == pytest.approx(0)


def test_flat_values_are_the_same_in_every_state_and_zero_for_a_hold():
    opps = pd.DataFrame({"took_extra_base": [1, 1, 0], "thrown_out": [0, 1, 0], "rv_safe": [0.9, 0.1, 0.5], "rv_out": [-0.2, -1.3, -0.4]})
    flat = apply_flat_values(opps, extra_base=0.25, out=0.7)
    assert (flat["rv_safe"] == 0.25).all() and (flat["rv_out"] == -0.7).all()
    assert flat["rv_actual"].tolist() == [0.25, -0.7, 0.0]
