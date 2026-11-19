"""Tests for the batter and sac fly opportunities in src/extra_bases_stat/opportunities.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "extra_bases_stat"))

from extra_bases import NOT_NEGATED_BY_ERROR, regex_in
from opportunities import (drop_blocked_runners, drop_error_advances, drop_undecided_plays, find_batter_opportunities,
                           find_sac_fly_opportunities, game_ending_plays, is_walk_off)

COLS = ["event", "batter", "batteam", "single", "double", "triple", "br1_pre", "br2_pre", "br3_pre", "br2_post",
        "br3_post", "run3", "outs_pre", "hittype"]


def plays(rows):
    df = pd.DataFrame(rows, columns=COLS)
    return df.replace({None: np.nan})


def row(event, batter="bat01", single=0, double=0, triple=0, br1=None, br2=None, br3=None, br2_post=None,
        br3_post=None, run3=None, outs=0):
    return (event, batter, "BOS", single, double, triple, br1, br2, br3, br2_post, br3_post, run3, outs, "F")


def test_batter_for_2_single_held_double_made_it_thrown_out():
    df = plays([
        row("S7/L7S", single=1),                  # held at first
        row("D8/F8D", double=1),                  # stretched it (or hit it far enough): made second
        row("S9/L9S.BX2(94)", single=1),          # stretched, thrown out
    ])
    out = find_batter_opportunities(df)
    two = out[out["opportunity_type"] == "batter_for_2"]
    assert two["took_extra_base"].tolist() == [0, 1, 1]
    assert two["thrown_out"].tolist() == [0, 0, 1]


def test_batter_for_3_double_held_triple_made_it_thrown_out():
    df = plays([
        row("D7/F7S", double=1),                  # held at second
        row("T9/L9D", triple=1),                  # made third
        row("D9/L9D.BX3(95)", double=1),          # tried for third, thrown out
    ])
    out = find_batter_opportunities(df)
    three = out[out["opportunity_type"] == "batter_for_3"]
    assert three["took_extra_base"].tolist() == [0, 1, 1]
    assert three["thrown_out"].tolist() == [0, 0, 1]


def test_batter_chances_need_empty_bases_skip_error_advances_and_ground_rule_doubles():
    df = plays([
        row("S8/L8S", single=1, br1="run01"),                  # runner on first: not a batter chance
        row("S4/G34.B-2(E4/TH)", single=1, br2_post="bat01"),  # advanced on an error: dropped
        row("DGR/L9LD", double=1),                             # ground-rule double: no triple possible
    ])
    assert len(find_batter_opportunities(df)) == 0


def test_sac_fly_scored_thrown_out_and_held():
    df = plays([
        row("8/SF/F8D.3-H", br3="run01", run3="run01", outs=0),          # tagged and scored
        row("9/F9D.3XH(92)", br3="run01", outs=1),                       # thrown out at home
        row("7/F7S", br3="run01", outs=1),                               # stayed
        row("8/F8D", br3="run01", outs=2),                               # 2 outs: not a chance
        row("63/G6", br3="run01", outs=0),                               # infield out: not a chance
    ])
    out = find_sac_fly_opportunities(df)
    assert out["opportunity_type"].eq("3rd_on_fly").all()
    assert out["took_extra_base"].tolist() == [1, 1, 0]
    assert out["runner_id"].tolist() == ["run01"] * 3


def test_thrown_out_flag_and_error_cancelled_outs():
    df = plays([
        row("S9/L9S.BX2(94)", single=1),                # thrown out trying
        row("S8/L8S.B-2", single=1, br2_post="bat01"),  # safe
        row("S7/L7S.BX2(7E4)", single=1),               # out cancelled by an error: safe, not thrown out
    ])
    out = find_batter_opportunities(df)
    assert out["took_extra_base"].tolist() == [1, 1, 1]
    assert out["thrown_out"].tolist() == [1, 0, 0]


def test_rank_corr_is_spearman_without_scipy():
    from compare_to_savant import rank_corr
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert rank_corr(a, pd.Series([10.0, 20.0, 30.0, 400.0])) == 1.0   # same order, different spacing
    assert rank_corr(a, pd.Series([4.0, 3.0, 2.0, 1.0])) == -1.0


def chances(rows):
    """rows: (top_bot, inning, score_v, score_h, runs, outs_post, took_extra_base)"""
    df = pd.DataFrame(rows, columns=["top_bot", "inning", "score_v", "score_h", "runs", "outs_post", "took_extra_base"])
    df["event"] = "S7/L7"
    return df


def test_walk_off_is_the_home_team_taking_the_lead_in_the_9th_or_later():
    df = chances([
        (1, 9, 3, 3, 1, 1, 1),    # tied, run scores: walk-off
        (1, 9, 3, 2, 2, 0, 1),    # down 1, two runs score: walk-off
        (1, 10, 3, 1, 2, 1, 0),   # down 2, two runs score: only ties it, game goes on
        (1, 9, 3, 3, 0, 1, 0),    # tied, no run: not a walk-off
        (1, 8, 3, 3, 1, 1, 1),    # 8th inning: not a walk-off
        (0, 9, 3, 3, 1, 1, 1),    # top of the 9th: visitors cannot walk off
    ])
    assert is_walk_off(df).tolist() == [True, True, False, False, False, False]


def test_drop_undecided_plays_removes_walk_offs_and_inning_ending_holds_only():
    df = chances([
        (1, 9, 3, 3, 1, 1, 0),    # walk-off, runner "held": dropped
        (1, 9, 3, 3, 1, 1, 1),    # walk-off, runner scored: dropped too
        (0, 4, 0, 0, 0, 3, 0),    # third out made by someone else, runner did not go: dropped
        (0, 4, 0, 0, 0, 3, 1),    # third out made by this runner going (thrown out): kept
        (0, 4, 0, 0, 1, 3, 1),    # he scored before the third out: kept
        (0, 4, 0, 0, 0, 1, 0),    # an ordinary hold: kept
    ])
    kept = drop_undecided_plays(df)
    assert kept.index.tolist() == [3, 4, 5]


def test_drop_undecided_plays_removes_ground_rule_doubles():
    df = chances([(0, 4, 0, 0, 0, 1, 0), (0, 4, 0, 0, 0, 1, 0)])
    df["event"] = ["DGR/L9LD.3-H;1-3", "D9/L9L.1-3"]   # runner on 1st is sent to 3rd by rule on the first one
    assert drop_undecided_plays(df).index.tolist() == [1]


def test_an_out_is_cancelled_only_by_an_error_in_the_fielding_sequence():
    events = pd.Series([
        "1X3(6E4)",                  # error in the fielding sequence: safe
        "2XH(UR)(8E3)(NR);B-2",      # same, after another parenthetical: safe
        "1XH(62)(E8);B-2",           # a plain putout, the error is elsewhere: the out stands
        "BX3(E9)(95)",               # error first, then the putout: the out stands
        "2XH(82)",
        "1X3",
    ])
    assert regex_in(events, r"[B123]X[123H]" + NOT_NEGATED_BY_ERROR).tolist() == [False, False, True, True, True, True]


def test_runner_advances_on_errors_are_dropped_and_real_attempts_kept():
    opps = pd.DataFrame({
        "opportunity_type": ["1st_on_single", "1st_on_single", "2nd_on_single", "1st_on_double", "1st_on_single"],
        "event": ["S7/G6.1-3(E7)", "S8/L8S.1-3", "S7/L7.2-H(UR)(E8)(NR)", "D9/L9L.1XH(92)(E8)", "S9/L9.1-3(TH)"],
    })
    kept = drop_error_advances(opps)
    assert kept["event"].tolist() == ["S8/L8S.1-3", "D9/L9L.1XH(92)(E8)", "S9/L9.1-3(TH)"]


def test_batter_advances_on_errors_are_dropped_but_an_out_cancelled_by_an_error_is_kept():
    df = plays([
        row("S4/G34.B-2(E4/TH)", single=1),        # reached second on an error: dropped
        row("S7/L56+.BX3(75)(E7)", single=1),      # error on the play, then out at third: dropped
        row("S7/L7S.BX2(7E4)", single=1),          # tried for second, out cancelled by an error: a real attempt
        row("S8/L8S.B-2", single=1, br2_post="bat01"),
    ])
    out = find_batter_opportunities(df)
    assert out["event"].tolist() == ["S7/L7S.BX2(7E4)", "S8/L8S.B-2"]


def test_sac_fly_chances_exclude_outfielders_fielding_a_ground_ball_for_a_force_out():
    df = plays([
        row("8/SF/F8D.3-H", br3="run01", run3="run01"),         # caught fly, scores
        row("7", br3="run01", outs=1),                          # a bare fly out: he stayed
        row("8(B)84(1)/LDP.3-H", br3="run01", run3="run01"),    # line drive double play, still the batter's catch
        row("84(1)/FO/F8RS.3-H", br3="run01", run3="run01"),    # force out at 2nd: no tag-up to decide
        row("96(1)/FO/G6MS.3-H", br3="run01", run3="run01"),
    ])
    assert len(find_sac_fly_opportunities(df)) == 3


def ended(rows):
    """rows: (gid, inning, top_bot, event, outs_post)"""
    return pd.DataFrame(rows, columns=["gid", "inning", "top_bot", "event", "outs_post"])


def test_game_ending_plays_are_the_last_play_of_a_half_inning_that_did_not_reach_three_outs():
    df = ended([
        ("G1", 7, 0, "K", 2), ("G1", 7, 0, "S8/L8", 2), ("G1", 7, 0, "63/G6", 3),     # an ordinary half-inning
        ("G1", 7, 1, "S9/L9", 1), ("G1", 7, 1, "S7/L7.2-H", 1), ("G1", 7, 1, "NP", 1),  # a 7-inning game ends here
        ("G2", 9, 1, "K", 2), ("G2", 9, 1, "NP", 2),                                      # NP rows do not count
    ])
    assert game_ending_plays(df).tolist() == [False, False, False, False, True, False, True, False]


def test_blocked_runners_on_first_are_dropped_only_when_a_runner_ahead_ends_on_third():
    cols = ["opportunity_type", "event", "batter", "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post", "br3_post",
            "run_b", "run1", "run2", "run3", "outs_post"]
    nan = np.nan
    opps = pd.DataFrame([
        # the runner on 2nd stops at 3rd, so the runner on 1st cannot take 3rd: blocked
        ("1st_on_single", "S8/L8S.2-3;1-2", "bat", "r1", "r2", nan, "bat", "r1", "r2", nan, nan, nan, nan, 0),
        # alone on 1st and takes 3rd: a real decision
        ("1st_on_single", "S9/L9D.1-3", "bat", "r1", nan, nan, "bat", nan, "r1", nan, nan, nan, nan, 0),
        # the runner on 2nd scores and the runner on 1st stops at 2nd: not blocked
        ("1st_on_single", "S7/L7.2-H;1-2", "bat", "r1", "r2", nan, "bat", "r1", nan, nan, nan, "r2", nan, 0),
        # other kinds are never blocked
        ("2nd_on_single", "S7/L7.2-3", "bat", nan, "r2", nan, "bat", nan, "r2", nan, nan, nan, nan, 0),
    ], columns=cols)
    assert drop_blocked_runners(opps).index.tolist() == [1, 2, 3]
