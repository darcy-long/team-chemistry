"""Tests for the batter and sac fly opportunities in src/extra_bases_stat/opportunities.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "extra_bases_stat"))

from opportunities import find_batter_opportunities, find_sac_fly_opportunities

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
