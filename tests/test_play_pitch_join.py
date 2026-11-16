"""Tests for the sequence matching in src/play_pitch_join.py using small made-up games."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from play_pitch_join import align_game


def ret_rows(rows):
    return pd.DataFrame(rows, columns=["inning", "top_bot", "batter_id", "pitcher_id", "pa"])


def sc_rows(rows):
    return pd.DataFrame(rows, columns=["inning", "half", "batter", "pitcher", "at_bat_number"])


def test_simple_game_matches_in_order():
    ret = ret_rows([(1, 0, 10, 99, 1), (1, 0, 11, 99, 1)])
    sc = sc_rows([(1, 0, 10, 99, 1), (1, 0, 11, 99, 2)])
    assert list(align_game(ret, sc)) == [1, 2]


def test_unfinished_plate_appearance_does_not_shift_later_matches():
    # Batter 11 is at the plate when a runner is caught stealing for the 3rd out.
    # Statcast numbers that at-bat (2); Retrosheet has only a non-PA row for it.
    ret = ret_rows([(1, 0, 10, 99, 1), (1, 0, 11, 99, 0), (2, 0, 11, 99, 1)])
    sc = sc_rows([(1, 0, 10, 99, 1), (1, 0, 11, 99, 2), (2, 0, 11, 99, 3)])
    assert list(align_game(ret, sc)) == [1, 2, 3]


def test_steal_during_plate_appearance_gets_that_at_bat():
    ret = ret_rows([(1, 0, 10, 99, 0), (1, 0, 10, 99, 1)])
    sc = sc_rows([(1, 0, 10, 99, 1)])
    assert list(align_game(ret, sc)) == [1, 1]


def test_no_match_returns_minus_one():
    ret = ret_rows([(1, 0, 10, 99, 1)])
    sc = sc_rows([(1, 0, 55, 99, 1)])
    assert list(align_game(ret, sc)) == [-1]


def test_game_ending_non_pa_matches_unfinished_at_bat():
    ret = ret_rows([(9, 0, 10, 99, 1), (9, 0, 11, 99, 0)])
    sc = sc_rows([(9, 0, 10, 99, 1), (9, 0, 11, 99, 2)])
    assert list(align_game(ret, sc)) == [1, 2]
