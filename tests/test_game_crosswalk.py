"""Tests for src/game_crosswalk.py using the saved crosswalk."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

CROSSWALK = Path(__file__).resolve().parent.parent / "data" / "processed" / "game_crosswalk.csv"


def load():
    return pd.read_csv(CROSSWALK)


def test_one_to_one():
    df = load()
    assert df["gid"].is_unique
    assert df["game_pk"].is_unique


def test_scores_agree_except_known_walkoff_quirk():
    # Statcast's last post-pitch score is sometimes one run short on walk-offs
    df = load()
    bad = df[(df["home_score"] != df["sc_home_score"]) | (df["away_score"] != df["sc_away_score"])]
    assert len(bad) <= 10
    assert (bad["match_method"] == "date_teams").all()


def test_doubleheaders_matched_by_score_not_order():
    # BAL 2015-06-28: Statcast's game_pk order is the reverse of Retrosheet's game number
    df = load().set_index("gid")
    assert df.loc["BAL201506281", "home_score"] == 4
    assert df.loc["BAL201506281", "game_pk"] == 414787


def test_doubleheaders_matched_by_starter_when_scores_tie():
    # PIT 2016-06-07: both games ended 3-1, and Statcast's game_pk order is reversed
    df = load().set_index("gid")
    assert df.loc["PIT201606071", "game_pk"] == 447743
    assert df.loc["PIT201606072", "game_pk"] == 447731
    assert df.loc["PIT201606071", "match_method"] == "doubleheader_starter"
    assert df.loc["PIT201606072", "match_method"] == "doubleheader_starter"
