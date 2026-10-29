"""Tests for src/hustle_index.py."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hustle_index import build_hustle_index, to_statcast_code, zscore


def test_zscore_has_mean_zero_and_std_one():
    z = zscore(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert abs(z.mean()) < 1e-12
    assert abs(z.std(ddof=1) - 1) < 1e-12


def test_zscore_of_constant_column_is_zero():
    assert (zscore(pd.Series([5.0, 5.0, 5.0])) == 0).all()


def test_team_codes():
    assert to_statcast_code("CHN") == "CHC"
    assert to_statcast_code("CHC") == "CHC"


def test_index_is_standardized_within_each_season():
    idx = build_hustle_index()
    by_season = idx.groupby("season")["hustle_z"]
    assert (by_season.mean().abs() < 1e-9).all()
    assert ((by_season.std(ddof=1) - 1).abs() < 1e-9).all()


def test_rank_one_has_highest_score():
    idx = build_hustle_index()
    for _, s in idx.groupby("season"):
        assert s.loc[s["season_rank"] == 1, "hustle_z"].iloc[0] == s["hustle_z"].max()


def test_adding_an_indicator_is_one_config_entry(tmp_path, monkeypatch):
    import hustle_index

    real = pd.read_csv(hustle_index.DATA_DIR / "extra_bases.csv")
    real["fake_score"] = range(len(real))
    real.to_csv(tmp_path / "fake.csv", index=False)

    monkeypatch.setitem(hustle_index.INDICATORS, "fake", {
        "file": "fake.csv", "team_col": "team", "score_col": "fake_score",
        "weight": 1.0, "higher_is_better": False,
    })
    monkeypatch.setattr(hustle_index, "DATA_DIR", tmp_path)
    for f in ["extra_bases.csv", "deg.csv", "standings.csv"]:
        (tmp_path / f).write_bytes((Path(__file__).resolve().parent.parent / "data" / "processed" / f).read_bytes())

    idx = hustle_index.build_hustle_index()
    assert "fake_z" in idx.columns
    # lower raw value should mean a higher z when higher_is_better is False
    assert idx["fake_z"].corr(idx["fake_raw"]) < 0
