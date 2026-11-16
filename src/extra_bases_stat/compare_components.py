"""
Compare our run-value version (--runs) to Baseball Savant's baserunning run value, piece by piece.

Savant splits a runner's value (runner_runs) into three parts that add up exactly:
    runner_runs_advances      what he got from the extra bases he took
    runner_runs_thrown_out    what he lost on the times he was thrown out going
    runner_runs_hold          what he lost on the times he held
We split our score the same way, by what he actually did on each chance: the sum of that chance's score
over his chances where he advanced safely (runs_advances), was thrown out (runs_thrown_out) or held (runs_hold).
They add up to runs_above_expected. So a gap in the total can be traced to a part.

It also compares the attempt models directly, since the parts depend on how many attempts each model expects:
    attempts above expected   ours: attempts - sum(p_safe + p_out)      Savant: n_att_xb - est_rate_att_generic_runner x n_opp_xb
and the plain counts (chances, attempts, thrown out), because we count about 11% more chances than Savant
(see the README), which scales every part.

For each pair it prints the Pearson and rank correlation, the slope of Savant on ours (1.0 = same scale),
and the standard deviation of each. Runner-seasons with at least --min-chances of our chances, every season
that has a Savant file.

Input:  the run version, computed here (extra_bases_statcast.compute(runs=True)), data/raw/savant/base_running_YYYY.csv
Output: data/processed/extra_bases_stat/savant_comparison/component_comparison.csv  (one row per runner-season in both)

Usage (from the repo root):
    python src/extra_bases_stat/compare_components.py
    python src/extra_bases_stat/compare_components.py --min-chances 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # this folder, for compare_to_savant
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for the shared modules

from compare_to_savant import STAT_DIR, load_savant, rank_corr
from extra_bases_statcast import add_runner_info, compute

OUT_PATH = STAT_DIR / "savant_comparison" / "component_comparison.csv"
RESULTS = {"advance": "advances", "thrown_out": "thrown_out", "hold": "hold"}


def our_components(opps):
    """One row per runner-season: our three parts, their total, and the attempt numbers."""
    opps = add_runner_info(opps).copy()
    opps["result"] = np.select([opps["thrown_out"] == 1, opps["took_extra_base"] == 1], ["thrown_out", "advance"], "hold")
    keys = ["mlbam_id", "season"]
    parts = opps.pivot_table(index=keys, columns="result", values="residual", aggfunc="sum", fill_value=0.0)
    parts = parts.rename(columns={k: f"runs_{v}" for k, v in RESULTS.items()})
    counts = opps.groupby(keys).agg(
        n_opportunities=("residual", "size"),
        n_attempts=("took_extra_base", "sum"),
        n_thrown_out=("thrown_out", "sum"),
        expected_attempts=("p_safe", "sum"),
        expected_out=("p_out", "sum"),
    )
    counts["expected_attempts"] += counts["expected_out"]
    names = opps.drop_duplicates(keys).set_index(keys)["name"]
    out = parts.join(counts).join(names).reset_index().dropna(subset=["mlbam_id"])
    out["mlbam_id"] = out["mlbam_id"].astype(int)
    out["runs_above_expected"] = out[["runs_advances", "runs_thrown_out", "runs_hold"]].sum(axis=1)
    out["attempts_above_expected"] = out["n_attempts"] - out["expected_attempts"]
    return out.drop(columns="expected_out")


def savant_all(seasons):
    frames = []
    for year in seasons:
        try:
            df = load_savant(year)
        except FileNotFoundError:
            continue
        df["season"] = year
        frames.append(df)
    sav = pd.concat(frames, ignore_index=True)
    sav["expected_attempts_sav"] = sav["est_rate_att_generic_runner"] * sav["n_opp_xb"]
    sav["attempts_above_expected_sav"] = sav["n_att_xb"] - sav["expected_attempts_sav"]
    return sav


def summarise(df, pairs):
    rows = []
    for label, ours, theirs in pairs:
        a, b = df[ours], df[theirs]
        rows.append({
            "piece": label, "pearson": a.corr(b), "rank": rank_corr(a, b),
            "slope_sav_on_ours": np.polyfit(a, b, 1)[0], "sd_ours": a.std(), "sd_savant": b.std(),
            "mean_ours": a.mean(), "mean_savant": b.mean(),
        })
    return pd.DataFrame(rows).set_index("piece")


def main(min_chances=50):
    _, opps = compute(runs=True, return_opportunities=True)
    ours = our_components(opps)
    sav = savant_all(sorted(ours["season"].unique()))
    keep = ["mlbam_id", "season", "runner_runs", "runner_runs_advances", "runner_runs_thrown_out", "runner_runs_hold",
            "n_opp_xb", "n_att_xb", "n_out", "expected_attempts_sav", "attempts_above_expected_sav"]
    df = ours.merge(sav[keep], on=["mlbam_id", "season"])
    df = df[df["n_opportunities"] >= min_chances].reset_index(drop=True)
    print(f"\n{len(df):,} runner-seasons in both with at least {min_chances} of our chances "
          f"({sorted(df['season'].unique())[0]}-{sorted(df['season'].unique())[-1]})\n")

    pd.set_option("display.width", 200)
    print("Value (runs)")
    print(summarise(df, [
        ("total", "runs_above_expected", "runner_runs"),
        ("advances", "runs_advances", "runner_runs_advances"),
        ("thrown out", "runs_thrown_out", "runner_runs_thrown_out"),
        ("hold", "runs_hold", "runner_runs_hold"),
    ]).round(3).to_string())
    print("\nCounts and the attempt model")
    print(summarise(df, [
        ("chances", "n_opportunities", "n_opp_xb"),
        ("attempts", "n_attempts", "n_att_xb"),
        ("thrown out", "n_thrown_out", "n_out"),
        ("expected attempts", "expected_attempts", "expected_attempts_sav"),
        ("attempts above expected", "attempts_above_expected", "attempts_above_expected_sav"),
    ]).round(3).to_string())

    # the same value pieces per chance, so our extra chances don't drive the comparison
    for col, sav_col in [("runs_above_expected", "runner_runs"), ("runs_advances", "runner_runs_advances"),
                         ("runs_thrown_out", "runner_runs_thrown_out"), ("runs_hold", "runner_runs_hold")]:
        df[f"{col}_per_chance"] = df[col] / df["n_opportunities"]
        df[f"{sav_col}_per_chance"] = df[sav_col] / df["n_opp_xb"]
    print("\nValue per chance (each divided by its own chance count)")
    print(summarise(df, [
        ("total", "runs_above_expected_per_chance", "runner_runs_per_chance"),
        ("advances", "runs_advances_per_chance", "runner_runs_advances_per_chance"),
        ("thrown out", "runs_thrown_out_per_chance", "runner_runs_thrown_out_per_chance"),
        ("hold", "runs_hold_per_chance", "runner_runs_hold_per_chance"),
    ]).round(4).to_string())

    # how much of the gap in the total sits in each part: regress the total difference on the part differences
    scale = np.polyfit(df["runs_above_expected"], df["runner_runs"], 1)[0]
    print(f"\nSavant's total is {scale:.2f}x ours on average (slope). Remaining disagreement, by part, after rescaling ours:")
    for label, o, t in [("advances", "runs_advances", "runner_runs_advances"),
                        ("thrown out", "runs_thrown_out", "runner_runs_thrown_out"),
                        ("hold", "runs_hold", "runner_runs_hold")]:
        diff = df[t] - scale * df[o]
        print(f"  {label:12s} sd of (Savant - rescaled ours) = {diff.std():.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} runner-seasons to {OUT_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare our run version to Savant, piece by piece.")
    ap.add_argument("--min-chances", type=int, default=50, help="minimum of our chances for a runner-season (default 50)")
    args = ap.parse_args()
    main(min_chances=args.min_chances)
