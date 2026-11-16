"""
Compare the smooth model (smooth_model.py) to the buckets, on the same chances and out of sample.

For the buckets and for each cumulative set of features (base; + location; + batted ball; + context; + park) it
prints three things:

  1. Fit: the three-way log loss (lower is better) overall and for each kind of chance, plus the attempt model
     and the safe-given-attempt model on their own. Every prediction comes from a model fitted on the other seasons.
  2. What the odds do to the run values: for the run version (rv_safe, rv_out from run_values.py), the total
     of the score over chances where the runner advanced / was thrown out / held, by kind, and how many chances
     the model puts near 0 or near 1 (a model that can tell clear no-goes from clear goes does that more).
  3. Agreement with Savant's runner_runs, piece by piece (as compare_components.py does), for runner-seasons with
     at least 50 of our chances.

The chances are the ones the run version scores (chances with no run value, and ones where a runner ahead made
going impossible, are left out; see the README). The bucket stat is not touched.

Output: data/processed/extra_bases_stat/smooth_model/evaluation.csv   (the fit table)

Usage (needs scikit-learn: use the repo's .venv, from the repo root):
    .venv/bin/python src/extra_bases_stat/evaluate_smooth.py
    .venv/bin/python src/extra_bases_stat/evaluate_smooth.py --groups base location    # only these cumulative sets
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))         # this folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for the shared modules

import compare_components as cc
import extra_bases_statcast as m
from extra_bases import BASE_BUCKETS
from run_expectancy import get_re_table
from run_values import add_run_values
from smooth_model import FEATURE_GROUPS, bucket_odds, make_features, smooth_odds, three_way_log_loss

OUT_PATH = m.OUT_DIR / "smooth_model" / "evaluation.csv"
BUCKET_COLS = BASE_BUCKETS + ["throw_dist_bin"]
MIN_CHANCES = 50


def binary_log_loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def load_scored_chances():
    """The chances the run version scores, with run values."""
    opps = add_run_values(m.build_opportunities(), get_re_table())
    opps = opps[opps["rv_ok"] & ~opps["safe_blocked"]].reset_index(drop=True)
    opps["result"] = np.select([opps["thrown_out"] == 1, opps["took_extra_base"] == 1], ["out", "adv"], "hold")
    return opps


def fit_table(opps, odds):
    """Log losses for one set of odds: three-way (overall and by kind), and the two stages alone."""
    row = {"three_way": three_way_log_loss(opps, odds["p_safe"], odds["p_out"])}
    for kind, d in opps.groupby("opportunity_type"):
        row[kind] = three_way_log_loss(d, odds.loc[d.index, "p_safe"], odds.loc[d.index, "p_out"])
    row["attempt_stage"] = binary_log_loss(opps["took_extra_base"], odds["p_attempt"])
    went = opps["took_extra_base"] == 1
    safe = (opps["took_extra_base"] - opps["thrown_out"])[went]
    row["safe_given_attempt_stage"] = binary_log_loss(safe, odds.loc[went, "p_success"])
    return row


def run_value_effect(opps, odds):
    """Score the chances with these odds: totals by what the runner did, for each kind, and sharpness."""
    d = opps.copy()
    d["p_safe"], d["p_out"] = odds["p_safe"], odds["p_out"]
    d["expected_rate"] = d["p_safe"] * d["rv_safe"] + d["p_out"] * d["rv_out"]
    d["residual"] = d["rv_actual"] - d["expected_rate"]
    by = d.pivot_table(index="opportunity_type", columns="result", values="residual", aggfunc="sum")
    batter = d[d["opportunity_type"] == "batter_for_2"]
    p_go = (batter["p_safe"] + batter["p_out"])
    sharp = {"b2_share_p_under_5pct": float((p_go < 0.05).mean()), "b2_share_p_over_95pct": float((p_go > 0.95).mean())}
    return d, by, sharp


def savant_agreement(scored, sav):
    ours = cc.our_components(scored)
    keep = ["mlbam_id", "season", "runner_runs", "runner_runs_advances", "runner_runs_thrown_out", "runner_runs_hold"]
    df = ours.merge(sav[keep], on=["mlbam_id", "season"])
    df = df[df["n_opportunities"] >= MIN_CHANCES]
    return {"runner_seasons": len(df),
            "total": df["runs_above_expected"].corr(df["runner_runs"]),
            "advances": df["runs_advances"].corr(df["runner_runs_advances"]),
            "thrown_out": df["runs_thrown_out"].corr(df["runner_runs_thrown_out"]),
            "hold": df["runs_hold"].corr(df["runner_runs_hold"]),
            "mean_hold_ours": df["runs_hold"].mean(), "mean_hold_savant": df["runner_runs_hold"].mean(),
            "mean_adv_ours": df["runs_advances"].mean(), "mean_adv_savant": df["runner_runs_advances"].mean()}


def main(group_names):
    t0 = time.time()
    opps = load_scored_chances()
    features = make_features(opps)
    sav = cc.savant_all(sorted(opps["season"].unique()))
    print(f"{len(opps):,} chances scored by the run version\n")

    models = {"buckets (current)": bucket_odds(opps, BUCKET_COLS)}
    cumulative = []
    for g in group_names:
        cumulative.append(g)
        name = "smooth: " + " + ".join(cumulative)
        models[name] = smooth_odds(opps, groups=tuple(cumulative), features=features)
        print(f"  {name}: done ({time.time() - t0:.0f}s)")

    fit, effect, agreement = {}, {}, {}
    for name, odds in models.items():
        fit[name] = fit_table(opps, odds)
        scored, by, sharp = run_value_effect(opps, odds)
        effect[name] = (by, sharp)
        agreement[name] = savant_agreement(scored, sav)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    fit_df = pd.DataFrame(fit).T
    print("\n1. Fit: log loss out of sample (lower is better)")
    print(fit_df.round(4).to_string())

    print("\n2. Run-value effect (sum of the score over chances where the runner advanced / held / was thrown out)")
    for kind in ["batter_for_2", "batter_for_3", "1st_on_single", "2nd_on_single"]:
        rows = {n: effect[n][0].loc[kind] for n in models}
        print(f"  {kind}")
        print(pd.DataFrame(rows).T[["adv", "hold", "out"]].round(0).to_string().replace("\n", "\n    ").join(["    ", ""]))
    print("  batter_for_2 sharpness (share of chances with odds of going under 5% / over 95%)")
    print(pd.DataFrame({n: effect[n][1] for n in models}).T.round(3).to_string())

    print("\n3. Agreement with Savant (correlation of each piece; runner-seasons with 50+ chances)")
    print(pd.DataFrame(agreement).T.round(3).to_string())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fit_df.to_csv(OUT_PATH)
    print(f"\nSaved the fit table to {OUT_PATH} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare the smooth model to the buckets.")
    ap.add_argument("--groups", nargs="+", default=[g for g in FEATURE_GROUPS if g != "speed"], choices=list(FEATURE_GROUPS),
                    help="feature groups, added one at a time in this order (default: all but the opt-in speed group)")
    main(ap.parse_args().groups)
