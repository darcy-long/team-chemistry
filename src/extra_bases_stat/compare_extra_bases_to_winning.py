"""
Do teams with higher extra-bases scores win more?

Takes the team-season files of each version of the stat (bases, runs, flat runs, smooth odds) and joins them to
data/processed/standings.csv. For each version it reports:

    1. Correlation of the team score (within each season, so seasons are comparable) with win%, run
       differential, runs scored, runs allowed and win_outperformance (wins above what runs scored and allowed
       predict). "total" is the team's whole season (bases or runs above expected); "per chance" divides it by the
       team's number of chances.
    2. Regression of win% on the score, with a separate intercept for every season (season fixed effects), so the
       size is "win% per standard deviation of the score". For the run versions the same regression with wins on
       runs gives wins per run of the score: a real run is worth about 0.10 wins, so that is the size to compare to.
    3. Regression of win% on run differential, with and without the score: does it add anything beyond runs?
    4. How stable the score is: season-to-season correlation of the team's score (a skill-like score repeats, a
       lucky one doesn't).

2020 is not in this project's data, so every season here is a full 162-game season (2015-2019, 2021-2025).

Input:  data/processed/extra_bases_stat/(variants/)extra_bases_statcast*.csv, data/processed/standings.csv
Output: data/processed/extra_bases_stat/winning/correlations.csv, regressions.csv

Usage (from the repo root):
    python src/extra_bases_stat/compare_extra_bases_to_winning.py
    python src/extra_bases_stat/compare_extra_bases_to_winning.py --regular-season

--regular-season reads the team files built with extra_bases_statcast.py --regular-season (suffix _regularseason, no
postseason chances), so the team totals cover the same games as the standings. Results go in winning_regularseason/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
STAT_DIR = ROOT / "data" / "processed" / "extra_bases_stat"
OUT_DIR = STAT_DIR / "winning"
REGULAR_SEASON = False   # set by --regular-season

# label: (file, column holding the team's season total: the sum of its residuals).
VERSIONS = {
    "bases (default)": ("extra_bases_statcast.csv", "net_bases_above_expected"),
    "bases, smooth odds": ("variants/extra_bases_statcast_smooth.csv", "net_bases_above_expected"),
    "runs": ("variants/extra_bases_statcast_runs.csv", "runs_above_expected"),
    "runs, smooth odds": ("variants/extra_bases_statcast_runs_smooth.csv", "runs_above_expected"),
    "runs, flat values": ("variants/extra_bases_statcast_runs_flat.csv", "runs_above_expected"),
    "runs, flat values, smooth odds": ("variants/extra_bases_statcast_runs_flat_smooth.csv", "runs_above_expected"),
    "runs, flat values, smooth odds, own speed": ("variants/extra_bases_statcast_runs_flat_smooth_ownspeed.csv", "runs_above_expected"),
}
OUTCOMES = ["win_pct", "run_diff", "R", "RA", "win_outperformance"]


def within_season_z(df, col):
    g = df.groupby("season")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def version_path(path):
    """The team file for a version; with --regular-season the file built without postseason chances."""
    if not REGULAR_SEASON:
        return path
    name = path.split("/")[-1].replace(".csv", "")
    return f"variants/{name}_regularseason.csv"


def load(label):
    path, total_col = VERSIONS[label]
    path = version_path(path)
    team = pd.read_csv(STAT_DIR / path)
    # the team files hold the true total (sum of residuals) and the true per-chance rate (total / chances).
    # raw_score, the plain average of the six kinds, is neither: it gives a 45-chance kind the same weight
    # as a 630-chance one.
    team["total"] = team[total_col]
    standings = pd.read_csv(ROOT / "data" / "processed" / "standings.csv")
    df = team.merge(standings, on=["team", "season"])
    for col in ("total", "per_chance"):
        df[col + "_z"] = within_season_z(df, col)
    return df


def corr_with_p(x, y):
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r * r))
    # two-sided p from the t distribution via the normal tail for large n is adequate here (n ~ 300)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return r, p


def fe_ols(df, y, xs):
    """OLS of y on xs with a separate intercept per season (demeaning within season). Returns coefs, se, t, R2 (within)."""
    d = df[[y] + xs + ["season"]].copy()
    for c in [y] + xs:
        d[c] = d[c] - d.groupby("season")[c].transform("mean")
    X, Y = d[xs].to_numpy(), d[y].to_numpy()
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = len(d) - len(xs) - d["season"].nunique()
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / (Y @ Y)
    return beta, se, beta / se, r2


def two_way_demean(df, col, rounds=50):
    """Remove each season's average and each team's average (a team's usual level, which includes its park)."""
    x = df[col].astype(float).copy()
    for _ in range(rounds):
        x = x - x.groupby(df["season"]).transform("mean")
        x = x - x.groupby(df["team"]).transform("mean")
    return x


def stability(df, col):
    nxt = df[["team", "season", col]].assign(season=lambda x: x["season"] - 1).rename(columns={col: "next"})
    m = df[["team", "season", col]].merge(nxt, on=["team", "season"])
    return m[col].corr(m["next"]), len(m)


def main():
    corr_rows, reg_rows = [], []
    for label in VERSIONS:
        full = load(label)
        for sample, df in (("all seasons", full),):
            for kind in ("total", "per_chance"):
                for out in OUTCOMES:
                    r, p = corr_with_p(df[kind + "_z"], within_season_z(df, out))
                    corr_rows.append(dict(version=label, sample=sample, score=kind, outcome=out, n=len(df), r=r, p=p))
                # win% on the score, season fixed effects
                b, se, t, r2 = fe_ols(df, "win_pct", [kind + "_z"])
                reg_rows.append(dict(version=label, sample=sample, score=kind, model="win% ~ score (per SD)", coef=b[0], se=se[0], t=t[0], r2_within=r2))
                # does it add to run differential?
                b0, _, _, r2_base = fe_ols(df, "win_pct", ["run_diff"])
                b2, se2, t2, r2_full = fe_ols(df, "win_pct", ["run_diff", kind + "_z"])
                reg_rows.append(dict(version=label, sample=sample, score=kind, model="win% ~ run_diff + score (score per SD)", coef=b2[1], se=se2[1], t=t2[1], r2_within=r2_full, r2_without_score=r2_base))
            # the same correlations after removing each team's usual level (a team against its own other seasons)
            for out in OUTCOMES:
                r, p = corr_with_p(two_way_demean(df, "total"), two_way_demean(df, out))
                corr_rows.append(dict(version=label, sample=sample, score="total, season and team adjusted", outcome=out, n=len(df), r=r, p=p))
            # wins per run of score (runs versions only)
            if VERSIONS[label][1]:
                b, se, t, r2 = fe_ols(df, "W", ["total"])
                reg_rows.append(dict(version=label, sample=sample, score="total", model="wins ~ runs of score (wins per run)", coef=b[0], se=se[0], t=t[0], r2_within=r2))
                b, se, t, r2 = fe_ols(df, "R", ["total"])
                reg_rows.append(dict(version=label, sample=sample, score="total", model="runs scored ~ runs of score", coef=b[0], se=se[0], t=t[0], r2_within=r2))
        s_tot, n1 = stability(full, "total_z")
        s_per, n2 = stability(full, "per_chance_z")
        reg_rows.append(dict(version=label, sample="all seasons (consecutive pairs only)", score="total", model="year-to-year correlation of the team score", coef=s_tot, se=np.nan, t=np.nan, r2_within=np.nan))
        reg_rows.append(dict(version=label, sample="all seasons (consecutive pairs only)", score="per_chance", model="year-to-year correlation of the team score", coef=s_per, se=np.nan, t=np.nan, r2_within=np.nan))

    corr = pd.DataFrame(corr_rows)
    reg = pd.DataFrame(reg_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(OUT_DIR / "correlations.csv", index=False)
    reg.to_csv(OUT_DIR / "regressions.csv", index=False)

    pd.set_option("display.width", 220)
    print("Correlation of the team score with each outcome (within season; all seasons, n = team-seasons)")
    for kind in ("total", "per_chance"):
        t = corr[(corr["sample"] == "all seasons") & (corr["score"] == kind)].pivot(index="version", columns="outcome", values="r")[OUTCOMES]
        print(f"\n  score = {kind}"); print(t.reindex(list(VERSIONS)).round(3).to_string())
    print("\n  score = total, after removing each team's usual level (a team against its own other seasons)")
    t = corr[corr["score"] == "total, season and team adjusted"].pivot(index="version", columns="outcome", values="r")[OUTCOMES]
    print(t.reindex(list(VERSIONS)).round(3).to_string())
    print("\nSignificance (p) of the correlation with win%, total score")
    t = corr[(corr["score"] == "total") & (corr["outcome"] == "win_pct")].set_index("version")["p"]
    print("  ", {k: round(v, 4) for k, v in t.reindex(list(VERSIONS)).items()})
    best = load("runs, flat values, smooth odds")
    best["fifth"] = pd.cut(best.groupby("season")["total"].rank(pct=True), [0, .2, .4, .6, .8, 1.0], labels=["lowest", "2nd", "3rd", "4th", "highest"])
    q = best.groupby("fifth", observed=True).agg(team_seasons=("win_pct", "size"), win_pct=("win_pct", "mean"), run_diff=("run_diff", "mean"), score_runs=("total", "mean"))
    q.to_csv(OUT_DIR / "by_fifth_flat_smooth.csv")
    print("\nTeam-seasons in fifths of the score (runs, flat values, smooth odds), within season"); print(q.round(3).to_string())
    print("\nRegressions (season fixed effects)")
    show = reg[(reg["sample"] == "all seasons") & (reg["score"] == "total")]
    print(show[["version", "sample", "model", "coef", "se", "t", "r2_within"]].round(4).to_string(index=False))
    print("\nStability (year-to-year correlation of the team score)")
    print(reg[reg["model"].str.startswith("year-to-year")][["version", "score", "coef"]].round(3).to_string(index=False))
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Do teams with higher extra-bases scores win more?")
    ap.add_argument("--regular-season", action="store_true",
                    help="use the team files built without postseason chances (written to winning_regularseason/)")
    if ap.parse_args().regular_season:
        REGULAR_SEASON = True
        OUT_DIR = STAT_DIR / "winning_regularseason"
    main()
