"""
Compare our runner numbers to Baseball Savant's published baserunning leaderboard.

Savant's download (one row per runner, one file per season) goes in data/raw/savant/ as
base_running_YYYY.csv. We join on MLBAM id (Savant's entity_id) and compare, for each season
that both have:
    chances        our n_opportunities     vs Savant n_opp_xb
    attempts       our n_attempts          vs Savant n_att_xb
    thrown out     our n_thrown_out        vs Savant n_out
    value          our net_bases_above_expected (bases)  vs Savant runner_runs (runs)
                   Pearson and rank correlation. They are in different units (bases vs runs),
                   so only the ordering and linear relationship are meaningful, not the size.
and print the top of both lists side by side. If the speed-neutral hustle file exists
(extra_bases_runners_hustle.csv) it also compares the two attempt models directly: how many
attempts each model expected from a runner (Savant: est_rate_att_generic_runner x chances, the
sigmoid on runner time vs fielder time; ours: the shrunk bucket rate) and the attempts above
expected each produces.

Compare it against the speed-neutral version (the default output). Savant's stat doesn't
use the runner's own speed.

Input:  data/processed/extra_bases_stat/extra_bases_runners.csv (postseason included; --regular-season
        uses variants/extra_bases_runners_regularseason.csv), data/raw/savant/base_running_YYYY.csv
Output: data/processed/extra_bases_stat/savant_comparison/savant_comparison_YYYY.csv  (one row per runner in both)

Usage (from the repo root):
    python src/extra_bases_stat/compare_to_savant.py            # every season with a Savant file
    python src/extra_bases_stat/compare_to_savant.py 2025
    python src/extra_bases_stat/compare_to_savant.py 2025 --min-chances 80
"""

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SAVANT_DIR = REPO / "data" / "raw" / "savant"
STAT_DIR = REPO / "data" / "processed" / "extra_bases_stat"


def stat_dir(suffix, without_batter_for_3=False):
    """The default outputs are in extra_bases_stat/, any other settings in extra_bases_stat/variants/.
    The version that leaves out batter_for_3 lives in extra_bases_stat/without_batter_for_3/."""
    root = STAT_DIR / "without_batter_for_3" if without_batter_for_3 else STAT_DIR
    return root / "variants" if suffix else root


def load_savant(year):
    path = SAVANT_DIR / f"base_running_{year}.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.rename(columns={"entity_id": "mlbam_id", "entity_name": "savant_name"})


def rank_corr(a, b):
    """Spearman correlation (Pearson on ranks)."""
    return a.rank().corr(b.rank())


def compare(year, ours, min_chances=50):
    """Join our runner-season rows to Savant's for one season. Returns the joined table."""
    savant = load_savant(year)
    mine = ours[ours["season"] == year]
    joined = mine.merge(savant, on="mlbam_id", how="inner", suffixes=("", "_savant"))
    joined = joined.rename(columns={"n_opp_xb": "savant_chances", "n_att_xb": "savant_attempts",
                                    "n_out": "savant_thrown_out", "runner_runs": "savant_runner_runs"})
    return joined, len(mine), len(savant)


def report_attempt_models(both, suffix="", without_batter3=False):
    """Compare expected attempts and attempts above expected, ours (hustle, speed-neutral) vs Savant's."""
    path = stat_dir("_hustle" + suffix, without_batter3) / f"extra_bases_runners_hustle{suffix}.csv"
    if not path.exists():
        return
    hustle = pd.read_csv(path)
    hustle = hustle[hustle["season"] == both["season"].iloc[0]][["mlbam_id", "expected", "net_bases_above_expected"]]
    m = both[["mlbam_id", "savant_chances", "savant_attempts", "est_rate_att_generic_runner"]].merge(
        hustle, on="mlbam_id")
    sav_expected = m["est_rate_att_generic_runner"] * m["savant_chances"]
    sav_above = m["savant_attempts"] - sav_expected
    print(f"  attempt models (hustle, {len(m)} runners)")
    print(f"    expected attempts   ours {m['expected'].sum():7.0f}  Savant sigmoid {sav_expected.sum():7.0f}  "
          f"per-runner correlation {m['expected'].corr(sav_expected):.3f}")
    print(f"    attempts above expected  Pearson {m['net_bases_above_expected'].corr(sav_above):.3f}   "
          f"rank {rank_corr(m['net_bases_above_expected'], sav_above):.3f}")


def report(year, ours, min_chances=50, suffix="", without_batter3=False):
    joined, n_ours, n_savant = compare(year, ours, min_chances)
    print(f"\n===== {year} =====")
    print(f"Runners: ours {n_ours}, Savant {n_savant}, in both {len(joined)}")

    both = joined[joined["savant_chances"] >= min_chances]
    print(f"Runners with {min_chances}+ Savant chances in both: {len(both)}")

    for label, mine, theirs in [
        ("chances", "n_opportunities", "savant_chances"),
        ("attempts", "n_attempts", "savant_attempts"),
        ("thrown out", "n_thrown_out", "savant_thrown_out"),
    ]:
        print(f"  {label:11s} ours {both[mine].sum():8.0f}  Savant {both[theirs].sum():8.0f}  "
              f"correlation per runner {both[mine].corr(both[theirs]):.3f}")

    x, y = both["net_bases_above_expected"], both["savant_runner_runs"]
    print(f"  value        Pearson {x.corr(y):.3f}   rank {rank_corr(x, y):.3f}   "
          f"(our net bases vs Savant runner runs)")

    report_attempt_models(both, suffix, without_batter3)

    cols = ["name", "savant_chances", "n_opportunities", "net_bases_above_expected", "savant_runner_runs"]
    show = both[cols].rename(columns={"savant_chances": "S_chances", "n_opportunities": "our_chances",
                                      "net_bases_above_expected": "our_bases", "savant_runner_runs": "S_runs"})
    print("\n  Top 8 by Savant runner runs:")
    print(show.sort_values("S_runs", ascending=False).head(8).round(2).to_string(index=False))
    print("\n  Top 8 by our net bases:")
    print(show.sort_values("our_bases", ascending=False).head(8).round(2).to_string(index=False))
    return joined


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare runner values to the Baseball Savant leaderboard.")
    ap.add_argument("years", nargs="*", type=int, help="seasons to compare (default: every Savant file we have)")
    ap.add_argument("--min-chances", type=int, default=50, help="minimum Savant chances to include a runner")
    ap.add_argument("--without-batter-for-3", action="store_true",
                    help="use the outputs that leave out batter_for_3 (the without_batter_for_3 folder)")
    ap.add_argument("--regular-season", action="store_true",
                    help="use the outputs that left playoff games out (the _regularseason files)")
    args = ap.parse_args()

    suffix = "_regularseason" if args.regular_season else ""
    runners_file = f"extra_bases_runners{suffix}.csv"
    ours = pd.read_csv(stat_dir(suffix, args.without_batter_for_3) / runners_file)
    available = sorted(int(p.stem.split("_")[-1]) for p in SAVANT_DIR.glob("base_running_*.csv"))
    years = args.years or available
    for year in years:
        if year not in set(ours["season"]):
            print(f"\n===== {year} =====\nNo {year} season in {runners_file} "
                  f"(our play-by-play covers {int(ours['season'].min())}-{int(ours['season'].max())}); skipping")
            continue
        joined = report(year, ours, args.min_chances, suffix, args.without_batter_for_3)
        (STAT_DIR / "savant_comparison").mkdir(exist_ok=True)
        path = STAT_DIR / "savant_comparison" / f"savant_comparison_{year}{suffix}.csv"
        joined.to_csv(path, index=False)
        print(f"\n  Saved {len(joined)} runners to {path}")
