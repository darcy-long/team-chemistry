"""
Extra Bases Taken

Baserunning indicator. Looks at every chance a runner had to take an extra base:
    - runner on 1st, single  -> stop at 2nd, or go to 3rd (or score)
    - runner on 2nd, single  -> stop at 3rd, or score
    - runner on 1st, double  -> stop at 3rd, or score
A runner counts as going for it if he took the base OR was thrown out trying,
so a team isn't punished for being aggressive.

Each chance is compared to how often runners go in the same situation (hit type,
outs, how deep the ball was, gap ball or not, runner speed decile). A team's
score is how much more often its runners went than expected. See models.py.

Input:  data/raw/retrosheet/YYYYplays.parquet  (pull_retrosheet.py)
        sprint speed + crosswalk               (pull_sprint_speed.py, player_crosswalk.py)
Output: data/processed/extra_bases.csv

Usage:
    python src/extra_bases.py                 # the settings used for the project
    python src/extra_bases.py --diagnostics   # also print bucket sizes first
    python src/extra_bases.py --no-depth --no-gap --no-sprint-speed --by-season   # try other settings
Runs with non-default settings save to extra_bases_experiment.csv, not extra_bases.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from models import aggregate_team_season, compute_residuals, compute_shrunk_expected_rate
from player_crosswalk import get_crosswalk
from pull_retrosheet import SEASONS, pull_season
from pull_sprint_speed import pull_all_seasons
from team_codes import RETROSHEET_TO_STATCAST

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MIN_OPPORTUNITIES = 50
SHRINKAGE_M = 20.0
BASE_BUCKETS = ["hittype", "outs_pre", "opportunity_type"]

# name, hit column, column with the runner's ID before the play,
# column that holds the ID of a runner who scored from that base,
# pattern in the play text for "thrown out trying for the extra base",
# (optional) column holding who ended up on 3rd
OPPORTUNITIES = [
    ("1st_on_single", "single", "br1_pre", "run1", r"1X[3H]", "br3_post"),
    ("2nd_on_single", "single", "br2_pre", "run2", r"2XH", None),
    ("1st_on_double", "double", "br1_pre", "run1", r"1XH", None),
]


def find_opportunities(plays):
    """One row per baserunning chance, with took_extra_base = 1 or 0."""
    frames = []
    for name, hit_col, runner_col, scored_col, out_pattern, third_col in OPPORTUNITIES:
        opp = plays[(plays[hit_col] == 1) & plays[runner_col].notna()].copy()
        opp["opportunity_type"] = name
        opp["runner_id"] = opp[runner_col]

        went = (opp[runner_col] == opp[scored_col]) | opp["event"].str.contains(out_pattern, na=False)
        if third_col:
            went |= opp[runner_col] == opp[third_col]
        opp["took_extra_base"] = went.astype(int)
        frames.append(opp)

    opps = pd.concat(frames, ignore_index=True)
    opps["team"] = opps["batteam"].map(RETROSHEET_TO_STATCAST).fillna(opps["batteam"])

    # How deep the ball was fielded (S, M, MD, D, XD), parsed from Retrosheet's loc.
    # Ground balls have no outfield depth, so they get "Infield".
    opps["depth"] = opps["loc"].str.extract(r"(XD|D|MD|M|S)", expand=False).fillna("Infield")
    # Balls hit to the gaps (78 = left-center, 89 = right-center) give runners more time
    opps["is_gap"] = opps["loc"].str.contains(r"^(?:78|89)", na=False, regex=True)
    opps["hittype"] = opps["hittype"].fillna("Unknown")
    return opps


def add_sprint_speed_decile(opps):
    """Attach each runner's sprint speed decile (1 = slowest tenth, within season).
    Retrosheet ID -> MLBAM ID (crosswalk) -> that season's sprint speed.
    Runners we can't match are dropped, since we can't tell how fast they are."""
    xwalk = get_crosswalk()
    speed = pull_all_seasons(sorted(opps["season"].unique()))

    opps = opps.merge(xwalk, left_on="runner_id", right_on="key_retro", how="left")
    print(f"Crosswalk matched {opps['key_mlbam'].notna().mean():.1%} of opportunities")

    opps = opps.merge(
        speed[["season", "player_id", "sprint_speed"]],
        left_on=["season", "key_mlbam"],
        right_on=["season", "player_id"],
        how="left",
    )
    print(f"Sprint speed matched {opps['sprint_speed'].notna().mean():.1%} of opportunities")

    opps["sprint_speed_decile"] = opps.groupby("season")["sprint_speed"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates="drop")
    )
    return opps.dropna(subset=["sprint_speed_decile"])


def check_bucket_sizes(opps, bucket_cols, min_size=20):
    """Print how many situations have fewer than min_size opportunities.
    More columns = more precise situations but thinner samples, so compare a few before choosing."""
    sizes = opps.groupby(bucket_cols).size()
    thin = (sizes < min_size).sum()
    print(f"{bucket_cols}")
    print(f"  {len(sizes)} situations, {thin} ({thin / len(sizes):.1%}) under {min_size} opportunities, "
          f"median size {sizes.median():.0f}, smallest {sizes.min()}\n")


def compute(use_depth=True, use_gap=True, use_sprint_speed=True, pool_seasons=True,
            bucket_cols=None, show_diagnostics=False, shrinkage_m=SHRINKAGE_M,
            min_opportunities=MIN_OPPORTUNITIES, return_opportunities=False):
    """Plays -> opportunities -> expected rate -> residuals -> one score per team-season.

    use_depth        : add ball depth (shallow ... extra deep) to the situation
    use_gap          : add gap ball (yes/no) to the situation
    use_sprint_speed : add the runner's sprint speed decile (needs the crosswalk + sprint speed
                       files; runners with no match are dropped)
    pool_seasons     : True = expected rates use all seasons together (bigger samples, but rule
                       changes like the 2023 bigger bases could shift the baseline).
                       False = each season gets its own expected rates (thinner samples).
    bucket_cols      : set the situation columns by hand; overrides the three use_ switches
    show_diagnostics : print bucket sizes for a few column choices before running
    return_opportunities : also return the per-play table (with residuals)

    The defaults are the settings used for the project's extra_bases.csv.
    """
    plays = []
    for year in SEASONS:
        df = pull_season(year)
        df["season"] = year
        plays.append(df)
    opps = find_opportunities(pd.concat(plays, ignore_index=True))

    if use_sprint_speed:
        opps = add_sprint_speed_decile(opps)

    if show_diagnostics:
        check_bucket_sizes(opps, BASE_BUCKETS)
        check_bucket_sizes(opps, BASE_BUCKETS + ["depth"])
        check_bucket_sizes(opps, BASE_BUCKETS + ["depth", "is_gap"])
        if use_sprint_speed:
            check_bucket_sizes(opps, BASE_BUCKETS + ["depth", "is_gap", "sprint_speed_decile"])
        if not pool_seasons:
            check_bucket_sizes(opps, ["season"] + BASE_BUCKETS)

    if bucket_cols is None:
        bucket_cols = BASE_BUCKETS + ["depth"] * use_depth + ["is_gap"] * use_gap \
            + ["sprint_speed_decile"] * use_sprint_speed
    print(f"Situation columns: {bucket_cols}")

    if pool_seasons:
        opps = compute_shrunk_expected_rate(opps, "took_extra_base", bucket_cols, m=shrinkage_m)
    else:
        opps = pd.concat(
            [compute_shrunk_expected_rate(g, "took_extra_base", bucket_cols, m=shrinkage_m)
             for _, g in opps.groupby("season")],
            ignore_index=True,
        )

    opps = compute_residuals(opps, "took_extra_base")
    scores = aggregate_team_season(opps, "team", "season", min_opportunities=min_opportunities)
    return (scores, opps) if return_opportunities else scores


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute Extra Bases Taken.")
    ap.add_argument("--diagnostics", action="store_true", help="print bucket sizes first")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--no-gap", action="store_true")
    ap.add_argument("--no-sprint-speed", action="store_true")
    ap.add_argument("--by-season", action="store_true", help="separate expected rates for each season")
    args = ap.parse_args()

    scores = compute(
        use_depth=not args.no_depth,
        use_gap=not args.no_gap,
        use_sprint_speed=not args.no_sprint_speed,
        pool_seasons=not args.by_season,
        show_diagnostics=args.diagnostics,
    )

    is_default = not (args.no_depth or args.no_gap or args.no_sprint_speed or args.by_season)
    out_path = OUT_DIR / ("extra_bases.csv" if is_default else "extra_bases_experiment.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_path, index=False)
    print(f"Saved {len(scores)} team-seasons to {out_path}")
