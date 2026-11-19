"""
Extra bases stat: Statcast batted-ball buckets, all six kinds of chance. A pure stat for now, not connected to extra_bases.py or the Hustle Index.

Same idea as extra_bases.py (did the runner go for the extra base more often than
runners usually do in the same situation?), but the Retrosheet "how deep was the ball"
and "gap ball" buckets are replaced with where the ball actually was, from Statcast
hit coordinates (hc_x, hc_y) in plays_statcast (play_pitch_join.py), and the batter's
chances and sac flies are added (see opportunities.py for the six kinds).

The situation is:
    hit type, outs, opportunity type          (as before)
    throw distance bin                        how far the fielded ball was from the base the
                                              runner was heading for, in feet. Quantile bins
                                              within each opportunity type
    fielding position (off by default)        who fielded it: LF, left-center, CF, right-center,
                                              RF, or infield, from Retrosheet's loc.
                                              Tested out of sample and it made the model worse
                                              (too many thin situations), so it is opt-in
    runner's own sprint speed decile (opt-in) the runner's own speed, so the score is about
                                              aggressiveness given his ability. Off by default:
                                              a published leaderboard is speed-neutral (same play,
                                              same outcome, same credit regardless of how fast the
                                              runner is), so the default is comparable to that
The expected value of a chance is the average result of its situation. (--shrinkage M pulls small
situations toward the average of their kind of chance, see models.py; it is off by default.)

What is rewarded (reward_hustle):
    False (default)  the result: safe extra base = +1, holding = 0, thrown out = -out_penalty
                     (default 1.0, so an out cancels one extra base). Closer to a published
                     leaderboard, which credits bases gained and charges for outs. It is NOT
                     the same number: those use run values and their own go/no-go model.
    True  (--hustle) the attempt: going = 1 even if thrown out, holding = 0. This is the
                     project's hustle stat.
The expected value is computed the same way in both cases (the average of that outcome in the
situation).

Score: each opportunity's residual is outcome - expected. Per team-season we report the
mean residual for each kind of chance (score_<type>), the team's total (net_bases_above_expected,
or runs_above_expected in the run version: the sum of all its residuals), and per_chance (that
total divided by n_opportunities). Use these for comparing teams. raw_score is the average of the
score_<type> columns, each kind counting equally: it is only a summary of the six kinds, and it is
NOT the total divided by the chances (tiny kinds such as 3rd_on_fly weigh as much as batter_for_2).

Smooth odds (default, --odds smooth): the expected odds come from smooth_model.py (gradient-boosted
trees fitted out of sample by season; see README 3c). --odds buckets uses the situation buckets instead,
and writes to variants/ with _buckets added.

Flat values (--runs --values flat): the same scoring, but an extra base is worth +0.25 runs and a thrown out costs
0.70 in every base-out state, instead of the run expectancy of the actual state. Savant's numbers behave this way
(an inference, see README 3b). Outputs get _flat in variants/.

Run values (--runs): instead of bases, each chance is scored in runs. The situation buckets give the odds
of each result (safe, held, thrown out); run_values.py gives what each result is worth for that chance's
own base-out state (the runner's result changed, everyone else's kept; values from a per-season run
expectancy table, run_expectancy.py). score = value of what happened - sum(odds * value), and the
unit becomes runs: runs_above_expected, runs_<kind>. Every chance has a run value: the plays that would not
(the ones that end the game) and the runner-on-1st singles that a runner ahead blocked are dropped from every
version when the chances are built (see opportunities.py). Outputs go in variants/ with _runs added to the file
names. Hustle and out-penalty are bases-only settings and can't be combined with it.

Postseason games are included by default (about 1.4% of chances). --regular-season leaves them out,
which matches standings.py and DEG, both regular season only.

Opportunities with no Statcast hit location are dropped (and, with --own-speed, those with no
sprint speed).

Player level: runners are also scored on their own. extra_bases_runners.csv has one row per
runner-season, with net_bases_above_expected = the sum of that runner's residuals (a leaderboard
number: net bases gained over what a typical runner would have on the same plays), and
extra_bases_runners_career.csv sums across all seasons.

Input:  data/processed/plays_statcast/YYYY.parquet  (play_pitch_join.py)
Output: data/processed/extra_bases_stat/extra_bases_statcast.csv           team-seasons
        data/processed/extra_bases_stat/extra_bases_runners.csv            runner-seasons
        data/processed/extra_bases_stat/extra_bases_runners_career.csv     runners, all seasons
        (the default settings: rewarding results, speed-neutral, postseason included.)
        Any other settings go in data/processed/extra_bases_stat/variants/: --hustle adds _hustle to
        the file names, --own-speed adds _ownspeed and --regular-season adds _regularseason, e.g.
        variants/extra_bases_statcast_hustle_ownspeed.csv

Usage (from the repo root):
    python src/extra_bases_stat/extra_bases_statcast.py                  # the default: results, speed-neutral
    python src/extra_bases_stat/extra_bases_statcast.py --evaluate       # compare bucket choices out of sample first
    python src/extra_bases_stat/extra_bases_statcast.py --position --bins 8
    python src/extra_bases_stat/extra_bases_statcast.py --runs           # score in runs, not bases
    python src/extra_bases_stat/extra_bases_statcast.py --hustle         # reward attempts, not results
    python src/extra_bases_stat/extra_bases_statcast.py --regular-season    # leave the playoffs out
    python src/extra_bases_stat/extra_bases_statcast.py --own-speed      # compare each runner to runners of his own speed
    python src/extra_bases_stat/extra_bases_statcast.py --hustle --own-speed   # the hustle stat
    python src/extra_bases_stat/extra_bases_statcast.py --out-penalty 1.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for the shared modules

from player_crosswalk import get_crosswalk, get_names
from extra_bases import BASE_BUCKETS, MIN_OPPORTUNITIES, add_sprint_speed_decile
from models import compute_residuals, compute_shrunk_expected_rate
from opportunities import find_all_opportunities
from pull_retrosheet import SEASONS
from run_expectancy import get_re_table
from run_values import FLAT_EXTRA_BASE, FLAT_OUT, add_run_values, apply_flat_values, check_construction

# Shrinkage strength: 0 = each situation's expected value is just its own average. Tested on held-out
# seasons, it made no difference to the default stat (0.1% of plays are in a situation with under 20
# plays), so it is off. Finer situations (--own-speed, --position) are thinner and can use --shrinkage.
SHRINKAGE_M = 0.0

REPO = Path(__file__).resolve().parents[2]
PLAYS_DIR = REPO / "data" / "processed" / "plays_statcast"
OUT_DIR = REPO / "data" / "processed" / "extra_bases_stat"   # the default outputs; variants go in OUT_DIR/variants
# All six kinds are counted by default. --exclude-batter-for-3 leaves out the batter-going-for-3 chances (we
# count about 11% more chances than Savant and it is the best match for the gap, but dropping it lowers
# agreement with Savant; see the README's open question) and writes to OUT_DIR/without_batter_for_3 so the
# two versions never overwrite each other.
DEFAULT_EXCLUDED_KINDS = ()

# Statcast hc_x / hc_y are pixels with home plate at (125.42, 198.27); about 2.5 feet per pixel
HOME_X, HOME_Y, FEET_PER_PIXEL = 125.42, 198.27, 2.5
BASE_FEET = 90 / np.sqrt(2)  # a base's x/y offset from home plate, in feet
# The base the runner is going to (x, y in feet, home plate at 0,0, +y toward center field)
TARGET_BASE_XY = {
    "1st_on_single": (-BASE_FEET, BASE_FEET),   # third base
    "2nd_on_single": (0.0, 0.0),                # home
    "1st_on_double": (0.0, 0.0),                # home
    "batter_for_2": (0.0, 2 * BASE_FEET),       # second base
    "batter_for_3": (-BASE_FEET, BASE_FEET),    # third base
    "3rd_on_fly": (0.0, 0.0),                   # home
}

PLAY_COLUMNS = [
    "gid", "event", "inning", "top_bot", "batteam", "batter", "single", "double", "triple",
    "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post", "br3_post", "run_b", "run1", "run2", "run3",
    "outs_pre", "outs_post", "runs", "loc", "hittype", "sc_status", "sc_hc_x", "sc_hc_y",
    "sc_launch_speed", "sc_launch_angle", "gametype",
    "score_v", "score_h",                      # for dropping walk-offs (opportunities.drop_undecided_plays)
    "bathand", "site", "sc_hit_distance_sc",   # only used by smooth_model.py
]


def regular_season_only(plays):
    """Drop playoff games (Retrosheet gametype: divisionseries, lcs, worldseries, wildcard, playoff)."""
    return plays[plays["gametype"] == "regular"]


def load_plays(include_postseason=True):
    frames = []
    for year in SEASONS:
        df = pd.read_parquet(PLAYS_DIR / f"{year}.parquet", columns=PLAY_COLUMNS)
        df["season"] = year
        frames.append(df)
    plays = pd.concat(frames, ignore_index=True)
    return plays if include_postseason else regular_season_only(plays)


def add_batted_ball_location(opps, n_bins=5):
    """Add throw_dist (feet from the fielded ball to the target base), a quantile bin of it
    within each opportunity type, and a left/center/right spray direction."""
    opps = opps[(opps["sc_status"] == "ok") & opps["sc_hc_x"].notna() & opps["sc_hc_y"].notna()].copy()
    x = (opps["sc_hc_x"] - HOME_X) * FEET_PER_PIXEL
    y = (HOME_Y - opps["sc_hc_y"]) * FEET_PER_PIXEL

    target = opps["opportunity_type"].map(TARGET_BASE_XY)
    opps["throw_dist"] = np.hypot(x - target.str[0], y - target.str[1])

    opps["throw_dist_bin"] = opps.groupby("opportunity_type")["throw_dist"].transform(
        lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop")
    )
    # Angle off straightaway center: negative = left field side, positive = right field side
    angle = np.degrees(np.arctan2(x, y))
    opps["spray"] = pd.cut(angle, [-180, -15, 15, 180], labels=["left", "center", "right"]).astype(str)
    return opps


OUTFIELD_ZONES = {"7": "LF", "78": "LC", "8": "CF", "89": "RC", "9": "RF"}  # 78 / 89 are the gaps


def add_fielder_position(opps):
    """Add fielder_pos: who fielded the ball, from the digits at the start of Retrosheet's loc
    (7 = LF, 78 = left-center gap, 8 = CF, 89 = right-center gap, 9 = RF; 1-6 and combos
    like 56 or 34 are the infield). Plays with no loc are "Unknown"."""
    zone = opps["loc"].str.extract(r"^(\d+)", expand=False)
    opps = opps.copy()
    fallback = pd.Series(np.where(zone.notna(), "IF", "Unknown"), index=zone.index)
    opps["fielder_pos"] = zone.map(OUTFIELD_ZONES).fillna(fallback)
    return opps


GRAVITY = 32.174        # ft/s^2
CONTACT_HEIGHT = 3.0    # ft
LOW_BALL_ANGLE = 10     # below this launch angle (ground balls, low liners) hang time isn't a useful idea


def add_hang_time(opps, n_bins=4):
    """Add hang_time (seconds, rough) and hang_bin.

    Hang time is projectile physics from exit velocity and launch angle: no air drag, ball
    leaves the bat 3 ft off the ground. That overstates real hang time on long flies, but it
    keeps the right order (popups > high flies > line drives), which is all quantile bins use.
        hang_bin  -1 = low ball (launch angle under 10: ground balls, low liners; or bunts)
                  -2 = no launch data
                  0.. = quantile bins of hang time among the rest, 0 = shortest, e.g. line drives
    Tested as a bucket column and it made the model worse out of sample, so it is not in the default.
    """
    opps = opps.copy()
    speed_fps = opps["sc_launch_speed"] * 5280 / 3600
    angle = np.radians(opps["sc_launch_angle"])
    vy = speed_fps * np.sin(angle)
    opps["hang_time"] = (vy + np.sqrt(vy ** 2 + 2 * GRAVITY * CONTACT_HEIGHT)) / GRAVITY

    has_launch = opps["sc_launch_speed"].notna() & opps["sc_launch_angle"].notna()
    high = has_launch & (opps["sc_launch_angle"] >= LOW_BALL_ANGLE)
    opps["hang_bin"] = -2
    opps.loc[has_launch & ~high, "hang_bin"] = -1
    opps.loc[high, "hang_bin"] = pd.qcut(opps.loc[high, "hang_time"], n_bins, labels=False, duplicates="drop")
    return opps


def drop_kinds(opps, kinds):
    """Leave out whole kinds of chance (e.g. ("batter_for_3",))."""
    return opps[~opps["opportunity_type"].isin(kinds)] if kinds else opps


def build_opportunities(n_bins=5, hang_bins=4, with_speed=False, include_postseason=True,
                        exclude_kinds=DEFAULT_EXCLUDED_KINDS):
    """with_speed=True also adds sprint_speed_decile and drops runners with no sprint speed."""
    opps = drop_kinds(find_all_opportunities(load_plays(include_postseason)), exclude_kinds)
    n_all = len(opps)
    opps = add_batted_ball_location(opps, n_bins=n_bins)
    print(f"{len(opps):,} of {n_all:,} opportunities have a Statcast hit location")
    opps = add_fielder_position(opps)
    opps = add_hang_time(opps, n_bins=hang_bins)
    return add_sprint_speed_decile(opps) if with_speed else opps


def add_outcome(opps, reward_hustle=False, out_penalty=1.0):
    """Add the outcome column that gets scored. See the module docstring.
    (The model-comparison code always predicts the plain go / hold choice, took_extra_base.)"""
    opps = opps.copy()
    if reward_hustle:
        opps["outcome"] = opps["took_extra_base"].astype(float)
    else:
        safe = opps["took_extra_base"] - opps["thrown_out"]
        opps["outcome"] = safe - out_penalty * opps["thrown_out"]
    return opps


def leave_one_season_out_expected(opps, bucket_cols, m=SHRINKAGE_M, outcome_col="took_extra_base"):
    """Out-of-sample expected value for every opportunity: each season's expectation comes from the
    OTHER seasons only, so a play never helps set its own baseline and a model with more or finer
    situations isn't rewarded for memorizing its own data. Shrinks toward the average of the
    opportunity's own kind. outcome_col is what is being predicted (the go/hold choice by default;
    use "outcome" for the scored result)."""
    expected = pd.Series(np.nan, index=opps.index)
    for season in sorted(opps["season"].unique()):
        train = opps[opps["season"] != season]
        test = opps[opps["season"] == season]
        league = train.groupby("opportunity_type")[outcome_col].mean()
        buckets = train.groupby(bucket_cols)[outcome_col].agg(n="count", rate="mean").reset_index()
        buckets["p"] = (buckets["n"] * buckets["rate"] + m * buckets["opportunity_type"].map(league)) \
            / (buckets["n"] + m)
        p = test[bucket_cols].merge(buckets[bucket_cols + ["p"]], on=bucket_cols, how="left")["p"].to_numpy()
        # a situation never seen in the other seasons falls back to the average of its kind (as arrays: the merge
        # result has a fresh 0..n index, so filling by index would pair the fallback with the wrong plays)
        fallback = test["opportunity_type"].map(league).to_numpy()
        expected.loc[test.index] = np.where(np.isnan(p), fallback, p)
    return expected


def log_loss(y, p):
    p = p.clip(1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def evaluate(opps, m=SHRINKAGE_M):
    """Compare bucket choices on the same opportunities, out of sample (lower log loss is better).
    Shows the overall number and each kind of chance, since the rare kinds barely move the overall.
    Needs sprint_speed_decile (build_opportunities(with_speed=True))."""
    speed = ["sprint_speed_decile"]
    dist = BASE_BUCKETS + ["throw_dist_bin"]
    options = {
        "no location": BASE_BUCKETS,
        "throw distance (default)": dist,
        "throw distance + own speed": dist + speed,
        "throw distance + fielding position": dist + ["fielder_pos"],
        "throw distance + hang time": dist + ["hang_bin"],
    }
    types = sorted(opps["opportunity_type"].unique())
    print("\nOut-of-sample log loss (leave one season out; lower is better)")
    print(f"  {'':40s}{'overall':>9s}" + "".join(f"{t:>15s}" for t in types) + "   situations")
    for name, cols in options.items():
        p = leave_one_season_out_expected(opps, cols, m=m)
        row = f"  {name:40s}{log_loss(opps['took_extra_base'], p):9.4f}"
        for t in types:
            mask = opps["opportunity_type"] == t
            row += f"{log_loss(opps.loc[mask, 'took_extra_base'], p[mask]):15.4f}"
        print(row + f"   {len(opps.groupby(cols))}")


def aggregate_by_type(opps, min_opportunities=MIN_OPPORTUNITIES, runs=False):
    """One row per team-season: score_<type> and n_<type> for each kind of chance, n_opportunities (all
    kinds together), the team's total (net_bases_above_expected, or runs_above_expected when runs=True:
    the sum of its residuals), per_chance (total / n_opportunities), and raw_score (the plain average of
    the score_<type> columns, which is NOT total / n_opportunities: see the module docstring)."""
    g = opps.groupby(["team", "season", "opportunity_type"])["residual"].agg(["mean", "count"]).reset_index()
    means = g.pivot(index=["team", "season"], columns="opportunity_type", values="mean").add_prefix("score_")
    counts = g.pivot(index=["team", "season"], columns="opportunity_type", values="count").add_prefix("n_")
    out = means.join(counts)
    out["raw_score"] = means.mean(axis=1)
    out["n_opportunities"] = counts.sum(axis=1)
    total_col = "runs_above_expected" if runs else "net_bases_above_expected"
    out[total_col] = opps.groupby(["team", "season"])["residual"].sum()
    out["per_chance"] = out[total_col] / out["n_opportunities"]
    out = out.reset_index()
    dropped = (out["n_opportunities"] < min_opportunities).sum()
    if dropped:
        print(f"Dropping {dropped} team-seasons with under {min_opportunities} opportunities")
    out = out[out["n_opportunities"] >= min_opportunities].reset_index(drop=True)
    first = ["team", "season", "raw_score", "n_opportunities", total_col, "per_chance"]
    return out[first + [c for c in out.columns if c not in first]]


def runner_names():
    """Retrosheet ID -> "First Last", from the Chadwick register (see player_crosswalk.get_names)."""
    return get_names().set_index("key_retro")["name"]


def add_runner_info(opps):
    """Add mlbam_id and name for each runner (name falls back to the Retrosheet ID)."""
    if "key_mlbam" not in opps.columns:
        opps = opps.merge(get_crosswalk(), left_on="runner_id", right_on="key_retro", how="left")
    opps = opps.rename(columns={"key_mlbam": "mlbam_id"})
    opps["name"] = opps["runner_id"].map(runner_names()).fillna(opps["runner_id"])
    return opps


def aggregate_runners(opps, by_season=True, runs=False):
    """One row per runner-season (or per runner, all seasons) with leaderboard numbers.

    net_bases_above_expected  sum of residuals: net bases gained over what a typical runner
                              would have gained on the same plays (a safe extra base is +1,
                              thrown out is -out_penalty in the default mode)
    actual / expected         the two sums it is the difference of
    n_attempts, n_thrown_out, n_safe   how often he went, was thrown out going, and made it
    per_100                   net_bases_above_expected per 100 chances
    above_<type>              the same sum for each kind of chance
    A runner traded mid-season keeps one row; team is where he had the most chances.
    runs=True (the run-value version): the same numbers in runs, with net_bases_above_expected named
    runs_above_expected and above_<type> named runs_<type> (per_100 is runs per 100 chances).
    """
    opps = add_runner_info(opps)
    keys = ["runner_id", "mlbam_id", "name"] + (["season"] if by_season else [])
    # a stable team label for the row: the team with the most chances (first seen if tied)
    team = (opps.groupby(keys + ["team"], dropna=False).size().reset_index(name="k")
            .sort_values("k", ascending=False, kind="stable").drop_duplicates(keys)[keys + ["team"]])

    out = opps.groupby(keys, dropna=False).agg(
        n_opportunities=("residual", "size"),
        n_attempts=("took_extra_base", "sum"),
        n_thrown_out=("thrown_out", "sum"),
        actual=("outcome", "sum"),
        expected=("expected_rate", "sum"),
        net_bases_above_expected=("residual", "sum"),
    ).reset_index()
    out["n_safe"] = out["n_attempts"] - out["n_thrown_out"]
    out["per_100"] = out["net_bases_above_expected"] / out["n_opportunities"] * 100
    by_type = (opps.groupby(keys + ["opportunity_type"], dropna=False)["residual"].sum()
               .unstack("opportunity_type", fill_value=0.0).add_prefix("above_").reset_index())
    out = out.merge(by_type, on=keys).merge(team, on=keys)
    front = ["name", "runner_id", "mlbam_id"] + (["season"] if by_season else []) + ["team"]
    out = out[front + [c for c in out.columns if c not in front]]
    if runs:
        out = out.rename(columns={"net_bases_above_expected": "runs_above_expected"})
        out = out.rename(columns={c: "runs_" + c[len("above_"):] for c in out.columns if c.startswith("above_")})
    sort_col = "runs_above_expected" if runs else "net_bases_above_expected"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def compute(n_bins=5, use_position=False, own_speed=False, reward_hustle=False, out_penalty=1.0,
            include_postseason=True, exclude_kinds=DEFAULT_EXCLUDED_KINDS, shrinkage_m=SHRINKAGE_M,
            min_opportunities=MIN_OPPORTUNITIES, runs=False, odds="smooth",
            values="re24", extra_base_value=FLAT_EXTRA_BASE, out_value=FLAT_OUT, show_evaluation=False, return_opportunities=False):
    if runs and (reward_hustle or out_penalty != 1.0):
        raise ValueError("runs can't be combined with reward_hustle or out_penalty (those score bases)")
    if values not in ("re24", "flat"):
        raise ValueError(f"values must be 're24' or 'flat', not {values!r}")
    if values == "flat" and not runs:
        raise ValueError("values='flat' scores in runs, so it needs runs=True")
    if odds not in ("buckets", "smooth"):
        raise ValueError(f"odds must be 'buckets' or 'smooth', not {odds!r}")
    opps = build_opportunities(n_bins=n_bins, with_speed=own_speed or show_evaluation,
                               include_postseason=include_postseason, exclude_kinds=exclude_kinds)
    if runs:
        opps = add_run_values(opps, get_re_table())
        print("\nRun values: does the built state match the real play?")
        print(check_construction(opps).to_string(), "\n")
        n_before = len(opps)
        opps = opps[opps["rv_ok"] & ~opps["safe_blocked"]].reset_index(drop=True)   # both should be empty now
        print(f"{n_before - len(opps):,} of {n_before:,} chances left out for having no run value or a blocked runner")
        if values == "flat":
            print(f"Flat values: extra base +{extra_base_value:g}, thrown out -{out_value:g}, whatever the base-out state")
            opps = apply_flat_values(opps, extra_base_value, out_value)
    if show_evaluation:
        evaluate(opps, m=shrinkage_m)

    bucket_cols = BASE_BUCKETS + ["throw_dist_bin"] + ["fielder_pos"] * use_position \
        + ["sprint_speed_decile"] * own_speed
    print(f"Situation columns: {bucket_cols}")
    if odds == "smooth":
        # the smooth model's odds (smooth_model.py, needs scikit-learn: run with .venv/bin/python); the rewards are the
        # same as for the buckets, only the expected value changes
        from smooth_model import SMOOTH_GROUPS, smooth_odds
        groups = SMOOTH_GROUPS + (("speed",) if own_speed else ())   # --own-speed: also the runner's sprint speed
        print(f"Odds: smooth model, features {groups} (out of sample by season)")
        sm = smooth_odds(opps, groups=groups, verbose=True)
        opps["p_safe"], opps["p_out"] = sm["p_safe"], sm["p_out"]
        if runs:
            print("Rewarding: run value of the result")
            opps["expected_rate"] = opps["p_safe"] * opps["rv_safe"] + opps["p_out"] * opps["rv_out"]
            opps["outcome"] = opps["rv_actual"]
        elif reward_hustle:
            print("Rewarding: the attempt (thrown out counts as going)")
            opps = add_outcome(opps, reward_hustle=True, out_penalty=out_penalty)
            opps["expected_rate"] = opps["p_safe"] + opps["p_out"]
        else:
            print(f"Rewarding: the result (safe +1, thrown out -{out_penalty:g})")
            opps = add_outcome(opps, reward_hustle=False, out_penalty=out_penalty)
            opps["expected_rate"] = opps["p_safe"] - out_penalty * opps["p_out"]
    elif runs:
        print("Rewarding: run value of the result (odds of safe / held / out from the situation)")
        safe, out = (opps["took_extra_base"] - opps["thrown_out"]).astype(float), opps["thrown_out"].astype(float)
        odds = {}
        for name, outcome in (("safe", safe), ("out", out)):
            tmp = compute_shrunk_expected_rate(opps.assign(_y=outcome), "_y", bucket_cols, m=shrinkage_m,
                                               league_group_col="opportunity_type")
            odds[name] = tmp["expected_rate"].to_numpy()
        opps["p_safe"], opps["p_out"] = odds["safe"], odds["out"]   # kept for compare_components.py
        opps["expected_rate"] = odds["safe"] * opps["rv_safe"] + odds["out"] * opps["rv_out"]
        opps["outcome"] = opps["rv_actual"]
    else:
        print("Rewarding: " + ("the attempt (thrown out counts as going)" if reward_hustle
                               else f"the result (safe +1, thrown out -{out_penalty:g})"))
        opps = add_outcome(opps, reward_hustle=reward_hustle, out_penalty=out_penalty)
        opps = compute_shrunk_expected_rate(opps, "outcome", bucket_cols, m=shrinkage_m,
                                            league_group_col="opportunity_type")
    opps = compute_residuals(opps, "outcome")
    scores = aggregate_by_type(opps, min_opportunities=min_opportunities, runs=runs)
    return (scores, opps) if return_opportunities else scores


def print_summary(scores, opps, out_dir, suffix, runs):
    """What was written and the headline numbers, so a run ends with something to read."""
    unit = "runs" if runs else "bases"
    runners = aggregate_runners(opps, by_season=False, runs=runs)
    value_col = "runs_above_expected" if runs else "net_bases_above_expected"
    qualified = runners[runners["n_opportunities"] >= 300]
    team_col = "runs_above_expected" if runs else "net_bases_above_expected"
    print(f"\nDone. Files in {out_dir} (suffix '{suffix}'):")
    for name in ("extra_bases_statcast", "extra_bases_runners", "extra_bases_runners_career"):
        print(f"  {name}{suffix}.csv")
    print(f"\n{len(opps):,} chances, seasons {opps['season'].min()}-{opps['season'].max()}, "
          f"{len(scores)} team-seasons, {opps['runner_id'].nunique():,} runners")
    print(f"Went for the extra base: {opps['took_extra_base'].mean():.1%}; thrown out trying: "
          f"{opps['thrown_out'].mean():.1%}")
    print(f"\nCareer leaders, {unit} above expected (300+ chances):")
    print(qualified.head(5)[["name", "n_opportunities", value_col]].round(1).to_string(index=False))
    print("\nCareer trailers:")
    print(qualified.tail(5)[["name", "n_opportunities", value_col]].round(1).to_string(index=False))
    print(f"\nBest team-seasons ({team_col}):")
    print(scores.nlargest(5, team_col)[["team", "season", team_col, "n_opportunities"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extra Bases Taken with Statcast batted-ball buckets.")
    ap.add_argument("--bins", type=int, default=5, help="throw distance bins per opportunity type")
    ap.add_argument("--position", action="store_true", help="add fielding position to the situation")
    ap.add_argument("--own-speed", action="store_true",
                    help="add the runner's own sprint speed decile to the situation (compares him to runners "
                         "of his own speed; the default is speed-neutral)")
    ap.add_argument("--regular-season", action="store_true",
                    help="leave playoff games out (the default counts them)")
    ap.add_argument("--exclude-batter-for-3", action="store_true",
                    help="leave out the batter-going-for-3 chances (counted by default, see the README)")
    ap.add_argument("--runs", action="store_true",
                    help="score in runs (run expectancy) instead of bases; outputs get _runs in variants/")
    ap.add_argument("--odds", choices=["buckets", "smooth"], default="smooth",
                    help="where the expected odds come from: the smooth model (default; needs scikit-learn: "
                         "run with .venv/bin/python) or the situation buckets (--odds buckets; outputs get "
                         "_buckets in variants/)")
    ap.add_argument("--values", choices=["re24", "flat"], default="re24",
                    help="with --runs: what each result is worth. re24 (default) = run expectancy of the actual "
                         "base-out state; flat = the same value in every state (see --extra-base-value, --out-value); "
                         "outputs get _flat in variants/")
    ap.add_argument("--extra-base-value", type=float, default=FLAT_EXTRA_BASE,
                    help=f"runs for taking the extra base, with --values flat (default {FLAT_EXTRA_BASE})")
    ap.add_argument("--out-value", type=float, default=FLAT_OUT,
                    help=f"runs lost when thrown out, with --values flat (default {FLAT_OUT})")
    ap.add_argument("--hustle", action="store_true",
                    help="reward attempts instead of results: going = 1 even if thrown out")
    ap.add_argument("--out-penalty", type=float, default=1.0,
                    help="how much being thrown out costs when rewarding results (default 1.0)")
    ap.add_argument("--shrinkage", type=float, default=SHRINKAGE_M,
                    help="pull small situations toward the average of their kind (default 0 = off)")
    ap.add_argument("--evaluate", action="store_true", help="compare bucket choices out of sample first")
    args = ap.parse_args()
    if args.runs and (args.hustle or args.out_penalty != 1.0):
        ap.error("--runs can't be combined with --hustle or --out-penalty (those score bases)")
    if args.values == "flat" and not args.runs:
        ap.error("--values flat scores in runs: add --runs")

    scores, opps = compute(n_bins=args.bins, use_position=args.position, own_speed=args.own_speed,
                           reward_hustle=args.hustle, out_penalty=args.out_penalty,
                           include_postseason=not args.regular_season, shrinkage_m=args.shrinkage, runs=args.runs, odds=args.odds, values=args.values,
                           extra_base_value=args.extra_base_value, out_value=args.out_value,
                           exclude_kinds=("batter_for_3",) if args.exclude_batter_for_3 else DEFAULT_EXCLUDED_KINDS,
                           show_evaluation=args.evaluate, return_opportunities=True)

    if args.bins != 5 or args.position or args.shrinkage != SHRINKAGE_M or (not args.hustle and args.out_penalty != 1.0):
        suffix = "_experiment"  # non-default settings don't overwrite the real output
    else:
        suffix = ("_hustle" if args.hustle else "") + ("_runs" if args.runs else "") + ("_flat" if args.values == "flat" else "") \
            + ("_buckets" if args.odds == "buckets" else "") + ("_ownspeed" if args.own_speed else "") \
            + ("_regularseason" if args.regular_season else "")
    root = OUT_DIR / "without_batter_for_3" if args.exclude_batter_for_3 else OUT_DIR
    out_dir = root / "variants" if suffix else root
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / f"extra_bases_statcast{suffix}.csv", index=False)
    aggregate_runners(opps, runs=args.runs).to_csv(out_dir / f"extra_bases_runners{suffix}.csv", index=False)
    aggregate_runners(opps, by_season=False, runs=args.runs).to_csv(
        out_dir / f"extra_bases_runners_career{suffix}.csv", index=False)
    print_summary(scores, opps, out_dir, suffix, args.runs)
