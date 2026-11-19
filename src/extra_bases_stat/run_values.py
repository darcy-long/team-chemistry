"""
Run value of each outcome of a baserunning chance, read off the run-expectancy table.

For a chance, one runner (the focal runner: the batter, or the runner on a base) decided whether
to take the extra base. Everyone else's result on the play is kept as it actually happened, and only
the focal runner's result is changed:

    held   he stays at the base he would have had without the extra base          (value 0, the baseline)
    safe   he takes the extra base (one base past held; home for runners on 2nd/3rd)
    out    he is thrown out trying

Each state is valued as RE(bases, outs after) + runs scored, and an outcome's run value is that minus
the same for held. rv_safe and rv_out come out for every chance, whatever the runner really did, and
rv_actual is the one that matches what he really did (0 if he held). So, as in the bases version, taking
the base counts the same whether he stopped at 3rd or scored from 1st on a single.

Runners behind him are held to the base the hit gives them (the batter at 1st on a single, the runner on
1st at 2nd, ...) in all three states. Their extra bases on the throw are a result of his decision, and if
they were kept as they really happened, successes would be credited with them while the expected value
(which has to be the same for every outcome) could not be.

Rules for the parts that depend on the focal runner's result:
    - Runners can't share a base. If a runner ahead of him ended up on the base he would take, he is
      held back to the base behind that runner (safe_blocked = True; if that is where he held, safe
      is worth the same as held, and the chance is not really a decision: the scorer leaves it out).
      Runners behind him who ended on or past his base are moved back
      to the base behind him (e.g. the batter who took 2nd on the throw is back at 1st if the
      runner held at 2nd).
    - A third out ends the half-inning. Runs only count for runners ahead of the last runner out
      (runs by runners behind him came too late). Every out here is a tag play, never a force: the
      focal runner, or a batter thrown out trying to stretch a hit. (A batter's out on a fly ball is
      the first out of the play, and sac fly chances need fewer than 2 outs, so it is never the third.)

Check: with every other runner left exactly as he really ended up, building the focal runner's real
result should give the real value of the play (this is only a check of the machinery, not the values). rv_gap is the difference (NaN when the real
result isn't one of the three, e.g. he scored from first, or stayed a base short of "held").
check_construction() summarises it.

Plays where the outs found from the runner columns don't match outs_post - outs_pre (nearly all walk-offs,
where the game ends and the batter is never put on a base) get no run values: rv_* are NaN and
rv_ok is False. A play that makes the third out clears the bases, so there the play text decides who
was out (see actual_destinations).

Destinations are coded: base 1/2/3, 4 = scored, OUT = out, ABSENT = no runner there.
Origins are coded: 0 = batter, 1/2/3 = the runner on that base.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for the shared modules

from extra_bases import NOT_NEGATED_BY_ERROR, regex_in
from run_expectancy import lookup_re, state_from_runner_columns

OUT, ABSENT, SCORED = -1, -9, 4

# kind: (origin of the focal runner, base he has if he holds, base he has if he takes the extra base)
KIND_SPEC = {
    "1st_on_single": (1, 2, 3),
    "2nd_on_single": (2, 3, SCORED),
    "1st_on_double": (1, 3, SCORED),
    "3rd_on_fly":    (3, 3, SCORED),
    "batter_for_2":  (0, 1, 2),
    "batter_for_3":  (0, 2, 3),
}

NEEDED_COLUMNS = [
    "season", "opportunity_type", "event", "batter", "br1_pre", "br2_pre", "br3_pre", "br1_post", "br2_post",
    "br3_post", "run_b", "run1", "run2", "run3", "outs_pre", "outs_post", "runs",
    "took_extra_base", "thrown_out",
]


def actual_destinations(opps, hit_bases):
    """n x 4 array: where the batter and the runners on 1st, 2nd, 3rd ended up (see the module docstring).

    A play that makes the third out clears the bases afterwards, so a runner who is missing from the
    bases and didn't score may not be out at all. On those plays only the play text can say who was put
    out (a "1X3" or "BX2" that no error cancelled; the batter on a caught fly ball); everyone else is put
    at the base the hit gives him (origin + hit_bases, at most 3)."""
    ids = [opps["batter"], opps["br1_pre"], opps["br2_pre"], opps["br3_pre"]]
    scored = [opps["run_b"], opps["run1"], opps["run2"], opps["run3"]]
    dest = np.full((len(opps), 4), ABSENT)
    for origin in range(4):
        d = np.full(len(opps), OUT)
        for base in (1, 2, 3):
            d = np.where((ids[origin] == opps[f"br{base}_post"]).to_numpy(), base, d)
        d = np.where((ids[origin] == scored[origin]).to_numpy(), SCORED, d)
        dest[:, origin] = np.where(ids[origin].notna().to_numpy(), d, ABSENT)

    inning_over = (opps["outs_post"] == 3).to_numpy()
    event = opps["event"].fillna("")
    for origin in range(4):
        letter = "B" if origin == 0 else str(origin)
        put_out = regex_in(event, letter + r"X[123H]" + NOT_NEGATED_BY_ERROR).to_numpy()
        if origin == 0:
            put_out = put_out | (opps["opportunity_type"] == "3rd_on_fly").to_numpy()   # caught fly ball
        unclear = inning_over & (dest[:, origin] == OUT) & ~put_out
        dest[:, origin] = np.where(unclear, np.minimum(origin + hit_bases, 3), dest[:, origin])
    return dest


def place_focal_runner(dest, focal, target):
    """Copy of dest with the focal runner's result set to target (a base, SCORED or OUT), and the
    other runners moved only where they would share his base. focal, target: one value per row.
    Returns (new dest, base he actually gets, which can be less than target if he is blocked)."""
    rows = np.arange(len(dest))
    d = dest.copy()
    on_base = (target >= 1) & (target <= 3)

    result = target.copy()
    for ahead in range(1, 4):
        blocks = on_base & (ahead > focal) & (d[:, ahead] >= 1) & (d[:, ahead] <= 3)
        result = np.where(blocks, np.minimum(result, d[:, ahead] - 1), result)
    result = np.where(on_base, np.maximum(result, np.maximum(focal, 1)), result)
    d[rows, focal] = result

    limit = result.copy()
    for behind in (2, 1, 0):
        moved = on_base & (behind < focal) & (d[:, behind] >= 1) & (d[:, behind] <= 3)
        new = np.maximum(np.minimum(d[:, behind], limit - 1), max(behind, 1))
        d[:, behind] = np.where(moved, new, d[:, behind])
        limit = np.where(moved, d[:, behind], limit)
    return d, result


# bases the hit gives a runner who is forced or just running with the ball in play (a fly out gives none)
HIT_BASES = {
    "1st_on_single": 1, "2nd_on_single": 1, "1st_on_double": 2, "3rd_on_fly": 0,
    "batter_for_2": 0, "batter_for_3": 0,  # bases empty: nobody is behind the batter
}


def hold_trailing_runners(dest, focal, hit_bases):
    """Copy of dest with runners behind the focal runner (lower origin) no further than origin + hit_bases."""
    d = dest.copy()
    for behind in (0, 1, 2):
        trailing = (behind < focal) & (d[:, behind] >= 1)
        d[:, behind] = np.where(trailing, np.minimum(d[:, behind], behind + hit_bases), d[:, behind])
    return d


def state_value(dest, outs_pre, season, re_table):
    """RE(after) + runs scored for a set of destinations (see the module docstring for the runs rule)."""
    origin = np.arange(4)[None, :]
    is_out = dest == OUT
    outs = np.asarray(outs_pre) + is_out.sum(axis=1)
    scored = dest == SCORED
    last_out = np.where(is_out, origin, -1).max(axis=1)
    after_third = (scored & (origin > last_out[:, None])).sum(axis=1)
    runs = np.where(outs >= 3, after_third, scored.sum(axis=1))
    state = ((dest == 1).any(axis=1).astype(int) + 2 * (dest == 2).any(axis=1) + 4 * (dest == 3).any(axis=1))
    return lookup_re(re_table, season, state, outs) + runs


def add_run_values(opps, re_table):
    """Add rv_safe, rv_out, rv_actual (all relative to holding), rv_gap and safe_blocked.
    opps needs NEEDED_COLUMNS (build_opportunities keeps them as long as extra_bases_statcast.PLAY_COLUMNS
    does) and one row per chance."""
    missing = [c for c in NEEDED_COLUMNS if c not in opps.columns]
    if missing:
        raise ValueError(f"opportunities are missing columns needed for run values: {missing}")
    opps = opps.reset_index(drop=True).copy()
    unknown = set(opps["opportunity_type"]) - set(KIND_SPEC)
    if unknown:
        raise ValueError(f"no run-value rule for opportunity types: {sorted(unknown)}")

    spec = opps["opportunity_type"].map(KIND_SPEC)
    focal = np.array([s[0] for s in spec])
    held = np.array([s[1] for s in spec])
    safe = np.array([s[2] for s in spec])
    season, outs_pre = opps["season"].to_numpy(), opps["outs_pre"].to_numpy()
    hit_bases = opps["opportunity_type"].map(HIT_BASES).to_numpy()
    dest = actual_destinations(opps, hit_bases)
    value = lambda d: state_value(d, outs_pre, season, re_table)  # noqa: E731
    out_target = np.full(len(opps), OUT)

    # the values: runners behind the focal runner held to the base the hit gives them
    iso = hold_trailing_runners(dest, focal, hit_bases)
    v_held = value(place_focal_runner(iso, focal, held)[0])
    v_safe = value(place_focal_runner(iso, focal, safe)[0])
    v_out = value(place_focal_runner(iso, focal, out_target)[0])

    # the check: everyone else exactly as they really ended up
    c_held = value(place_focal_runner(dest, focal, held)[0])
    c_safe_d, safe_base = place_focal_runner(dest, focal, safe)
    c_safe, c_out = value(c_safe_d), value(place_focal_runner(dest, focal, out_target)[0])

    actual_state = state_from_runner_columns(opps, "post")
    v_actual = lookup_re(re_table, season, actual_state, opps["outs_post"].to_numpy()) + opps["runs"].to_numpy()

    outs_ok = (dest == OUT).sum(axis=1) == (opps["outs_post"] - opps["outs_pre"]).to_numpy()

    went_out = opps["thrown_out"].to_numpy() == 1
    went_safe = (opps["took_extra_base"].to_numpy() == 1) & ~went_out
    rv_actual = np.where(went_out, v_out - v_held, np.where(went_safe, v_safe - v_held, 0.0))

    # does the built state for the real result reproduce the real play?
    focal_dest = dest[np.arange(len(opps)), focal]
    built = np.select(
        [went_out, went_safe & (focal_dest == safe_base), ~went_out & ~went_safe & (focal_dest == held)],
        [c_out, c_safe, c_held], default=np.nan)
    opps["rv_safe"] = np.where(outs_ok, v_safe - v_held, np.nan)
    opps["rv_out"] = np.where(outs_ok, v_out - v_held, np.nan)
    opps["rv_actual"] = np.where(outs_ok, rv_actual, np.nan)
    opps["rv_gap"] = np.where(outs_ok, built - v_actual, np.nan)
    opps["safe_blocked"] = safe_base < safe
    opps["rv_ok"] = outs_ok
    return opps


def check_construction(opps, tolerance=1e-9):
    """Per kind: chances, how many could be checked, how many built states differ from the real play,
    and how many safe states were blocked by a runner ahead. Needs add_run_values output."""
    g = opps.groupby("opportunity_type")
    out = pd.DataFrame({
        "chances": g.size(),
        "no_run_value": g["rv_ok"].apply(lambda s: int((~s).sum())),
        "checked": g["rv_gap"].count(),
        "mismatch": g["rv_gap"].apply(lambda s: int((s.abs() > tolerance).sum())),
        "safe_blocked": g["safe_blocked"].sum(),
    })
    out["mismatch_pct"] = (out["mismatch"] / out["checked"] * 100).round(2)
    return out


# Flat values (--values flat). Savant's numbers behave as if every extra base were worth about the same and every
# thrown out cost about the same, whatever the base-out state: its cost per thrown out is -0.81 to -0.87 in all ten of
# its situations, and its largest credit for an extra base is about 0.25 in every kind. This is an inference from
# its published numbers, not something Savant states. Defaults found by a grid search against Savant's runner-season
# totals (see the README); the agreement is flat for extra base 0.25-0.35 and out 0.6-0.8.
FLAT_EXTRA_BASE = 0.25
FLAT_OUT = 0.70


def apply_flat_values(opps, extra_base=FLAT_EXTRA_BASE, out=FLAT_OUT):
    """Replace rv_safe, rv_out and rv_actual by flat values: +extra_base for taking the base, -out for being
    thrown out, 0 for holding, whatever the base-out state. Rows keep their rv_ok / safe_blocked flags."""
    opps = opps.copy()
    opps["rv_safe"] = float(extra_base)
    opps["rv_out"] = -float(out)
    went_out = opps["thrown_out"].to_numpy() == 1
    went_safe = (opps["took_extra_base"].to_numpy() == 1) & ~went_out
    opps["rv_actual"] = np.where(went_out, opps["rv_out"], np.where(went_safe, opps["rv_safe"], 0.0))
    return opps
