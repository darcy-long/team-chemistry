"""
Every kind of extra-base chance, in one table.

Six kinds, each with took_extra_base = 1 (went for it, even if thrown out) or 0 (held),
and thrown_out = 1 if he was thrown out trying (so safe = took_extra_base - thrown_out):
    1st_on_single   runner on 1st, single: stop at 2nd, or go to 3rd/home      (extra_bases.py)
    1st_on_double   runner on 1st, double: stop at 3rd, or score               (extra_bases.py)
    2nd_on_single   runner on 2nd, single: stop at 3rd, or score               (extra_bases.py)
    batter_for_2    bases empty, single or double: held at 1st (single), or stretched it
                    to 2nd (a double), or was thrown out trying
    batter_for_3    bases empty, double or triple: held at 2nd (double), or stretched it
                    to 3rd (a triple), or was thrown out trying
    3rd_on_fly      runner on 3rd, fewer than 2 outs, fly/line out to the outfield:
                    stay, or tag up and go home (sac fly)

Batter chances are limited to the bases empty so the batter's own decision is what
is measured: with a runner ahead of him, he only gets the extra base on a throw to
another base. Batter advances on an error are dropped (not a decision, and we can't
tell if he was stretching). A stretch that works is scored as the bigger hit, so a double
counts as "went for 2" even when it was never a close call (down the line, off the wall);
the throw-distance buckets are what separate those from real decisions.
"""

import pandas as pd

from extra_bases import NOT_NEGATED_BY_ERROR, find_opportunities as find_runner_opportunities
from team_codes import RETROSHEET_TO_STATCAST

# (name, hit that means he held, hit that means he stretched it and made it,
#  column showing where he ended up on a throw, pattern for thrown out trying)
# A stretch that works is scored as the bigger hit (single -> double, double -> triple), so the
# successful ones can only be found by looking at the bigger hit.
BATTER_OPPORTUNITIES = [
    ("batter_for_2", "single", "double", "br2_post", r"BX2"),
    ("batter_for_3", "double", "triple", "br3_post", r"BX[3H]"),
]
ERROR_ADVANCE = r"B-[23H]\([^)]*E\d"  # e.g. S4/G34.B-2(E4/TH)


def _finish(opps, name, runner_col):
    opps = opps.copy()
    opps["opportunity_type"] = name
    opps["runner_id"] = opps[runner_col]
    opps["team"] = opps["batteam"].map(RETROSHEET_TO_STATCAST).fillna(opps["batteam"])
    opps["hittype"] = opps["hittype"].fillna("Unknown")
    return opps


def find_batter_opportunities(plays):
    """Hits with the bases empty: did the batter stretch it?
    batter_for_2: single (held), double (made it), single with an out at 2nd (thrown out)
    batter_for_3: double (held), triple (made it), double with an out at 3rd (thrown out)
    Ground-rule doubles are skipped (he can't go for a triple)."""
    bases_empty = plays["br1_pre"].isna() & plays["br2_pre"].isna() & plays["br3_pre"].isna()
    not_ground_rule = ~plays["event"].str.startswith("DGR", na=False)
    frames = []
    for name, held_col, stretch_col, end_col, out_pattern in BATTER_OPPORTUNITIES:
        opp = plays[((plays[held_col] == 1) | (plays[stretch_col] == 1)) & bases_empty & not_ground_rule]
        opp = opp[~opp["event"].str.contains(ERROR_ADVANCE, na=False)]
        went = (
            (opp[stretch_col] == 1)
            | (opp[end_col] == opp["batter"])
            | opp["event"].str.contains(out_pattern, na=False)
        )
        thrown_out = opp["event"].str.contains(out_pattern + NOT_NEGATED_BY_ERROR, na=False)
        opp = _finish(opp, name, "batter")
        opp["took_extra_base"] = went.astype(int).to_numpy()
        opp["thrown_out"] = thrown_out.astype(int).to_numpy()
        frames.append(opp)
    return pd.concat(frames, ignore_index=True)


def find_sac_fly_opportunities(plays):
    """Runner on 3rd, fewer than 2 outs, ball caught by an outfielder: did he tag and go?"""
    caught_by_outfielder = plays["event"].str.match(r"^[789]", na=False)
    opp = plays[plays["br3_pre"].notna() & (plays["outs_pre"] < 2) & caught_by_outfielder]
    went = (opp["br3_pre"] == opp["run3"]) | opp["event"].str.contains("3XH", na=False)
    opp = _finish(opp, "3rd_on_fly", "br3_pre")
    opp["took_extra_base"] = went.astype(int).to_numpy()
    opp["thrown_out"] = opp["event"].str.contains("3XH" + NOT_NEGATED_BY_ERROR, na=False).astype(int)
    return opp


def find_all_opportunities(plays):
    """plays needs a season column plus the Retrosheet columns used above."""
    return pd.concat(
        [find_runner_opportunities(plays), find_batter_opportunities(plays), find_sac_fly_opportunities(plays)],
        ignore_index=True,
    )
