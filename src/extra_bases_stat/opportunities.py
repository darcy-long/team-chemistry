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

Plays where nobody really decided, or where we can't tell what the runner decided, are dropped
(find_all_opportunities says which, and why):
    - the play that ended the game (walk-offs, rain: the other runners stop wherever they are)
    - the inning ended without the runner going or being put out (someone else made the third out)
    - ground-rule doubles (a runner on 1st is sent to 3rd by rule)
    - runners who advanced on an error (the batter chances already drop these)
    - a runner on 1st on a single whose way to 3rd was blocked by a runner ahead who ended on 3rd
"""

import numpy as np
import pandas as pd

from extra_bases import NOT_NEGATED_BY_ERROR, find_opportunities as find_runner_opportunities, regex_in
from run_values import HIT_BASES, actual_destinations
from team_codes import RETROSHEET_TO_STATCAST

# (name, hit that means he held, hit that means he stretched it and made it,
#  column showing where he ended up on a throw, pattern for thrown out trying)
# A stretch that works is scored as the bigger hit (single -> double, double -> triple), so the
# successful ones can only be found by looking at the bigger hit.
BATTER_OPPORTUNITIES = [
    ("batter_for_2", "single", "double", "br2_post", r"BX2"),
    ("batter_for_3", "double", "triple", "br3_post", r"BX[3H]"),
]
# A batter advance that came with an error: B-2(E4/TH), B-2(UR)(E4), or an out with an error somewhere else on
# the play: BX3(75)(E7). (BX2(7E4), an out the error cancelled, is a real attempt and is kept.)
ERROR_ADVANCE = (r"B-[23H](?:\([^)]*\))*?\([^)]*E\d"
                 r"|BX[23H](?=(?:\([^)]*\))*\(\d+\))(?:\([^)]*\))*?\([^)]*E\d")

# A runner who took his extra base on an error, by kind: 1-3(E7), 2-H(E8), 1-H(UR)(E8)
RUNNER_ERROR_ADVANCE = {
    "1st_on_single": r"1-[3H](?:\([^)]*\))*?\([^)]*E\d",
    "2nd_on_single": r"2-H(?:\([^)]*\))*?\([^)]*E\d",
    "1st_on_double": r"1-H(?:\([^)]*\))*?\([^)]*E\d",
    "3rd_on_fly": r"3-H(?:\([^)]*\))*?\([^)]*E\d",
}


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
        opp = opp[~regex_in(opp["event"], ERROR_ADVANCE)]
        went = (
            (opp[stretch_col] == 1)
            | (opp[end_col] == opp["batter"])
            | regex_in(opp["event"], out_pattern)
        )
        thrown_out = regex_in(opp["event"], out_pattern + NOT_NEGATED_BY_ERROR)
        opp = _finish(opp, name, "batter")
        opp["took_extra_base"] = went.astype(int).to_numpy()
        opp["thrown_out"] = thrown_out.astype(int).to_numpy()
        frames.append(opp)
    return pd.concat(frames, ignore_index=True)


def find_sac_fly_opportunities(plays):
    """Runner on 3rd, fewer than 2 outs, ball caught by an outfielder: did he tag and go?
    The out is the batter's own: 8/SF/F8D, 8(B)84(1)/LDP, or a bare 7 or 8.3-H. A longer putout such as 84(1)/FO
    is an outfielder fielding a ground ball for a force out, which gives the runner on 3rd no tag-up to decide."""
    caught_by_outfielder = regex_in(plays["event"], r"^[789](?:/|\(B\)|\.|$)")
    opp = plays[plays["br3_pre"].notna() & (plays["outs_pre"] < 2) & caught_by_outfielder]
    went = (opp["br3_pre"] == opp["run3"]) | regex_in(opp["event"], "3XH")
    opp = _finish(opp, "3rd_on_fly", "br3_pre")
    opp["took_extra_base"] = went.astype(int).to_numpy()
    opp["thrown_out"] = regex_in(opp["event"], "3XH" + NOT_NEGATED_BY_ERROR).astype(int)
    return opp


def game_ending_plays(plays):
    """The last play of a game that ended mid-inning: a walk-off, or a game stopped by rain.
    The last play of the last half-inning does not end it on 3 outs, and Retrosheet then leaves the runners
    after the play blank, so the other runners look like they held. (Doing it this way, not with the score and
    "9th inning or later", also catches the 7-inning doubleheaders of 2021.)
    plays needs gid, inning, top_bot, event and outs_post in game order, with a unique index."""
    real = plays[plays["event"] != "NP"]    # NP rows are substitutions and other non-plays
    last_of_half = real.groupby(["gid", "inning", "top_bot"], sort=False).tail(1)
    return plays.index.isin(last_of_half.index[last_of_half["outs_post"] < 3])


def is_walk_off(plays):
    """The home team takes the lead in the bottom of the 9th or later, so the game ends on this play."""
    return ((plays["top_bot"] == 1) & (plays["inning"] >= 9) & (plays["score_h"] <= plays["score_v"])
            & (plays["score_h"] + plays["runs"] > plays["score_v"]))


def drop_undecided_plays(opps):
    """Drop chances where the runner could not really decide, or where we cannot tell what he decided.

    Walk-offs: the game ends when the winning run scores, so the other runners stop wherever they are. They
    would show up as "held" (and the runner who scores the winning run as "went") when nobody chose either.

    Inning ending without the runner going or being put out: someone else made the third out (a double play
    on a fly ball, the batter thrown out stretching a hit). The run may not have counted, so a runner who
    tagged up and went looks like one who held, and we cannot tell which he was.

    Ground-rule doubles: by rule every runner is sent two bases, so a runner on 1st stops at 3rd whatever he
    would have done. (The batter chances already skip these.)"""
    inning_ended = (opps["outs_post"] == 3) & (opps["took_extra_base"] == 0)
    ground_rule = opps["event"].str.startswith("DGR", na=False)
    return opps[~is_walk_off(opps) & ~inning_ended & ~ground_rule]


def drop_error_advances(opps):
    """Drop runners who took their extra base on an error (1-3(E7)): the error gave it to them, so it says nothing
    about the decision, and the batter chances already drop these (ERROR_ADVANCE)."""
    drop = np.zeros(len(opps), dtype=bool)
    for kind, pattern in RUNNER_ERROR_ADVANCE.items():
        drop |= ((opps["opportunity_type"] == kind) & regex_in(opps["event"], pattern)).to_numpy()
    return opps[~drop]


def drop_blocked_runners(opps):
    """Drop the runners on 1st on a single who could not go to 3rd because a runner ahead of them ended the play
    on 3rd. Going wasn't an option, so holding isn't a decision: left in, those plays count as holds and pull
    the go rate down. (Only this kind can be blocked: the other runners' extra base is home, and the batters
    chances have the bases empty.)"""
    first = (opps["opportunity_type"] == "1st_on_single").to_numpy()
    if not first.any():
        return opps
    dest = actual_destinations(opps[first], np.full(int(first.sum()), HIT_BASES["1st_on_single"]))
    ahead_on_base = ((dest[:, 2] >= 1) & (dest[:, 2] <= 3)) | ((dest[:, 3] >= 1) & (dest[:, 3] <= 3))
    blocked = np.zeros(len(opps), dtype=bool)
    blocked[np.flatnonzero(first)[ahead_on_base]] = True
    return opps[~blocked]


def find_all_opportunities(plays):
    """plays needs a season column plus the Retrosheet columns used above, in game order with a unique index,
    plus score_v, score_h, runs, outs_post, inning, top_bot, gid, event for dropping the plays nobody decided.
    The plays dropped, and why, are in the module docstring."""
    plays = plays[~game_ending_plays(plays)]
    opps = pd.concat(
        [find_runner_opportunities(plays), find_batter_opportunities(plays), find_sac_fly_opportunities(plays)],
        ignore_index=True,
    )
    opps = drop_undecided_plays(opps)
    opps = drop_error_advances(opps)
    opps = drop_blocked_runners(opps)
    return opps.reset_index(drop=True)
