"""
A smooth model for the odds of each result of a chance, as an alternative to the buckets.

The bucket model (extra_bases_statcast.py) gives every chance in a situation the same odds. This model
instead predicts the odds from the chance's own details, so two chances in the same bucket can differ:
where the ball was and how far the throw is, how hard and how high it was hit, who else is on base, and so on.
The buckets are not changed or replaced: this is a separate module, and nothing here runs unless asked for.

Two models for each kind of chance, the way Savant describes theirs:
    P(attempt)            did the runner go for the extra base (thrown out counts as going)
    P(safe | attempt)     given that he went, did he make it
and the odds the scorer needs follow from them:
    p_safe = P(attempt) x P(safe | attempt)        p_out = P(attempt) x (1 - P(safe | attempt))
(held = 1 - P(attempt)). These are the same p_safe and p_out the bucket version computes.

Each model is a gradient-boosted tree model (scikit-learn's HistGradientBoostingClassifier). A longer throw
can only make going and being safe more likely, so throw distance is constrained to push the odds up, never
down. Features come in groups (FEATURE_GROUPS) so each can be tested on its own. By default the runner's own speed
and identity are not used: the stat stays speed-neutral. The opt-in "speed" group adds his sprint speed that season
(ft/s), so each runner is compared with runners of his own speed (more speed can only make going and being safe
more likely).

Every prediction is out of sample: each season's odds come from a model fitted on the other seasons only
(like extra_bases_statcast.leave_one_season_out_expected), so a play never helps set its own odds.

Needs scikit-learn, which is imported only when a model is fitted (so the rest of the project runs without it).
In this repo it is installed in .venv: run  .venv/bin/python ...

Public pieces:
    make_features(opps)                         the feature table (one row per chance)
    smooth_odds(opps, groups)                   out-of-sample p_attempt, p_success, p_safe, p_out
    bucket_odds(opps, bucket_cols)              the same four columns from buckets, for comparison
    three_way_log_loss(opps, p_safe, p_out)     how well odds fit what happened (lower is better)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/, for the shared modules

from extra_bases_statcast import FEET_PER_PIXEL, HOME_X, HOME_Y

# Features, in groups that are added one at a time when comparing. "base" is roughly what the buckets use.
FEATURE_GROUPS = {
    "base": ["throw_dist", "outs_pre", "hittype"],
    "location": ["hc_x_ft", "hc_y_ft", "angle", "home_dist", "fielder_pos"],
    "batted_ball": ["launch_speed", "launch_angle", "hang_time", "hit_distance"],
    "context": ["on1", "on2", "on3", "bat_left", "season"],
    "park": ["park"],
    "speed": ["sprint_speed"],   # opt-in (needs build_opportunities(with_speed=True)); not in SMOOTH_GROUPS
}
SMOOTH_GROUPS = ("base", "location", "batted_ball", "context")   # the best set in evaluate_smooth.py (park added nothing)
CATEGORICAL = ["hittype", "fielder_pos", "park"]
MONOTONIC_UP = ["throw_dist", "sprint_speed"]   # a longer throw / a faster runner never makes going / being safe less likely


def _float(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def make_features(opps):
    """One row per chance: every feature in FEATURE_GROUPS (categories as integer codes).
    opps comes from extra_bases_statcast.build_opportunities (it needs throw_dist, hang_time, fielder_pos)."""
    x = (_float(opps["sc_hc_x"]) - HOME_X) * FEET_PER_PIXEL
    y = (HOME_Y - _float(opps["sc_hc_y"])) * FEET_PER_PIXEL
    f = pd.DataFrame(index=opps.index)
    f["throw_dist"] = _float(opps["throw_dist"])
    f["outs_pre"] = opps["outs_pre"].to_numpy(dtype="float64")
    f["hittype"] = opps["hittype"].fillna("Unknown").to_numpy()
    f["hc_x_ft"], f["hc_y_ft"] = x, y
    f["angle"] = np.degrees(np.arctan2(x, y))        # off straightaway center; negative = left field side
    f["home_dist"] = np.hypot(x, y)
    f["fielder_pos"] = opps["fielder_pos"].fillna("Unknown").to_numpy()
    f["launch_speed"] = _float(opps["sc_launch_speed"])
    f["launch_angle"] = _float(opps["sc_launch_angle"])
    f["hang_time"] = _float(opps["hang_time"])
    f["hit_distance"] = _float(opps["sc_hit_distance_sc"])
    for base in (1, 2, 3):
        f[f"on{base}"] = opps[f"br{base}_pre"].notna().to_numpy(dtype="float64")
    f["bat_left"] = (opps["bathand"] == "L").to_numpy(dtype="float64")
    f["season"] = opps["season"].to_numpy(dtype="float64")
    f["park"] = opps["site"].fillna("Unknown").to_numpy()
    f["sprint_speed"] = _float(opps["sprint_speed"]) if "sprint_speed" in opps.columns else np.nan
    for col in CATEGORICAL:   # integer codes, the same in every fold
        f[col] = pd.Categorical(f[col]).codes.astype("float64")
    return f


def feature_columns(groups):
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def _model(columns, stage):
    from sklearn.ensemble import HistGradientBoostingClassifier
    small = stage == "success"   # far fewer thrown outs than attempts: fewer, smaller trees
    return HistGradientBoostingClassifier(
        max_iter=60 if small else 150, learning_rate=0.08, max_leaf_nodes=15,
        min_samples_leaf=60 if small else 100, l2_regularization=1.0,
        categorical_features=[c in CATEGORICAL for c in columns],
        monotonic_cst={c: 1 for c in MONOTONIC_UP if c in columns}, random_state=0)


def _fit_predict(X_train, y_train, X_test, columns, stage):
    if y_train.nunique() < 2:
        return np.full(len(X_test), float(y_train.iloc[0]) if len(y_train) else 0.5)
    return _model(columns, stage).fit(X_train[columns], y_train).predict_proba(X_test[columns])[:, 1]


def smooth_odds(opps, groups=("base",), features=None, verbose=False):
    """Out-of-sample odds from the smooth model, one row per chance (same index as opps):
    p_attempt, p_success (given an attempt), p_safe, p_out. Fitted separately for each kind of chance,
    each season predicted by a model fitted on the other seasons."""
    f = make_features(opps) if features is None else features
    columns = feature_columns(groups)
    attempt = opps["took_extra_base"].astype(int)
    safe = (opps["took_extra_base"] - opps["thrown_out"]).astype(int)
    out = pd.DataFrame(index=opps.index, columns=["p_attempt", "p_success"], dtype="float64")
    for kind, rows in opps.groupby("opportunity_type").groups.items():
        for season in sorted(opps.loc[rows, "season"].unique()):
            test = rows[(opps.loc[rows, "season"] == season).to_numpy()]
            train = rows[(opps.loc[rows, "season"] != season).to_numpy()]
            went = train[(attempt.loc[train] == 1).to_numpy()]
            out.loc[test, "p_attempt"] = _fit_predict(f.loc[train], attempt.loc[train], f.loc[test], columns, "attempt")
            out.loc[test, "p_success"] = _fit_predict(f.loc[went], safe.loc[went], f.loc[test], columns, "success")
        if verbose:
            print(f"  fitted {kind}")
    out["p_safe"] = out["p_attempt"] * out["p_success"]
    out["p_out"] = out["p_attempt"] * (1 - out["p_success"])
    return out


def bucket_odds(opps, bucket_cols):
    """The bucket model's odds, out of sample the same way: a chance's odds are the rates of its bucket in the
    other seasons (a bucket never seen there falls back to its kind's rate). Same four columns as smooth_odds."""
    attempt = opps["took_extra_base"].astype(int)
    safe = (opps["took_extra_base"] - opps["thrown_out"]).astype(int)
    out = pd.DataFrame(index=opps.index, columns=["p_attempt", "p_success"], dtype="float64")
    for season in sorted(opps["season"].unique()):
        train = opps[opps["season"] != season]
        test = opps[opps["season"] == season]
        went = train[attempt.loc[train.index] == 1]
        for col, y_all, fit_rows in (("p_attempt", attempt, train), ("p_success", safe, went)):
            y = y_all.loc[fit_rows.index]
            by_bucket = y.groupby([fit_rows[c] for c in bucket_cols]).mean().rename("p")
            by_kind = y.groupby(fit_rows["opportunity_type"]).mean()
            p = test[bucket_cols].merge(by_bucket.reset_index(), on=bucket_cols, how="left")["p"].to_numpy()
            fallback = test["opportunity_type"].map(by_kind).to_numpy()
            out.loc[test.index, col] = np.where(np.isnan(p), fallback, p)
    out["p_safe"] = out["p_attempt"] * out["p_success"]
    out["p_out"] = out["p_attempt"] * (1 - out["p_success"])
    return out


def three_way_log_loss(opps, p_safe, p_out):
    """Mean of -log(probability given to what happened: safe, thrown out, or held) for each chance.
    Lower is better. A model that gave every chance the same odds would score the entropy of the three results."""
    p_safe, p_out = np.asarray(p_safe, dtype=float), np.asarray(p_out, dtype=float)
    went_out = opps["thrown_out"].to_numpy() == 1
    went_safe = (opps["took_extra_base"].to_numpy() == 1) & ~went_out
    p = np.where(went_out, p_out, np.where(went_safe, p_safe, 1 - p_safe - p_out))
    return float(-np.log(np.clip(p, 1e-6, 1)).mean())
