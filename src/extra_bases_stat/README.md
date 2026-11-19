# Extra Bases Taken: process and current status

> **TODO: the Extra Bases indicator still needs to be updated with the new Extra Bases Taken stat models.**
> The indicator that feeds the Hustle Index is [`src/extra_bases.py`](../extra_bases.py) (with [`src/models.py`](../models.py)). It still uses the original three kinds of chance and the depth/gap buckets. The models in this folder (six kinds of chance, Statcast throw-distance buckets, scored results) are not wired into it yet, so the Hustle Index doesn't use them. See the [project README](../../README.md#31-extra-bases-taken) for how the indicator fits in.

A baserunning stat: **does a runner take the extra base more often than a typical runner would in the same situation?** The aim is a version of the stat Statcast publishes (Baseball Savant's baserunning run value), built from public data, that we can check against theirs.

Seasons: 2015-2025 (no 2020). 2026 is not included: Retrosheet had not published its play-by-play when this was built.

---

## 1. The six kinds of chance

| Kind | What the runner decides | Went |
|---|---|---|
| `1st_on_single` | runner on 1st, single: stop at 2nd, or go to 3rd/home | 34% |
| `1st_on_double` | runner on 1st, double: stop at 3rd, or score | 45% |
| `2nd_on_single` | runner on 2nd, single: stop at 3rd, or score | 62% |
| `3rd_on_fly` | runner on 3rd, under 2 outs, ball caught by an outfielder: stay, or tag up and score | 80% |
| `batter_for_2` | bases empty, single or double: held at 1st, or stretched to 2nd | 24% |
| `batter_for_3` | bases empty, double or triple: held at 2nd, or stretched to 3rd | 9% |

Notes on the definitions (`opportunities.py`):
- Thrown out trying counts as going. An out cancelled by an error counts as safe, and the whole chain of parentheticals after the out is read: `1X3(6E4)` and `2XH(UR)(8E3)(NR)` are safe, but `1XH(62)(E8)` is still an out (the error was elsewhere on the play).
- Batter chances are bases-empty only, so the batter's own decision is what is measured. A stretch that works is recorded by Retrosheet as the bigger hit (single becomes a double), so a double counts as "went for 2" (and, if he didn't try for third, as "held" at 2nd in `batter_for_3`: two separate decisions, which is why a double is in both kinds).
- Plays where nobody really decided, or where we can't tell what the runner decided, are dropped before scoring, in every version of the stat (`find_all_opportunities`). 14,143 of the 409,308 chances:

| Dropped | Chances | Why |
|---|---|---|
| Walk-offs, games stopped early, and plays where the inning ended without the runner going or being put out | 2,159 | The game ends when the winning run scores, so the other runners stop where they are and look like holds (and Retrosheet leaves the runners after a game-ending play blank). If someone else makes the third out, the run may not have counted, so a runner who tagged up looks like he held. Games that end mid-inning are found from the play data, not "9th inning or later", so the 7-inning doubleheaders of 2021 are caught |
| Ground-rule doubles | 1,384 | By rule every runner is sent two bases, so a runner on 1st stops at 3rd whatever he would have done |
| Runners who took the extra base on an error (`1-3(E7)`) | 2,172 | The error gave it to them. Batters' advances on errors (24 chances, including a batter put out after one) were already dropped |
| A runner on 1st on a single who could not go to 3rd because a runner ahead ended the play on 3rd | 8,354 | Going wasn't an option, so holding isn't a decision. Left in, they counted as holds and pulled the go rate for the kind from 34.7% to 31.3% |
| Outfielders fielding a ground ball for a force out, counted as "caught fly balls" for the runner on 3rd | 50 | There was no tag-up to decide |

That leaves 395,165 chances.

## 2. Pipeline

```
pull_retrosheet.py   -> data/raw/retrosheet/YYYYplays.parquet     every play (Retrosheet)
pull_statcast.py     -> data/raw/statcast_YYYY.parquet            every pitch (Baseball Savant)
pull_sprint_speed.py -> data/raw/sprint_speed/YYYY.parquet        sprint speed (Savant)
player_crosswalk.py  -> data/raw/register_*.parquet               player IDs and names (Chadwick register)
game_crosswalk.py    -> data/processed/game_crosswalk.csv         Retrosheet game <-> Statcast game_pk
play_pitch_join.py   -> data/processed/plays_statcast/YYYY.parquet  Retrosheet plays + Statcast pitch data
extra_bases_stat/run_expectancy.py       -> .../run_expectancy.csv            run expectancy (RE24) for each season
extra_bases_stat/extra_bases_statcast.py -> data/processed/extra_bases_stat/   the stat (--runs scores it in runs)
extra_bases_stat/compare_to_savant.py    -> .../savant_comparison/            check against Savant
extra_bases_stat/compare_components.py   -> .../savant_comparison/component_comparison.csv   run version vs Savant, by part
extra_bases_stat/smooth_model.py + evaluate_smooth.py -> .../smooth_model/evaluation.csv        smooth odds vs buckets (opt-in: --odds smooth)
```

**Game crosswalk.** 24,682 games match one-to-one. Single games match on date and teams. Doubleheaders match on final score, then on home starting pitcher when scores tie (game_pk order is not always schedule order).

**Play join.** Counting plate appearances fails, because Statcast numbers a plate appearance that never finishes (e.g. a caught stealing for the third out) and Retrosheet does not. Instead each Retrosheet plate appearance is matched to the next Statcast at-bat with the same inning, half, batter and pitcher. Every match is re-checked, and a play that fails gets no Statcast data. 99.9-100% of plate appearances match.

**Savant's published leaderboard** (`base_running_YYYY.csv`, 2016-2026) was downloaded by hand. There is no code for it. The comparison scripts read `data/raw/savant/`, which is not in the repo (it is gitignored). The yearly files are committed in `data/extra_bases_statcast/`; to run the comparisons, copy them over: `mkdir -p data/raw/savant && cp data/extra_bases_statcast/base_running_*.csv data/raw/savant/`. The by-situation downloads mentioned in section 3b are not in the repo.

## 3. The model (bucket model)

1. Each chance gets a result: **+1** if the runner made the extra base, **-1** if thrown out trying, **0** if he held.
2. Chances are grouped into situations: **kind of chance, hit type (ground/line/fly/popup/bunt), outs, and throw distance**. Throw distance is the straight-line distance from where the ball was fielded (Statcast `hc_x`/`hc_y`, converted to feet) to the base the runner is heading for, split into 5 quantile bins within each kind.
3. The **expected result** is the average result in that situation. (`--shrinkage M` can pull small situations toward the average for their kind, `expected = (n * situation_avg + M * kind_avg) / (n + M)`. It is off by default: for the default stat it makes no difference, because only 0.1% of plays are in a situation with under 20 plays.)
4. A chance's **score = result - expected**. A runner's or team's score is the sum of these, and a team's per-chance score is that sum divided by its chances.

Reading a runner's number: `net_bases_above_expected` is net bases gained over what a typical runner would have gained on the same plays. Kiermaier's 2015, for example, is +21.9 bases in 132 chances.

### Settings
| Default | Option |
|---|---|
| Speed-neutral (runner's own speed not used) | `--own-speed` compares a runner with runners of his own speed |
| Scores results (+1 / 0 / -1) | `--hustle` scores the attempt (thrown out counts as going) |
| Postseason included | `--regular-season` leaves it out |
| All six kinds counted | `--exclude-batter-for-3` leaves out the batter-for-3 chances; those outputs go in `without_batter_for_3/` |
| 5 throw-distance bins | `--bins N`, `--position` adds fielding position |
| Run values from the base-out state | `--runs --values flat` uses one value for every state: +0.25 runs for the extra base, -0.70 for a thrown out (`--extra-base-value`, `--out-value`); `_flat` in `variants/`. See 3b |
| Odds from buckets | `--odds smooth` uses the smooth model instead (3c); `_smooth` in `variants/` |
| Scored in bases | `--runs` scores in runs (run expectancy); outputs go in `variants/` with `_runs` added. See 3b |

`--evaluate` compares bucket choices on held-out seasons (each season predicted from the other nine).

## 3b. The run version (`--runs`)

The default stat counts every extra base as +1 and every out as -1. `--runs` replaces that with what each result was worth in runs, using run expectancy (RE24).

**Run expectancy** (`run_expectancy.py`). For each season, the average runs scored from a base-out state (8 base states x 3 outs) to the end of the half-inning, from every play. Half-innings that did not finish (walk-offs) and `NP` rows are left out. Bases empty with 0 outs comes out at 0.47-0.53 depending on the season. The run value of a play is RE(after) + runs - RE(before), and over a season those average to about zero.

**Run value of each result** (`run_values.py`). For each chance the focal runner (the batter, or the runner on the base) either stays at the base he'd have without the extra base (held, the baseline), takes the extra base (safe), or is thrown out. Each result is valued as RE(bases, outs after) + runs scored, minus the same for held. Only the focal runner's result changes:
- Runners ahead of him are kept as they really ended up. (If one ended on the base he would take, he would be held back. Those chances are dropped before scoring, see section 1, so it no longer comes up.)
- Runners behind him are held to the base the hit gives them (batter at 1st on a single, and so on) in all three states. Their extra bases on the throw are a result of his decision; crediting them to successes only would make the expected value unfair.
- A third out ends the half-inning. Only runners ahead of the last runner out score. Every out here is a tag play.
- A play that makes the third out clears the bases afterwards, so who is out on that play comes from the play text (an uncancelled `1X3`, `BX2`, and so on, and the batter on a caught fly ball); everyone else is put where the hit leaves him. (An early version treated every runner missing from the bases as out and dropped those chances, which removed about 40% of all thrown-out chances and made the run version's out counts too low. Fixed, with tests.)

**Score.** The buckets of section 3 give the odds of safe and of out. Then `expected = p_safe x rv_safe + p_out x rv_out` using that chance's own base-out state, `score = rv(what happened) - expected`, and the unit is runs: `runs_above_expected`, `runs_<kind>`, and `per_100` is runs per 100 chances. As in the bases version, taking the base is the same whether he stopped at 3rd or scored from 1st on a single (a possible refinement: split that one kind into 3rd and home).

**Check built in.** With every other runner left exactly as he really ended up, building the focal runner's real result should reproduce the real play's run value. It does for all of the 393,853 checkable chances (`check_construction`, printed on every `--runs` run). Before the fixes in section 1 it missed 1 (a play where a chained error went unread).

**Nothing is left out of the run version.** Every chance has a run value. (Before the fixes in section 1, 13 chances had none, nearly all of them plays that ended the game, and 8,354 blocked runner-on-1st singles were dropped from this version only, to stop them pulling down the go rate: left in, the 1st-on-single total was +402 runs league-wide instead of -28. They are now dropped from every version.)

**What it looks like.** The league total is +109 runs over about 395,000 chances. The seasons are -159 (2019) to +114 (2025): the same pooled-baseline drift as the bases version (runners go more often after 2021), not something the run values add. Runner-seasons with 50+ chances correlate 0.905 with the bases version and 0.845 with Savant's `runner_runs`, the same as the bases version (0.845). Career leaders (300+ chances): Elvis Andrus, Jose Ramirez, Trea Turner.

**Piece by piece against Savant** (`compare_components.py`, 2,644 runner-seasons). Savant splits `runner_runs` into advances + thrown out + hold, which add up exactly; we split our score the same way by what the runner did. Correlations: advances 0.865, thrown out 0.874, hold 0.775, total 0.845. The attempt models agree closely: attempts 0.969, expected attempts 0.942, attempts above expected 0.860 (slope 0.85). Per thrown-out the cost is the same (about -0.8 runs). Two differences remain, both traced further (the readings below, by regressing Savant's per-runner numbers on ours by kind, 2,657 runner-seasons; coefficients are noisy because the kinds are correlated, so read them as direction, not as exact values; they were run before the fixes in section 1 and have not been rerun):
- **Hold penalty (ours -2.8 vs Savant -1.5 per runner-season).** About three-quarters of the gap is the two batter kinds. Savant's hold penalty is about -0.068 per hold on runner chances (se 0.001) but about +0.003 on batter chances (se 0.0006): in their numbers, a batter who doesn't stretch a hit is not charged. We charge -0.019 per held `batter_for_2` and -0.015 per held `batter_for_3` (about -0.9 runs per runner-season in total). On the four runner kinds our per-hold charge is about 1.3x theirs, so that part is a smaller scale difference. Their advance credit on `batter_for_2` is also small (about 0.03 per advance vs our 0.08). The likely reason is that their odds for batter chances are close to 0 or close to 1 (from fielder and runner times), where ours are an average of about 24%, so a held single is nearly free for them. That is a reading, not shown: sharper buckets don't fix it. With 5, 10, 20 and 40 throw-distance bins the held `batter_for_2` penalty only goes from -2,677 to -2,442 runs, and exit speed and launch angle as extra buckets made out-of-sample fit worse. What would test it is a smooth model with a fielder-time stand-in (not built).
- **Thrown outs (ours 1.29 vs Savant 1.10 per runner-season).** Half of the gap is the `1st_on_single` chances with a runner also on 2nd (see the open question below), which Savant doesn't count. Most of the rest is thrown-out plays that end the inning on 2 outs: Savant has about 83% as many of those as we do for runner chances (se 0.01), against 91-96% for the others, mostly `1st_on_single` and `2nd_on_single` with 2 outs. Why is not known.

**Flat values (`--runs --values flat`).** Savant's numbers behave as if every extra base were worth about the same and every thrown out cost about the same, whatever the base-out state: its cost per thrown out is -0.81 to -0.87 in all ten of its situations (ours ranged from -0.46 to -1.23), and its largest credit for an extra base is about 0.25 in every kind. That is an inference from its published numbers; the Savant page describes the success-probability model but not how it converts to runs. Scoring with +0.25 for an extra base and -0.70 for a thrown out (same odds, same chances) raised the runner-season agreement with Savant's total from 0.865 to 0.897 (smooth odds; 0.845 to 0.874 with buckets), brought each kind's size to within about 0.8-1.1 of Savant's (the situation values made the run-scoring kinds about twice Savant's), and balanced the league total (+196 runs to +28 with smooth odds; about 0 with buckets). Agreement is flat for extra base 0.25-0.35 and out 0.6-0.8, so the exact constants matter little; they were chosen on the same data they are scored on. Situation-specific averages (per kind, or kind and outs) did worse than flat (0.882, 0.865). The two situation-average figures (0.882, 0.865) were computed before the fixes in section 1 and have not been rerun; the rest of this paragraph was.

Where the remaining difference sits (Savant situation downloads in `data/raw/savant/by_situation/`, all ten situations, 2016-2025; computed before the fixes in section 1): Savant's probability model is sharper than ours (it has runner speed, fielder arm and fielder distance), so its credits and hold penalties are 1.2-1.6 times smaller; Savant's batter chances include batters with runners on, ours are bases-empty only (Savant has 12-68% more batter attempts from fewer chances); and our `1st_on_single` count is 46-75% above Savant's, which falls to 6-10% above when only a runner on 1st counts (dropping the cases with a runner on 2nd raises agreement to 0.903).

## 3c. A smooth model for the odds (opt-in: `--odds smooth`)

> After the fixes in section 1 (which removed 14,143 chances) the run-version agreement (0.97 with the buckets) and the own-speed paragraph below were rerun. The table of feature groups, from `evaluate_smooth.py`, was computed before them and has not been rerun.

**Use it:** `.venv/bin/python src/extra_bases_stat/extra_bases_statcast.py --odds smooth` (add `--runs` for runs). Needs scikit-learn (in `requirements.txt`). Outputs go in `variants/` with `_smooth` added; the default stays the buckets. It uses the feature set base + location + batted ball + context (`SMOOTH_GROUPS`). With `--runs`, the runner-season totals correlate 0.97 with the bucket version.

**Controlling for the runner's speed (`--odds smooth --own-speed`).** Adds his sprint speed that season (ft/s) to the smooth model's features, so each runner is compared with runners of his own speed (files get `_ownspeed`). With `--runs --values flat` (2,628 runner-seasons, 50+ chances): agreement with Savant falls from 0.897 to 0.602, because Savant's leaderboard is not speed-adjusted (its runner runs correlate 0.69 with sprint speed; ours 0.64 without the adjustment, 0.07 with it). The spread of runner scores shrinks (SD 1.9 to 1.4 runs); fast and slow thirds go from about +1.4 / −1.3 runs to +0.1 / −0.1. Read it as "how much did he run beyond what his speed predicts", not as a Savant comparison.

`smooth_model.py` is an alternative to the buckets, used only with `--odds smooth`. It predicts the odds of each result from the chance's own details instead of its bucket: P(attempt) and P(safe | attempt), each a gradient-boosted tree model fitted per kind of chance, with throw distance constrained so a longer throw never lowers the odds. Features come in groups: base (throw distance, outs, hit type); location (hit coordinates in feet, spray angle, distance from home, who fielded it); batted ball (exit speed, launch angle, rough hang time, hit distance); context (who is on base, batter handedness, season); park. Runner speed and identity are never used. Every season is predicted by a model fitted on the other seasons.

It needs scikit-learn (in `requirements.txt`; use a fresh environment, since SciPy older than 1.14 clashes with NumPy 2): `.venv/bin/python src/extra_bases_stat/evaluate_smooth.py` compares it to the buckets on the chances the run version scores (395,165 now). Results (no tuning was done; the settings were fixed in advance):

| Odds | Three-way log loss | `batter_for_2` | Total vs Savant | Hold vs Savant | Mean hold (Savant -1.47) | Mean advances (Savant 2.44) |
|---|---|---|---|---|---|---|
| Buckets (current) | 0.460 | 0.357 | 0.840 | 0.764 | -2.90 | 4.10 |
| Smooth: base only | 0.444 | 0.335 | 0.844 | 0.770 | -2.77 | 3.98 |
| + location | 0.387 | 0.248 | 0.856 | 0.785 | -2.41 | 3.59 |
| + batted ball | 0.367 | 0.222 | 0.859 | 0.790 | -2.25 | 3.44 |
| + context | **0.360** | **0.218** | 0.860 | 0.789 | -2.23 | 3.41 |
| + park | 0.361 | 0.219 | 0.861 | 0.793 | -2.24 | 3.43 |

- **The fit improves a lot, mostly from where the ball went and how it was hit.** Park adds nothing. Context helps only a little, and since each season is predicted from the others, "season" can't be doing much.
- **It sharpens the batter odds the way the hold-penalty explanation predicted:** the share of `batter_for_2` chances with under 5% odds of going rises from 24% to 59%, and the total held `batter_for_2` penalty falls from -2,688 to -1,458 runs. Mean hold moves about half the way to Savant's, and advances move about a third of the way.
- **Agreement with Savant improves a little** (total 0.840 to 0.860, advances 0.860 to 0.877, hold 0.764 to 0.789), but **thrown-out agreement is slightly worse** (0.873 to 0.866); thrown outs are rare, so that model has little to learn from.
- **What it does not fix:** the held `batter_for_3` penalty barely moves (-643 to -583), and Savant's batter hold penalty is still close to zero where ours is -1,458 for `batter_for_2`. The extra thrown outs and chances are definitions, not odds, so they are unchanged. The rest of the gap may need information we don't have (fielder and runner times), or may be definitional.

## 4. Outputs (`data/processed/extra_bases_stat/`)

> Updated for every dropped play listed in section 1: sections 1, 3b (the correlations and totals; the regression readings and smooth-odds figures in it are not), 4 and 5. The feature-group table in 3c was computed before and has not been rerun.

| File | Rows | What |
|---|---|---|
| `extra_bases_runners.csv` | 7,137 | runner-seasons. The leaderboard. |
| `extra_bases_runners_career.csv` | 2,130 | runners, all seasons |
| `extra_bases_statcast.csv` | 300 | team-seasons: `score_<kind>` for each kind, the team's total (`net_bases_above_expected`) and `per_chance` (total divided by chances), `raw_score` (plain average of the six kinds, not the per-chance rate), counts |
| `run_expectancy.csv` | | RE24 table per season (used by `--runs`) |
| `smooth_model/evaluation.csv` | | smooth odds vs buckets, from `evaluate_smooth.py` |

Not in the repo (gitignored, large, regenerate by running the stat with the options in section 3): `variants/` (the same tables for the other settings: `_hustle`, `_ownspeed`, `_regularseason`, `_runs`, `_smooth`, `_flat`; 57 files here), `without_batter_for_3/` (the same files with `batter_for_3` left out) and `savant_comparison/` (ours joined to Savant, by runner, plus the report page from `explore_vs_savant.py`).

Runner columns: `n_opportunities`, `n_attempts`, `n_thrown_out`, `n_safe`, `actual`, `expected`, **`net_bases_above_expected`**, `per_100`, and `above_<kind>` (the total split by kind). `per_100` is `net_bases_above_expected / n_opportunities * 100`: net bases above expected per 100 chances, so runners with different numbers of chances share a scale. It is a plain rate with no adjustment for sample size (one lucky play in one chance reads +95), so only rank runners above a minimum: 50 chances for a season, 300 for a career.

Career leaders (all seasons, speed-neutral): Jose Ramirez, Elvis Andrus, Kevin Kiermaier; Hamilton is 11th and Betts 12th. With `batter_for_3` left out, Kiermaier is 5th and Hamilton 17th.

## 5. How it compares to Savant

Runners with 50+ Savant chances, 2016-2025 (no 2020):

| Check | Result |
|---|---|
| Our runner value vs Savant `runner_runs` | correlation 0.83-0.87 each year (average 0.85), rank correlation 0.81-0.85 (average 0.83) |
| Predicted attempts (ours vs Savant's sigmoid) | correlation 0.94-0.96 |
| Chances | we count 2-11% more (1.02-1.11x, 1.07x overall) |
| Attempts | 1-8% fewer (0.92-0.99x) |
| Thrown out | we count 10-24% more |
| Speed-neutral vs own-speed (agreement with Savant; flat values, smooth odds) | 0.897 vs 0.602, so speed-neutral is the comparable version |

How the version of the stat changes the agreement (2,628 runner-seasons with 50+ Savant chances, 2016-2025). Correlation is how closely our values line up with Savant's; rank correlation is how closely the order matches (see below):

| Version | Correlation | Rank correlation |
|---|---|---|
| Bases (the default stat) | 0.845 | 0.827 |
| Run expectancy, bucket odds | 0.845 | 0.831 |
| Run expectancy, smooth odds | 0.865 | 0.854 |
| Flat values (+0.25 / -0.70), bucket odds | 0.874 | 0.856 |
| Flat values, smooth odds | **0.897** | **0.885** |

The flat constants were tuned on this same Savant data, so the flat rows are a little flattering. Run expectancy values each play by its situation, which Savant's numbers don't seem to do.

Leaving `batter_for_3` out (`--exclude-batter-for-3`) brings chances to 0.89-0.98x (average 0.94) but lowers value correlation to 0.81-0.84 (average 0.82) and attempts to 0.88-0.96x.

Rank correlation first turns every value into its place in the list (1st, 2nd, 3rd...) and then correlates the places. Correlation can be pulled by a few extreme runners, while rank correlation only asks whether the order is the same, so it ignores how big the gaps are. "Agree" means the rankings look alike: the best and worst runners match closely, and a typical runner's rank is about 28 places apart out of 300. Our score is in bases and Savant's is in runs.

Two plays checked by hand, converting our bases to runs with a run-expectancy table built from our Retrosheet data.

| Play | Ours | Savant |
|---|---|---|
| Carroll triple, 2025-09-17 | +0.25 runs | +0.23 |
| Turner triple, 2025-08-20 | +0.19 runs | +0.24 |

## 6. Running it

Use the project's environment (`pip install -r requirements.txt`), then from the repo root:

```
python src/pull_retrosheet.py
python src/pull_statcast.py
python src/pull_sprint_speed.py
python src/player_crosswalk.py
python src/game_crosswalk.py
python src/play_pitch_join.py
python src/extra_bases_stat/extra_bases_statcast.py
python src/extra_bases_stat/extra_bases_statcast.py --runs
python src/extra_bases_stat/extra_bases_statcast.py --hustle --own-speed
python src/extra_bases_stat/run_expectancy.py
python src/extra_bases_stat/compare_to_savant.py
python src/extra_bases_stat/explore_vs_savant.py --open
python -m pytest tests
```

- `pull_statcast.py` is slow (about 15 minutes per season). The pull scripts skip seasons already downloaded.
- `extra_bases_statcast.py` with no options is the default stat. `--runs` scores in runs and `--hustle --own-speed` is the hustle stat. Anything other than the default goes in `variants/`.
- `run_expectancy.py` builds the RE24 table; the `--runs` run builds it if it is missing.
- `compare_to_savant.py` and `explore_vs_savant.py` need the Savant files in `data/raw/savant/` (see section 2).
- Each script prints where it saved its output and a short summary when it finishes.
