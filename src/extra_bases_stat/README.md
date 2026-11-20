# Extra Bases Taken: process and current status

> **TODO: the Extra Bases indicator still needs to be updated with the new Extra Bases Taken stat models.**
> The indicator that feeds the Hustle Index is [`src/extra_bases.py`](../extra_bases.py) (with [`src/models.py`](../models.py)). It still uses the original three kinds of chance and the depth/gap buckets. The models in this folder (six kinds of chance, Statcast throw-distance buckets, scored results) are not wired into it yet, so the Hustle Index doesn't use them. See the [project README](../../README.md#31-extra-bases-taken) for how the indicator fits in.

A baserunning stat: **does a runner take the extra base more often than a typical runner would in the same situation?** The aim is a version of the stat Statcast publishes (Baseball Savant's baserunning run value), built from public data, that we can check against theirs (see [NOTES.md](NOTES.md)).

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
extra_bases_stat/smooth_model.py + evaluate_smooth.py -> .../smooth_model/evaluation.csv        smooth odds vs buckets
```

**Game crosswalk.** 24,682 games match one-to-one. Single games match on date and teams. Doubleheaders match on final score, then on home starting pitcher when scores tie (game_pk order is not always schedule order).

**Play join.** Counting plate appearances fails, because Statcast numbers a plate appearance that never finishes (e.g. a caught stealing for the third out) and Retrosheet does not. Instead each Retrosheet plate appearance is matched to the next Statcast at-bat with the same inning, half, batter and pitcher. Every match is re-checked, and a play that fails gets no Statcast data. 99.9-100% of plate appearances match.

**Savant's published leaderboard** (`base_running_YYYY.csv`, 2016-2026) was downloaded by hand. There is no code for it. The comparison scripts read `data/raw/savant/`, which is not in the repo (it is gitignored). The yearly files are committed in `data/extra_bases_statcast/`; to run the comparisons, copy them over: `mkdir -p data/raw/savant && cp data/extra_bases_statcast/base_running_*.csv data/raw/savant/`. The by-situation downloads used in NOTES.md are not in the repo.

## 3. The model

1. Each chance gets a result: **+1** if the runner made the extra base, **-1** if thrown out trying, **0** if he held.
2. The **expected result** is the average result runners get in the same situation, predicted by a **smooth model** (`smooth_model.py`): gradient-boosted trees, one set per kind of chance, that predict the odds of going and of being safe from the chance's own details. These are the throw distance (feet from where the ball was fielded to the base the runner is heading for), outs and hit type, where the ball went and how hard it was hit, who is on base, and the batter's handedness. Runner speed and identity are never used. Every season is predicted by a model fitted on the other seasons. Needs scikit-learn (in `requirements.txt`).
3. A chance's **score = result - expected**. A runner's or team's score is the sum of these, and a team's per-chance score is that sum divided by its chances.

`--odds buckets` uses the older bucket model instead: the average result in the situation (kind of chance, hit type, outs, and throw distance in 5 quantile bins within each kind). The smooth model fits better (log loss 0.360 vs 0.460, see NOTES.md).

Reading a runner's number: `net_bases_above_expected` is net bases gained over what a typical runner would have gained on the same plays. Kiermaier's 2015, for example, is +22.8 bases in 132 chances (+11.5 when compared with runners of his own speed, `--own-speed`).

### Settings
| Default | Option |
|---|---|
| Smooth-model odds | `--odds buckets` uses the situation buckets; outputs get `_buckets` in `variants/` |
| Scored in bases (+1 / 0 / -1) | `--runs` scores in runs using the run expectancy of the base-out state (`_runs`). `--runs --values flat` uses +0.25 runs for the extra base and -0.70 for a thrown out in every state (`--extra-base-value`, `--out-value`; `_flat`) |
| Scores results | `--hustle` scores the attempt (thrown out counts as going) |
| Speed-neutral (runner's own speed not used) | `--own-speed` compares a runner with runners of his own speed (`_ownspeed`) |
| Postseason included | `--regular-season` leaves it out |
| All six kinds counted | `--exclude-batter-for-3` leaves out the batter-for-3 chances; those outputs go in `without_batter_for_3/` |
| 5 throw-distance bins (buckets) | `--bins N`, `--position` adds fielding position |

`--evaluate` compares bucket choices on held-out seasons (each season predicted from the other nine). Anything other than the default is written to `variants/`.

**Against Savant** (2,628 runner-seasons with 50+ Savant chances, 2016-2025; correlation / rank correlation with Savant's `runner_runs`): bases 0.845 / 0.827; run expectancy with smooth odds 0.865 / 0.854; flat values (+0.25 / -0.70) with smooth odds 0.897 / 0.885. The own-speed version correlates 0.602 (flat values), because Savant's leaderboard isn't speed-adjusted, so only the speed-neutral versions are comparable. Details, and where the two differ, are in [NOTES.md](NOTES.md).

## 4. Outputs (`data/processed/extra_bases_stat/`)

| File | Rows | What |
|---|---|---|
| `extra_bases_runners.csv` | 7,137 | runner-seasons. The leaderboard. |
| `extra_bases_runners_career.csv` | 2,130 | runners, all seasons |
| `extra_bases_statcast.csv` | 300 | team-seasons: `score_<kind>` for each kind, the team's total (`net_bases_above_expected`) and `per_chance` (total divided by chances), `raw_score` (plain average of the six kinds, not the per-chance rate), counts |
| `run_expectancy.csv` | | RE24 table per season (used by `--runs`) |
| `smooth_model/evaluation.csv` | | smooth odds vs buckets, from `evaluate_smooth.py` |

Not in the repo (gitignored, large, regenerate by running the stat with the options in section 3): `variants/` (the same tables for the other settings: `_hustle`, `_ownspeed`, `_regularseason`, `_runs`, `_buckets`, `_flat`), `without_batter_for_3/` (the same files with `batter_for_3` left out) and `savant_comparison/` (ours joined to Savant, by runner, plus the report page from `explore_vs_savant.py`).

Runner columns: `n_opportunities`, `n_attempts`, `n_thrown_out`, `n_safe`, `actual`, `expected`, **`net_bases_above_expected`**, `per_100`, and `above_<kind>` (the total split by kind). `per_100` is `net_bases_above_expected / n_opportunities * 100`: net bases above expected per 100 chances, so runners with different numbers of chances share a scale. It is a plain rate with no adjustment for sample size (one lucky play in one chance reads +95), so only rank runners above a minimum: 50 chances for a season, 300 for a career.

Career leaders (all seasons, speed-neutral): Jose Ramirez, Elvis Andrus, Kevin Kiermaier; Hamilton is 11th and Betts 12th. With `batter_for_3` left out, Kiermaier is 5th and Hamilton 17th.

## 5. Running it

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
