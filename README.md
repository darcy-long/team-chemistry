# SARG Team Chemistry

**Name:** Darcy Long
**Date:** Fall 2026

---

## 1. Overview
Can team chemistry, quantified as a composite score of discretionary, selfless "hustle" plays positively predict a team's outperformance of its expected win-loss record?

## 2. Background / Motivation

Team chemistry is one of the most common explanations in baseball. A team that "just clicks" gets credit when it beats its talent, and a team full of good players that underperforms gets blamed for bad clubhouse chemistry. But nobody really measures it. Most of the evidence is anecdotal, and the stats we do have (WAR, OAA, UBR) measure how good players are, not whether they play selflessly for each other.

This project is attempting to put a number on it. The idea is to look at plays that are discretionary, where a player can choose to put in extra effort that helps the team but doesn't show up in his own stat line, such as taking the extra base or not giving up on a play in a blowout. Each indicator rewards the *attempt*, not just the outcome, so a runner thrown out going for third still counts.


## 3. Indicators (WORK IN PROGRESS)

### 3.1 Extra Bases Taken
- **What it measures:** Do a team's runners go for the extra base more often than runners usually do in the same situation?
- **Opportunities:** every chance a runner had to take the extra base:
  - runner on 1st, single (stop at 2nd, or go to 3rd or score)
  - runner on 2nd, single (stop at 3rd, or score)
  - runner on 1st, double (stop at 3rd, or score)
- **How it is calculated:**
  1. Each opportunity gets a 1 if the runner took the extra base *or was thrown out trying*, and a 0 if he stopped. Getting thrown out still counts as a 1 because we want to reward the attempt, not the outcome.
  2. Group opportunities into situations: hit type, outs, opportunity type, how deep the ball was fielded (shallow to extra deep, or infield), whether it was hit to a gap, and the runner's sprint speed decile that season.
  3. Expected rate = how often runners go in that situation. Rare situations are pulled toward the league average so a few plays don't decide the rate: `expected = (n * situation_rate + 20 * league_rate) / (n + 20)`, where n is the number of opportunities in the situation.
  4. Residual = what happened (1 or 0) minus the expected rate.
  5. Team score = average residual over all of the team's opportunities that season. Positive means the team's runners go more than expected.
- **Why these choices:** a fast runner on a shallow single is a different decision than a slow runner on a deep one, so the situation includes ball depth and sprint speed. Attempts count so the indicator isn't just "who got lucky."
- **Data used:** Retrosheet play-by-play, Statcast sprint speed, and the Chadwick Register to match Retrosheet player IDs to Statcast IDs. 2015-2025 (no 2020).
- **Limitations:** about 1% of opportunities are dropped because the runner has no sprint speed. Team-seasons with under 50 opportunities are dropped. Retrosheet doesn't say how the throw or the outfielder's arm affected the decision.
- **Code:** `src/extra_bases.py` finds the opportunities, and `src/models.py` does the expected rate and averaging. Output is `data/processed/extra_bases.csv`.
- **Trying other settings:** `extra_bases.py` can turn the situation columns on or off (depth, gap, sprint speed) and compute expected rates per season instead of pooled, e.g. `python src/extra_bases.py --no-gap --by-season`. Add `--diagnostics` to print how many situations are thin under each choice. With all columns on, about half of the situations have under 20 opportunities, which is why the shrinkage in step 3 matters. Runs with non-default settings save to `extra_bases_experiment.csv` so they don't overwrite the real output.

*Also will be turned into a separate statistic potentially*


### 3.2 Defensive Effort Gap (DEG) -- TO BE REMOVED
- **What it measures:** Do fielders turn balls in play into outs as well in blowouts as in close games?
- **How it is calculated:**
  - Expected outs = sum of (1 − xBA) over balls in play
  - Conversion gap = (actual outs − expected outs) / balls in play
  - DEG-Ahead = gap when ahead by 5+ − gap in close games
  - DEG-Behind = gap when behind by 5+ − gap in close games
- **Score states:** close (within 2 runs), ahead (5+), behind (5+), all 7th inning or later
- **Data used:** Statcast, 2015–2025 (no 2020)
- **Why these choices:** home runs can't be fielded and bunts are a strategy choice, so both are dropped. Sac flies count as outs, and since Statcast leaves their xBA blank I fill it in from similar exit speed / launch angle.
- **Limitations:** team level only, and blowout samples are small (some teams only have ~70 balls in play in a state). Also a defensive indicator and we may want to focus on offensive only, hence why will be removed.


## 4. Hustle Index (Composite)
- **How it is built:**
  1. Z-score each indicator within each season
  2. Weighted average of the z-scores (weights currently equal)
  3. Z-score the average again so the index is "standard deviations from league average"
- **Indicators included:** Extra Bases Taken, DEG-Behind
- **How to read it:** 0 = league average, +1 = one standard deviation above, -1 = one standard deviation below

### Adding an indicator
1. Put a CSV in `data/processed/` with one row per team-season (a team column, a `season` column and a score column).
2. Add an entry to `INDICATORS` at the top of `src/hustle_index.py`:
   ```python
   "my_indicator": {
       "file": "my_indicator.csv",
       "team_col": "team",
       "score_col": "score",
       "weight": 1.0,
       "higher_is_better": True,   # False if a lower value means more hustle
   },
   ```
3. Run `python src/hustle_index.py`. The new column shows up in the output, leaderboard and correlations.

## 5. Repository Layout
```
team-chemistry/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   │  # 1. get the data
│   ├── pull_statcast.py       # download Statcast pitch data
│   ├── pull_retrosheet.py     # download Retrosheet play-by-play
│   ├── pull_sprint_speed.py   # download Statcast sprint speed
│   ├── player_crosswalk.py    # Retrosheet player ID -> Statcast player ID
│   ├── team_codes.py          # Retrosheet -> Statcast team code lookup
│   ├── check_team_codes.py    # checks the two sources' team codes agree
│   │  # 2. build the indicators
│   ├── extra_bases.py         # Extra Bases Taken
│   ├── models.py              # expected-rate math used by extra_bases.py
│   ├── deg.py                 # Defensive Effort Gap
│   ├── standings.py           # W/L, run differential, Pythagorean wins
│   │  # 3. combine and check
│   ├── hustle_index.py        # Hustle Index composite (add indicators here)
│   ├── compare_to_winning.py  # correlations + regression against win%
│   └── check_data.py          # sanity check on the processed CSVs
├── tests/
│   ├── test_deg.py
│   ├── test_hustle_index.py
│   └── test_models.py
└── data/
    ├── raw/                   # created by the pull scripts (not included, too large)
    │   ├── statcast_YYYY.parquet
    │   ├── retrosheet/
    │   ├── sprint_speed/
    │   └── register_crosswalk.parquet
    └── processed/
        ├── extra_bases.csv    # Extra Bases Taken output
        ├── deg.csv            # DEG output
        ├── standings.csv      # W/L, run differential, etc.
        └── hustle_index.csv   # final Hustle Index
```

## 6. How to Run
```bash
pip install -r requirements.txt

# 0. Download the raw data (slow, needs internet). Skip this if you only want to
#    rerun the Hustle Index, since the processed CSVs are included.
python src/pull_statcast.py
python src/pull_retrosheet.py
python src/pull_sprint_speed.py

# 1. Indicators
python src/extra_bases.py     # needs the Retrosheet + sprint speed files from step 0 (--diagnostics for bucket sizes)
python src/deg.py             # needs the Statcast files from step 0
python src/standings.py       # needs the Retrosheet files from step 0

# 2. Check the processed CSVs look right
python src/check_data.py

# 3. Hustle Index (uses the CSVs in data/processed/)
python src/hustle_index.py          # add a year to see that season, e.g. 2023

# 4. Does it relate to winning?
python src/compare_to_winning.py

# 5. Tests
python -m pytest
```

## 7. Data Sources
- Statcast via `pybaseball`
- Retrosheet (baserunning, standings)
- Statcast sprint speed via `pybaseball`
- Chadwick Bureau Register (player ID crosswalk)

## 8. Notes
- The z-scoring lives in `hustle_index.py` only. An earlier version of `models.py` had its own z-score function that used a slightly different standard deviation, but the index always starts from the raw scores, so I dropped it to avoid having two.

## 9. References / Acknowledgments
Dr. D, Tony Tarasco.
