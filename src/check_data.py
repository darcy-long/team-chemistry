"""
Quick sanity check on the CSVs in data/processed/. Run this before trusting the index.

Usage:
    python src/check_data.py
"""

from pathlib import Path

import pandas as pd

from hustle_index import INDICATORS, to_statcast_code

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    problems = 0
    for name, indicator in INDICATORS.items():
        df = pd.read_csv(DATA_DIR / indicator["file"])
        df["team"] = df[indicator["team_col"]].map(to_statcast_code)
        teams = df.groupby("season")["team"].nunique()
        n_missing = df[indicator["score_col"]].isna().sum()
        print(f"{name}: {len(df)} rows, {n_missing} missing scores, teams per season {teams.min()}-{teams.max()}")
        if n_missing or teams.min() != 30 or df.duplicated(["team", "season"]).any():
            print(f"  -> check {indicator['file']} (missing scores, a team count that isn't 30, or duplicate rows)")
            problems += 1

    print("\nAll good." if not problems else f"\n{problems} indicator file(s) need a look.")


if __name__ == "__main__":
    main()
