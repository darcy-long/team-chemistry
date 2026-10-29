"""
Does the Hustle Index say anything about winning?

Checks, all using data/processed/hustle_index.csv:
    1. Correlation of each score with win%, run differential and win_outperformance
       (wins above what runs scored/allowed predict).
    2. Regression of win% on run differential with and without hustle.

Usage:
    python src/compare_to_winning.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"


def ols(y, X):
    """Least squares with an intercept. Returns (coefficients, R^2)."""
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return beta, r2


def main():
    df = pd.read_csv(DATA_DIR / "hustle_index.csv")
    y = df["win_pct"].to_numpy()

    print("Correlation with winning:")
    z_cols = [c for c in df.columns if c.endswith("_z")]
    outcomes = ["win_pct", "run_diff", "win_outperformance"]
    print(df[z_cols + outcomes].corr().loc[z_cols, outcomes].round(3))

    _, r2_base = ols(y, df[["run_diff"]].to_numpy())
    beta, r2_full = ols(y, df[["run_diff", "hustle_z"]].to_numpy())

    print(f"\nwin% ~ run_diff            R^2 = {r2_base:.3f}")
    print(f"win% ~ run_diff + hustle   R^2 = {r2_full:.3f}  (hustle coef = {beta[2]:+.4f} per SD)")


if __name__ == "__main__":
    main()
