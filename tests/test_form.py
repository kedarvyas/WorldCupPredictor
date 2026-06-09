"""Form feature tests — leakage and window semantics."""

import numpy as np
import pandas as pd

from src.features.form import compute_form


def _matches(rows):
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_first_match_is_nan_not_zero():
    out = compute_form(_matches([("2020-01-01", "A", "B", 1, 0)]))
    assert np.isnan(out.loc[0, "home_ppg"]) and np.isnan(out.loc[0, "away_rest_days"])


def test_no_leakage_and_window():
    out = compute_form(_matches([
        ("2020-01-01", "A", "B", 3, 0),   # A wins by 3
        ("2020-01-08", "A", "C", 0, 1),   # A loses
        ("2020-01-15", "B", "A", 1, 1),   # then A draws away
    ]), window=2)
    # Match 2: A's form = match 1 only (win, +3): ppg 3.0, gd +3.0
    assert out.loc[1, "home_ppg"] == 3.0 and out.loc[1, "home_gd_form"] == 3.0
    # Match 3 (A away): form over matches 1-2 = (3+0)/2 pts, (3-1)/2 gd
    assert out.loc[2, "away_ppg"] == 1.5 and out.loc[2, "away_gd_form"] == 1.0
    # Rest days: A played Jan 8, faces Jan 15 -> 7
    assert out.loc[2, "away_rest_days"] == 7


def test_window_caps_history():
    rows = [(f"2020-01-{d:02d}", "A", "B", 1, 0) for d in range(1, 8)]
    out = compute_form(_matches(rows), window=3)
    # By the last match, A's window holds 3 wins regardless of 6 prior.
    assert out.loc[6, "home_ppg"] == 3.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All form tests passed.")
