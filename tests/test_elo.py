"""Tests for the Elo computation — above all, the no-leakage property.

Runnable two ways:
    .venv/bin/python -m tests.test_elo     (plain asserts, no pytest needed)
    pytest tests/                           (if/when pytest is installed)
"""

import pandas as pd

from src.features.elo import compute_elo, expected_score, mov_multiplier


def _matches(rows):
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team",
                                     "home_score", "away_score",
                                     "tournament", "neutral"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_expected_score_sanity():
    assert expected_score(1500, 1500) == 0.5
    assert abs(expected_score(1900, 1500) - 0.909) < 0.001
    # Symmetry: P(A beats B) + P(B beats A) == 1
    assert abs(expected_score(1700, 1500) + expected_score(1500, 1700) - 1) < 1e-12


def test_mov_multiplier():
    assert [mov_multiplier(d) for d in (0, 1, 2, 3, 4)] == [1.0, 1.0, 1.5, 1.75, 1.875]
    assert mov_multiplier(-3) == 1.75  # margin, not direction


def test_zero_sum():
    df = _matches([("2020-01-01", "A", "B", 3, 0, "Friendly", False),
                   ("2020-02-01", "B", "C", 1, 1, "FIFA World Cup", True),
                   ("2020-03-01", "C", "A", 0, 2, "UEFA Euro", True)])
    _, ratings = compute_elo(df)
    assert abs(sum(ratings.values()) - 3 * 1500) < 1e-9, "rating must be conserved"


def test_no_leakage():
    """THE test: pre-match features for match i must not depend on match i's
    result. We flip match 2's result and check match 2's features are
    unchanged while match 3's (which legitimately depend on match 2) move."""
    base = [("2020-01-01", "A", "B", 2, 0, "Friendly", False),
            ("2020-02-01", "A", "B", 1, 0, "Friendly", False),
            ("2020-03-01", "A", "B", 1, 1, "Friendly", False)]
    flipped = list(base)
    flipped[1] = ("2020-02-01", "A", "B", 0, 5, "Friendly", False)

    out_base, _ = compute_elo(_matches(base))
    out_flip, _ = compute_elo(_matches(flipped))

    cols = ["home_elo_pre", "away_elo_pre"]
    assert out_base.loc[1, cols].equals(out_flip.loc[1, cols]), \
        "LEAKAGE: match 2's features changed with match 2's own result"
    assert not out_base.loc[2, cols].equals(out_flip.loc[2, cols]), \
        "match 3's features should reflect match 2's result"


def test_home_advantage_only_when_not_neutral():
    """Same upset, neutral vs not: the home side is 'expected' to do better
    at home, so an identical home win earns FEWER points at home."""
    home, _ = compute_elo(_matches([("2020-01-01", "A", "B", 1, 0, "Friendly", False)]))
    neut, _ = compute_elo(_matches([("2020-01-01", "A", "B", 1, 0, "Friendly", True)]))
    gain_home = compute_elo(_matches([("2020-01-01", "A", "B", 1, 0, "Friendly", False)]))[1]["A"] - 1500
    gain_neut = compute_elo(_matches([("2020-01-01", "A", "B", 1, 0, "Friendly", True)]))[1]["A"] - 1500
    assert gain_home < gain_neut


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All Elo tests passed.")
