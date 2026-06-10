"""Betting math + ledger settlement tests.

Runnable without pytest:
    .venv/bin/python -m tests.test_betting
"""

import pandas as pd

from src.models.betting import LEDGER_COLS, edge, kelly_fraction, settle


def test_edge_and_kelly():
    assert abs(edge(0.5, 2.2) - 0.10) < 1e-12       # 50% at 2.2 -> +10%
    assert abs(edge(0.5, 2.0) - 0.0) < 1e-12        # fair odds -> no edge
    assert kelly_fraction(0.5, 2.0) == 0.0          # no edge -> no bet
    assert abs(kelly_fraction(0.5, 2.2) - 0.1 / 1.2) < 1e-12
    assert kelly_fraction(0.2, 1.5) == 0.0          # negative edge -> 0


def _ledger(rows):
    return pd.DataFrame(rows, columns=LEDGER_COLS)


def test_settlement_all_markets():
    rows = [
        ["t", "A", "B", "1X2", "home", 2.0, 0.5, 10],     # A 2-1 -> won
        ["t", "A", "B", "1X2", "draw", 3.5, 0.3, 10],     # -> lost
        ["t", "A", "B", "O/U 2.5", "over", 1.9, 0.55, 10],   # 3 goals -> won
        ["t", "A", "B", "O/U 2.5", "under", 2.0, 0.5, 10],   # -> lost
        ["t", "A", "B", "Correct score", "2-1", 9.0, 0.12, 5],  # exact -> won
        ["t", "C", "D", "1X2", "away", 2.5, 0.4, 10],     # unplayed
    ]
    out = settle(_ledger(rows), {("A", "B"): (2, 1)})
    assert list(out["status"]) == ["won", "lost", "won", "lost", "won",
                                   "pending"]
    assert abs(out.loc[0, "profit"] - 10.0) < 1e-12   # 10 * (2.0 - 1)
    assert abs(out.loc[1, "profit"] + 10.0) < 1e-12
    assert abs(out.loc[4, "profit"] - 40.0) < 1e-12   # 5 * (9 - 1)
    assert out.loc[5, "profit"] == 0.0


def test_draw_settlement():
    out = settle(_ledger([["t", "A", "B", "1X2", "draw", 3.2, 0.3, 10],
                          ["t", "A", "B", "O/U 2.5", "under", 1.8, 0.6, 10]]),
                 {("A", "B"): (1, 1)})
    assert list(out["status"]) == ["won", "won"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All betting tests passed.")
