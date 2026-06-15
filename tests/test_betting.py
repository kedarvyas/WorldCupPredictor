"""Betting math + ledger settlement tests.

Runnable without pytest:
    .venv/bin/python -m tests.test_betting
"""

import numpy as np
import pandas as pd

from src.models.betting import (LEDGER_COLS, clv, devig, edge, equity_curve,
                                kelly_fraction, max_drawdown, settle)


def test_odds_format_conversion():
    from src.models.betting import american_to_decimal, decimal_to_american
    assert american_to_decimal(250) == 3.5
    assert american_to_decimal(-400) == 1.25
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(-100) == 2.0
    assert decimal_to_american(3.5) == 250
    assert decimal_to_american(1.25) == -400
    assert decimal_to_american(2.0) == 100
    # round-trips
    for a in (-400, -150, -101, 100, 137, 603, 1581):
        assert abs(decimal_to_american(american_to_decimal(a)) - a) <= 1


def test_edge_and_kelly():
    assert abs(edge(0.5, 2.2) - 0.10) < 1e-12       # 50% at 2.2 -> +10%
    assert abs(edge(0.5, 2.0) - 0.0) < 1e-12        # fair odds -> no edge
    assert kelly_fraction(0.5, 2.0) == 0.0          # no edge -> no bet
    assert abs(kelly_fraction(0.5, 2.2) - 0.1 / 1.2) < 1e-12
    assert kelly_fraction(0.2, 1.5) == 0.0          # negative edge -> 0


def test_devig():
    # Two-way market at 1.90/1.90: raw implied 0.526+0.526 = 1.053 overround;
    # de-vig is symmetric back to 0.5/0.5.
    dv = devig([1.9, 1.9])
    assert abs(dv.sum() - 1.0) < 1e-12
    assert abs(dv[0] - 0.5) < 1e-12
    # Lopsided 3-way still normalizes to 1 and preserves ordering.
    dv3 = devig([1.4, 4.5, 9.0])
    assert abs(dv3.sum() - 1.0) < 1e-12
    assert dv3[0] > dv3[1] > dv3[2]


def test_clv():
    # Took 2.10, market closed at 2.00 -> beat the close by +5%.
    assert abs(clv(2.10, 2.00) - 0.05) < 1e-12
    # Took a worse price than the close -> negative CLV.
    assert clv(1.90, 2.00) < 0
    # No closing line recorded -> NaN.
    assert np.isnan(clv(2.0, None))
    assert np.isnan(clv(2.0, float("nan")))


def test_over_prob_ladder_monotone():
    """P(total > line) must fall as the line rises, and sum DC matrix mass."""
    from src.models.dixon_coles import MAX_GOALS
    from src.models.recalibrate import over_prob

    n = MAX_GOALS + 1
    rng = np.random.default_rng(0)
    M = rng.random((n, n))
    M /= M.sum()

    class _StubDC:
        def score_matrix(self, home, away, true_home=False):
            return M

    dc = _StubDC()
    p15 = over_prob(dc, "A", "B", line=1.5)
    p25 = over_prob(dc, "A", "B", line=2.5)
    p35 = over_prob(dc, "A", "B", line=3.5)
    assert p15 > p25 > p35
    tot = np.add.outer(np.arange(n), np.arange(n))
    assert abs(p25 - M[tot > 2.5].sum()) < 1e-12


def test_ledger_backfill():
    """Old-schema rows (pre bet_type/edge/closing_odds) load with the new
    columns filled deterministically from the row's own model_p/odds."""
    import src.models.betting as bet

    old_cols = ["placed_at_utc", "home_team", "away_team", "market",
                "selection", "decimal_odds", "model_p", "stake"]
    old = pd.DataFrame([["t", "A", "B", "1X2", "home", 2.0, 0.6, 10],   # +edge
                        ["t", "A", "B", "1X2", "draw", 3.0, 0.2, 5]],   # -edge
                       columns=old_cols)
    orig = bet.LEDGER_PATH
    tmp = orig.parent / "_test_backfill.csv"
    try:
        old.to_csv(tmp, index=False)
        bet.LEDGER_PATH = tmp
        out = bet.load_ledger()
        assert list(out.columns) == LEDGER_COLS
        assert abs(out.loc[0, "edge"] - 0.2) < 1e-12      # 0.6*2.0 - 1
        assert out.loc[0, "bet_type"] == "value"
        assert out.loc[1, "bet_type"] == "hunch"          # 0.2*3.0 - 1 < 0
        assert out["closing_odds"].isna().all()
    finally:
        bet.LEDGER_PATH = orig
        tmp.unlink(missing_ok=True)


def _ledger(rows):
    # Rows are written with the original 8 columns; pad to the current
    # schema (bet_type/edge/closing_odds) so settlement tests stay terse.
    base = ["placed_at_utc", "home_team", "away_team", "market", "selection",
            "decimal_odds", "model_p", "stake"]
    return pd.DataFrame(rows, columns=base).reindex(columns=LEDGER_COLS)


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


def test_new_markets_settlement():
    # A 2-1 over B: home win, 3 goals, both scored.
    rows = [
        ["t", "A", "B", "BTTS", "yes", 1.8, 0.55, 10],          # both scored
        ["t", "A", "B", "BTTS", "no", 2.0, 0.45, 10],
        ["t", "A", "B", "Double chance", "1X", 1.3, 0.7, 10],   # home or draw
        ["t", "A", "B", "Double chance", "X2", 2.5, 0.5, 10],   # draw or away
        ["t", "A", "B", "Draw no bet", "home", 1.6, 0.6, 10],   # home -> win
    ]
    out = settle(_ledger(rows), {("A", "B"): (2, 1)})
    assert list(out["status"]) == ["won", "lost", "won", "lost", "won"]


def test_draw_no_bet_push():
    # Draw refunds the stake: status push, zero profit; the away DNB on a
    # 1-0 home win loses as normal.
    out = settle(_ledger([["t", "A", "B", "Draw no bet", "home", 1.6, 0.6, 10],
                          ["t", "A", "B", "Draw no bet", "away", 2.4, 0.4, 10]]),
                 {("A", "B"): (1, 1)})
    assert list(out["status"]) == ["push", "push"]
    assert (out["profit"] == 0.0).all()
    out2 = settle(_ledger([["t", "A", "B", "Draw no bet", "away", 2.4, 0.4, 10]]),
                  {("A", "B"): (1, 0)})
    assert out2.loc[0, "status"] == "lost"


def test_btts_no_when_blank():
    # 1-0: BTTS no wins, yes loses.
    out = settle(_ledger([["t", "A", "B", "BTTS", "yes", 1.8, 0.5, 10],
                          ["t", "A", "B", "BTTS", "no", 2.0, 0.5, 10]]),
                 {("A", "B"): (1, 0)})
    assert list(out["status"]) == ["lost", "won"]


def test_equity_curve_and_drawdown():
    rows = [["t1", "A", "B", "1X2", "home", 2.0, 0.5, 10],   # win  +10
            ["t2", "A", "C", "1X2", "home", 2.0, 0.5, 10],   # lose -10
            ["t3", "A", "D", "1X2", "home", 2.0, 0.5, 10]]   # lose -10
    settled = settle(_ledger(rows),
                     {("A", "B"): (2, 0), ("A", "C"): (0, 1),
                      ("A", "D"): (0, 1)})
    curve = equity_curve(settled)
    assert list(curve["cum_profit"]) == [10.0, 0.0, -10.0]
    # Peak +10 then down to -10 -> drawdown -20.
    assert abs(max_drawdown(curve["cum_profit"]) - (-20.0)) < 1e-12
    assert max_drawdown([]) == 0.0
    assert max_drawdown([1.0, 2.0, 3.0]) == 0.0   # monotone up -> no dip


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All betting tests passed.")
