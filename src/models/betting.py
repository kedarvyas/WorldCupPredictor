"""Paper-trading betting board: model prices, edges, Kelly stakes, ledger.

Markets and engines:
- 1X2 (moneyline): the LOCKED champion forecast (frozen pre-tournament)
- Over/under 2.5 goals: DC score matrix + Platt recalibration
- Correct score: DC score matrix (display/track only — weakest validation,
  fattest bookmaker margins)

Edge and staking:
- edge = p_model * decimal_odds - 1   (expected profit per unit staked)
- Kelly fraction = (p*o - 1) / (o - 1), bet only when positive; we use
  fractional Kelly (default 1/4) because full Kelly assumes p is exactly
  right, and ours carries model risk.

The ledger (reports/paper_ledger.csv) is append-only and settles itself
against real results — same philosophy as the locked-forecast scoreboard:
predictions first, grading by reality, no edits after the fact.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

LEDGER_PATH = PROJECT_ROOT / "reports" / "paper_ledger.csv"
LEDGER_COLS = ["placed_at_utc", "home_team", "away_team", "market",
               "selection", "decimal_odds", "model_p", "stake"]


def edge(p: float, odds: float) -> float:
    return p * odds - 1.0


def kelly_fraction(p: float, odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        return pd.read_csv(LEDGER_PATH)
    return pd.DataFrame(columns=LEDGER_COLS)


def append_bet(home, away, market, selection, odds, model_p, stake) -> None:
    row = pd.DataFrame([{
        "placed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "home_team": home, "away_team": away, "market": market,
        "selection": selection, "decimal_odds": float(odds),
        "model_p": float(model_p), "stake": float(stake)}])
    header = not LEDGER_PATH.exists()
    LEDGER_PATH.parent.mkdir(exist_ok=True)
    row.to_csv(LEDGER_PATH, mode="a", header=header, index=False)


def _won(row, gh: int, ga: int) -> bool:
    if row["market"] == "1X2":
        actual = "home" if gh > ga else ("away" if ga > gh else "draw")
        return row["selection"] == actual
    if row["market"] == "O/U 2.5":
        return (gh + ga > 2.5) == (row["selection"] == "over")
    if row["market"] == "Correct score":
        return row["selection"] == f"{gh}-{ga}"
    raise ValueError(f"unknown market {row['market']}")


def settle(ledger: pd.DataFrame, results: dict) -> pd.DataFrame:
    """results: {(home, away): (gh, ga)} of played matches.
    Adds status (pending/won/lost) and profit columns."""
    out = ledger.copy()
    status, profit = [], []
    for _, row in out.iterrows():
        res = results.get((row["home_team"], row["away_team"]))
        if res is None:
            status.append("pending")
            profit.append(0.0)
            continue
        win = _won(row, *res)
        status.append("won" if win else "lost")
        profit.append(row["stake"] * (row["decimal_odds"] - 1.0) if win
                      else -row["stake"])
    out["status"] = status
    out["profit"] = profit
    return out
