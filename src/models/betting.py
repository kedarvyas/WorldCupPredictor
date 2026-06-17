"""Paper-trading betting board: model prices, edges, Kelly stakes, ledger.

Markets and engines:
- 1X2 (moneyline): the LOCKED champion forecast (frozen pre-tournament)
- Over/under totals ladder (1.5/2.5/3.5/4.5): DC score matrix. Only the 2.5
  line is Platt-recalibrated (recalibrate.fit fits that line only); the
  other lines are raw DC mass, labelled as such in the dashboard. A ladder
  matters because a single 2.5 line on a lopsided fixture (e.g. a heavy
  favourite vs a minnow) sits far from 50/50, where the book's price leaves
  no realistic value — a higher line is where any disagreement shows up.
- Correct score: DC score matrix (display/track only — weakest validation,
  fattest bookmaker margins)

Edge and staking:
- edge = p_model * decimal_odds - 1   (expected profit per unit staked)
- Kelly fraction = (p*o - 1) / (o - 1), bet only when positive; we use
  fractional Kelly (default 1/4) because full Kelly assumes p is exactly
  right, and ours carries model risk.
- devig() strips the book's margin from a complete market so model edge can
  be split into "the book's vig" vs "genuine model disagreement".

The ledger (reports/paper_ledger.csv) records each bet's edge and bet_type
(value vs hunch) at placement so model-advised bets can be graded apart from
hunches, plus a closing_odds field for closing-line value (clv) — the
cleanest skill signal. Bet terms are append-only and the ledger settles
itself against real results; only the observed closing line is filled in
after the fact (save_closing_odds). Same philosophy as the locked-forecast
scoreboard: predictions first, grading by reality, no edits after the fact.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

LEDGER_PATH = PROJECT_ROOT / "reports" / "paper_ledger.csv"
LEDGER_COLS = ["placed_at_utc", "home_team", "away_team", "market",
               "selection", "decimal_odds", "model_p", "stake",
               "bet_type", "edge", "closing_odds"]


def american_to_decimal(a: float) -> float:
    """+250 -> 3.50; -400 -> 1.25. |a| must be >= 100."""
    if abs(a) < 100:
        raise ValueError(f"American odds must be <= -100 or >= +100, got {a}")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def decimal_to_american(d: float) -> float:
    """3.50 -> +250; 1.25 -> -400. d must be > 1."""
    if d <= 1.0:
        raise ValueError(f"decimal odds must exceed 1, got {d}")
    return round((d - 1.0) * 100.0) if d >= 2.0 else -round(100.0 / (d - 1.0))


def edge(p: float, odds: float) -> float:
    return p * odds - 1.0


def kelly_fraction(p: float, odds: float) -> float:
    if odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def devig(decimal_odds) -> np.ndarray:
    """Proportional de-vig of a complete market's decimal odds.

    Raw implied probs q_i = 1/o_i sum to MORE than 1 (the book's margin);
    proportional normalization q_i / sum(q) strips it. Same method as the
    title-odds de-vig (src/models/market.py); use only on a COMPLETE market
    (e.g. all three 1X2 outcomes, both O/U sides) — not a partial set like
    the top-3 correct scores."""
    q = 1.0 / np.asarray(decimal_odds, dtype=float)
    return q / q.sum()


def clv(placed_odds: float, closing_odds: float) -> float:
    """Closing-line value: how much better your price was than the close.

    Measured on de-vig-free decimal odds as placed/closing - 1. Positive
    means you beat the close (locked a better price than the market settled
    at) — the cleanest skill signal in betting, less variance-bound than P/L.
    Returns NaN when no closing line was recorded."""
    if not closing_odds or closing_odds <= 1.0:
        return float("nan")
    return placed_odds / closing_odds - 1.0


def load_ledger() -> pd.DataFrame:
    """Load the ledger, backfilling columns added after early rows were
    written. edge and bet_type are recomputed deterministically from the
    bet's own model_p/odds; closing_odds stays blank until recorded.

    Defensive: a malformed CSV (e.g. a row corrupted by an older append bug,
    possibly restored from a gist) is salvaged by skipping unparseable rows
    rather than crashing the whole page; a totally unreadable file falls back
    to an empty ledger."""
    if not LEDGER_PATH.exists():
        return pd.DataFrame(columns=LEDGER_COLS)
    try:
        df = pd.read_csv(LEDGER_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LEDGER_COLS)
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(LEDGER_PATH, on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=LEDGER_COLS)
    if "edge" not in df.columns:
        df["edge"] = df["model_p"] * df["decimal_odds"] - 1.0
    if "bet_type" not in df.columns:
        df["bet_type"] = np.where(df["edge"] > 0, "value", "hunch")
    if "closing_odds" not in df.columns:
        df["closing_odds"] = np.nan
    return df.reindex(columns=LEDGER_COLS)


def append_bet(home, away, market, selection, odds, model_p, stake,
               bet_type=None, edge_val=None, closing_odds=None) -> None:
    odds, model_p = float(odds), float(model_p)
    edge_val = float(edge_val) if edge_val is not None else edge(model_p, odds)
    bet_type = bet_type or ("value" if edge_val > 0 else "hunch")
    row = pd.DataFrame([{
        "placed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "home_team": home, "away_team": away, "market": market,
        "selection": selection, "decimal_odds": odds,
        "model_p": model_p, "stake": float(stake),
        "bet_type": bet_type, "edge": edge_val,
        "closing_odds": float(closing_odds) if closing_odds else np.nan}])
    # Read-normalize-write rather than a raw file append: an existing ledger
    # may still be in an older, narrower schema (pre bet_type/edge/closing_odds).
    # load_ledger backfills it, so the file is always rewritten with one
    # consistent header. Appending a wider row to a narrower CSV corrupts it
    # (the next read_csv throws ParserError).
    LEDGER_PATH.parent.mkdir(exist_ok=True)
    out = pd.concat([load_ledger(), row], ignore_index=True)
    out.reindex(columns=LEDGER_COLS).to_csv(LEDGER_PATH, index=False)


def save_closing_odds(updated: pd.DataFrame) -> None:
    """Persist edited closing_odds back to the ledger. Only the closing
    line is observed reference data recorded after the fact; bet terms
    (stake, odds, selection) stay immutable, so the rest is rewritten
    verbatim from the in-memory ledger."""
    updated.reindex(columns=LEDGER_COLS).to_csv(LEDGER_PATH, index=False)


def _won(row, gh: int, ga: int):
    """True (win), False (loss), or None (push — stake returned)."""
    actual = "home" if gh > ga else ("away" if ga > gh else "draw")
    mkt, sel = row["market"], row["selection"]
    if mkt == "1X2":
        return sel == actual
    if mkt.startswith("O/U "):
        line = float(mkt.split()[1])
        return (gh + ga > line) == (sel == "over")
    if mkt == "Correct score":
        return sel == f"{gh}-{ga}"
    if mkt == "BTTS":
        return (gh >= 1 and ga >= 1) == (sel == "yes")
    if mkt == "Double chance":
        return {"1X": actual in ("home", "draw"),
                "12": actual in ("home", "away"),
                "X2": actual in ("draw", "away")}[sel]
    if mkt == "Draw no bet":
        if actual == "draw":
            return None  # stake returned
        return sel == actual
    raise ValueError(f"unknown market {mkt}")


def settle(ledger: pd.DataFrame, results: dict) -> pd.DataFrame:
    """results: {(home, away): (gh, ga)} of played matches.
    Adds status (pending/won/lost/push) and profit columns."""
    out = ledger.copy()
    status, profit = [], []
    for _, row in out.iterrows():
        res = results.get((row["home_team"], row["away_team"]))
        if res is None:
            status.append("pending")
            profit.append(0.0)
            continue
        win = _won(row, *res)
        if win is None:                      # push: stake returned, no P/L
            status.append("push")
            profit.append(0.0)
        elif win:
            status.append("won")
            profit.append(row["stake"] * (row["decimal_odds"] - 1.0))
        else:
            status.append("lost")
            profit.append(-row["stake"])
    out["status"] = status
    out["profit"] = profit
    return out


def equity_curve(settled: pd.DataFrame) -> pd.DataFrame:
    """Settled bets in placement order with running cumulative P/L.

    Pushes count (profit 0) so the step count matches the ledger; pending
    bets are excluded (no P/L yet). Returns the settled rows plus a
    cum_profit column — the bankroll trajectory."""
    done = settled[settled["status"].isin(["won", "lost", "push"])].copy()
    done = done.sort_values("placed_at_utc")
    done["cum_profit"] = done["profit"].cumsum()
    return done


def max_drawdown(cum_profit) -> float:
    """Largest peak-to-trough drop in a cumulative-P/L series (<= 0).
    0.0 for an empty series or one that only ever climbs."""
    arr = np.asarray(cum_profit, dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr - np.maximum.accumulate(arr)).min())


def open_exposure(settled: pd.DataFrame) -> dict:
    """Risk carried by still-pending bets: how many, total stake at risk
    (the most they can lose), and combined profit if they all win.

    Potential profit ignores correlation — open legs on the same match can't
    all win — so it's an upper bound, not an expectation."""
    pend = settled[settled["status"] == "pending"]
    return {
        "n": int(len(pend)),
        "at_risk": float(pend["stake"].sum()),
        "potential_profit": float(
            (pend["stake"] * (pend["decimal_odds"] - 1.0)).sum()),
    }
