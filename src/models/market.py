"""Market title odds -> de-vigged implied probabilities.

Source: DraftKings outright (championship) odds via ESPN, 2026-06-05,
hand-entered to data/raw/market_odds_2026.csv with team names mapped to
our dataset's conventions (Czechia -> Czech Republic, Congo DR -> DR Congo).

De-vig: raw implied probabilities q_i = 100 / (100 + american_odds) sum to
MORE than 1 — the excess is the bookmaker's margin (the "vig"). We use
proportional normalization p_i = q_i / sum(q). Caveat documented in the
dashboard: proportional de-vig is the simplest method and slightly
overstates longshots (favorite-longshot bias); fine for model comparison,
not for betting.
"""

import pandas as pd

from src.config import DATA_RAW

SOURCE = "DraftKings via ESPN, 2026-06-05"


def market_probs() -> tuple[pd.Series, float]:
    """(de-vigged title probabilities by team, overround).
    Overround = sum of raw implied probs; ~1.0 means no margin."""
    odds = pd.read_csv(DATA_RAW / "market_odds_2026.csv", index_col="team")
    raw = 100.0 / (100.0 + odds["american_odds"])
    overround = float(raw.sum())
    return raw / overround, overround
