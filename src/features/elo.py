"""Elo ratings computed from scratch over the full match history.

Design (confirmed in Phase 2 discussion), following World Football Elo Ratings:
- Importance-weighted K: World Cup 60, continental finals 50, qualifiers 40,
  other tournaments 30, friendlies 20. K is the learning rate: how fast
  ratings chase results.
- Margin-of-victory multiplier: x1.5 for 2-goal wins, x1.75 for 3,
  +1/8 per goal beyond.
- Home advantage offset: +60 rating points to the home side inside the
  expectation when the venue is not neutral, so home wins aren't
  over-rewarded and venue effects stay OUT of the ratings (the supervised
  model handles venue explicitly with its own features).

LEAKAGE DISCIPLINE: for each match, both teams' ratings are recorded as
pre-match features BEFORE the update is applied. The post-match rating
arithmetically contains the result; recording it as a feature would hand
the model the label. tests/test_elo.py enforces this property.
"""

from collections import defaultdict

import pandas as pd

from src.config import ELO_BASE_RATING

HOME_ADVANTAGE = 60.0

K_BY_TIER = {"world_cup": 60.0, "continental": 50.0, "qualifier": 40.0,
             "other": 30.0, "friendly": 20.0}

CONTINENTAL_FINALS = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup",
    "Confederations Cup",
}


def match_tier(tournament: str) -> str:
    if tournament == "FIFA World Cup":
        return "world_cup"
    if "qualification" in tournament.lower():
        return "qualifier"
    if tournament in CONTINENTAL_FINALS:
        return "continental"
    if tournament == "Friendly":
        return "friendly"
    return "other"


def expected_score(rating_a: float, rating_b: float) -> float:
    """P(A scores) under Elo's logistic curve. 400-pt gap ~= 91%."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / 400.0))


def mov_multiplier(goal_diff: int) -> float:
    """Bigger wins move ratings more: 1.0 / 1.5 / 1.75 / +1/8 per extra goal."""
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run Elo chronologically over `matches`.

    Returns (copy of matches with pre-match rating columns added,
             final ratings dict for use in the 2026 simulation).

    Requires `matches` sorted by date — asserted, because a silently
    unsorted frame would corrupt every rating.
    """
    assert matches["date"].is_monotonic_increasing, "matches must be date-sorted"

    ratings: dict[str, float] = defaultdict(lambda: ELO_BASE_RATING)
    home_pre, away_pre = [], []

    for m in matches.itertuples():
        r_home, r_away = ratings[m.home_team], ratings[m.away_team]

        # (1) Record features FIRST — pre-match information only.
        home_pre.append(r_home)
        away_pre.append(r_away)

        # (2) THEN update using the result.
        adv = 0.0 if m.neutral else HOME_ADVANTAGE
        e_home = expected_score(r_home + adv, r_away)
        goal_diff = m.home_score - m.away_score
        s_home = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)
        k = K_BY_TIER[match_tier(m.tournament)]
        delta = k * mov_multiplier(goal_diff) * (s_home - e_home)
        ratings[m.home_team] = r_home + delta
        ratings[m.away_team] = r_away - delta  # zero-sum

    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    return out, dict(ratings)
