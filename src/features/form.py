"""Recent-form features: points-per-game and goal difference over each
team's last N matches, plus days since last match.

Same leakage discipline as Elo: for each match, a team's form is computed
from strictly PRIOR matches — the current match's result enters the rolling
window only after its features are recorded.

Teams with no prior history get NaN (a genuine "we don't know"), not 0 —
zero would mean "terrible form", which is a different claim.
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.config import FORM_WINDOW


def compute_form(matches: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Add per-team rolling form columns. Requires date-sorted input."""
    assert matches["date"].is_monotonic_increasing, "matches must be date-sorted"

    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    last_played: dict[str, pd.Timestamp] = {}
    cols: dict[str, list] = {c: [] for c in [
        "home_ppg", "home_gd_form", "home_rest_days",
        "away_ppg", "away_gd_form", "away_rest_days"]}

    def snapshot(team, date, side):
        h = history[team]
        if h:
            pts = [p for p, _ in h]
            gds = [g for _, g in h]
            cols[f"{side}_ppg"].append(sum(pts) / len(pts))
            cols[f"{side}_gd_form"].append(sum(gds) / len(gds))
        else:
            cols[f"{side}_ppg"].append(np.nan)
            cols[f"{side}_gd_form"].append(np.nan)
        rest = (date - last_played[team]).days if team in last_played else np.nan
        cols[f"{side}_rest_days"].append(rest)

    for m in matches.itertuples():
        # (1) Record features from prior matches only.
        snapshot(m.home_team, m.date, "home")
        snapshot(m.away_team, m.date, "away")

        # (2) THEN fold this match into both teams' histories.
        gd = m.home_score - m.away_score
        home_pts = 3 if gd > 0 else (1 if gd == 0 else 0)
        away_pts = 3 if gd < 0 else (1 if gd == 0 else 0)
        history[m.home_team].append((home_pts, gd))
        history[m.away_team].append((away_pts, -gd))
        last_played[m.home_team] = m.date
        last_played[m.away_team] = m.date

    out = matches.copy()
    for c, values in cols.items():
        out[c] = values
    return out
