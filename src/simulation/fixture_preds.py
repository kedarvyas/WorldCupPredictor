"""Pre-tournament fixture predictions, frozen to a locked file.

The dashboard's predicted-vs-actual comparison is only honest if the
predictions are immutable once the tournament starts: recomputing them
after results arrive would let post-match Elo leak back into "pre-match"
forecasts (the same leakage rule from Phase 2, at tournament scale).

`python -m src.simulation.fixture_preds` writes the lock file once and
refuses to overwrite it thereafter.
"""

import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, PROJECT_ROOT
from src.simulation.bracket import GROUPS
from src.simulation.simulate import build_prob_tables, team_state

LOCK_PATH = PROJECT_ROOT / "reports" / "predictions_2026_locked.csv"
CLASS_LABELS = ["home", "draw", "away"]


def scoreline_dists(before=None) -> dict:
    """P(scoreline | outcome) from modern-era matches."""
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[m["date"] >= "2010-01-01"]
    if before is not None:
        m = m[m["date"] < before]
    gd = m["home_score"] - m["away_score"]
    oc = np.select([gd > 0, gd < 0], ["win", "loss"], "draw")
    dists = {}
    for o in ("win", "draw", "loss"):
        c = Counter(zip(m.loc[oc == o, "home_score"].astype(int),
                        m.loc[oc == o, "away_score"].astype(int)))
        tot = sum(c.values())
        dists[o] = {s: n / tot for s, n in c.items()}
    return dists


def fixture_predictions(model, state) -> pd.DataFrame:
    """All 72 group fixtures: model probabilities, predicted pick, and the
    exact (analytic) top-2 scorelines — outcome probs x P(score|outcome),
    the same mixture the Monte Carlo samples, without sampling noise."""
    probs = build_prob_tables(model, state)
    dists = scoreline_dists()
    members = {t: g for g, ts in GROUPS.items() for t in ts}
    fx = pd.read_csv(DATA_PROCESSED / "wc2026_fixtures.csv",
                     parse_dates=["date"])
    rows = []
    for r in fx.itertuples():
        p = probs[(r.home_team, r.away_team)]
        mix = Counter()
        for o, pi in zip(("win", "draw", "loss"), p):
            for s, q in dists[o].items():
                mix[s] += pi * q
        (s1, p1), (s2, p2) = mix.most_common(2)
        rows.append({
            "date": r.date.date(), "group": members[r.home_team],
            "home_team": r.home_team, "away_team": r.away_team,
            "venue": f"{r.city} ({r.country})",
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "pick": CLASS_LABELS[int(np.argmax(p))],
            "top1_score": f"{s1[0]}-{s1[1]}", "top1_p": p1,
            "top2_score": f"{s2[0]}-{s2[1]}", "top2_p": p2,
        })
    return pd.DataFrame(rows).sort_values(["date", "group"])


def main():
    if LOCK_PATH.exists() and "--force" not in sys.argv:
        sys.exit(f"{LOCK_PATH.name} already exists — predictions are locked.\n"
                 "(--force to regenerate; only legitimate BEFORE kickoff.)")
    model = joblib.load(MODELS_DIR / "outcome_model.joblib")
    preds = fixture_predictions(model, team_state())
    assert len(preds) == 72, f"expected 72 fixtures, got {len(preds)}"
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(LOCK_PATH, index=False)
    print(f"Locked {len(preds)} predictions -> {LOCK_PATH.name}")


if __name__ == "__main__":
    main()
