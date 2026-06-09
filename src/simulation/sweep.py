"""Rating-uncertainty sweep for the 2026 simulation.

Re-runs the tournament Monte Carlo with per-team Elo jitter ~ N(0, sigma)
drawn fresh each simulated tournament, for sigma in SIGMAS. Outputs one
CSV per sigma for the dashboard's uncertainty slider (sigma=0 reproduces
the primary published numbers).

JitterTable quacks like the dict that simulate_once expects (probs[(a,b)])
but computes probabilities on demand from the model's own coefficients,
shifting only the elo_diff term — so the existing tournament machinery is
reused without modification.

Usage:
    .venv/bin/python -m src.simulation.sweep [n_sims]
"""

import sys

import joblib
import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, MODELS_DIR, PROJECT_ROOT
from src.models.evaluate import CLASSES
from src.simulation.bracket import GROUPS, HOSTS
from src.simulation.simulate import (ALL_TEAMS, FEATURES, ROUNDS,
                                     _feature_row, build_prob_tables,
                                     scoreline_sampler, simulate_once,
                                     team_state)

SIGMAS = (0.0, 50.0, 100.0, 150.0)
REPORTS = PROJECT_ROOT / "reports"


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


class JitterTable:
    """Dict-like probs table: [p_a, p_draw, p_b] with current Elo jitter."""

    def __init__(self, model, state):
        scaler = model.named_steps["standardscaler"]
        lr = model.named_steps["logisticregression"]
        order = [list(lr.classes_).index(c) for c in CLASSES]
        self.coef = lr.coef_[order] / scaler.scale_
        self.intercept = (lr.intercept_[order]
                          - lr.coef_[order] @ (scaler.mean_ / scaler.scale_))
        self.c_elo = self.coef[:, FEATURES.index("elo_diff")]
        self.jitter = dict.fromkeys(ALL_TEAMS, 0.0)
        self.z = {}
        for a in ALL_TEAMS:
            for b in ALL_TEAMS:
                if a != b:
                    x = np.asarray(_feature_row(state, a, b,
                                                neutral=(a not in HOSTS)),
                                   dtype=float)
                    self.z[(a, b)] = self.coef @ x + self.intercept

    def set_jitter(self, rng, sigma):
        if sigma == 0:
            self.jitter = dict.fromkeys(ALL_TEAMS, 0.0)
        else:
            self.jitter = dict(zip(ALL_TEAMS,
                                   rng.normal(0.0, sigma, len(ALL_TEAMS))))

    def __getitem__(self, key):
        a, b = key
        d = self.jitter[a] - self.jitter[b]
        p_ab = _softmax(self.z[(a, b)] + self.c_elo * d)
        p_ba = _softmax(self.z[(b, a)] - self.c_elo * d)[::-1]
        if a in HOSTS and b not in HOSTS:
            return p_ab
        if b in HOSTS and a not in HOSTS:
            return p_ba
        return (p_ab + p_ba) / 2


def run(n_sims=10_000):
    model = joblib.load(MODELS_DIR / "outcome_model.joblib")
    state = team_state()
    table = JitterTable(model, state)

    # Sanity: at zero jitter the on-demand table must match the static one.
    static = build_prob_tables(model, state)
    for pair in [("Spain", "Brazil"), ("Mexico", "Qatar"), ("Japan", "Iran")]:
        assert np.allclose(table[pair], static[pair], atol=1e-9), pair

    fx = pd.read_csv(DATA_PROCESSED / "wc2026_fixtures.csv")
    fixtures = {g: [] for g in GROUPS}
    members = {t: g for g, ts in GROUPS.items() for t in ts}
    for r in fx.itertuples():
        fixtures[members[r.home_team]].append((r.home_team, r.away_team))
    assert all(len(f) == 6 for f in fixtures.values())

    for sigma in SIGMAS:
        rng = np.random.default_rng(2026)
        sample_score = scoreline_sampler(rng)
        counts = {t: dict.fromkeys(ROUNDS, 0) for t in ALL_TEAMS}
        for _ in range(n_sims):
            table.set_jitter(rng, sigma)
            for rnd, teams in simulate_once(table, fixtures,
                                            sample_score, rng).items():
                for t in teams:
                    counts[t][rnd] += 1
        out = (pd.DataFrame(counts).T / n_sims)[ROUNDS]
        out = out.sort_values("champion", ascending=False)
        path = REPORTS / f"sim_2026_sigma{int(sigma)}.csv"
        out.to_csv(path)
        top = out.iloc[0]
        print(f"sigma={sigma:>5.0f}: favorite {out.index[0]} "
              f"{top['champion']:.1%} -> {path.name}")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
