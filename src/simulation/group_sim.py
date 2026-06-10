"""Live group-stage simulator: played matches are facts, the rest is sampled.

The pre-tournament Monte Carlo answers "what did we expect before kickoff?"
This module answers "what do we expect NOW?" — each simulation fixes the
group matches that have actually been played at their real scores and
samples only the remaining fixtures, so the forecast updates as results
arrive (a Bayesian update by conditioning, not refitting: model parameters
stay frozen, only the evidence grows).

Output per team: P(finish 1st/2nd/3rd/4th), P(top-2), P(advance as
best-third), P(advance overall), expected points.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED
from src.simulation.bracket import GROUPS
from src.simulation.simulate import rank_teams

ALL_TEAMS = sorted(t for g in GROUPS.values() for t in g)


def group_fixtures() -> dict[str, list[tuple[str, str]]]:
    fx = pd.read_csv(DATA_PROCESSED / "wc2026_fixtures.csv")
    fixtures = {g: [] for g in GROUPS}
    for r in fx.itertuples():
        g = next(k for k, t in GROUPS.items() if r.home_team in t)
        fixtures[g].append((r.home_team, r.away_team))
    assert all(len(f) == 6 for f in fixtures.values())
    return fixtures


def played_results() -> dict[tuple[str, str], tuple[int, int]]:
    """Actual WC2026 group results so far, keyed like the fixtures."""
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[(m["tournament"] == "FIFA World Cup") & (m["date"] >= "2026-06-11")]
    return {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
            for r in m.itertuples()}


def simulate_group_stage(probs, sample_score, n_sims=5000, seed=2026,
                         results=None) -> pd.DataFrame:
    """probs/sample_score: same contracts as simulate.simulate_once
    (outcome table keyed by ordered pair; sampler(outcome_idx, pair)).
    `results`: {(home, away): (gh, ga)} of matches to hold fixed."""
    rng = np.random.default_rng(seed)
    fixtures = group_fixtures()
    results = results if results is not None else {}

    pos_counts = {t: np.zeros(4) for t in ALL_TEAMS}
    best_third = defaultdict(int)
    pts_sum = defaultdict(float)

    for _ in range(n_sims):
        placements, third_stats = {}, {}
        for g, teams in GROUPS.items():
            stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
            h2h = {}
            for a, b in fixtures[g]:
                if (a, b) in results:
                    ga, gb = results[(a, b)]
                else:
                    p = probs[(a, b)]
                    oc = rng.choice(3, p=p / p.sum())
                    ga, gb = sample_score(oc, (a, b))
                h2h[(a, b)] = (ga, gb)
                stats[a]["pts"] += 3 if ga > gb else (1 if ga == gb else 0)
                stats[b]["pts"] += 3 if gb > ga else (1 if ga == gb else 0)
                stats[a]["gd"] += ga - gb; stats[b]["gd"] += gb - ga
                stats[a]["gf"] += ga;      stats[b]["gf"] += gb
            placements[g] = rank_teams(stats, rng, h2h_results=h2h)
            third_stats[g] = stats[placements[g][2]]
            for pos, t in enumerate(placements[g]):
                pos_counts[t][pos] += 1
            for t in teams:
                pts_sum[t] += stats[t]["pts"]

        for g in rank_teams(third_stats, rng)[:8]:
            best_third[placements[g][2]] += 1

    members = {t: g for g, ts in GROUPS.items() for t in ts}
    rows = {}
    for t in ALL_TEAMS:
        p1, p2, p3, p4 = pos_counts[t] / n_sims
        b3 = best_third[t] / n_sims
        rows[t] = {"group": members[t], "1st": p1, "2nd": p2, "3rd": p3,
                   "4th": p4, "top2": p1 + p2, "best_third": b3,
                   "advance": p1 + p2 + b3, "xPts": pts_sum[t] / n_sims}
    return pd.DataFrame.from_dict(rows, orient="index")
