"""Live tournament simulator: played matches are facts, the rest is sampled.

The pre-tournament Monte Carlo answers "what did we expect before kickoff?"
This module answers "what do we expect NOW?" — each simulation fixes
everything that has actually happened (group matches at their real scores,
knockout matches at their real winners) and samples only the remaining
fixtures. A Bayesian update by conditioning, not refitting: model
parameters stay frozen, only the evidence grows.

Knockout subtlety: the results data records a match decided on penalties
as a DRAW — the actual winner lives in shootouts.csv (downloaded in
Phase 0, needed for the first time here).

Output per team: group-finish distribution (P 1st..4th, top-2, best-third,
advance, xPts) plus round survival (R16 ... champion).
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, FIXTURES_FROZEN
from src.simulation.bracket import (FINAL, GROUPS, QF, R16, R32, SF,
                                    allocate_thirds)
from src.simulation.simulate import rank_teams

ALL_TEAMS = sorted(t for g in GROUPS.values() for t in g)
KO_START = "2026-06-28"  # day after the last group match


def group_fixtures() -> dict[str, list[tuple[str, str]]]:
    fx = pd.read_csv(FIXTURES_FROZEN)  # full 72; wc2026_fixtures.csv shrinks
    fixtures = {g: [] for g in GROUPS}
    for r in fx.itertuples():
        g = next(k for k, t in GROUPS.items() if r.home_team in t)
        fixtures[g].append((r.home_team, r.away_team))
    assert all(len(f) == 6 for f in fixtures.values())
    return fixtures


def played_results() -> dict[tuple[str, str], tuple[int, int]]:
    """Actual WC2026 GROUP results so far, keyed like the fixtures.
    Filtered to the 72 known group pairings — knockout matches must not
    inflate the count or pollute the group-results dict."""
    pairs = {p for fx in group_fixtures().values() for p in fx}
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[(m["tournament"] == "FIFA World Cup") & (m["date"] >= "2026-06-11")]
    return {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
            for r in m.itertuples()
            if (r.home_team, r.away_team) in pairs}


def played_ko_results() -> dict[frozenset, str]:
    """{frozenset({team_a, team_b}): winner} for played knockout matches.
    A knockout match is any played WC2026 match not in the group fixture
    list; drawn knockouts were decided on penalties -> shootouts.csv."""
    group_pairs = {(a, b) for fx in group_fixtures().values() for a, b in fx}
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[(m["tournament"] == "FIFA World Cup") & (m["date"] >= KO_START)]
    m = m[~m.apply(lambda r: (r["home_team"], r["away_team"]) in group_pairs,
                   axis=1)]
    if m.empty:
        return {}
    so = pd.read_csv(DATA_RAW / "shootouts.csv", parse_dates=["date"])
    so = so[so["date"] >= KO_START]
    pens = {(r.home_team, r.away_team): r.winner for r in so.itertuples()}

    out = {}
    for r in m.itertuples():
        if r.home_score > r.away_score:
            w = r.home_team
        elif r.home_score < r.away_score:
            w = r.away_team
        else:
            w = pens.get((r.home_team, r.away_team))
            assert w is not None, \
                f"drawn KO {r.home_team}-{r.away_team} missing from shootouts"
        out[frozenset((r.home_team, r.away_team))] = w
    return out


def simulate_live(probs, sample_score, n_sims=5000, seed=2026,
                  group_results=None, ko_results=None, return_nodes=False):
    """Full-tournament Monte Carlo conditioned on everything played so far.

    probs/sample_score: same contracts as simulate.simulate_once (outcome
    table keyed by ordered pair; sampler(outcome_idx, pair)).
    group_results: {(home, away): (gh, ga)} held fixed.
    ko_results: {frozenset({a, b}): winner} held fixed.
    return_nodes: also return {match_no: {team: appearance_prob}} for
    every bracket node — feeds the dashboard's bracket view.
    """
    rng = np.random.default_rng(seed)
    fixtures = group_fixtures()
    group_results = group_results or {}
    ko_results = ko_results or {}

    pos_counts = {t: np.zeros(4) for t in ALL_TEAMS}
    best_third = defaultdict(int)
    pts_sum = defaultdict(float)
    reach = {t: defaultdict(int) for t in ALL_TEAMS}
    node_counts = defaultdict(lambda: defaultdict(int))

    def ko(a, b):
        real = ko_results.get(frozenset((a, b)))
        if real is not None:
            return real
        p = probs[(a, b)]
        oc = rng.choice(3, p=p / p.sum())
        if oc == 1:  # draw -> strength-weighted ET/pens coin flip
            return a if rng.random() < p[0] / (p[0] + p[2]) else b
        return a if oc == 0 else b

    for _ in range(n_sims):
        placements, third_stats = {}, {}
        for g, teams in GROUPS.items():
            stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
            h2h = {}
            for a, b in fixtures[g]:
                if (a, b) in group_results:
                    ga, gb = group_results[(a, b)]
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

        qualified = set(rank_teams(third_stats, rng)[:8])
        for g in qualified:
            best_third[placements[g][2]] += 1
        third_slot = allocate_thirds(qualified)

        winners, entrants = {}, {}
        for match_no, sa, sb in R32:
            pick = lambda s: (placements[s[1]][0] if s[0] == "W" else
                              placements[s[1]][1] if s[0] == "R" else
                              placements[third_slot[match_no]][2])
            a, b = pick(sa), pick(sb)
            entrants[match_no] = (a, b)
            winners[match_no] = ko(a, b)
        for rnd in (R16, QF, SF, FINAL):
            for match_no, (fa, fb) in rnd.items():
                a, b = winners[fa], winners[fb]
                entrants[match_no] = (a, b)
                winners[match_no] = ko(a, b)

        for rnd_name, nums in [("R32", [m for m, _, _ in R32]), ("R16", R16),
                               ("QF", QF), ("SF", SF), ("final", FINAL)]:
            for m_no in nums:
                for t in entrants[m_no]:
                    reach[t][rnd_name] += 1
        reach[winners[104]]["champion"] += 1
        if return_nodes:
            for m_no, (a, b) in entrants.items():
                node_counts[m_no][a] += 1
                node_counts[m_no][b] += 1
            node_counts["champion"][winners[104]] += 1

    members = {t: g for g, ts in GROUPS.items() for t in ts}
    rows = {}
    for t in ALL_TEAMS:
        p1, p2, p3, p4 = pos_counts[t] / n_sims
        b3 = best_third[t] / n_sims
        rows[t] = {"group": members[t], "1st": p1, "2nd": p2, "3rd": p3,
                   "4th": p4, "top2": p1 + p2, "best_third": b3,
                   "advance": p1 + p2 + b3, "xPts": pts_sum[t] / n_sims,
                   **{r: reach[t][r] / n_sims
                      for r in ("R32", "R16", "QF", "SF", "final",
                                "champion")}}
    df = pd.DataFrame.from_dict(rows, orient="index")
    if return_nodes:
        nodes = {m: {t: c / n_sims for t, c in teams.items()}
                 for m, teams in node_counts.items()}
        return df, nodes
    return df


def bracket_tree_order() -> dict[str, list[int]]:
    """Match numbers per round, ordered by a depth-first walk from the
    final so vertically adjacent nodes share a parent (bracket layout)."""
    feeders = {**R16, **QF, **SF, **FINAL}
    order = {"R32": [], "R16": [], "QF": [], "SF": [], "final": []}
    rnd_of = {m: "R16" for m in R16} | {m: "QF" for m in QF} \
        | {m: "SF" for m in SF} | {104: "final"}

    def visit(m):
        if m in feeders:
            a, b = feeders[m]
            visit(a)
            visit(b)
            order[rnd_of[m]].append(m)
        else:
            order["R32"].append(m)

    visit(104)
    return order
