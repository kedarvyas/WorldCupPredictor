"""Monte Carlo simulation of the 2026 World Cup.

Per simulated tournament: sample all 72 group matches from the model's
probability vectors, rank groups (points/GD/goals/h2h/lots), select the
8 best thirds, allocate them to R32 slots (bracket.allocate_thirds), then
play the knockout bracket to the final.

Modeling choices (confirmed):
- Ratings/form FROZEN at tournament start.
- Neutral matches are SYMMETRIZED: average of model(A vs B) and the
  mirror of model(B vs A) — kills the spurious home-label signal found
  in EDA. Hosts get genuine home treatment in all their matches.
- Knockout draws -> strength-weighted coin flip: P(A wins ET/pens) =
  P(A win) / (P(A win) + P(B win)).
- Scorelines (needed for GD/goals tiebreakers): champion samples the
  empirical modern-era distribution conditional on the sampled outcome
  (team-blind); with --dc, outcomes AND scorelines come from the fitted
  Dixon-Coles tau-corrected score matrix per pairing (team-specific, and
  mutually consistent by construction). Fair-play tiebreaker unavailable
  -> drawing of lots (random).

Usage:
    .venv/bin/python -m src.simulation.simulate [n_sims] [--dc]
"""

import sys
from collections import Counter, defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from src.config import (DATA_PROCESSED, FIXTURES_FROZEN, MODELS_DIR,
                        PROJECT_ROOT, TOURNAMENT_START)
from src.models.train import TIERS, proba_in_class_order
from src.simulation.bracket import (FINAL, GROUPS, HOSTS, QF, R16, R32, SF,
                                    allocate_thirds)

FEATURES = ["elo_diff", "neutral", *(f"tier_{t}" for t in TIERS),
            "ppg_diff", "gd_form_diff", "rest_diff"]
ALL_TEAMS = sorted(t for g in GROUPS.values() for t in g)
ROUNDS = ["group_top2", "third_qualified", "R32", "R16", "QF", "SF",
          "final", "champion"]


# --- Frozen team state ------------------------------------------------------

def team_state() -> pd.DataFrame:
    elo = pd.read_csv(DATA_PROCESSED / "current_elo.csv", index_col="team")["elo"]
    matches = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    hist = defaultdict(lambda: deque(maxlen=10))
    for m in matches.itertuples():
        gd = m.home_score - m.away_score
        hist[m.home_team].append((3 if gd > 0 else 1 if gd == 0 else 0, gd))
        hist[m.away_team].append((3 if gd < 0 else 1 if gd == 0 else 0, -gd))
    rows = {}
    for t in ALL_TEAMS:
        h = hist[t]
        assert h, f"no match history for {t}"
        rows[t] = {"elo": elo[t],
                   "ppg": sum(p for p, _ in h) / len(h),
                   "gd_form": sum(g for _, g in h) / len(h)}
    return pd.DataFrame.from_dict(rows, orient="index")


# --- Probability tables (computed once; sims only sample) -------------------

def _feature_row(state, home, away, neutral):
    return [state.loc[home, "elo"] - state.loc[away, "elo"], int(neutral),
            1, 0, 0, 0,  # tier_world_cup one-hot
            state.loc[home, "ppg"] - state.loc[away, "ppg"],
            state.loc[home, "gd_form"] - state.loc[away, "gd_form"],
            0.0]  # rest_diff: shared tournament schedule


def build_prob_tables(model, state):
    """Returns {(team_a, team_b): [p_a_win, p_draw, p_b_win]} for every
    ordered pair, host-aware and symmetrized, plus the 72 group fixtures
    keyed by (home, away) with their real venue flags."""
    pairs = [(a, b) for a in ALL_TEAMS for b in ALL_TEAMS if a != b]
    X = pd.DataFrame(
        [_feature_row(state, a, b, neutral=(a not in HOSTS)) for a, b in pairs],
        columns=FEATURES)
    p = proba_in_class_order(model, X)  # columns: home_win, draw, away_win
    raw = {pair: p[i] for i, pair in enumerate(pairs)}

    table = {}
    for a, b in pairs:
        if a in HOSTS and b not in HOSTS:
            table[(a, b)] = raw[(a, b)]                      # a genuinely home
        elif b in HOSTS and a not in HOSTS:
            table[(a, b)] = raw[(b, a)][::-1]                # mirror of b home
        else:  # both neutral (or host vs host): symmetrize the label bias
            table[(a, b)] = (raw[(a, b)] + raw[(b, a)][::-1]) / 2
    return table


def scoreline_sampler(rng, before=TOURNAMENT_START):
    """Empirical scoreline distributions conditional on outcome, 2010+.
    `before` caps the window: TOURNAMENT_START by default (frozen-parameters
    rule — tournament results must never feed the distributions), or an
    earlier cutoff for as-of backtests."""
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[(m["date"] >= "2010-01-01") & (m["date"] < before)]
    gd = m["home_score"] - m["away_score"]
    outcomes = np.select([gd > 0, gd < 0], ["win", "loss"], "draw")
    dists = {}
    for oc in ("win", "draw", "loss"):
        counts = Counter(zip(m.loc[outcomes == oc, "home_score"].astype(int),
                             m.loc[outcomes == oc, "away_score"].astype(int)))
        lines = list(counts)
        w = np.array(list(counts.values()), dtype=float)
        dists[oc] = (lines, w / w.sum())

    def sample(outcome_idx, pair=None):  # 0 = first wins, 1 draw, 2 second
        oc = ("win", "draw", "loss")[outcome_idx]   # pair ignored: global dist
        lines, w = dists[oc]
        return lines[rng.choice(len(lines), p=w)]
    return sample


def dc_tables(dc, rng):
    """Outcome table + PAIR-AWARE scoreline sampler from a fitted
    Dixon-Coles model. Outcomes and scorelines both come from the same
    tau-corrected score matrix per pairing, so tiebreaker goal differences
    are team-specific and consistent with the sampled outcomes.

    No symmetrization needed: at neutral venues DC is structurally
    symmetric (M(a,b) == M(b,a).T), unlike the LR's home-label bias."""
    probs, regions = {}, {}
    for a in ALL_TEAMS:
        for b in ALL_TEAMS:
            if a == b:
                continue
            if b in HOSTS and a not in HOSTS:
                M = dc.score_matrix(b, a, true_home=True).T
            else:
                M = dc.score_matrix(a, b,
                                    true_home=(a in HOSTS and b not in HOSTS))
            probs[(a, b)] = np.array([np.tril(M, -1).sum(), np.trace(M),
                                      np.triu(M, 1).sum()])
            per_oc = []
            n = M.shape[0]
            for mask in (np.tri(n, k=-1, dtype=bool),        # a wins: i > j
                         np.eye(n, dtype=bool),               # draw
                         np.tri(n, k=-1, dtype=bool).T):      # b wins: i < j
                lines = [(i, j) for i in range(n) for j in range(n)
                         if mask[i, j]]
                w = np.array([M[i, j] for i, j in lines])
                per_oc.append((lines, w / w.sum()))
            regions[(a, b)] = per_oc

    def sample(outcome_idx, pair):
        lines, w = regions[pair][outcome_idx]
        return lines[rng.choice(len(lines), p=w)]
    return probs, sample


# --- One tournament ---------------------------------------------------------

def rank_teams(stats, rng, h2h_results=None, depth=0):
    """Order teams by points, GD, goals; full ties -> head-to-head among the
    tied (same criteria), then drawing of lots."""
    order = sorted(stats, key=lambda t: (-stats[t]["pts"], -stats[t]["gd"],
                                         -stats[t]["gf"], rng.random()))
    if h2h_results is None or depth > 0:
        return order
    ranked, i = [], 0
    while i < len(order):
        key = lambda t: (stats[t]["pts"], stats[t]["gd"], stats[t]["gf"])
        tied = [t for t in order[i:] if key(t) == key(order[i])]
        if len(tied) > 1:
            sub = {t: {"pts": 0, "gd": 0, "gf": 0} for t in tied}
            for (a, b), (ga, gb) in h2h_results.items():
                if a in sub and b in sub:
                    sub[a]["pts"] += 3 if ga > gb else (1 if ga == gb else 0)
                    sub[b]["pts"] += 3 if gb > ga else (1 if ga == gb else 0)
                    sub[a]["gd"] += ga - gb; sub[b]["gd"] += gb - ga
                    sub[a]["gf"] += ga;      sub[b]["gf"] += gb
            tied = rank_teams(sub, rng, depth=1)
        ranked.extend(tied)
        i = len(ranked)
    return ranked


def simulate_once(probs, fixtures, sample_score, rng):
    placements = {}   # group letter -> [1st, 2nd, 3rd, 4th]
    third_stats = {}
    for g, teams in GROUPS.items():
        stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        results = {}
        for a, b in fixtures[g]:
            p = probs[(a, b)]
            oc = rng.choice(3, p=p / p.sum())
            ga, gb = sample_score(oc, (a, b))
            results[(a, b)] = (ga, gb)
            stats[a]["pts"] += 3 if ga > gb else (1 if ga == gb else 0)
            stats[b]["pts"] += 3 if gb > ga else (1 if ga == gb else 0)
            stats[a]["gd"] += ga - gb; stats[b]["gd"] += gb - ga
            stats[a]["gf"] += ga;      stats[b]["gf"] += gb
        placements[g] = rank_teams(stats, rng, h2h_results=results)
        third_stats[g] = stats[placements[g][2]]

    thirds_ranked = rank_teams(third_stats, rng)  # keys are group letters
    qualified = set(thirds_ranked[:8])
    third_slot = allocate_thirds(qualified)

    def ko_winner(a, b):
        p = probs[(a, b)]
        oc = rng.choice(3, p=p / p.sum())
        if oc == 1:  # draw -> strength-weighted ET/pens coin flip
            return a if rng.random() < p[0] / (p[0] + p[2]) else b
        return a if oc == 0 else b

    winners = {}
    entrants = {}
    for match_no, sa, sb in R32:
        pick = lambda s: (placements[s[1]][0] if s[0] == "W" else
                          placements[s[1]][1] if s[0] == "R" else
                          placements[third_slot[match_no]][2])
        a, b = pick(sa), pick(sb)
        entrants[match_no] = (a, b)
        winners[match_no] = ko_winner(a, b)
    for rnd in (R16, QF, SF, FINAL):
        for match_no, (fa, fb) in rnd.items():
            a, b = winners[fa], winners[fb]
            entrants[match_no] = (a, b)
            winners[match_no] = ko_winner(a, b)

    reached = defaultdict(set)
    for g in GROUPS:
        reached["group_top2"].update(placements[g][:2])
    reached["third_qualified"].update(placements[g][2] for g in qualified)
    for rnd_name, nums in [("R32", [m for m, _, _ in R32]), ("R16", R16),
                           ("QF", QF), ("SF", SF), ("final", FINAL)]:
        for m in nums:
            reached[rnd_name].update(entrants[m])
    reached["champion"] = {winners[104]}
    return reached


def run(n_sims=10_000, seed=2026, use_dc=False):
    rng = np.random.default_rng(seed)
    if use_dc:
        from src.models.dixon_coles import TRAIN_START, DixonColes
        matches = pd.read_csv(DATA_PROCESSED / "matches.csv",
                              parse_dates=["date"])
        dc = DixonColes(half_life_years=10.0).fit(
            matches[(matches["date"] >= TRAIN_START)
                    & (matches["date"] < TOURNAMENT_START)])
        print(f"DC fit: gamma={dc.gamma:.3f} rho={dc.rho:.4f}")
        probs, sample_score = dc_tables(dc, rng)
    else:
        model = joblib.load(MODELS_DIR / "outcome_model.joblib")
        probs = build_prob_tables(model, team_state())
        sample_score = scoreline_sampler(rng)

    fx = pd.read_csv(FIXTURES_FROZEN)
    fixtures = {g: [] for g in GROUPS}
    for r in fx.itertuples():
        g = next(k for k, t in GROUPS.items() if r.home_team in t)
        fixtures[g].append((r.home_team, r.away_team))
    assert all(len(f) == 6 for f in fixtures.values())

    counts = {t: dict.fromkeys(ROUNDS, 0) for t in ALL_TEAMS}
    for i in range(n_sims):
        for rnd, teams in simulate_once(probs, fixtures, sample_score, rng).items():
            for t in teams:
                counts[t][rnd] += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{n_sims:,} tournaments simulated")

    out = (pd.DataFrame(counts).T / n_sims).sort_values("champion",
                                                        ascending=False)
    dest = PROJECT_ROOT / "reports" / ("sim_2026_dc.csv" if use_dc
                                       else "sim_2026.csv")
    dest.parent.mkdir(exist_ok=True)
    out.to_csv(dest)
    print(f"\nSaved {dest.relative_to(PROJECT_ROOT)} ({n_sims:,} sims)")
    print("\n=== Title odds (top 15) ===")
    cols = ["R32", "R16", "QF", "SF", "final", "champion"]
    print((out[cols].head(15) * 100).round(1).to_string())
    return out


if __name__ == "__main__":
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    run(int(nums[0]) if nums else 10_000, use_dc="--dc" in sys.argv)
