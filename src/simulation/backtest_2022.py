"""Backtest: simulate WC2022 as-of 2022-11-19 and score against reality.

Question under test: are our COMPOUNDED tournament odds overconfident,
even though per-match probabilities are calibrated? And does injecting
rating uncertainty (Elo jitter ~ N(0, sigma) per team per simulated
tournament) improve agreement with what actually happened?

Everything is as-of: Elo from matches strictly before the cutoff, model
trained on 2010 -> cutoff, scorelines from 2010 -> cutoff. The leakage
discipline in features.csv makes the training set a simple date filter.

For jitter speed we hand-roll the pipeline's math (standardize -> linear
scores -> softmax): only elo_diff changes per simulation, so each pair's
non-Elo score is precomputed and the jitter enters as a scalar adjustment.
Verified against model.predict_proba at sigma=0.

Usage:
    .venv/bin/python -m src.simulation.backtest_2022 [n_sims]
"""

import sys
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED
from src.features.elo import compute_elo
from src.models.baselines import load_features
from src.models.evaluate import CLASSES
from src.models.train import fit_model, make_xy
from src.simulation.simulate import FEATURES, rank_teams, scoreline_sampler

CUTOFF = "2022-11-19"  # tournament began 2022-11-20
HOST = "Qatar"

WC2022_GROUPS = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}
TEAMS = sorted(t for g in WC2022_GROUPS.values() for t in g)

# R16 template (match -> (group, rank) pairs), then feeders. Verified
# against the actual 2022 fixtures (e.g. m49 NED-USA, m60 ENG-FRA... wait,
# m60 was FRA(1D)-POL? no: m51). See Wikipedia 2022 knockout stage.
R16 = {49: (("A", 0), ("B", 1)), 50: (("C", 0), ("D", 1)),
       51: (("D", 0), ("C", 1)), 52: (("B", 0), ("A", 1)),
       53: (("E", 0), ("F", 1)), 54: (("G", 0), ("H", 1)),
       55: (("F", 0), ("E", 1)), 56: (("H", 0), ("G", 1))}
QF = {57: (53, 54), 58: (49, 50), 59: (55, 56), 60: (51, 52)}
SF = {61: (57, 58), 62: (59, 60)}
FINAL = {63: (61, 62)}
ROUNDS = ["R16", "QF", "SF", "final", "champion"]

ACTUAL = {
    "R16": {"Netherlands", "Senegal", "England", "United States", "Argentina",
            "Poland", "France", "Australia", "Japan", "Spain", "Morocco",
            "Croatia", "Brazil", "Switzerland", "Portugal", "South Korea"},
    "QF": {"Netherlands", "Argentina", "France", "England", "Croatia",
           "Brazil", "Morocco", "Portugal"},
    "SF": {"Argentina", "Croatia", "France", "Morocco"},
    "final": {"Argentina", "France"},
    "champion": {"Argentina"},
}


# --- As-of state and hand-rolled predictor ----------------------------------

def as_of_state():
    matches = pd.read_csv(DATA_PROCESSED / "matches.csv",
                          index_col="match_id", parse_dates=["date"])
    matches = matches[matches["date"] < CUTOFF]
    _, ratings = compute_elo(matches)

    hist = defaultdict(list)
    for m in matches.itertuples():
        gd = m.home_score - m.away_score
        hist[m.home_team].append((3 if gd > 0 else 1 if gd == 0 else 0, gd))
        hist[m.away_team].append((3 if gd < 0 else 1 if gd == 0 else 0, -gd))
    state = {}
    for t in TEAMS:
        h = hist[t][-10:]
        state[t] = {"elo": ratings[t],
                    "ppg": sum(p for p, _ in h) / len(h),
                    "gd_form": sum(g for _, g in h) / len(h)}
    return pd.DataFrame.from_dict(state, orient="index")


def train_as_of():
    df = load_features()
    tr = df[(df["date"] >= "2010-01-01") & (df["date"] < CUTOFF)]
    X, y = make_xy(tr)
    return fit_model(X, y)


class PairPredictor:
    """Softmax over precomputed per-pair linear scores; Elo jitter enters
    as delta_z = coef_elo_scaled * (jitter_a - jitter_b)."""

    def __init__(self, model, state):
        scaler = model.named_steps["standardscaler"]
        lr = model.named_steps["logisticregression"]
        order = [list(lr.classes_).index(c) for c in CLASSES]
        self.coef = lr.coef_[order] / scaler.scale_     # (3, n_features)
        self.intercept = (lr.intercept_[order]
                          - lr.coef_[order] @ (scaler.mean_ / scaler.scale_))
        self.i_elo = FEATURES.index("elo_diff")
        self.coef_elo = self.coef[:, self.i_elo]        # (3,)

        self.pairs = [(a, b) for a in TEAMS for b in TEAMS if a != b]
        X = np.array([[state.loc[a, "elo"] - state.loc[b, "elo"],
                       float(a != HOST), 1, 0, 0, 0,
                       state.loc[a, "ppg"] - state.loc[b, "ppg"],
                       state.loc[a, "gd_form"] - state.loc[b, "gd_form"],
                       0.0] for a, b in self.pairs])
        self.z_base = {p: self.coef @ X[i] + self.intercept
                       for i, p in enumerate(self.pairs)}

    def probs(self, a, b, jitter):
        """Symmetrized (host-aware) [p_a, p_draw, p_b] under current jitter."""
        d = jitter[a] - jitter[b]
        z_ab = self.z_base[(a, b)] + self.coef_elo * d
        z_ba = self.z_base[(b, a)] - self.coef_elo * d
        p_ab = _softmax(z_ab)
        p_ba = _softmax(z_ba)[::-1]
        if a == HOST:
            return p_ab
        if b == HOST:
            return p_ba
        return (p_ab + p_ba) / 2


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# --- Tournament -------------------------------------------------------------

def simulate_once(pred, jitter, sample_score, rng):
    placements = {}
    for g, teams in WC2022_GROUPS.items():
        stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        results = {}
        for a, b in combinations(teams, 2):
            p = pred.probs(a, b, jitter)
            oc = rng.choice(3, p=p / p.sum())
            ga, gb = sample_score(oc)
            results[(a, b)] = (ga, gb)
            stats[a]["pts"] += 3 if ga > gb else (1 if ga == gb else 0)
            stats[b]["pts"] += 3 if gb > ga else (1 if ga == gb else 0)
            stats[a]["gd"] += ga - gb; stats[b]["gd"] += gb - ga
            stats[a]["gf"] += ga;      stats[b]["gf"] += gb
        placements[g] = rank_teams(stats, rng, h2h_results=results)

    def ko(a, b):
        p = pred.probs(a, b, jitter)
        oc = rng.choice(3, p=p / p.sum())
        if oc == 1:
            return a if rng.random() < p[0] / (p[0] + p[2]) else b
        return a if oc == 0 else b

    winners, reached = {}, defaultdict(set)
    for m, ((ga, ra), (gb, rb)) in R16.items():
        a, b = placements[ga][ra], placements[gb][rb]
        reached["R16"].update((a, b))
        winners[m] = ko(a, b)
    for rnd_name, rnd in (("QF", QF), ("SF", SF), ("final", FINAL)):
        for m, (fa, fb) in rnd.items():
            a, b = winners[fa], winners[fb]
            reached[rnd_name].update((a, b))
            winners[m] = ko(a, b)
    reached["champion"] = {winners[63]}
    return reached


def run(n_sims=10_000, sigmas=(0.0, 50.0, 100.0)):
    print(f"As-of {CUTOFF}: recomputing Elo, training model...")
    state = as_of_state()
    model = train_as_of()
    pred = PairPredictor(model, state)

    # Sanity: hand-rolled softmax must equal sklearn at zero jitter.
    a, b = "Brazil", "Serbia"
    X1 = pd.DataFrame([[state.loc[a, "elo"] - state.loc[b, "elo"], 1.0,
                        1, 0, 0, 0,
                        state.loc[a, "ppg"] - state.loc[b, "ppg"],
                        state.loc[a, "gd_form"] - state.loc[b, "gd_form"],
                        0.0]], columns=FEATURES)
    from src.models.train import proba_in_class_order
    zero = defaultdict(float)
    z_ab = pred.z_base[(a, b)]
    assert np.allclose(_softmax(z_ab), proba_in_class_order(model, X1)[0],
                       atol=1e-9), "hand-rolled predictor != sklearn"

    summary = {}
    for sigma in sigmas:
        rng = np.random.default_rng(2022)
        sample_score = scoreline_sampler(rng, before=CUTOFF)
        counts = {t: dict.fromkeys(ROUNDS, 0) for t in TEAMS}
        for _ in range(n_sims):
            jitter = (defaultdict(float) if sigma == 0 else
                      defaultdict(float, zip(TEAMS, rng.normal(0, sigma, len(TEAMS)))))
            for rnd, ts in simulate_once(pred, jitter, sample_score, rng).items():
                for t in ts:
                    counts[t][rnd] += 1
        probs = pd.DataFrame(counts).T / n_sims

        ll = br = 0.0
        for rnd in ROUNDS:
            p = probs[rnd].clip(1e-6, 1 - 1e-6)
            y = probs.index.isin(ACTUAL[rnd]).astype(float)
            ll += float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
            br += float(((p - y) ** 2).mean())
        fav = probs["champion"].idxmax()
        summary[f"sigma={sigma:.0f}"] = {
            "log_loss_sum": round(ll, 4), "brier_sum": round(br, 4),
            "favorite": fav,
            "fav_champ_%": round(100 * probs["champion"].max(), 1),
            "ARG_champ_%": round(100 * probs.loc["Argentina", "champion"], 1),
            "ARG_rank": int((probs["champion"] >
                             probs.loc["Argentina", "champion"]).sum()) + 1,
        }
        print(f"  sigma={sigma:>5.0f}: done")

    table = pd.DataFrame(summary).T
    print("\n=== WC2022 backtest: predicted odds vs reality "
          f"({n_sims:,} sims; lower loss = better) ===")
    print(table.to_string())
    return table


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000)
