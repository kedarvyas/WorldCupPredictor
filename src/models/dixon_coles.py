"""Dixon-Coles (1997) goals model, fit from scratch.

Home goals ~ Poisson(lambda), away ~ Poisson(mu):
    log lambda = attack_home - defense_away + gamma * is_true_home
    log mu     = attack_away - defense_home

plus the tau correction: independent Poissons underpredict low-scoring
draws, so the (0,0), (1,0), (0,1), (1,1) cells are adjusted by a fitted
dependence parameter rho (mass-preserving by construction).

Fit by maximum likelihood with exponential time decay (recent matches
count more), via L-BFGS-B with an analytic gradient — finite differences
over ~460 parameters would be impractical (verified against finite
differences in tests/test_dixon_coles.py).

Identifiability: adding a constant c to every attack AND every defense
leaves every lambda/mu unchanged (a flat direction in the likelihood);
a quadratic penalty on mean(attack) pins it.

Usage:
    .venv/bin/python -m src.models.dixon_coles            # validate vs LR
    .venv/bin/python -m src.models.dixon_coles --lock     # freeze WC2026 forecast
"""

import sys
from math import lgamma

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import (DATA_PROCESSED, FIXTURES_FROZEN, PROJECT_ROOT,
                        TOURNAMENT_START, VALIDATION_START)

MAX_GOALS = 12          # score-matrix truncation
RHO_BOUND = 0.2         # keeps all tau cells positive for realistic rates
MEAN_PENALTY = 100.0    # weight on mean(attack)^2 identifiability pin
RIDGE = 5.0             # Gaussian prior toward the average team: pins the
                        # 1-match micro-teams (Ryukyu, Kernow, ...) whose
                        # unregularized MLE params explode exp() downstream
TRAIN_START = "2010-01-01"

DC_LOCK_PATH = PROJECT_ROOT / "reports" / "predictions_2026_dc_locked.csv"


class DixonColes:
    def __init__(self, half_life_years=5.0):
        self.half_life = half_life_years
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.gamma = 0.0
        self.rho = 0.0

    # --- Fitting -------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "DixonColes":
        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        h = matches["home_team"].map(idx).to_numpy()
        a = matches["away_team"].map(idx).to_numpy()
        x = matches["home_score"].to_numpy(dtype=float)
        y = matches["away_score"].to_numpy(dtype=float)
        home = (~matches["neutral"]).to_numpy(dtype=float)
        age = (matches["date"].max() - matches["date"]).dt.days / 365.25
        w = np.exp(-np.log(2.0) / self.half_life * age.to_numpy())

        m00 = (x == 0) & (y == 0)
        m10 = (x == 1) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m11 = (x == 1) & (y == 1)

        def nll_grad(p):
            alpha, beta = p[:n], p[n:2 * n]
            gamma, rho = p[-2], p[-1]
            lam = np.exp(alpha[h] - beta[a] + gamma * home)
            mu = np.exp(alpha[a] - beta[h])

            tau = np.ones_like(lam)
            tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau[m10] = 1.0 + mu[m10] * rho
            tau[m01] = 1.0 + lam[m01] * rho
            tau[m11] = 1.0 - rho
            tau = np.clip(tau, 1e-10, None)

            ll = w * (x * np.log(lam) - lam + y * np.log(mu) - mu + np.log(tau))
            pen = (MEAN_PENALTY * alpha.mean() ** 2
                   + RIDGE * (alpha @ alpha + beta @ beta))

            # d log(tau) / d log(lam or mu), nonzero only in the low cells
            t_lam = np.zeros_like(lam)
            t_mu = np.zeros_like(mu)
            t_lam[m00] = -lam[m00] * mu[m00] * rho / tau[m00]
            t_mu[m00] = t_lam[m00]
            t_mu[m10] = mu[m10] * rho / tau[m10]
            t_lam[m01] = lam[m01] * rho / tau[m01]

            g_lam = w * (x - lam + t_lam)   # dLL / d log(lam), per match
            g_mu = w * (y - mu + t_mu)      # dLL / d log(mu)

            ga = np.zeros(n)
            gb = np.zeros(n)
            np.add.at(ga, h, g_lam)         # log lam = alpha_h - beta_a + ...
            np.add.at(ga, a, g_mu)          # log mu  = alpha_a - beta_h
            np.add.at(gb, a, -g_lam)
            np.add.at(gb, h, -g_mu)
            ga -= 2.0 * MEAN_PENALTY * alpha.mean() / n + 2.0 * RIDGE * alpha
            gb -= 2.0 * RIDGE * beta

            t_rho = np.zeros_like(lam)
            t_rho[m00] = -lam[m00] * mu[m00] / tau[m00]
            t_rho[m10] = mu[m10] / tau[m10]
            t_rho[m01] = lam[m01] / tau[m01]
            t_rho[m11] = -1.0 / tau[m11]

            grad = np.concatenate([ga, gb,
                                   [np.sum(home * g_lam)],
                                   [np.sum(w * t_rho)]])
            return -(ll.sum() - pen), -grad

        p0 = np.zeros(2 * n + 2)
        p0[-2] = 0.25  # gamma: home teams score more, start plausible
        bounds = ([(None, None)] * (2 * n + 1) + [(-RHO_BOUND, RHO_BOUND)])
        res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 1000})
        if not res.success:
            print(f"WARNING: optimizer stopped: {res.message}")
        self.attack = dict(zip(self.teams, res.x[:n]))
        self.defense = dict(zip(self.teams, res.x[n:2 * n]))
        self.gamma, self.rho = res.x[-2], res.x[-1]
        return self

    # --- Prediction ----------------------------------------------------------

    def rates(self, home_team, away_team, true_home=False):
        """(lambda, mu). Unseen teams get 0 = roughly average params."""
        a_h = self.attack.get(home_team, 0.0)
        d_h = self.defense.get(home_team, 0.0)
        a_a = self.attack.get(away_team, 0.0)
        d_a = self.defense.get(away_team, 0.0)
        lam = np.exp(a_h - d_a + self.gamma * true_home)
        mu = np.exp(a_a - d_h)
        # Belt and braces alongside RIDGE: no plausible international match
        # has a 30-goal rate; the cap keeps exp()/Poisson tails finite.
        return min(lam, 30.0), min(mu, 30.0)

    def score_matrix(self, home_team, away_team, true_home=False):
        """P(home goals=i, away goals=j), tau-corrected, renormalized."""
        lam, mu = self.rates(home_team, away_team, true_home)
        k = np.arange(MAX_GOALS + 1)
        logf = np.array([lgamma(i + 1) for i in k])
        ph = np.exp(k * np.log(lam) - lam - logf)
        pa = np.exp(k * np.log(mu) - mu - logf)
        M = np.outer(ph, pa)
        M[0, 0] *= 1.0 - lam * mu * self.rho
        M[1, 0] *= 1.0 + mu * self.rho
        M[0, 1] *= 1.0 + lam * self.rho
        M[1, 1] *= 1.0 - self.rho
        return M / M.sum()

    def outcome_probs(self, home_team, away_team, true_home=False):
        """[P(home win), P(draw), P(away win)] from the score matrix."""
        M = self.score_matrix(home_team, away_team, true_home)
        return np.array([np.tril(M, -1).sum(), np.trace(M),
                         np.triu(M, 1).sum()])


# --- Validation against the LR champion --------------------------------------

def load_matches():
    return pd.read_csv(DATA_PROCESSED / "matches.csv", index_col="match_id",
                       parse_dates=["date"])


def validate():
    from src.models.baselines import load_features
    from src.models.evaluate import evaluate_probs, results_table
    from src.models.train import make_xy

    matches = load_matches()
    train = matches[(matches["date"] >= TRAIN_START)
                    & (matches["date"] < VALIDATION_START)]

    # Same validation rows the LR was scored on (make_xy's NaN-form mask),
    # so the comparison is apples-to-apples.
    df = load_features()
    val = df[df["date"] >= VALIDATION_START]
    X_val, y_val = make_xy(val)
    val_rows = matches.loc[X_val.index]

    rows = {}
    for hl in (3.0, 5.0, 10.0):
        dc = DixonColes(half_life_years=hl).fit(train)
        probs = np.array([dc.outcome_probs(r.home_team, r.away_team,
                                           true_home=not r.neutral)
                          for r in val_rows.itertuples()])
        rows[f"DC half-life {hl:.0f}y"] = evaluate_probs(y_val, probs)
        print(f"  fit hl={hl:.0f}y: gamma={dc.gamma:.3f} rho={dc.rho:.4f}")

    rows["LR champion (Phase 4)"] = {"accuracy": 0.6014, "log_loss": 0.8707,
                                     "brier": 0.5117, "n": len(y_val)}
    print()
    print(results_table(rows).sort_values("log_loss").to_string())


# --- Freeze the WC2026 challenger forecast ------------------------------------

def lock():
    from src.simulation.bracket import GROUPS

    if DC_LOCK_PATH.exists() and "--force" not in sys.argv:
        sys.exit(f"{DC_LOCK_PATH.name} already exists — locked.")

    matches = load_matches()
    train = matches[(matches["date"] >= TRAIN_START)
                    & (matches["date"] < TOURNAMENT_START)]
    dc = DixonColes(half_life_years=10.0).fit(train)  # best by validation
    print(f"Full fit: gamma={dc.gamma:.3f} rho={dc.rho:.4f}")

    members = {t: g for g, ts in GROUPS.items() for t in ts}
    fx = pd.read_csv(FIXTURES_FROZEN, parse_dates=["date"])
    rows = []
    for r in fx.itertuples():
        M = dc.score_matrix(r.home_team, r.away_team, true_home=not r.neutral)
        p = np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])
        flat = [((i, j), M[i, j]) for i in range(MAX_GOALS + 1)
                for j in range(MAX_GOALS + 1)]
        (s1, p1), (s2, p2) = sorted(flat, key=lambda t: -t[1])[:2]
        rows.append({
            "date": r.date.date(), "group": members[r.home_team],
            "home_team": r.home_team, "away_team": r.away_team,
            "venue": f"{r.city} ({r.country})",
            "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "pick": ["home", "draw", "away"][int(np.argmax(p))],
            "top1_score": f"{s1[0]}-{s1[1]}", "top1_p": p1,
            "top2_score": f"{s2[0]}-{s2[1]}", "top2_p": p2,
        })
    preds = pd.DataFrame(rows).sort_values(["date", "group"])
    assert len(preds) == 72
    preds.to_csv(DC_LOCK_PATH, index=False)
    print(f"Locked {len(preds)} DC predictions -> {DC_LOCK_PATH.name}")


if __name__ == "__main__":
    if "--lock" in sys.argv:
        lock()
    else:
        validate()
