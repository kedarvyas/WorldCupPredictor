"""Platt-style recalibration of the DC over/under-2.5 probability.

The validation check showed DC's totals probabilities are calibrated
through the middle but compressed at the tails (predicted 0.29 -> observed
0.39; predicted 0.75 -> 0.70). One-parameter-pair fix: fit
    p' = sigmoid(a + b * logit(p))
on the validation window's (predicted, actual) pairs. b < 1 stretches
tail predictions back toward the middle.

Estimated on validation-window predictions from the validation-window fit,
then applied to tournament predictions from the deployment fit — the
recalibration never sees tournament data (frozen-parameters rule).

Usage:
    .venv/bin/python -m src.models.recalibrate    # fit + save params
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import MODELS_DIR, VALIDATION_START
from src.models.dixon_coles import (MAX_GOALS, TRAIN_START, DixonColes,
                                    load_matches)

PARAMS_PATH = MODELS_DIR / "ou25_recalibration.json"


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def over_prob(dc, home, away, line=2.5, true_home=False) -> float:
    """P(total goals > line) from the DC score matrix, for any .5 line.

    Only the 2.5 line is recalibrated (see recalibrate.fit); other lines are
    raw DC matrix mass and should be labelled as such (display-grade)."""
    M = dc.score_matrix(home, away, true_home)
    tot = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    return float(M[tot > line].sum())


def over25_prob(dc, home, away, true_home=False) -> float:
    return over_prob(dc, home, away, line=2.5, true_home=true_home)


def fit() -> dict:
    m = load_matches()
    train = m[(m["date"] >= TRAIN_START) & (m["date"] < VALIDATION_START)]
    val = m[m["date"] >= VALIDATION_START]
    dc = DixonColes(half_life_years=10.0).fit(train)

    p = np.array([over25_prob(dc, r.home_team, r.away_team,
                              true_home=not r.neutral)
                  for r in val.itertuples()])
    y = (val["home_score"] + val["away_score"] > 2.5).to_numpy(dtype=int)

    lr = LogisticRegression(C=1e6)  # plain Platt: no regularization wanted
    lr.fit(_logit(p).reshape(-1, 1), y)
    params = {"a": float(lr.intercept_[0]), "b": float(lr.coef_[0][0]),
              "fit_window": [VALIDATION_START, str(val["date"].max().date())],
              "n": int(len(y))}

    # Report before/after log-loss on the fit window (in-sample for the
    # recalibration, out-of-sample for DC itself).
    p_cal = apply(p, params)
    for name, probs in (("raw", p), ("recalibrated", p_cal)):
        ll = -np.mean(y * np.log(np.clip(probs, 1e-9, 1))
                      + (1 - y) * np.log(np.clip(1 - probs, 1e-9, 1)))
        print(f"O/U 2.5 log-loss {name}: {ll:.4f}")
    return params


def apply(p, params) -> np.ndarray:
    z = params["a"] + params["b"] * _logit(np.asarray(p, dtype=float))
    return 1.0 / (1.0 + np.exp(-z))


def load_params() -> dict:
    return json.loads(PARAMS_PATH.read_text())


if __name__ == "__main__":
    params = fit()
    PARAMS_PATH.write_text(json.dumps(params, indent=2))
    print(f"a={params['a']:.4f} b={params['b']:.4f} -> {PARAMS_PATH.name}")
