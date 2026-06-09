"""Multinomial logistic regression with temporal validation.

Experiments (per project spec):
- Training-window start year as a hyperparameter: 1990 vs 2000 vs 2010,
  selected by validation log-loss.
- Recency weighting (exponential decay by year, sklearn sample_weight) as
  an alternative to a hard cutoff.

The model's class order follows sklearn's model.classes_ (alphabetical) —
`proba_in_class_order` re-maps columns to our CLASSES order explicitly,
because implicit column-order assumptions already bit us once (Phase 3).

Usage:
    .venv/bin/python -m src.models.train
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, VALIDATION_START
from src.features.elo import match_tier
from src.models.baselines import b2_elo_curve, load_features
from src.models.evaluate import CLASSES, evaluate_probs, results_table

# Long gaps (off-cycle teams) make raw rest-days an outlier farm; beyond a
# month, extra rest carries no plausible signal.
REST_CLIP_DAYS = 30

TIERS = ["world_cup", "continental", "qualifier", "friendly"]  # ref = "other"


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(index=df.index)
    X["elo_diff"] = df["elo_diff"]
    X["neutral"] = df["neutral"].astype(int)
    tier = df["tournament"].map(match_tier)
    for t in TIERS:
        X[f"tier_{t}"] = (tier == t).astype(int)
    X["ppg_diff"] = df["home_ppg"] - df["away_ppg"]
    X["gd_form_diff"] = df["home_gd_form"] - df["away_gd_form"]
    X["rest_diff"] = (df["home_rest_days"].clip(upper=REST_CLIP_DAYS)
                      - df["away_rest_days"].clip(upper=REST_CLIP_DAYS))
    keep = X.notna().all(axis=1)  # drops new-team matches with no form history
    return X[keep], df.loc[keep, "outcome"]


def fit_model(X, y, sample_weight=None):
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000))
    model.fit(X, y, logisticregression__sample_weight=sample_weight)
    return model


def proba_in_class_order(model, X) -> np.ndarray:
    """predict_proba columns follow model.classes_ (alphabetical); re-map
    to CLASSES order so every consumer sees [home_win, draw, away_win]."""
    raw = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return raw[:, order]


def run():
    df = load_features()
    train_all = df[df["date"] < VALIDATION_START]
    val = df[df["date"] >= VALIDATION_START]
    X_val, y_val = make_xy(val)
    val_year = pd.Timestamp(VALIDATION_START).year

    rows, fitted = {}, {}

    # --- Hard-cutoff windows ------------------------------------------------
    for start in (1990, 2000, 2010):
        tr = train_all[train_all["date"].dt.year >= start]
        X_tr, y_tr = make_xy(tr)
        m = fit_model(X_tr, y_tr)
        name = f"LR window {start}+"
        fitted[name] = m
        rows[name] = evaluate_probs(y_val, proba_in_class_order(m, X_val))

    # --- Recency weighting (1990+, exponential decay by year) ---------------
    X_tr, y_tr = make_xy(train_all)
    years = train_all.loc[X_tr.index, "date"].dt.year
    for half_life in (5, 10, 20):
        w = 0.5 ** ((val_year - years) / half_life)
        m = fit_model(X_tr, y_tr, sample_weight=w)
        name = f"LR 1990+ decay hl={half_life}y"
        fitted[name] = m
        rows[name] = evaluate_probs(y_val, proba_in_class_order(m, X_val))

    # --- Reference: the baseline to beat ------------------------------------
    rows["B2 Elo curve (baseline)"] = evaluate_probs(
        val.loc[X_val.index, "outcome"],
        b2_elo_curve(train_all, val.loc[X_val.index]))

    table = results_table(rows).sort_values("log_loss")
    print(table.to_string())

    best_name = table.drop("B2 Elo curve (baseline)").index[0]
    best = fitted[best_name]
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODELS_DIR / "outcome_model.joblib")
    meta = {"config": best_name, "features": list(X_val.columns),
            "classes_order": CLASSES,
            "validation_log_loss": float(table.loc[best_name, "log_loss"])}
    (MODELS_DIR / "outcome_model.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved best config: {best_name} -> models/outcome_model.joblib")

    # --- Teaching check: does P(draw) now depend on closeness? --------------
    probe = pd.DataFrame(0, index=range(5), columns=X_val.columns)
    probe["elo_diff"] = [-400, -150, 0, 150, 400]
    probe["neutral"] = 1
    p = proba_in_class_order(best, probe)
    print("\nP(draw) by Elo gap (neutral friendly-ish, baseline B2 said 23% flat):")
    for d, pr in zip(probe["elo_diff"], p):
        print(f"  elo_diff {d:+5d}: home {pr[0]:.2f}  draw {pr[1]:.2f}  away {pr[2]:.2f}")
    return best, X_val, y_val


if __name__ == "__main__":
    run()
