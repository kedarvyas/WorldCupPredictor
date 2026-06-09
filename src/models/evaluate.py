"""Probability-quality metrics for three-class outcome predictions.

Accuracy only checks the argmax. Log-loss and Brier score the full
probability vector — which is what the Monte Carlo simulation will consume,
so probability quality is the metric that matters here.

- log-loss: -log P(true class). Unbounded; brutally punishes confident
  mistakes (P=0 on the realized outcome costs infinity).
- Brier (multiclass): mean squared error between the probability vector and
  the one-hot outcome. Bounded [0, 2]; gentler tails than log-loss.
"""

import numpy as np
import pandas as pd

CLASSES = ["home_win", "draw", "away_win"]


def evaluate_probs(y_true: pd.Series, probs: np.ndarray) -> dict:
    """probs: (n, 3) array in CLASSES order.

    Log-loss is computed directly from the one-hot matrix rather than via
    sklearn.metrics.log_loss: sklearn assumes probability columns follow
    LEXICOGRAPHIC label order, which silently mis-scored our column order
    (home/away swapped). Direct computation has no ordering convention.
    """
    y_onehot = pd.get_dummies(y_true)[CLASSES].to_numpy(dtype=float)
    p_true = np.clip((probs * y_onehot).sum(axis=1), 1e-15, 1.0)
    return {
        "accuracy": float((probs.argmax(axis=1) ==
                           y_onehot.argmax(axis=1)).mean()),
        "log_loss": float(-np.log(p_true).mean()),
        "brier": float(((probs - y_onehot) ** 2).sum(axis=1).mean()),
        "n": len(y_true),
    }


def results_table(rows: dict[str, dict]) -> pd.DataFrame:
    return (pd.DataFrame(rows).T[["accuracy", "log_loss", "brier", "n"]]
              .round(4))
