"""Reliability curves: is a stated 70% really a 70%?

For each class, bin validation predictions by stated probability and plot
observed frequency against it. Calibrated = on the diagonal.
Also reports per-class predicted vs actual outcome shares.

Usage:
    .venv/bin/python -m src.models.calibration
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROJECT_ROOT, VALIDATION_START
from src.models.baselines import load_features
from src.models.evaluate import CLASSES
from src.models.train import make_xy, proba_in_class_order

FIGURES = PROJECT_ROOT / "reports" / "figures"


def run():
    df = load_features()
    val = df[df["date"] >= VALIDATION_START]
    X_val, y_val = make_xy(val)
    model = joblib.load(MODELS_DIR / "outcome_model.joblib")
    probs = proba_in_class_order(model, X_val)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    bins = np.linspace(0, 1, 11)
    for i, (cls, ax) in enumerate(zip(CLASSES, axes)):
        p, hit = probs[:, i], (y_val == cls).to_numpy()
        idx = np.digitize(p, bins) - 1
        centers, observed, counts = [], [], []
        for b in range(10):
            mask = idx == b
            if mask.sum() >= 20:  # skip near-empty bins (noise)
                centers.append(p[mask].mean())
                observed.append(hit[mask].mean())
                counts.append(int(mask.sum()))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
        ax.plot(centers, observed, "o-", label="model")
        ax.set_title(f"{cls}  (pred {p.mean():.3f} vs actual {hit.mean():.3f})")
        ax.set_xlabel("stated probability")
        if i == 0:
            ax.set_ylabel("observed frequency")
        ax.legend()
        print(f"{cls:9s}: mean predicted {p.mean():.3f}, actual {hit.mean():.3f}, "
              f"bins used {len(centers)}, max |gap| "
              f"{max(abs(c - o) for c, o in zip(centers, observed)):.3f}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "calibration.png"
    fig.suptitle(f"Reliability curves — validation {VALIDATION_START}+ "
                 f"(n={len(y_val):,})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
