# World Cup 2026 Predictor

Predicts international match outcomes (home win / draw / away win) with
calibrated probabilities, then Monte Carlo–simulates the 2026 World Cup.

## Pipeline

1. **Data** (`src/data/`) — download + clean match history (Kaggle, martj42 dataset)
2. **Features** (`src/features/`) — Elo (computed from scratch), recent form,
   match importance. Strictly pre-match information only.
3. **Models** (`src/models/`) — baselines, multinomial logistic regression,
   temporal validation, probability calibration
4. **Simulation** (`src/simulation/`) — 48-team 2026 bracket, 10k+ Monte Carlo runs
5. **Dashboard** (`app/`) — Streamlit

Notebooks in `notebooks/` are for exploration; anything load-bearing gets
promoted to `src/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Kaggle token at ~/.kaggle/kaggle.json, then:
.venv/bin/python -m src.data.download
```

## Methodology notes

- **Temporal validation only** — train on older matches, validate on newer.
- **No feature leakage** — every feature uses only pre-match information.
- **Training window is a hyperparameter** — full history warms up Elo;
  the supervised model trains on 1990+ (vs 2000+/2010+ by validation log-loss).
- **Probability quality over accuracy** — log-loss, Brier score, calibration curves.
