# World Cup 2026 Predictor ⚽

A FIFA World Cup 2026 forecasting system built from scratch: international
match data → calibrated outcome probabilities → a 10,000-run Monte Carlo of
the real 48-team tournament → a live Streamlit dashboard that updates as
results come in and grades its own pre-registered forecasts.

## Two models, openly disagreeing

| | Champion | Challenger |
|---|---|---|
| Approach | Multinomial logistic regression | Dixon-Coles (1997) goals model |
| Core signal | Hand-computed Elo (importance-weighted K, margin-of-victory, home offset) + form, venue, importance | Per-team attack/defense Poisson rates, low-score τ correction, time-decayed MLE with hand-derived gradient |
| Validation | Best overall log-loss (0.871 vs 0.895 Elo-curve baseline) | Best on elite-vs-elite matches (1.016 vs 1.038) |
| 2026 title pick | Spain ~35% | Brazil ~21% |

The disagreement is displayed, diagnosed (DC's 10-year memory vs Elo's
recency), and documented rather than averaged away — the tournament itself
adjudicates.

## What the dashboard does

- **Tournament odds** — round-by-round survival probabilities for all 48
  teams, with a rating-uncertainty slider
- **Schedule & predictions** — all 72 group fixtures with locked forecasts,
  graded against reality as results land (log-loss vs baselines, scoreline
  hits, a calibration check that fills in live)
- **Live simulator** — re-simulates the remaining tournament conditioned on
  every played match (shootout winners included); probabilistic bracket,
  pre-tournament-vs-now movers, and a 🎲 one-tournament mode
- **Market vs models** — de-vigged DraftKings title odds against both
  models, gap diagnosis included
- **Betting board** — paper-trading with model-priced 1X2 / over-under /
  correct-score markets, Kelly staking, and an append-only ledger that
  settles itself against real results

## Methodology (the actual point)

1. **Temporal validation only** — train on the past, validate on the future
2. **Zero feature leakage** — strictly pre-match features, enforced by
   tests that flip a result and assert the match's own features don't move
3. **Baselines first** — the model must beat "pick the higher Elo" before
   it earns trust
4. **Probabilities, not picks** — log-loss, Brier, reliability curves
5. **Pre-registration** — all forecasts locked and SHA256-receipted
   (`reports/LOCKS.sha256`) before kickoff; the WC2022 backtest validated
   the full pipeline end-to-end before any 2026 number was published

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Kaggle credentials (kaggle.com -> Settings -> API), then:
.venv/bin/python -m src.data.download
.venv/bin/python -m src.data.clean
.venv/bin/python -m src.features.build
.venv/bin/python -m src.models.train
.venv/bin/python -m src.simulation.simulate          # champion Monte Carlo
.venv/bin/streamlit run app/dashboard.py
```

Tests run without pytest: `for t in tests/test_*.py; do .venv/bin/python -m tests.$(basename $t .py); done`

## Structure

```
src/data/        download + cleaning (incl. dissolved-nations successor merges)
src/features/    Elo from scratch, rolling form — leakage-tested
src/models/      baselines, LR training, Dixon-Coles, calibration, betting math
src/simulation/  2026 bracket (best-8-thirds + FIFA tiebreakers), Monte Carlo,
                 live conditioning, WC2022 backtest
app/             Streamlit dashboard (8 views)
tests/           leakage, gradient-vs-finite-difference, bracket, settlement
notebooks/       EDA (class balance, era drift, entity audit)
```

Data: [martj42 international results](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
(1872–present) · FIFA rankings · DraftKings futures (snapshot 2026-06-05).
