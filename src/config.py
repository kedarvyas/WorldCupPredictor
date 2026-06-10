"""Central configuration: paths and hyperparameters.

Anything we intend to tune (training window, Elo K-factor, form window N)
lives here as a named constant rather than a magic number in pipeline code.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Data source ----------------------------------------------------------
# Kaggle mirror of martj42's international results dataset.
# (Slug still says "2017" but the dataset is actively updated.)
KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"

# --- Temporal split ---------------------------------------------------------
# Train on matches before this date, validate on matches after. Chosen so
# validation contains a full World Cup cycle (WC2022 + Euro/Copa 2024) and
# avoids COVID empty-stadium matches straddling the boundary.
VALIDATION_START = "2022-01-01"

# FIFA rankings snapshot (Kaggle, cashncarry/fifaworldranking) ends here;
# rank-based comparisons are restricted to matches before this date.
FIFA_RANKS_END = "2024-06-20"

# --- Tournament freeze line --------------------------------------------------
# NOTHING may train/fit/estimate on matches from this date onward: model
# parameters are frozen at their pre-tournament fit, and only the locked
# forecasts + live conditioning consume tournament results. Every fit and
# distribution estimate must cap its data at this date.
TOURNAMENT_START = "2026-06-11"

# Frozen snapshot of the 72 group fixtures, written once pre-tournament.
# wc2026_fixtures.csv shrinks as matches are played (clean.py keeps only
# unplayed rows) — simulators must use this stable copy instead.
FIXTURES_FROZEN = DATA_PROCESSED / "wc2026_fixtures_frozen.csv"

# --- Modeling hyperparameters (tuned later; defaults to start) -------------
# Training-window start year is a hyperparameter: compare 1990 / 2000 / 2010
# by validation log-loss in Phase 4. Full history before this is used only
# to warm up Elo.
TRAIN_START_YEAR = 1990

# Elo parameters (Phase 2) — placeholders, explained and tuned there.
ELO_BASE_RATING = 1500.0
ELO_K_FACTOR = 32.0

# Recent-form window: last N matches (Phase 2).
FORM_WINDOW = 10
