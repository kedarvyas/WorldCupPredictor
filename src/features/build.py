"""Feature pipeline: cleaned matches -> Elo + form -> features.csv.

Also writes current_elo.csv (every team's latest rating) for the Phase 5
simulation, which needs ratings as of tournament start.

Usage:
    .venv/bin/python -m src.features.build
"""

import pandas as pd

from src.config import DATA_PROCESSED, TOURNAMENT_START
from src.features.elo import compute_elo
from src.features.form import compute_form


def build() -> pd.DataFrame:
    matches = pd.read_csv(DATA_PROCESSED / "matches.csv",
                          index_col="match_id", parse_dates=["date"])
    # Frozen-parameters rule: Elo/form/training features must never absorb
    # tournament results, even if the pipeline is rerun mid-tournament.
    matches = matches[matches["date"] < TOURNAMENT_START]

    with_elo, final_ratings = compute_elo(matches)
    features = compute_form(with_elo)
    features["elo_diff"] = features["home_elo_pre"] - features["away_elo_pre"]

    features.to_csv(DATA_PROCESSED / "features.csv")
    (pd.Series(final_ratings, name="elo").rename_axis("team")
       .sort_values(ascending=False)
       .to_csv(DATA_PROCESSED / "current_elo.csv"))

    print(f"features.csv: {len(features):,} rows, "
          f"{features.shape[1]} columns")
    return features


if __name__ == "__main__":
    build()
