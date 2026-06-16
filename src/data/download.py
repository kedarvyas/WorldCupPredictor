"""Download the raw match dataset from Kaggle into data/raw/.

Principles:
- data/raw/ is immutable: we download here and never edit these files.
  All cleaning happens downstream, producing files in data/processed/.
- We record a snapshot timestamp so we always know what data vintage a
  trained model saw (matters for reproducibility as the dataset updates).

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(kaggle.com -> Settings -> API -> Create New Token).

Usage:
    .venv/bin/python -m src.data.download
"""

import json
from datetime import datetime, timezone

import pandas as pd

from src.config import DATA_RAW, KAGGLE_DATASET


def download() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Import inside the function: the kaggle package authenticates at import
    # time and fails with an unhelpful error if credentials are missing, so
    # we guard it and raise a clear, *catchable* error (not sys.exit, which
    # raises SystemExit and would hang a long-running caller like the
    # dashboard instead of surfacing as an error).
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            f"Kaggle credentials problem: {exc}. Create a token at "
            "kaggle.com -> Settings -> API and place it at "
            "~/.kaggle/kaggle.json (chmod 600), or set KAGGLE_USERNAME / "
            "KAGGLE_KEY in the environment.") from exc

    print(f"Downloading {KAGGLE_DATASET} -> {DATA_RAW}")
    api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_RAW), unzip=True)

    # Record the snapshot date alongside the data.
    snapshot = {
        "dataset": KAGGLE_DATASET,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (DATA_RAW / "snapshot.json").write_text(json.dumps(snapshot, indent=2))

    # Sanity checks: did we get what we expect?
    results = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    print(f"\nresults.csv: {len(results):,} matches, "
          f"{results['date'].min().date()} to {results['date'].max().date()}")
    print(f"Columns: {list(results.columns)}")

    expected = {"date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "city", "country", "neutral"}
    missing = expected - set(results.columns)
    if missing:
        raise RuntimeError(f"Schema check FAILED, missing columns: {missing}")
    print("Schema check passed.")


if __name__ == "__main__":
    download()
