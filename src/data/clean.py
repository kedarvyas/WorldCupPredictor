"""Clean the raw match data into data/processed/.

Outputs:
- matches.csv        — played matches, chronologically sorted, renames applied
- wc2026_fixtures.csv — the pre-loaded (unplayed) 2026 World Cup group fixtures,
                        kept for the Phase 5 simulation

Decisions from the Phase 1 audit (see notebooks/01_eda.ipynb):
- Czechoslovakia -> Czech Republic, Yugoslavia -> Serbia (FIFA successors;
  avoids Elo cold-starts in 1994, inside the training window)
- German DR and Saarland kept as separate teams (no successor restart)
- Drop the spurious 1950 Croatia-Serbia friendly (sub-national exhibition)

Usage:
    .venv/bin/python -m src.data.clean
"""

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, FIXTURES_FROZEN

# Successor merges: applied to team columns before any Elo computation.
# (West Germany -> Germany and USSR -> Russia are already merged upstream.)
TEAM_RENAMES = {
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
}


def clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])

    # Split played matches from the pre-loaded WC2026 fixtures — explicitly,
    # not by letting NaN scores fall out downstream.
    unplayed = df["home_score"].isna() | df["away_score"].isna()
    fixtures = df[unplayed]
    df = df[~unplayed].copy()
    assert (df["date"] <= pd.Timestamp.now()).all(), \
        "a 'played' match is dated in the future"

    # Drop the spurious 1950 Croatia-Serbia friendly.
    spurious = (
        (df["date"] == "1950-02-26")
        & (df["home_team"] == "Croatia") & (df["away_team"] == "Serbia")
    )
    assert spurious.sum() == 1, f"expected 1 spurious row, found {spurious.sum()}"
    df = df[~spurious]

    # Successor merges.
    df[["home_team", "away_team"]] = (
        df[["home_team", "away_team"]].replace(TEAM_RENAMES)
    )

    # Chronological order is load-bearing: Elo and form are running
    # computations over this ordering. Stable match_id for joins later.
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    df.index.name = "match_id"

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PROCESSED / "matches.csv")
    fixtures.to_csv(DATA_PROCESSED / "wc2026_fixtures.csv", index=False)

    # Freeze the full 72-fixture snapshot exactly once (pre-tournament).
    # wc2026_fixtures.csv shrinks as results arrive; simulators need the
    # complete stable schedule.
    if not FIXTURES_FROZEN.exists():
        assert len(fixtures) == 72, \
            f"cannot freeze fixtures: expected 72 unplayed, got {len(fixtures)}"
        fixtures.to_csv(FIXTURES_FROZEN, index=False)
        print(f"Froze fixtures snapshot -> {FIXTURES_FROZEN.name}")

    print(f"matches.csv:         {len(df):,} played matches "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    print(f"wc2026_fixtures.csv: {len(fixtures):,} unplayed fixtures")
    for old in TEAM_RENAMES:
        remaining = ((df["home_team"] == old) | (df["away_team"] == old)).sum()
        assert remaining == 0, f"{old} still present after rename"
    return df


if __name__ == "__main__":
    clean()
