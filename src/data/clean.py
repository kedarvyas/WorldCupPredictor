"""Clean the raw match data into data/processed/.

Outputs:
- matches.csv        — played matches, chronologically sorted, renames applied
- wc2026_fixtures.csv — the pre-loaded (unplayed) 2026 World Cup group fixtures,
                        kept for the Phase 5 simulation

Manual results overlay (data/manual_results.csv): the upstream Kaggle
dataset lags live tournament results by hours-to-days. Final scores entered
manually (via the dashboard's Schedule view or by editing the CSV) fill in
any fixture upstream still lists as unplayed. Upstream wins once it catches
up — the overlay only fills gaps, so no duplication or conflict is possible.

Decisions from the Phase 1 audit (see notebooks/01_eda.ipynb):
- Czechoslovakia -> Czech Republic, Yugoslavia -> Serbia (FIFA successors;
  avoids Elo cold-starts in 1994, inside the training window)
- German DR and Saarland kept as separate teams (no successor restart)
- Drop the spurious 1950 Croatia-Serbia friendly (sub-national exhibition)

Usage:
    .venv/bin/python -m src.data.clean
"""

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, FIXTURES_FROZEN, PROJECT_ROOT

MANUAL_RESULTS = PROJECT_ROOT / "data" / "manual_results.csv"

# Successor merges: applied to team columns before any Elo computation.
# (West Germany -> Germany and USSR -> Russia are already merged upstream.)
TEAM_RENAMES = {
    "Czechoslovakia": "Czech Republic",
    "Yugoslavia": "Serbia",
}


def apply_manual_results(df: pd.DataFrame) -> pd.DataFrame:
    """Fill scores from the manual overlay into rows upstream lists as
    unplayed. Only NaN-score rows are touched: once upstream publishes a
    result, the manual entry for that match becomes inert."""
    if not MANUAL_RESULTS.exists():
        return df
    manual = pd.read_csv(MANUAL_RESULTS)
    applied = 0
    for r in manual.itertuples():
        mask = (df["home_team"] == r.home_team) \
            & (df["away_team"] == r.away_team) \
            & df["home_score"].isna()
        if mask.any():
            df.loc[mask, "home_score"] = float(r.home_score)
            df.loc[mask, "away_score"] = float(r.away_score)
            applied += mask.sum()
    if applied:
        print(f"manual overlay: filled {applied} result(s) upstream "
              f"hasn't published yet")
    return df


def add_manual_result(home: str, away: str, home_score: int,
                      away_score: int) -> pd.DataFrame:
    """Record a final score and reflect it immediately, WITHOUT the raw
    Kaggle pipeline (which isn't present on the cloud deploy — data/raw is
    not committed). Appends to the manual overlay (audit trail; also re-
    applied by clean() on a local refresh) and overlays the score directly
    onto the committed data/processed/matches.csv.

    Note: on Streamlit Cloud the processed file is on an ephemeral disk, so a
    manual entry survives the session but not a redeploy."""
    rec = pd.DataFrame([{"home_team": home, "away_team": away,
                         "home_score": int(home_score),
                         "away_score": int(away_score)}])
    MANUAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    rec.to_csv(MANUAL_RESULTS, mode="a", index=False,
               header=not MANUAL_RESULTS.exists())

    # The fixture identifies the exact match: the same two teams have met
    # historically, so we must scope by date + tournament, not team names
    # alone, or we'd overwrite an old friendly's score.
    fx = pd.read_csv(FIXTURES_FROZEN)
    frow = fx[(fx["home_team"] == home) & (fx["away_team"] == away)]
    if frow.empty:
        raise ValueError(f"{home} vs {away} not in frozen fixtures")
    fixture = frow.iloc[[0]].copy()
    fixture["date"] = pd.to_datetime(fixture["date"])

    matches = pd.read_csv(DATA_PROCESSED / "matches.csv")
    matches = matches.drop(columns="match_id", errors="ignore")
    matches["date"] = pd.to_datetime(matches["date"])
    mask = ((matches["home_team"] == home) & (matches["away_team"] == away)
            & (matches["date"] == fixture["date"].iloc[0])
            & (matches["tournament"] == "FIFA World Cup"))
    if mask.any():
        matches.loc[mask, ["home_score", "away_score"]] = [float(home_score),
                                                           float(away_score)]
    else:
        fixture[["home_score", "away_score"]] = [float(home_score),
                                                 float(away_score)]
        matches = pd.concat([matches, fixture], ignore_index=True)

    # Re-sort and re-id exactly as clean() does, so downstream joins hold.
    matches = matches.sort_values(["date", "home_team"]).reset_index(drop=True)
    matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")
    matches.index.name = "match_id"
    matches.to_csv(DATA_PROCESSED / "matches.csv")
    return matches


def clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    df = apply_manual_results(df)

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
