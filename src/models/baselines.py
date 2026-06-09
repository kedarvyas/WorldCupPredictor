"""Baselines the Phase 4 model must beat.

B0  base rates      — always predict training class frequencies (log-loss floor)
B1  Elo hard pick   — higher pre-match Elo (+home adv) wins, probability ~1.
                      Demonstrates why hard picks are toxic under log-loss.
B2  Elo curve       — Elo's logistic gives P(home prevails); a constant
                      draw share (training draw rate) is carved out.
B3  FIFA rank pick  — better-ranked team wins (point-in-time ranks via
                      as-of join). Compared on the pre-2024-06 sub-window
                      where the rankings snapshot is fresh.

Usage:
    .venv/bin/python -m src.models.baselines
"""

import numpy as np
import pandas as pd

from src.config import (DATA_PROCESSED, DATA_RAW, FIFA_RANKS_END,
                        TRAIN_START_YEAR, VALIDATION_START)
from src.features.elo import HOME_ADVANTAGE, expected_score
from src.models.evaluate import CLASSES, evaluate_probs, results_table

EPS = 1e-9  # "probability" assigned off-pick by the hard baseline

# results.csv name -> FIFA ranking name (only diffs that matter recently)
FIFA_NAME_MAP = {
    "United States": "USA", "South Korea": "Korea Republic",
    "North Korea": "Korea DPR", "Iran": "IR Iran", "China": "China PR",
    "Ivory Coast": "Côte d'Ivoire", "DR Congo": "Congo DR",
    "Czech Republic": "Czechia", "Taiwan": "Chinese Taipei",
    "Brunei": "Brunei Darussalam", "Cape Verde": "Cabo Verde",
    "Curaçao": "Curacao", "Kyrgyzstan": "Kyrgyz Republic",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Saint Kitts and Nevis": "St Kitts and Nevis",
    "Saint Lucia": "St Lucia",
    "Saint Vincent and the Grenadines": "St Vincent and the Grenadines",
    "Gambia": "The Gambia",
    "United States Virgin Islands": "US Virgin Islands",
}


def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "features.csv",
                     index_col="match_id", parse_dates=["date"])
    df = df[df["date"] >= f"{TRAIN_START_YEAR}-01-01"].copy()
    gd = df["home_score"] - df["away_score"]
    df["outcome"] = np.select([gd > 0, gd < 0], ["home_win", "away_win"], "draw")
    return df


def add_fifa_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join: each match gets the most recent ranking BEFORE
    its date. Using a later ranking would leak the match's own result
    (rankings respond to results), so the as-of direction matters."""
    ranks = pd.read_csv(DATA_RAW / "fifa_ranking-2024-06-20.csv",
                        parse_dates=["rank_date"])
    ranks = (ranks[["rank_date", "country_full", "rank"]]
             .rename(columns={"country_full": "team"})
             .sort_values("rank_date"))
    out = df.sort_values("date").copy()
    for side in ("home", "away"):
        keyed = out[[f"{side}_team", "date"]].copy()
        keyed["team"] = keyed[f"{side}_team"].replace(FIFA_NAME_MAP)
        merged = pd.merge_asof(keyed.reset_index().sort_values("date"),
                               ranks, left_on="date", right_on="rank_date",
                               by="team", allow_exact_matches=False)
        out[f"{side}_fifa_rank"] = merged.set_index("match_id")["rank"]
    return out


# --- Baselines (each returns an (n, 3) array in CLASSES order) -------------

def b0_base_rates(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    rates = train["outcome"].value_counts(normalize=True)[CLASSES].to_numpy()
    return np.tile(rates, (len(val), 1))


def _hard_pick(home_better: pd.Series) -> np.ndarray:
    """P~1 on the picked side, EPS elsewhere. Never picks a draw."""
    probs = np.full((len(home_better), 3), EPS)
    probs[home_better.to_numpy(), 0] = 1 - 2 * EPS
    probs[~home_better.to_numpy(), 2] = 1 - 2 * EPS
    return probs


def b1_elo_pick(val: pd.DataFrame) -> np.ndarray:
    adv = np.where(val["neutral"], 0.0, HOME_ADVANTAGE)
    return _hard_pick(val["home_elo_pre"] + adv >= val["away_elo_pre"])


def b2_elo_curve(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    adv = np.where(val["neutral"], 0.0, HOME_ADVANTAGE)
    e_home = expected_score(val["home_elo_pre"] + adv, val["away_elo_pre"])
    p_draw = (train["outcome"] == "draw").mean()
    return np.column_stack([e_home * (1 - p_draw),
                            np.full(len(val), p_draw),
                            (1 - e_home) * (1 - p_draw)])


def b3_fifa_pick(val: pd.DataFrame) -> np.ndarray:
    return _hard_pick(val["home_fifa_rank"] <= val["away_fifa_rank"])


def run() -> pd.DataFrame:
    df = add_fifa_ranks(load_features())
    train = df[df["date"] < VALIDATION_START]
    val = df[df["date"] >= VALIDATION_START]

    rows = {
        "B0 base rates": evaluate_probs(val["outcome"], b0_base_rates(train, val)),
        "B1 Elo hard pick": evaluate_probs(val["outcome"], b1_elo_pick(val)),
        "B2 Elo curve": evaluate_probs(val["outcome"], b2_elo_curve(train, val)),
    }
    print(f"=== Full validation window ({VALIDATION_START} ->) ===")
    print(results_table(rows).to_string())

    # FIFA comparison: fresh-rankings sub-window, all baselines re-scored
    # there so the comparison is apples-to-apples.
    sub = val[(val["date"] < FIFA_RANKS_END)
              & val["home_fifa_rank"].notna() & val["away_fifa_rank"].notna()]
    sub_rows = {
        "B0 base rates": evaluate_probs(sub["outcome"], b0_base_rates(train, sub)),
        "B1 Elo hard pick": evaluate_probs(sub["outcome"], b1_elo_pick(sub)),
        "B2 Elo curve": evaluate_probs(sub["outcome"], b2_elo_curve(train, sub)),
        "B3 FIFA rank pick": evaluate_probs(sub["outcome"], b3_fifa_pick(sub)),
    }
    print(f"\n=== Fresh-FIFA sub-window (-> {FIFA_RANKS_END}) ===")
    print(results_table(sub_rows).to_string())
    missing = val["home_fifa_rank"].isna() | val["away_fifa_rank"].isna()
    print(f"\n(validation matches lacking a FIFA rank: {missing.sum()})")
    return results_table(rows)


if __name__ == "__main__":
    run()
