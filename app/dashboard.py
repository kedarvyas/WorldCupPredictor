"""World Cup 2026 predictor dashboard.

Run from the project root:
    .venv/bin/streamlit run app/dashboard.py
"""

import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED, MODELS_DIR  # noqa: E402
from src.features.elo import match_tier  # noqa: E402
from src.models.train import TIERS, proba_in_class_order  # noqa: E402
from src.models.dixon_coles import DC_LOCK_PATH  # noqa: E402
from src.simulation.bracket import GROUPS, HOSTS  # noqa: E402
from src.simulation.fixture_preds import (LOCK_PATH,  # noqa: E402
                                          fixture_predictions)
from src.simulation.fixture_preds import scoreline_dists as _scoreline_dists  # noqa: E402
from src.simulation.simulate import team_state  # noqa: E402

REPORTS = ROOT / "reports"
ROUND_LABELS = {"group_top2": "Top 2 in group", "third_qualified": "Best-8 third",
                "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarterfinal",
                "SF": "Semifinal", "final": "Final", "champion": "Champion"}

st.set_page_config(page_title="WC2026 Predictor", page_icon="⚽", layout="wide")


# --- Cached loaders ----------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "outcome_model.joblib")


@st.cache_data
def load_state():
    return team_state()


@st.cache_data
def load_sim(sigma: int) -> pd.DataFrame | None:
    path = (REPORTS / "sim_2026.csv" if sigma == 0
            else REPORTS / f"sim_2026_sigma{sigma}.csv")
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def scoreline_dists():
    return _scoreline_dists()


MODEL_LOCKS = {
    "LR + Elo (champion)": LOCK_PATH,
    "Dixon-Coles goals model (challenger)": DC_LOCK_PATH,
}


@st.cache_data
def locked_predictions(lock_file: str) -> tuple[pd.DataFrame, bool]:
    """A frozen pre-tournament forecast. Falls back to live computation
    (champion only) if the lock file is missing."""
    path = Path(lock_file)
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df, True
    return fixture_predictions(load_model(), load_state()), False


@st.cache_data
def wc2026_results() -> pd.DataFrame:
    """Played 2026 WC matches from the (refreshable) cleaned data."""
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[(m["tournament"] == "FIFA World Cup") & (m["date"] >= "2026-06-11")]
    m = m[["home_team", "away_team", "home_score", "away_score"]].copy()
    gd = m["home_score"] - m["away_score"]
    m["actual"] = np.select([gd > 0, gd < 0], ["home", "away"], "draw")
    return m


def view_schedule():
    st.header("Group-stage schedule: predictions vs reality")
    model_choice = st.selectbox(
        "Forecast model", list(MODEL_LOCKS),
        help="Champion/challenger: both forecasts were frozen before kickoff "
             "and face the same results. Champion = outcome model (Elo + "
             "logistic regression; better validation log-loss: 0.871 vs "
             "0.891) with global scorelines. Challenger = Dixon-Coles "
             "goals model: team-specific attack/defense rates, so its "
             "scorelines distinguish Spain–Cape Verde (2-0) from "
             "France–England (1-1).")
    locked, is_locked = locked_predictions(str(MODEL_LOCKS[model_choice]))
    if is_locked:
        scoreline_note = (
            "team-specific Poisson score matrices (attack × defense)"
            if "Dixon" in model_choice else
            "outcome-conditional modern-era distributions (not team-specific)")
        st.caption(f"Predictions **frozen 2026-06-10**, the day before "
                   f"kickoff — results update; the forecast cannot. "
                   f"Scorelines are {scoreline_note}; true home advantage "
                   f"applies only to USA/Mexico/Canada matches.")
    else:
        st.warning("Lock file missing — showing live-computed predictions. "
                   "Run `python -m src.simulation.fixture_preds` to freeze.")

    if st.button("↻ Refresh results from data source"):
        with st.spinner("Downloading latest results from Kaggle…"):
            from src.data.clean import clean
            from src.data.download import download
            download()
            clean()
        st.cache_data.clear()
        st.rerun()

    res = wc2026_results()
    fx = locked.merge(res, on=["home_team", "away_team"], how="left")
    played = fx["actual"].notna()

    # --- Running scoreboard: forecast vs reality so far ---------------------
    if played.any():
        pf = fx[played]
        p_actual = np.select(
            [pf["actual"] == "home", pf["actual"] == "draw"],
            [pf["p_home"], pf["p_draw"]], pf["p_away"])
        hits1 = (pf["top1_score"] ==
                 pf["home_score"].astype(int).astype(str) + "-"
                 + pf["away_score"].astype(int).astype(str))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches played", f"{played.sum()}/72")
        c2.metric("Outcome picks correct",
                  f"{(pf['pick'] == pf['actual']).mean():.0%}",
                  help="Argmax pick — remember the model is honest about "
                       "uncertainty; ~60% was the validation rate.")
        c3.metric("Log-loss (vs 1.05 = know-nothing)",
                  f"{-np.log(np.clip(p_actual, 1e-15, 1)).mean():.3f}",
                  help="Mean -log P(realized outcome). Validation: 0.871. "
                       "Below 1.05 means beating the base-rate forecaster.")
        c4.metric("Exact-score hits",
                  f"{int(hits1.sum())} (exp. {pf['top1_p'].sum():.1f})",
                  help="Likeliest-scoreline hits vs the number the forecast "
                       "itself expected by now. Close = calibrated.")
    else:
        st.info("No results yet — kickoff June 11. The scoreboard appears "
                "here as matches finish.")

    # --- Fixture table -------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    g = c1.selectbox("Group", ["All"] + list(GROUPS))
    days = ["All"] + sorted({str(d) for d in fx["date"]})
    d = c2.selectbox("Date", days)
    only = c3.selectbox("Show", ["All", "Played", "Upcoming"])
    show = fx
    if g != "All":
        show = show[show["group"] == g]
    if d != "All":
        show = show[show["date"].astype(str) == d]
    if only == "Played":
        show = show[show["actual"].notna()]
    elif only == "Upcoming":
        show = show[show["actual"].isna()]

    disp = pd.DataFrame({
        "date": show["date"], "grp": show["group"],
        "match": show["home_team"] + " vs " + show["away_team"],
        "home %": show["p_home"], "draw %": show["p_draw"],
        "away %": show["p_away"],
        "likeliest": show["top1_score"] + show["top1_p"].map("  ({:.0%})".format),
        "result": np.where(
            show["actual"].notna(),
            show["home_score"].fillna(0).astype(int).astype(str) + "-"
            + show["away_score"].fillna(0).astype(int).astype(str), "—"),
        "pick": np.select(
            [show["actual"].isna(), show["pick"] == show["actual"]],
            ["·", "✓"], "✗"),
    })
    # st.dataframe ignores Styler na_rep (renders NaN as "None"), so
    # P(actual) is a pre-formatted string column with a hand-painted
    # gradient computed from the numeric values.
    p_act = pd.Series(np.select(
        [show["actual"] == "home", show["actual"] == "draw",
         show["actual"] == "away"],
        [show["p_home"], show["p_draw"], show["p_away"]], np.nan),
        index=show.index, dtype=float)
    disp["P(actual)"] = p_act.map(
        lambda v: "·" if pd.isna(v) else f"{v:.0%}")
    cmap = plt.get_cmap("RdYlGn")
    pa_styles = [
        "" if pd.isna(v) else
        f"background-color: {plt.matplotlib.colors.to_hex(cmap((min(max(v, .1), .75) - .1) / .65))}"
        for v in p_act
    ]
    styled = (disp.reset_index(drop=True)  # Styler needs a UNIQUE index
                  .style.format({"home %": "{:.0%}", "draw %": "{:.0%}",
                                 "away %": "{:.0%}"})
                  .background_gradient(subset=["home %", "draw %", "away %"],
                                       cmap="Blues", vmin=0, vmax=0.85)
                  .apply(lambda s: pa_styles, subset=["P(actual)"]))
    st.dataframe(styled, height=600, use_container_width=True,
                 hide_index=True)
    st.caption("**P(actual)** = probability the frozen forecast gave the "
               "outcome that actually happened — green is good. A calibrated "
               "model should average ~45–50% here, not 100%: upsets are "
               "supposed to happen at their stated rates.")


def predict_pair(team_a, team_b, venue, tier):
    """Symmetrized [P(A), P(draw), P(B)] for an arbitrary pairing."""
    model, state = load_model(), load_state()

    def row(h, a, neutral):
        return [state.loc[h, "elo"] - state.loc[a, "elo"], int(neutral),
                *(1 if t == tier else 0 for t in TIERS),
                state.loc[h, "ppg"] - state.loc[a, "ppg"],
                state.loc[h, "gd_form"] - state.loc[a, "gd_form"], 0.0]

    cols = ["elo_diff", "neutral", *(f"tier_{t}" for t in TIERS),
            "ppg_diff", "gd_form_diff", "rest_diff"]
    if venue == f"{team_a} at home":
        X = pd.DataFrame([row(team_a, team_b, False)], columns=cols)
        return proba_in_class_order(model, X)[0]
    if venue == f"{team_b} at home":
        X = pd.DataFrame([row(team_b, team_a, False)], columns=cols)
        return proba_in_class_order(model, X)[0][::-1]
    X = pd.DataFrame([row(team_a, team_b, True), row(team_b, team_a, True)],
                     columns=cols)
    p = proba_in_class_order(model, X)
    return (p[0] + p[1][::-1]) / 2


# --- Views -------------------------------------------------------------------

def view_tournament():
    st.header("Tournament odds — 10,000 simulated World Cups")
    sigma = st.select_slider(
        "Rating uncertainty σ (Elo points jittered per simulated tournament)",
        options=[0, 50, 100, 150], value=0,
        help="σ=0 is the primary published forecast. Higher σ models more "
             "pre-tournament uncertainty about true team strength; favorites "
             "shrink toward the field. The WC2022 backtest found no clear "
             "evidence favoring any value — see the Model card.")
    if sigma == 0:
        st.caption("**Primary published numbers** (σ=0, ratings frozen at "
                   "tournament start).")
    sim = load_sim(sigma)
    if sim is None:
        st.warning(f"σ={sigma} sweep not yet computed — run "
                   "`python -m src.simulation.sweep`.")
        return

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Title odds (top 15)")
        top = sim["champion"].head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(top.index, top.values * 100,
                color=["#c0392b" if t in HOSTS else "#2c3e50"
                       for t in top.index])
        ax.set_xlabel("P(champion) %")
        for i, v in enumerate(top.values):
            ax.text(v * 100 + 0.3, i, f"{v:.1%}", va="center", fontsize=8)
        ax.margins(y=0.01)
        st.pyplot(fig)
        st.caption("Hosts in red.")

    with right:
        st.subheader("Round-by-round survival")
        styled = (sim.rename(columns=ROUND_LABELS)
                     .style.format("{:.1%}")
                     .background_gradient(cmap="Blues", axis=None))
        st.dataframe(styled, height=520)

    st.subheader("Groups")
    cols_per_row = 3
    letters = list(GROUPS)
    for i in range(0, len(letters), cols_per_row):
        for col, g in zip(st.columns(cols_per_row), letters[i:i + cols_per_row]):
            with col:
                rows = sim.loc[GROUPS[g],
                               ["group_top2", "third_qualified", "R32"]]
                rows = rows.sort_values("R32", ascending=False)
                st.markdown(f"**Group {g}**")
                st.dataframe(rows.rename(columns=ROUND_LABELS)
                                 .style.format("{:.0%}"))


def view_match():
    st.header("Match explorer")
    teams = sorted(load_state().index)
    c1, c2, c3, c4 = st.columns(4)
    team_a = c1.selectbox("Team A", teams, index=teams.index("Spain"))
    team_b = c2.selectbox("Team B", teams, index=teams.index("Argentina"))
    venue = c3.selectbox("Venue", ["Neutral", f"{team_a} at home",
                                   f"{team_b} at home"])
    tier_label = c4.selectbox("Match importance", [
        "World Cup", "Continental final", "Qualifier", "Friendly", "Other"])
    tier = {"World Cup": "world_cup", "Continental final": "continental",
            "Qualifier": "qualifier", "Friendly": "friendly",
            "Other": "other"}[tier_label]
    if team_a == team_b:
        st.info("Pick two different teams.")
        return

    p = predict_pair(team_a, team_b, venue, tier)
    state = load_state()
    st.caption(f"Elo: {team_a} {state.loc[team_a, 'elo']:.0f} vs "
               f"{team_b} {state.loc[team_b, 'elo']:.0f}"
               + (" · neutral venue: prediction symmetrized to remove "
                  "home-label bias" if venue == "Neutral" else ""))

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{team_a} wins", f"{p[0]:.1%}")
    c2.metric("Draw", f"{p[1]:.1%}")
    c3.metric(f"{team_b} wins", f"{p[2]:.1%}")
    st.progress(float(p[0]))

    dists = scoreline_dists()
    mix = Counter()
    for o, pi in zip(("win", "draw", "loss"), p):
        for s, q in dists[o].items():
            mix[s] += pi * q
    top5 = mix.most_common(5)
    st.subheader("Most likely scorelines")
    st.caption("From the modern-era scoreline distribution conditional on "
               "outcome — not team-specific (Dixon-Coles is the stretch goal).")
    st.table(pd.DataFrame(
        [(f"{a}-{b}", f"{q:.1%}") for (a, b), q in top5],
        columns=[f"{team_a} - {team_b}", "probability"]))


def view_model_card():
    st.header("Model card")
    st.markdown(f"""
**Model.** Multinomial logistic regression (scikit-learn), trained on
international matches **2010 → 2022** ({4533:,}-match temporal validation on
2022+). Features: pre-match Elo difference (computed from scratch, importance-
weighted K, margin-of-victory, +60 home-advantage offset), venue neutrality,
match importance tier, recent form (PPG and goal-diff over last 10), rest-day
difference. Every feature uses **only pre-match information** (unit-tested).

**Validation (temporal — train on past, validate on future):**

| Model | Accuracy | Log-loss | Brier |
|---|---|---|---|
| B0 — base rates (no team info) | 49.0% | 1.051 | — |
| B1 — Elo hard pick (clipped) | 60.2% | 8.27 | — |
| B2 — Elo curve | 60.2% | 0.895 | 0.526 |
| B3 — FIFA-rank pick | 57.3% | — | — |
| **LR (champion, published)** | **60.1%** | **0.871** | **0.512** |
| Dixon-Coles (challenger) | 59.3% | 0.891 | 0.524 |

The model's edge over B2 is **probability quality** (draw probability varies
with closeness: ~28% for even matches → ~14% at a 400-pt Elo gap), not pick
accuracy — and probability quality is what the simulation compounds.
""")
    st.image(str(REPORTS / "figures" / "calibration.png"),
             caption="Reliability curves on the 2022+ validation window: "
                     "stated probabilities match observed frequencies.")
    st.markdown("""
**WC2022 backtest (full-system test).** Pipeline rebuilt as-of 2022-11-19 and
the 2022 World Cup simulated 10,000×: Brazil favorite at 34.7%, **Argentina
2nd at 25.0% — and Argentina won**. Injecting rating uncertainty
(σ ∈ {50, 100}) did not measurably improve agreement with reality
(log-loss 1.64 vs 1.63 — within single-tournament noise), so the primary
forecast uses σ=0 and the slider is provided as sensitivity analysis.

**Challenger: Dixon-Coles goals model.** Per-team attack/defense Poisson
rates (hand-written likelihood, analytic gradient, L-BFGS; τ low-score
correction ρ=−0.04, home boost γ=0.30, 10y-half-life decay, ridge-regularized).
Slightly worse on *outcomes* than the champion (0.891 vs 0.871), but its
scorelines are team-specific — Spain–Cape Verde 2-0, not the global 1-0.
Both forecasts were frozen pre-kickoff; the Schedule view scores them
against reality side by side.

**Known limitations**
- No squad/injury/lineup information — ratings summarize results only.
- Ratings frozen at tournament start (no in-tournament updating).
- Champion scorelines are outcome-conditional global distributions, not
  team-specific (the Dixon-Coles challenger fixes this **in the Schedule
  view only**).
- Tournament odds & Match explorer still run entirely on the champion:
  the Monte Carlo's scoreline sampling (and thus group tiebreakers) is
  team-blind, and DC's softer mismatch probabilities do not affect the
  published title odds.
- FIFA ranking feature excluded (source data ends 2024-06).
- Penalty shootouts modeled as strength-weighted coin flips.
- Single-tournament backtest: weak power to detect compounding bias.
""")


PAGES = {"🏆 Tournament odds": view_tournament,
         "📅 Schedule & predictions": view_schedule,
         "⚔️ Match explorer": view_match,
         "📋 Model card": view_model_card}

st.sidebar.title("WC2026 Predictor")
choice = st.sidebar.radio("View", list(PAGES))
st.sidebar.caption("Trained through 2026-06-08 · 10k Monte Carlo runs · "
                   "probabilities, not promises.")
PAGES[choice]()
