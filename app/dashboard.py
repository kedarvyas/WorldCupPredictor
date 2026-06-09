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
from src.simulation.bracket import GROUPS, HOSTS  # noqa: E402
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
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[m["date"] >= "2010-01-01"]
    gd = m["home_score"] - m["away_score"]
    oc = np.select([gd > 0, gd < 0], ["win", "loss"], "draw")
    dists = {}
    for o in ("win", "draw", "loss"):
        c = Counter(zip(m.loc[oc == o, "home_score"].astype(int),
                        m.loc[oc == o, "away_score"].astype(int)))
        tot = sum(c.values())
        dists[o] = {s: n / tot for s, n in c.items()}
    return dists


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
| **LR (published)** | **60.1%** | **0.871** | **0.512** |

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

**Known limitations**
- No squad/injury/lineup information — ratings summarize results only.
- Ratings frozen at tournament start (no in-tournament updating).
- Scorelines are outcome-conditional global distributions, not team-specific.
- FIFA ranking feature excluded (source data ends 2024-06).
- Penalty shootouts modeled as strength-weighted coin flips.
- Single-tournament backtest: weak power to detect compounding bias.
""")


PAGES = {"🏆 Tournament odds": view_tournament,
         "⚔️ Match explorer": view_match,
         "📋 Model card": view_model_card}

st.sidebar.title("WC2026 Predictor")
choice = st.sidebar.radio("View", list(PAGES))
st.sidebar.caption("Trained through 2026-06-08 · 10k Monte Carlo runs · "
                   "probabilities, not promises.")
PAGES[choice]()
