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

from src.config import DATA_PROCESSED, FIXTURES_FROZEN, MODELS_DIR  # noqa: E402
from src.features.elo import match_tier  # noqa: E402
from src.models.train import TIERS, proba_in_class_order  # noqa: E402
from src.models.dixon_coles import DC_LOCK_PATH  # noqa: E402
from src.simulation.bracket import GROUPS, HOSTS  # noqa: E402
from src.simulation.fixture_preds import (LOCK_PATH,  # noqa: E402
                                          fixture_predictions)
from src.simulation.fixture_preds import scoreline_dists as _scoreline_dists  # noqa: E402
from src.simulation.group_sim import (bracket_tree_order,  # noqa: E402
                                      played_ko_results, played_results,
                                      simulate_live)
from src.simulation.simulate import (build_prob_tables, dc_tables,  # noqa: E402
                                     scoreline_sampler, team_state)

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
def load_sim(sigma: int, dc: bool = False) -> pd.DataFrame | None:
    if dc:
        path = REPORTS / "sim_2026_dc.csv"
    else:
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
        from src.data.clean import clean
        from src.data.download import download
        try:
            with st.spinner("Downloading latest results from Kaggle…"):
                download()
                clean()
        except (Exception, SystemExit) as exc:
            st.error(
                f"Refresh failed: {exc}\n\nThis needs Kaggle API credentials "
                "(`~/.kaggle/kaggle.json` locally, or `KAGGLE_USERNAME` / "
                "`KAGGLE_KEY` in the cloud app's secrets). Meanwhile, enter "
                "the score by hand below — it updates every view immediately.")
        else:
            st.cache_data.clear()
            st.success("Results refreshed from the data source.")
            st.rerun()

    with st.expander("✍️ Enter a result manually (upstream data lags live "
                     "matches)"):
        st.caption("The Kaggle dataset updates on its maintainer's schedule "
                   "— often hours or days behind a final whistle. Enter the "
                   "full-time score here to update every view immediately; "
                   "once upstream publishes the result, its data takes over "
                   "and the manual entry becomes inert.")
        res_now = wc2026_results()
        frozen = pd.read_csv(FIXTURES_FROZEN)
        done_pairs = set(zip(res_now["home_team"], res_now["away_team"])) \
            if len(res_now) else set()
        open_fx = frozen[~frozen.apply(
            lambda r: (r["home_team"], r["away_team"]) in done_pairs, axis=1)]
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        label = c1.selectbox("Match", [
            f"{r.home_team} vs {r.away_team}" for r in open_fx.itertuples()])
        gh = c2.number_input("Home", min_value=0, max_value=15, value=0)
        ga = c3.number_input("Away", min_value=0, max_value=15, value=0)
        if c4.button("Save", type="primary"):
            home, away = label.split(" vs ")
            from src.data.clean import MANUAL_RESULTS, clean
            row = pd.DataFrame([{"home_team": home, "away_team": away,
                                 "home_score": int(gh),
                                 "away_score": int(ga)}])
            row.to_csv(MANUAL_RESULTS, mode="a", index=False,
                       header=not MANUAL_RESULTS.exists())
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


@st.cache_resource(show_spinner="Fitting Dixon-Coles (first run only)…")
def fitted_dc():
    from src.config import TOURNAMENT_START
    from src.models.dixon_coles import TRAIN_START, DixonColes
    m = pd.read_csv(DATA_PROCESSED / "matches.csv", parse_dates=["date"])
    m = m[(m["date"] >= TRAIN_START) & (m["date"] < TOURNAMENT_START)]
    return DixonColes(half_life_years=10.0).fit(m)  # frozen pre-tournament


@st.cache_data
def current_results():
    return played_results()


@st.cache_data
def current_ko_results():
    return played_ko_results()


def _sim_engine(model_choice, seed):
    rng = np.random.default_rng(seed)
    if "Dixon" in model_choice:
        return dc_tables(fitted_dc(), rng)
    return (build_prob_tables(load_model(), load_state()),
            scoreline_sampler(rng))


def _single_winners(nodes):
    """For a 1-sim nodes dict, derive each match's winner: the entrant who
    also appears in its parent node (the champion, for the final)."""
    from src.simulation.bracket import FINAL, QF, R16, SF
    feeders = {**R16, **QF, **SF, **FINAL}
    parent = {f: m for m, (fa, fb) in feeders.items() for f in (fa, fb)}
    champion = max(nodes["champion"], key=nodes["champion"].get)
    winners = {104: champion}
    for m, p in parent.items():
        cand = [t for t in nodes[m] if t in nodes[p]]
        if cand:
            winners[m] = cand[0]
    return winners, champion


def view_live_sim():
    st.header("Live simulator — groups & knockout")
    results = current_results()
    ko_res = current_ko_results()
    st.caption(f"Simulates the WHOLE remaining tournament **conditioned on "
               f"reality**: {len(results)}/72 group results and "
               f"{len(ko_res)} knockout result(s) are held fixed; only "
               f"unplayed matches are sampled (drawn knockouts resolve via "
               f"the real shootout winner). Model parameters stay frozen at "
               f"their pre-tournament fit — the forecast updates because "
               f"the evidence grows, not the model. Refresh results on the "
               f"Schedule view.")

    c1, c2, c3 = st.columns([2, 1, 1])
    model_choice = c1.selectbox("Forecast model", list(MODEL_LOCKS))
    n_sims = c2.selectbox("Simulations", [2000, 5000, 10000], index=1)
    run = c3.button("▶ Run simulation", type="primary")
    resample = st.checkbox(
        "Fresh randomness each run", value=False,
        help="Off (default): a fixed seed per evidence-state, so repeated "
             "runs are identical and any change you see reflects new "
             "results, never noise. On: each run draws new random "
             "tournaments — re-running shows the Monte Carlo error "
             "(~±0.7pp on a champion probability at 5k sims). The wobble "
             "is sampling noise, not the model changing its mind.")

    if run:
        rng_seed = (int(np.random.default_rng().integers(2**31)) if resample
                    else 2026 + len(results) + 100 * len(ko_res))
        with st.spinner(f"Simulating {n_sims:,} tournaments…"):
            probs, sampler = _sim_engine(model_choice, rng_seed)
            out, nodes = simulate_live(probs, sampler, n_sims=n_sims,
                                       seed=rng_seed, group_results=results,
                                       ko_results=ko_res, return_nodes=True)
        st.session_state["live_sim"] = (out, nodes, model_choice, n_sims,
                                        len(results), len(ko_res), rng_seed)

    if "live_sim" not in st.session_state:
        st.info("Pick a model and hit **Run simulation**.")
        return
    (out, nodes, used_model, used_n,
     used_gr, used_ko, used_seed) = st.session_state["live_sim"]
    st.markdown(f"*{used_n:,} simulations · {used_model} · {used_gr} group "
                f"+ {used_ko} knockout result(s) locked in · "
                f"seed {used_seed}*")

    tab_groups, tab_ko, tab_one = st.tabs(
        ["Groups", "Knockout", "🎲 One tournament"])

    with tab_groups:
        pct_cols = ["1st", "2nd", "3rd", "4th", "top2", "best_third",
                    "advance"]
        letters = sorted(GROUPS)
        for i in range(0, len(letters), 3):
            for col, g in zip(st.columns(3), letters[i:i + 3]):
                with col:
                    st.markdown(f"**Group {g}**")
                    tbl = (out[out["group"] == g][pct_cols + ["xPts"]]
                           .sort_values("advance", ascending=False))
                    st.dataframe(
                        tbl.style.format({c: "{:.0%}" for c in pct_cols}
                                         | {"xPts": "{:.1f}"})
                           .background_gradient(subset=["advance"],
                                                cmap="Greens",
                                                vmin=0, vmax=1),
                        height=178)

    with tab_ko:
        st.markdown(_bracket_html(nodes), unsafe_allow_html=True)
        st.caption("Each slot lists the two most likely occupants and how "
                   "often they appear there across simulations. Slots at "
                   "100% are locked in by real results; everything else is "
                   "still probability. **Adjacent slots are independent "
                   "probabilities, not one storyline** — a team can appear "
                   "in a QF slot even though it's the underdog in its R16 "
                   "slot: those are the simulations where it won. For a "
                   "single consistent storyline, use the 🎲 One tournament "
                   "tab.")
        st.subheader("Title odds — pre-tournament vs now")
        baseline = load_sim(0, dc="Dixon" in used_model)
        if baseline is None:
            st.warning("Pre-tournament baseline CSV missing for this model.")
        else:
            comp = pd.DataFrame({"pre-tournament": baseline["champion"],
                                 "now": out["champion"]}).fillna(0.0)
            comp["Δ (pp)"] = (comp["now"] - comp["pre-tournament"]) * 100
            comp = comp[comp[["pre-tournament", "now"]].max(axis=1) >= 0.005]
            comp = comp.reindex(
                comp["Δ (pp)"].abs().sort_values(ascending=False).index)
            st.dataframe(
                comp.style.format({"pre-tournament": "{:.1%}",
                                   "now": "{:.1%}", "Δ (pp)": "{:+.1f}"})
                    .background_gradient(subset=["Δ (pp)"], cmap="RdYlGn",
                                         vmin=-5, vmax=5),
                height=420)
            st.caption("Baseline = this model's LOCKED pre-tournament 10k-"
                       "sim (champion vs champion, DC vs DC — baselines are "
                       "never crossed; the two models' pre-tournament "
                       "numbers differ hugely, so mixing them would "
                       "manufacture fake movement). With few results in, "
                       "moves within ±1pp are Monte Carlo noise — real "
                       "signal looks like a result, e.g. a favorite "
                       "dropping points.")

        with st.expander("Full survival table"):
            ko_cols = ["R32", "R16", "QF", "SF", "final", "champion"]
            alive = out[out["champion"] > 0].sort_values("champion",
                                                         ascending=False)
            st.dataframe(
                alive[["group"] + ko_cols]
                .style.format({c: "{:.1%}" for c in ko_cols})
                .background_gradient(subset=ko_cols, cmap="Blues",
                                     axis=None),
                height=600)

    with tab_one:
        st.caption("🎲 **One complete simulated tournament** — a single "
                   "random draw from the same model, not the odds. This is "
                   "what most simulator sites show: one consistent "
                   "storyline, different every click (an 80% favorite "
                   "still loses this universe 1 time in 5). Played results "
                   "are respected; everything else is dice.")
        if st.button("🎲 Simulate one tournament"):
            one_seed = int(np.random.default_rng().integers(2**31))
            with st.spinner("Rolling one universe…"):
                probs1, sampler1 = _sim_engine(model_choice, one_seed)
                _, nodes1 = simulate_live(
                    probs1, sampler1, n_sims=1, seed=one_seed,
                    group_results=results, ko_results=ko_res,
                    return_nodes=True)
            st.session_state["one_sim"] = (nodes1, model_choice)
        if "one_sim" in st.session_state:
            nodes1, one_model = st.session_state["one_sim"]
            winners1, champ1 = _single_winners(nodes1)
            st.success(f"🏆 Champion of this universe: **{champ1}** "
                       f"({one_model})")
            st.markdown(_bracket_html(nodes1, winners=winners1),
                        unsafe_allow_html=True)


def _bracket_html(nodes, winners=None) -> str:
    order = bracket_tree_order()
    titles = {"R32": "Round of 32", "R16": "Round of 16",
              "QF": "Quarterfinals", "SF": "Semifinals", "final": "Final"}

    single = winners is not None

    def box(match_no):
        top = sorted(nodes.get(match_no, {}).items(),
                     key=lambda kv: -kv[1])[:2]
        lines = ""
        for t, p in top:
            won = single and winners.get(match_no) == t
            cls = "t w" if won else "t"
            val = "✓" if won else ("" if single else f"{p:.0%}")
            lines += f"<div class='{cls}'><em>{t}</em><span>{val}</span></div>"
        return f"<div class='bx'>{lines or '<div class=t>—</div>'}</div>"

    cols = ""
    for rnd, ms in order.items():
        boxes = "".join(box(m) for m in ms)
        cols += (f"<div class='rcol'><div class='hdr'>{titles[rnd]}</div>"
                 f"<div class='stack'>{boxes}</div></div>")
    champ = sorted(nodes.get("champion", {}).items(),
                   key=lambda kv: -kv[1])[:1 if single else 3]
    champ_lines = "".join(
        f"<div class='t w'><em>{t}</em>"
        + ("" if single else f"<span>{p:.0%}</span>") + "</div>"
        for t, p in champ)
    cols += (f"<div class='rcol'><div class='hdr'>🏆 Champion</div>"
             f"<div class='stack'><div class='bx champ'>{champ_lines}"
             f"</div></div></div>")

    return f"""
<style>
.bracket {{ display:flex; gap:10px; height:1000px; }}
.bracket .rcol {{ flex:1; display:flex; flex-direction:column; min-width:0; }}
.bracket .hdr {{ text-align:center; font-size:12px; font-weight:600;
                 color:#5a6172; padding-bottom:4px; }}
.bracket .stack {{ flex:1; display:flex; flex-direction:column;
                   justify-content:space-around; }}
.bracket .bx {{ border:1px solid #d6d9e0; border-radius:6px;
                padding:3px 7px; background:#f7f8fa; font-size:11px;
                line-height:1.55; }}
.bracket .bx.champ {{ border-color:#c9a227; background:#fdf8e7; }}
.bracket .t {{ display:flex; justify-content:space-between; gap:6px; }}
.bracket .t em {{ font-style:normal; overflow:hidden;
                  text-overflow:ellipsis; white-space:nowrap; }}
.bracket .t span {{ color:#5a6172; font-variant-numeric:tabular-nums; }}
.bracket .t.w em {{ font-weight:700; color:#1a7f37; }}
.bracket .t.w span {{ color:#1a7f37; }}
</style>
<div class="bracket">{cols}</div>"""


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
    model_choice = st.selectbox(
        "Forecast model", list(MODEL_LOCKS),
        help="Champion = Elo + logistic regression (better validation "
             "log-loss; the primary published numbers). Challenger = "
             "Dixon-Coles: outcomes AND scorelines from team-specific "
             "score matrices, so group tiebreakers use team-specific goal "
             "differences. The models disagree substantially about title "
             "odds — an honest display of model risk, not a bug.")
    use_dc = "Dixon" in model_choice
    if use_dc:
        sigma = 0
        st.caption("Challenger numbers (Dixon-Coles). The champion remains "
                   "the primary published forecast; σ-sweep applies to the "
                   "champion only.")
    else:
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
    sim = load_sim(sigma, dc=use_dc)
    if sim is None:
        st.warning("Simulation output missing — run "
                   "`python -m src.simulation.simulate"
                   + (" --dc`." if use_dc else "` / sweep."))
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


def view_market():
    st.header("Market vs models — who does the bookmaker agree with?")
    from src.models.market import SOURCE, market_probs

    mkt, over = market_probs()
    champ = load_sim(0)["champion"]
    dc = load_sim(0, dc=True)["champion"]
    if champ is None or dc is None:
        st.warning("Run the simulations first.")
        return
    df = pd.DataFrame({"market": mkt, "LR (champion)": champ,
                       "DC (challenger)": dc}).fillna(0.0)
    df = df.sort_values("market", ascending=False)
    df["LR − mkt"] = df["LR (champion)"] - df["market"]
    df["DC − mkt"] = df["DC (challenger)"] - df["market"]

    st.caption(f"Market: {SOURCE}. The raw odds embed a {over - 1:.1%} "
               f"bookmaker margin, removed here by proportional "
               f"normalization (the simplest de-vig; it slightly overstates "
               f"longshots — favorite-longshot bias). Model numbers are the "
               f"frozen 10k-simulation title odds. The market is not ground "
               f"truth — it's a third, independent forecaster that prices "
               f"squad information our models can't see.")

    top = df.head(20)
    lr_mad = (top["LR − mkt"].abs().mean())
    dc_mad = (top["DC − mkt"].abs().mean())
    lr_closer = int((top["LR − mkt"].abs() < top["DC − mkt"].abs()).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean |LR − market| (top 20)", f"{lr_mad:.1%}")
    c2.metric("Mean |DC − market| (top 20)", f"{dc_mad:.1%}")
    c3.metric("Teams where LR is closer", f"{lr_closer}/20")

    left, right = st.columns([3, 2])
    with left:
        pct = ["market", "LR (champion)", "DC (challenger)"]
        st.dataframe(
            df.style.format({c: "{:.1%}" for c in pct}
                            | {"LR − mkt": "{:+.1%}", "DC − mkt": "{:+.1%}"})
              .background_gradient(subset=pct, cmap="Blues", axis=None,
                                   vmin=0, vmax=0.20)
              .background_gradient(subset=["LR − mkt", "DC − mkt"],
                                   cmap="RdBu_r", vmin=-0.12, vmax=0.12),
            height=620)
    with right:
        floor = 5e-5  # half a sim out of 10k: log-scale floor for zeros
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for col, color in (("LR (champion)", "#2c3e50"),
                           ("DC (challenger)", "#c0392b")):
            ax.scatter(df["market"].clip(lower=floor),
                       df[col].clip(lower=floor),
                       s=22, alpha=0.75, color=color, label=col)
        lims = [floor, 0.5]
        ax.plot(lims, lims, "k--", alpha=0.4, label="perfect agreement")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("market (de-vigged)"); ax.set_ylabel("model")
        for t in ("Brazil", "Spain", "France"):
            ax.annotate(t, (df.loc[t, "market"],
                            df.loc[t, "DC (challenger)"]),
                        fontsize=7, xytext=(4, 2),
                        textcoords="offset points")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        st.caption("Log-log: distance from the diagonal = disagreement. "
                   "Points on the floor are teams our sims give <1-in-10k "
                   "title odds.")


def view_betting():
    st.header("Betting board — paper trading")
    st.caption("**Educational scorekeeping, not income.** Model prices vs "
               "book odds you enter; the ledger settles itself against real "
               "results. Remember: edges shown are vs *your entered odds* — "
               "the book's margin (~5-7%) and the squad information it "
               "prices (see Market vs models: France) are exactly what we "
               "must beat. Expectation set by the model card, not hype.")

    from src.models.betting import (LEDGER_PATH, american_to_decimal,
                                    append_bet, clv, decimal_to_american,
                                    devig, edge, equity_curve, kelly_fraction,
                                    load_ledger, max_drawdown,
                                    save_closing_odds, settle)
    from src.models.recalibrate import apply as recal_apply
    from src.models.recalibrate import load_params, over_prob
    import src.storage as storage

    # On cloud the filesystem is ephemeral; pull the persisted ledger from the
    # gist once per session (no-op locally / when persistence is unconfigured).
    if storage.enabled() and not st.session_state.get("ledger_pulled"):
        storage.pull(LEDGER_PATH)
        st.session_state["ledger_pulled"] = True

    results = current_results()
    locked, _ = locked_predictions(str(LOCK_PATH))
    upcoming = locked[~locked.apply(
        lambda r: (r["home_team"], r["away_team"]) in results, axis=1)]
    if upcoming.empty:
        st.info("No unplayed group fixtures left.")
    else:
        labels = [f"{r.date} — {r.home_team} vs {r.away_team} (Group {r.group})"
                  for r in upcoming.itertuples()]
        pick = st.selectbox("Fixture", labels)
        row = upcoming.iloc[labels.index(pick)]
        home, away = row["home_team"], row["away_team"]

        dc = fitted_dc()
        fx = pd.read_csv(DATA_PROCESSED / "wc2026_fixtures_frozen.csv")
        neutral = bool(fx[(fx["home_team"] == home)
                          & (fx["away_team"] == away)]["neutral"].iloc[0])

        # Totals ladder: a single 2.5 line on a lopsided fixture sits far
        # from 50/50, where the book's price leaves no value. Show a ladder
        # and flag the line nearest a coin-flip. Only 2.5 is recalibrated
        # (the Platt fit is 2.5-specific); other lines are raw DC mass.
        TOTAL_LINES = [1.5, 2.5, 3.5, 4.5]
        over_p = {ln: over_prob(dc, home, away, line=ln, true_home=not neutral)
                  for ln in TOTAL_LINES}
        over_p[2.5] = float(recal_apply(over_p[2.5], load_params()))
        balanced = min(TOTAL_LINES, key=lambda ln: abs(over_p[ln] - 0.5))

        M = dc.score_matrix(home, away, true_home=not neutral)
        flat = sorted(((f"{i}-{j}", M[i, j]) for i in range(7)
                       for j in range(7)), key=lambda t: -t[1])[:3]

        c0, c1 = st.columns([1, 3])
        bankroll = c0.number_input("Paper bankroll", value=1000.0, step=100.0)
        kelly_mult = c0.selectbox("Kelly fraction", [0.25, 0.5, 1.0],
                                  help="Full Kelly assumes the model "
                                       "probability is exactly right. "
                                       "It isn't. Bet smaller.")
        fmt = c0.radio("Odds format", ["Decimal", "American"],
                       help="Decimal 1.25 = American −400; "
                            "decimal 7.03 = American +603. Internally "
                            "everything is decimal; this only changes "
                            "what you type and see.")
        c0.caption("**The % is OUR model's probability; the prefilled "
                   "odds are the model's fair (break-even) price.** "
                   "Overwrite them with your book's odds — edge and Best "
                   "Bets compare model vs book.")

        markets = {"1X2": {f"{home} (home)": ("home", row["p_home"]),
                           "Draw": ("draw", row["p_draw"]),
                           f"{away} (away)": ("away", row["p_away"])}}
        engines = {"1X2": "locked champion forecast"}
        for ln in TOTAL_LINES:
            markets[f"O/U {ln}"] = {f"Over {ln}": ("over", over_p[ln]),
                                    f"Under {ln}": ("under", 1 - over_p[ln])}
            engines[f"O/U {ln}"] = (
                "DC matrix + recalibration" if ln == 2.5
                else "DC matrix — raw, uncalibrated")
            if ln == balanced:
                engines[f"O/U {ln}"] += " · ⚖️ balanced line (nearest 50/50)"

        # Derived 1X2 markets — same locked outcome probs as the moneyline.
        ph, pdr, pa = row["p_home"], row["p_draw"], row["p_away"]
        markets["Double chance"] = {
            f"{home} or draw (1X)": ("1X", ph + pdr),
            "Not draw (12)": ("12", ph + pa),
            f"draw or {away} (X2)": ("X2", pdr + pa)}
        engines["Double chance"] = "locked champion forecast"
        # Draw-no-bet: draw refunds the stake, so price the home/away outcome
        # conditional on no draw (p / (p_home + p_away)).
        no_draw = ph + pa
        markets["Draw no bet"] = {f"{home} (DNB)": ("home", ph / no_draw),
                                  f"{away} (DNB)": ("away", pa / no_draw)}
        engines["Draw no bet"] = "locked champion forecast · draw = push"
        # Both teams to score — goal-level, so from the DC matrix (raw).
        p_btts = float(1 - M[0, :].sum() - M[:, 0].sum() + M[0, 0])
        markets["BTTS"] = {"Yes": ("yes", p_btts), "No": ("no", 1 - p_btts)}
        engines["BTTS"] = "DC matrix — raw, uncalibrated"

        markets["Correct score"] = {f"{s}": (s, p) for s, p in flat}
        engines["Correct score"] = "DC matrix — display-grade only"
        entered = []  # (market, label, sel, p, odds, mkt_p) for Best Bets
        with c1:
            for mkt, sels in markets.items():
                st.markdown(f"**{mkt}** · *{engines[mkt]}*")
                cols = st.columns(len(sels))
                # Pass 1: render inputs and collect every leg's book odds, so
                # a complete market can be de-vigged before any leg is judged.
                legs = []  # (col, label, sel, p, odds)
                for col, (label, (sel, p)) in zip(cols, sels.items()):
                    with col:
                        st.metric(label, f"{p:.1%}", help="model probability")
                        fair = round(1 / p, 2) if p > 0.02 else 50.0
                        if fmt == "American":
                            raw = st.number_input(
                                "book odds (American)",
                                value=float(decimal_to_american(fair)),
                                step=5.0, key=f"odds_us_{mkt}_{sel}",
                                label_visibility="collapsed")
                            try:
                                odds = american_to_decimal(raw)
                            except ValueError:
                                st.caption("⚠️ needs ≤ −100 or ≥ +100")
                                continue
                        else:
                            odds = st.number_input(
                                "book odds (decimal)", min_value=1.01,
                                value=fair, step=0.05,
                                key=f"odds_{mkt}_{sel}",
                                label_visibility="collapsed")
                    legs.append((col, label, sel, p, odds))

                # De-vig only a COMPLETE market whose selections partition the
                # outcome space and sum to 1 (1X2, both O/U sides, BTTS yes/no,
                # DNB home/away). Double chance overlaps (sums to 2) and the
                # top-3 correct scores are partial — leave those un-de-vigged.
                complete = (mkt in ("1X2", "BTTS", "Draw no bet")
                            or mkt.startswith("O/U "))
                mkt_ps = (devig([o for *_, o in legs]) if complete
                          else [None] * len(legs))

                # Pass 2: judge each leg now the market's vig is stripped.
                for (col, label, sel, p, odds), mkt_p in zip(legs, mkt_ps):
                    entered.append((mkt, label, sel, p, odds, mkt_p))
                    e = edge(p, odds)
                    bet_type = "value" if e > 0 else "hunch"
                    with col:
                        if e > 0:
                            stake = round(bankroll * kelly_mult
                                          * kelly_fraction(p, odds), 2)
                            st.caption(f"edge {e:+.1%} · stake {stake:.0f}")
                        else:
                            # Kelly sizes negative edges to zero; hunch bets
                            # get a flat 1% tracking stake instead.
                            stake = round(bankroll * 0.01, 2)
                            st.caption(f"edge {e:+.1%} · no value "
                                       f"(hunch stake {stake:.0f})")
                        if mkt_p is not None:
                            # Split the picture: de-vigged fair market price
                            # vs the model — pure disagreement, margin removed.
                            st.caption(f"mkt {mkt_p:.0%} · disagree "
                                       f"{p - mkt_p:+.1%}")
                        if st.button(f"Log: {label}", key=f"log_{mkt}_{sel}"):
                            append_bet(home, away, mkt, sel, odds, p, stake,
                                       bet_type=bet_type, edge_val=e)
                            if storage.enabled() and not storage.push(
                                    LEDGER_PATH):
                                st.warning("Bet saved locally but the gist "
                                           "sync failed — it may not survive "
                                           "a redeploy.")
                            msg = f"Logged: {label} @ {odds:.2f}"
                            if e > 0:
                                st.success(msg)
                            else:
                                st.warning(msg + f" — model sees {e:+.1%} "
                                           "edge; logged as a 1% hunch "
                                           "stake against its advice")

        # --- Best Bets: verdicts on the odds entered above -------------------
        st.subheader("Best bets — this fixture")
        verdicts = pd.DataFrame(
            [{"market": m, "selection": lbl, "model p": p,
              "mkt (de-vig)": mp if mp is not None else float("nan"),
              "disagree": (p - mp) if mp is not None else float("nan"),
              "book odds": o if fmt == "Decimal" else decimal_to_american(o),
              "edge": edge(p, o),
              "verdict": ("🟢 undervalued" if edge(p, o) > 0.03 else
                          "🔴 overpriced" if edge(p, o) < -0.03 else
                          "⚪ fairly priced")}
             for m, lbl, s, p, o, mp in entered]).sort_values("edge",
                                                          ascending=False)
        if (verdicts["edge"].abs() < 0.005).all():
            st.info("All selections sit at the model's own fair price — "
                    "you haven't entered book odds yet, so there's nothing "
                    "to disagree about. Value only exists relative to a "
                    "price someone is offering.")
        st.dataframe(
            verdicts.style.format({"model p": "{:.1%}", "edge": "{:+.1%}",
                                   "mkt (de-vig)": "{:.1%}",
                                   "disagree": "{:+.1%}",
                                   "book odds": "{:.2f}" if fmt == "Decimal"
                                   else "{:+.0f}"}, na_rep="—")
                    .background_gradient(subset=["edge"], cmap="RdYlGn",
                                         vmin=-0.10, vmax=0.10),
            hide_index=True, use_container_width=True)
        st.caption("**edge** is real value at the price you typed (model p × "
                   "odds − 1). **mkt (de-vig)** strips the book's margin so "
                   "**disagree** = model − fair market is pure disagreement, "
                   "vig removed. 🟢 edge > +3% · ⚪ within ±3% · 🔴 below −3% "
                   "(margin or squad info winning). A 🟢 is a model opinion, "
                   "not a guarantee.")

    st.divider()
    st.subheader("Paper ledger")
    ledger = load_ledger()
    if ledger.empty:
        st.info("No paper bets yet. Log one above — the ledger settles "
                "itself as results arrive.")
        return
    settled = settle(ledger, results)
    done = settled[settled["status"] != "pending"]

    def _roi(df):
        s = df["stake"].sum()
        return df["profit"].sum() / s if s else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bets", f"{len(settled)} ({len(done)} settled)")
    c2.metric("Staked (settled)", f"{done['stake'].sum():.0f}")
    c3.metric("P/L", f"{done['profit'].sum():+.0f}")
    c4.metric("ROI", f"{_roi(done):+.1%}",
              help="Break-even means the model roughly priced the matches "
                   "as well as your book net of its margin — already a "
                   "strong result for a results-only model.")

    # Segment value (model-advised) from hunch (logged against advice) so the
    # one question that matters — did following the model's edges pay? — is
    # answerable instead of blended away.
    val_done = done[done["bet_type"] == "value"]
    hunch_done = done[done["bet_type"] == "hunch"]
    d1, d2, d3 = st.columns(3)
    d1.metric("Value bets ROI", f"{_roi(val_done):+.1%}",
              help=f"{len(val_done)} settled bets the model flagged as +edge.")
    d2.metric("Hunch bets ROI", f"{_roi(hunch_done):+.1%}",
              help=f"{len(hunch_done)} settled bets logged against the "
                   "model's advice (flat 1% stake).")

    # Closing-line value: the cleanest skill signal. Computed over settled
    # bets that have a closing line recorded below.
    have_close = done[done["closing_odds"].notna()]
    if len(have_close):
        avg_clv = np.mean([clv(o, c) for o, c
                           in zip(have_close["decimal_odds"],
                                  have_close["closing_odds"])])
        beat = np.mean([clv(o, c) > 0 for o, c
                        in zip(have_close["decimal_odds"],
                               have_close["closing_odds"])])
        d3.metric("Avg CLV", f"{avg_clv:+.1%}",
                  help=f"Beat the close on {beat:.0%} of {len(have_close)} "
                       "priced bets. Locking a better price than the market "
                       "settles at is the strongest evidence of edge — far "
                       "less variance-bound than P/L.")
    else:
        d3.metric("Avg CLV", "—",
                  help="Record closing odds below to track closing-line "
                       "value — whether you consistently beat the market's "
                       "settling price.")

    # Equity curve: cumulative P/L over settled bets in the order placed —
    # the bankroll trajectory, plus the worst peak-to-trough dip (drawdown)
    # the strategy lived through.
    curve = equity_curve(settled)
    if len(curve) >= 2:
        e1, e2 = st.columns([3, 1])
        with e1:
            chart = curve.reset_index(drop=True)[["cum_profit"]]
            chart.index = chart.index + 1          # 1-based bet number
            chart.index.name = "settled bet #"
            st.line_chart(chart, y="cum_profit", height=240)
        e2.metric("Max drawdown", f"{max_drawdown(curve['cum_profit']):+.0f}",
                  help="Largest peak-to-trough drop in cumulative P/L — the "
                       "deepest the bankroll fell from a high-water mark. "
                       "Variance is normal; this sizes it.")

    st.dataframe(settled.iloc[::-1], height=320, hide_index=True)

    # Record closing lines after the fact. Bet terms stay immutable; only the
    # observed closing line is editable, then persisted (save_closing_odds).
    with st.expander("Record closing odds (for CLV)"):
        st.caption("Enter each bet's decimal odds at kickoff. Only the "
                   "closing_odds column is editable — everything else is the "
                   "locked bet record.")
        edited = st.data_editor(
            ledger, hide_index=True, use_container_width=True,
            disabled=[c for c in ledger.columns if c != "closing_odds"],
            column_config={"closing_odds": st.column_config.NumberColumn(
                "closing_odds", min_value=1.01, step=0.05)},
            key="closing_editor")
        if st.button("Save closing odds"):
            save_closing_odds(edited)
            if storage.enabled():
                storage.push(LEDGER_PATH)
            st.success("Saved. CLV updates on the next rerun.")
            st.rerun()


def view_model_card():
    st.header("Model card")
    st.markdown(f"""
**Model.** Multinomial logistic regression (scikit-learn). Hyperparameters
(training-window start = 2010) selected by temporal validation
({4533:,} matches, 2022+); final model **refit on 2010 → 2026-06-08**
(15,742 matches, everything before the tournament freeze line) — the
standard deploy step, so coefficients also see the current cycle. Features:
pre-match Elo difference (computed from scratch, importance-weighted K,
margin-of-victory, +60 home-advantage offset), venue neutrality, match
importance tier, recent form (PPG and goal-diff over last 10), rest-day
difference. Every feature uses **only pre-match information** (unit-tested).
All parameters are frozen at the pre-tournament fit — enforced by date caps
in code, not just policy.

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
  team-specific (the Dixon-Coles challenger fixes this in the Schedule
  and Tournament-odds views; the Match explorer is still champion-only).
- The two models **disagree substantially on title odds** (champion:
  Spain 35%, Brazil ~5%; challenger: Brazil 21%, Spain 12%). Diagnosis:
  DC's 10y memory still credits Brazil's dominant 2010s. On the full
  validation set the champion wins (0.871 vs 0.891) — but on the 557
  **elite-vs-elite** validation matches (both teams WC2026 participants),
  DC hl=10y is the better model (log-loss 1.016 vs champion 1.038), and
  longer memory beats shorter there too. Each model is primary where it
  validates best; bookmakers' lower Brazil price likely reflects squad
  information neither model sees. The group stage adjudicates.
- FIFA ranking feature excluded (source data ends 2024-06).
- Penalty shootouts modeled as strength-weighted coin flips.
- Single-tournament backtest: weak power to detect compounding bias.
""")


def view_about():
    st.header("About this project")
    st.markdown("""
**A FIFA World Cup 2026 forecasting system built from scratch** — match
data to calibrated probabilities to a full tournament Monte Carlo to this
dashboard. Built as a first ML project with a teacher-style workflow:
every method explained before it was coded, every claim validated before
it was trusted.

#### What it does
- **Predicts match outcomes** (win/draw/loss) with two independently
  validated models:
  - *Champion*: multinomial logistic regression on hand-computed Elo
    ratings (importance-weighted K, margin-of-victory, home-advantage
    offset), recent form, venue and match-importance features
  - *Challenger*: a Dixon-Coles (1997) goals model — per-team attack and
    defense rates, low-score correlation correction, time-decayed maximum
    likelihood, hand-derived gradient
- **Simulates the 2026 World Cup** 10,000× — the real 48-team format,
  12 groups, best-8-thirds selection with the full FIFA tiebreaker
  cascade, the official Round-of-32 bracket — to produce title odds and
  round-by-round survival probabilities
- **Updates live during the tournament**: played matches are held fixed
  at their real scores (penalty shootouts resolved via the shootout
  record) and only the remaining fixtures are sampled
- **Compares against the betting market**: de-vigged sportsbook odds vs
  both models, plus a paper-trading ledger that settles itself against
  real results

#### Methodology principles (the actual point of the project)
1. **Temporal validation only** — train on the past, validate on the
   future; a random split would leak.
2. **No feature leakage** — every feature uses strictly pre-match
   information, enforced by unit tests that flip a match result and
   assert the match's own features don't move.
3. **Beat the baselines** — naive Elo-pick and base-rate baselines first;
   the model must earn its complexity.
4. **Probabilities over picks** — log-loss, Brier score, and reliability
   curves; accuracy alone can't see overconfidence.
5. **Pre-registration** — all 144 fixture forecasts (both models) were
   locked and SHA256-receipted before kickoff; the dashboard grades them
   against reality with no possibility of quiet revision.

#### Honest limitations
No squad/injury/lineup data (results only — the biggest gap vs
bookmakers), ratings frozen at tournament start, single-tournament
backtest power. Full details in the Model card.

#### Stack
Python · pandas · NumPy · scikit-learn · SciPy · Streamlit · Matplotlib ·
Seaborn — data from the open international results dataset (1872–present),
FIFA rankings, and DraftKings title odds.
""")


PAGES = {"🏆 Tournament odds": view_tournament,
         "📅 Schedule & predictions": view_schedule,
         "🧮 Live simulator": view_live_sim,
         "⚔️ Match explorer": view_match,
         "📈 Market vs models": view_market,
         "🎟️ Betting board": view_betting,
         "📋 Model card": view_model_card,
         "ℹ️ About": view_about}

def _require_password():
    """Gate the app behind a password when one is set in st.secrets (for the
    public cloud URL). No secret configured -> no gate, so local runs on the
    Mac are unchanged."""
    try:
        expected = st.secrets["password"]
    except Exception:
        return  # no secrets / no password -> open
    if st.session_state.get("authenticated"):
        return
    st.title("WC2026 Predictor")
    pw = st.text_input("Password", type="password")
    if not pw:
        st.stop()
    if pw == expected:
        st.session_state["authenticated"] = True
        st.rerun()
    st.error("Incorrect password.")
    st.stop()


_require_password()

st.sidebar.title("WC2026 Predictor")
choice = st.sidebar.radio("View", list(PAGES))
st.sidebar.caption("Trained through 2026-06-08 · 10k Monte Carlo runs · "
                   "probabilities, not promises.")
PAGES[choice]()
