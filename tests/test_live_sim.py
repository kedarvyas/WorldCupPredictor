"""End-to-end conditioning tests for the live tournament simulator.

Strategy: feed a COMPLETE fake group stage (the locked forecast's modal
scorelines as if they all happened), which makes placements and therefore
the R32 bracket deterministic. Then force a knockout winner via ko_results
and assert the conditioning propagates (winner reaches R16 with P=1).

Runnable without pytest:
    .venv/bin/python -m tests.test_live_sim
"""

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROJECT_ROOT
from src.simulation.bracket import GROUPS, R32, allocate_thirds
from src.simulation.group_sim import group_fixtures, simulate_live
from src.simulation.simulate import (build_prob_tables, rank_teams,
                                     scoreline_sampler, team_state)

_ENGINE = {}


def _engine():
    if not _ENGINE:
        model = joblib.load(MODELS_DIR / "outcome_model.joblib")
        _ENGINE["probs"] = build_prob_tables(model, team_state())
    return (_ENGINE["probs"],
            scoreline_sampler(np.random.default_rng(0)))


def _tie_free(results):
    """True if no group has two teams on identical (pts, gd, gf) and the
    8th/9th-best thirds are strictly separated — i.e. nothing anywhere
    falls through to head-to-head or drawing of lots."""
    thirds = []
    for g, teams in GROUPS.items():
        stats = {t: [0, 0, 0] for t in teams}
        for a, b in group_fixtures()[g]:
            ga, gb = results[(a, b)]
            for t, mine, theirs in ((a, ga, gb), (b, gb, ga)):
                stats[t][0] += 3 if mine > theirs else (1 if mine == theirs else 0)
                stats[t][1] += mine - theirs
                stats[t][2] += mine
        recs = sorted((tuple(v) for v in stats.values()), reverse=True)
        if len(set(recs)) < 4:
            return False
        thirds.append(recs[2])
    thirds.sort(reverse=True)
    return thirds[7] != thirds[8]


def _full_group_results():
    """One sampled, FULLY TIE-FREE realization of the group stage. NOT the
    modal scorelines — a universe where every match ends 1-0 leaves most
    groups perfectly tied (even joint winners go to drawing of lots), so
    nothing is deterministic. We sample varied scores and keep the first
    realization with no ties anywhere, verified by _tie_free."""
    probs, _ = _engine()
    for seed in range(100, 200):
        rng = np.random.default_rng(seed)
        sampler = scoreline_sampler(np.random.default_rng(seed))
        out = {}
        for fx in group_fixtures().values():
            for a, b in fx:
                p = probs[(a, b)]
                oc = rng.choice(3, p=p / p.sum())
                out[(a, b)] = sampler(oc, (a, b))
        if _tie_free(out):
            return out
    raise AssertionError("no tie-free realization in 100 seeds (suspicious)")


def _deterministic_bracket(results):
    """Reproduce placements + first R32 pairing from a complete group
    stage (mirrors simulate_live's group logic)."""
    rng = np.random.default_rng(99)
    placements, third_stats = {}, {}
    for g, teams in GROUPS.items():
        stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        h2h = {}
        for a, b in group_fixtures()[g]:
            ga, gb = results[(a, b)]
            h2h[(a, b)] = (ga, gb)
            stats[a]["pts"] += 3 if ga > gb else (1 if ga == gb else 0)
            stats[b]["pts"] += 3 if gb > ga else (1 if ga == gb else 0)
            stats[a]["gd"] += ga - gb; stats[b]["gd"] += gb - ga
            stats[a]["gf"] += ga;      stats[b]["gf"] += gb
        placements[g] = rank_teams(stats, rng, h2h_results=h2h)
        third_stats[g] = stats[placements[g][2]]
    # Use a pairing with NO third-place slot: thirds can be decided by
    # drawing of lots (genuinely random), which would make the pairing
    # unstable across simulations.
    match_no, sa, sb = next((m, a, b) for m, a, b in R32
                            if a[0] in "WR" and b[0] in "WR")
    pick = lambda s: (placements[s[1]][0] if s[0] == "W"
                      else placements[s[1]][1])
    return pick(sa), pick(sb)


def test_complete_groups_fix_positions_except_lots():
    """With all 72 results fixed, group POSITIONS are deterministic and all
    top-2 teams certainly reach R32. Third-placed teams may legitimately
    stay random: the modal-scoreline universe leaves many thirds on
    identical records, and the tiebreaker cascade correctly bottoms out at
    drawing of lots. R32 probabilities must still sum to exactly 32."""
    probs, sampler = _engine()
    results = _full_group_results()
    out = simulate_live(probs, sampler, n_sims=200, seed=1,
                        group_results=results)
    for col in ("1st", "2nd", "3rd", "4th"):
        assert set(np.round(out[col].unique(), 9)) <= {0.0, 1.0}, \
            f"group positions must be certain (col {col})"
    assert (out.loc[out["top2"] == 1.0, "R32"] == 1.0).all()
    assert abs(out["R32"].sum() - 32) < 1e-9
    fractional = out[(out["R32"] > 1e-9) & (out["R32"] < 1 - 1e-9)]
    assert (fractional["3rd"] == 1.0).all(), \
        "only third-placed (lots-tied) teams may have uncertain R32"


def test_forced_ko_winner_propagates():
    probs, sampler = _engine()
    results = _full_group_results()
    a, b = _deterministic_bracket(results)
    out = simulate_live(probs, sampler, n_sims=200, seed=1,
                        group_results=results,
                        ko_results={frozenset((a, b)): b})
    assert out.loc[b, "R16"] == 1.0, f"forced winner {b} must always reach R16"
    assert out.loc[a, "R16"] == 0.0, f"forced loser {a} must never reach R16"
    assert out.loc[a, "champion"] == 0.0, "eliminated team cannot win"


def test_no_conditioning_round_trip():
    """Sanity: pre-tournament live sim ~ static pre-tournament sim."""
    probs, sampler = _engine()
    out = simulate_live(probs, sampler, n_sims=2000, seed=7)
    static = pd.read_csv(PROJECT_ROOT / "reports" / "sim_2026.csv",
                         index_col=0)
    top5 = static["champion"].head(5)
    for team, p in top5.items():
        assert abs(out.loc[team, "champion"] - p) < 0.04, \
            f"{team}: live {out.loc[team, 'champion']:.3f} vs static {p:.3f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All live-sim tests passed.")
