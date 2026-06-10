"""Dixon-Coles tests: gradient correctness above all.

A wrong analytic gradient doesn't crash — L-BFGS just converges to the
wrong place, silently. Checking against finite differences on a small
problem is the standard defense.

Runnable without pytest:
    .venv/bin/python -m tests.test_dixon_coles
"""

import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColes


def _toy_matches(seed=0, n=200, teams=("A", "B", "C", "D")):
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp("2020-01-01")
    strength = {t: i * 0.4 for i, t in enumerate(teams)}  # D strongest
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        lam = np.exp(0.1 + strength[h] - strength[a] * 0.5)
        mu = np.exp(strength[a] - strength[h] * 0.5)
        rows.append({"date": base + pd.Timedelta(days=3 * k),
                     "home_team": h, "away_team": a,
                     "home_score": rng.poisson(lam),
                     "away_score": rng.poisson(mu),
                     "neutral": bool(rng.random() < 0.5)})
    return pd.DataFrame(rows)


def _nll_grad_fn(matches):
    """Capture fit()'s internal objective by intercepting the module's
    `minimize` reference, then calling the real one."""
    import src.models.dixon_coles as mod
    captured = {}
    real = mod.minimize

    def spy(fun, x0, **kw):
        captured["fun"], captured["x0"] = fun, x0
        return real(fun, x0, **kw)

    mod.minimize = spy
    try:
        DixonColes(half_life_years=5.0).fit(matches)
    finally:
        mod.minimize = real
    return captured["fun"], captured["x0"]


def test_gradient_matches_finite_differences():
    fun, x0 = _nll_grad_fn(_toy_matches())
    rng = np.random.default_rng(1)
    p = x0 + rng.normal(0, 0.1, size=x0.shape)
    p[-1] = 0.05  # rho inside bounds
    _, grad = fun(p)
    eps = 1e-6
    for i in list(range(3)) + [len(p) - 2, len(p) - 1]:  # params incl gamma,rho
        e = np.zeros_like(p)
        e[i] = eps
        fd = (fun(p + e)[0] - fun(p - e)[0]) / (2 * eps)
        assert abs(fd - grad[i]) < 1e-4 * max(1, abs(fd)), \
            f"param {i}: analytic {grad[i]:.6f} vs finite-diff {fd:.6f}"


def test_score_matrix_is_distribution():
    dc = DixonColes().fit(_toy_matches())
    M = dc.score_matrix("D", "A", true_home=True)
    assert abs(M.sum() - 1) < 1e-9
    assert (M >= 0).all()


def test_recovers_relative_strength():
    dc = DixonColes().fit(_toy_matches(n=600))
    # D was built strongest, A weakest: D should outscore A at home & away
    p = dc.outcome_probs("D", "A")
    assert p[0] > p[2], "stronger team should be favored"
    lam, mu = dc.rates("D", "A")
    assert lam > mu


def test_gradient_check_covers_attack_defense_block():
    """Also finite-diff a few params in the defense block specifically —
    its sign convention (beta enters both rates negatively) is the
    likeliest place for a derivation slip."""
    fun, x0 = _nll_grad_fn(_toy_matches(seed=2))
    rng = np.random.default_rng(3)
    p = x0 + rng.normal(0, 0.1, size=x0.shape)
    p[-1] = -0.05
    _, grad = fun(p)
    n = (len(p) - 2) // 2
    eps = 1e-6
    for i in range(n, n + 3):  # first three defense params
        e = np.zeros_like(p)
        e[i] = eps
        fd = (fun(p + e)[0] - fun(p - e)[0]) / (2 * eps)
        assert abs(fd - grad[i]) < 1e-4 * max(1, abs(fd)), \
            f"defense param {i}: analytic {grad[i]:.6f} vs fd {fd:.6f}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("All Dixon-Coles tests passed.")
