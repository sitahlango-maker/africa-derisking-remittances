"""
fe.py
Shared helper for the models: fixed-effects regression with standard errors clustered by
receiving country, plus the paper's three layers of evidence (Section 4.6):
  1. 95% confidence interval
  2. equivalence test (two one-sided tests) against +/-0.5 and +/-1 percentage point
  3. minimum detectable effect at 80% power

Imported by 06_models.py and 07_event_study.py. Not run on its own.
"""

import numpy as np
import pandas as pd
from scipy import stats

THRESHOLDS = (0.5, 1.0)      # smallest effects of interest, in percentage points (Section 4.6)


def _demean(values, groups, tol=1e-9, max_iter=1000):
    """Remove every fixed effect from each column by repeatedly subtracting group averages
    ("alternating projections"). What is left is each observation's difference from its own
    corridor's and quarter's normal level."""
    x = values.astype(float).copy()
    codes = [pd.factorize(g)[0] for g in groups]
    counts = [np.bincount(c) for c in codes]
    for _ in range(max_iter):
        before = x.copy()
        for c, n in zip(codes, counts):
            for j in range(x.shape[1]):
                x[:, j] -= (np.bincount(c, weights=x[:, j]) / n)[c]
        if np.abs(x - before).max() < tol:
            break
    return x


def regress(df, y, xs, fixed_effects, cluster="iso3", label="", scale=None):
    """Fixed-effects OLS of y on xs.
    scale: optional {variable: multiplier} to report an effect for a meaningful change,
           e.g. {"L1_log_dep": np.log(0.5)} = effect of deposits abroad halving."""
    d = df.dropna(subset=[y] + xs + fixed_effects + [cluster]).copy()
    # drop observations that are alone in a fixed-effect group: they carry no information
    for _ in range(20):
        keep = np.ones(len(d), dtype=bool)
        for f in fixed_effects:
            keep &= d.groupby(f)[f].transform("size").to_numpy() > 1
        if keep.all():
            break
        d = d[keep]

    z = _demean(d[[y] + xs].to_numpy(), [d[f] for f in fixed_effects])
    Y, X = z[:, 0], z[:, 1:]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ Y
    e = Y - X @ b

    # standard errors clustered by receiving country (CR1)
    g = pd.factorize(d[cluster])[0]
    G = g.max() + 1
    meat = np.zeros((len(xs), len(xs)))
    for k in range(G):
        s = X[g == k].T @ e[g == k]
        meat += np.outer(s, s)
    n = len(d)
    k_all = len(xs) + sum(d[f].nunique() for f in fixed_effects) - (len(fixed_effects) - 1)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / max(n - k_all, 1))
    se = np.sqrt(np.diag(V))
    dof = G - 1

    rows = []
    for i, v in enumerate(xs):
        m = (scale or {}).get(v, 1.0)          # report the effect of a meaningful change
        coef, s_e = b[i] * m, abs(se[i] * m)
        t95, t90, t80 = stats.t.ppf(0.975, dof), stats.t.ppf(0.95, dof), stats.t.ppf(0.80, dof)
        p = 2 * stats.t.sf(abs(coef / s_e), dof)
        row = {"model": label, "outcome": y, "variable": v, "scaled_by": m,
               "coef": coef, "se": s_e, "p": p,
               "ci95_lo": coef - t95 * s_e, "ci95_hi": coef + t95 * s_e,
               "ci90_lo": coef - t90 * s_e, "ci90_hi": coef + t90 * s_e,
               "mde80": (t95 + t80) * s_e,
               "n_obs": n, "n_clusters": G}
        for th in THRESHOLDS:
            # two one-sided tests: is the effect above -th AND below +th?
            p_low = stats.t.sf((coef + th) / s_e, dof)
            p_high = stats.t.cdf((coef - th) / s_e, dof)
            row[f"tost_p_{th}"] = max(p_low, p_high)
        row["verdict"] = verdict(row)
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(r):
    """Plain-language reading, following the decision rules fixed in Section 4.6 (headline threshold 0.5)."""
    significant = r["p"] < 0.05
    negligible = r["tost_p_0.5"] < 0.05
    if significant and negligible:
        return "real but negligible (<0.5)"
    if significant and abs(r["coef"]) >= 0.5:
        return "MEANINGFUL EFFECT"
    if significant:
        return "real, size unclear"
    if negligible:
        return "no meaningful effect"
    return "inconclusive"


def show(res, cols=("model", "outcome", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict")):
    """Print results in a readable table."""
    out = res[list(cols)].copy()
    for c in ("coef", "ci95_lo", "ci95_hi", "mde80"):
        out[c] = out[c].map(lambda v: f"{v:+.2f}" if c != "mde80" else f"{v:.2f}")
    out["p"] = res["p"].map(lambda v: f"{v:.3f}")
    print(out.to_string(index=False))
