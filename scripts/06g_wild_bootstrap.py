"""
06g_wild_bootstrap.py
Additional check (requested in review): wild cluster bootstrap inference for the headline estimates.

Why: with only 22-38 receiving countries, standard errors clustered by country can be too small,
even after the t(G-1) correction used in the paper. The wild cluster bootstrap (Cameron, Gelbach
& Miller 2008; Roodman et al. 2019) is the standard remedy. In plain words: we re-run the model
thousands of times, each time randomly flipping the sign of whole countries' "leftovers"
(residuals), and see how unusual the real estimate looks against that spread.

What it reports for each headline estimate:
  - the paper's 95% range (CR1, t with G-1 degrees of freedom), for comparison
  - the wild bootstrap p-value (is the effect different from zero?)
  - the wild bootstrap 95% range (found by testing many possible values: "test inversion")
  - the wild bootstrap equivalence tests at +/-0.5 and +/-1 point
  - the verdict under the bootstrap, using the same rules as Section 4.6

Settings: restricted bootstrap (WCR: the tested value is imposed), Rademacher weights (+1/-1),
9,999 draws, fixed seed so the results can be reproduced.

Needs: scripts/fe.py, the panels from step 5
Output: outputs/tables/table6d_wild_bootstrap.csv

Run from the repo folder:  python scripts/06g_wild_bootstrap.py   (takes a few minutes)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts")
from fe import _demean, regress   # noqa: E402

B = 9999
SEED = 20261116
THRESHOLDS = (0.5, 1.0)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
HALF = np.log(0.5)


def prepare(df, y, xs, fixed_effects, cluster="iso3"):
    """Same sample and fixed-effect removal as fe.regress, so the numbers match the paper."""
    d = df.dropna(subset=[y] + xs + fixed_effects + [cluster]).copy()
    for _ in range(20):
        keep = np.ones(len(d), dtype=bool)
        for f in fixed_effects:
            keep &= d.groupby(f)[f].transform("size").to_numpy() > 1
        if keep.all():
            break
        d = d[keep]
    z = _demean(d[[y] + xs].to_numpy(), [d[f] for f in fixed_effects])
    g = pd.factorize(d[cluster])[0]
    n = len(d)
    k_all = len(xs) + sum(d[f].nunique() for f in fixed_effects) - (len(fixed_effects) - 1)
    return z[:, 0], z[:, 1:], g, n, k_all


class WildBootstrap:
    """Restricted wild cluster bootstrap for one coefficient (column j of X)."""

    def __init__(self, Y, X, g, n, k_all, j):
        self.Y, self.X, self.g, self.j = Y, X, g, j
        self.G = g.max() + 1
        self.adj = (self.G / (self.G - 1)) * ((n - 1) / max(n - k_all, 1))
        self.XtX_inv = np.linalg.pinv(X.T @ X)
        self.c = self.XtX_inv[j]                       # picks out coefficient j
        self.XX_g = np.stack([X[g == k].T @ X[g == k] for k in range(self.G)])
        rng = np.random.default_rng(SEED)
        self.W = rng.choice([-1.0, 1.0], size=(B, self.G))   # same draws for every tested value
        b = self.XtX_inv @ X.T @ Y
        self.beta = b[j]
        self.t_obs_at = lambda v: (self.beta - v) / self._se(Y - X @ b)

    def _se(self, e):
        u = np.array([self.c @ (self.X[self.g == k].T @ e[self.g == k]) for k in range(self.G)])
        return np.sqrt(self.adj * (u ** 2).sum())

    def t_draws(self, value):
        """Bootstrap t-statistics for H0: coefficient j = value."""
        X, g, j = self.X, self.g, self.j
        others = [i for i in range(X.shape[1]) if i != j]
        Xo = X[:, others]
        y_adj = self.Y - value * X[:, j]
        b_r = np.linalg.lstsq(Xo, y_adj, rcond=None)[0] if others else np.zeros(0)
        fit = value * X[:, j] + (Xo @ b_r if others else 0.0)
        e_r = self.Y - fit
        Xf_g = np.stack([X[g == k].T @ fit[g == k] for k in range(self.G)])   # G x k
        Xe_g = np.stack([X[g == k].T @ e_r[g == k] for k in range(self.G)])   # G x k
        XtY = Xf_g.sum(0)[None, :] + self.W @ Xe_g                            # B x k
        b_star = XtY @ self.XtX_inv.T                                          # B x k
        S = Xf_g[None] + self.W[:, :, None] * Xe_g[None] - np.einsum("gkl,bl->bgk", self.XX_g, b_star)
        u = S @ self.c                                                         # B x G
        se_star = np.sqrt(self.adj * (u ** 2).sum(1))
        return (b_star[:, j] - value) / se_star

    def p_two_sided(self, value):
        t = self.t_obs_at(value)
        return np.mean(np.abs(self.t_draws(value)) >= abs(t))

    def p_one_sided(self, value, side):
        """side='greater': H1 coef > value; side='less': H1 coef < value."""
        t, ts = self.t_obs_at(value), self.t_draws(value)
        return np.mean(ts >= t) if side == "greater" else np.mean(ts <= t)

    def ci(self, se_guess, level=0.95):
        """95% range by test inversion: the values the bootstrap test does not reject."""
        alpha = 1 - level

        def edge(direction):
            inside, outside = self.beta, self.beta + direction * se_guess
            while self.p_two_sided(outside) > alpha:      # widen until rejected
                outside = self.beta + 2 * (outside - self.beta)
            for _ in range(30):                            # bisection
                mid = (inside + outside) / 2
                if self.p_two_sided(mid) > alpha:
                    inside = mid
                else:
                    outside = mid
            return (inside + outside) / 2

        return edge(-1), edge(+1)


def run(df, y, xs, fes, var, label, scale=1.0):
    paper = regress(df, y, xs, fes, label=label, scale={var: scale} if scale != 1.0 else None)
    paper = paper[paper["variable"] == var].iloc[0]
    Y, X, g, n, k_all = prepare(df, y, xs, fes)
    wb = WildBootstrap(Y, X, g, n, k_all, xs.index(var))
    m = scale
    lo, hi = wb.ci(se_guess=abs(paper["se"] / m))
    lo, hi = sorted((lo * m, hi * m))
    row = {"model": label, "variable": var, "coef": wb.beta * m,
           "paper_ci95_lo": paper["ci95_lo"], "paper_ci95_hi": paper["ci95_hi"], "paper_p": paper["p"],
           "wild_ci95_lo": lo, "wild_ci95_hi": hi, "wild_p": wb.p_two_sided(0.0),
           "n_obs": n, "n_clusters": wb.G}
    for th in THRESHOLDS:
        # equivalence: effect above -th AND below +th, in reported (scaled) units
        lo_null, hi_null = sorted((-th / m, th / m))
        row[f"wild_tost_p_{th}"] = max(wb.p_one_sided(lo_null, "greater"), wb.p_one_sided(hi_null, "less"))
    sig, neg = row["wild_p"] < 0.05, row["wild_tost_p_0.5"] < 0.05
    row["wild_verdict"] = ("real but negligible (<0.5)" if sig and neg else
                           "MEANINGFUL EFFECT" if sig and abs(row["coef"]) >= 0.5 else
                           "real, size unclear" if sig else
                           "no meaningful effect" if neg else "inconclusive")
    row["paper_verdict"] = paper["verdict"]
    print(f"  done: {label} | {var}")
    return row


if __name__ == "__main__":
    os.makedirs("outputs/tables", exist_ok=True)
    cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
    svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
    FE_C, FE_S = ["corridor", "period_q"], ["service_id", "period_q"]
    C_XS = ["grey_qend", "L1_log_dep"] + CONTROLS + ["log_n_services"]

    print(f"Wild cluster bootstrap, {B:,} draws per test (this takes a few minutes)...")
    rows = [
        run(cq, "cost", ["grey_qend"], FE_C, "grey_qend", "A: grey only"),
        run(cq, "cost", ["grey_qend"] + CONTROLS + ["log_n_services"], FE_C, "grey_qend", "B: + controls"),
        run(cq, "cost", C_XS, FE_C, "grey_qend", "C: + de-risking"),
        run(cq, "cost", C_XS, FE_C, "L1_log_dep", "C: + de-risking (deposits halve)", scale=HALF),
        run(svc, "cost_pct_200", ["grey_qend"], FE_S, "grey_qend", "S1: all services"),
    ]
    out = pd.DataFrame(rows)
    out.to_csv("outputs/tables/table6d_wild_bootstrap.csv", index=False)

    pd.set_option("display.width", 220)
    show = out[["model", "coef", "paper_ci95_lo", "paper_ci95_hi", "wild_ci95_lo", "wild_ci95_hi",
                "paper_p", "wild_p", "wild_tost_p_0.5", "n_clusters", "paper_verdict", "wild_verdict"]].copy()
    for col in ["coef", "paper_ci95_lo", "paper_ci95_hi", "wild_ci95_lo", "wild_ci95_hi"]:
        show[col] = show[col].map(lambda v: f"{v:+.2f}")
    for col in ["paper_p", "wild_p", "wild_tost_p_0.5"]:
        show[col] = show[col].map(lambda v: f"{v:.3f}")
    print("\nWILD CLUSTER BOOTSTRAP vs the paper's standard errors (total cost, percentage points)")
    print(show.to_string(index=False))
    print("\nIf the wild ranges are similar to the paper's and the verdicts match, the conclusions do not")
    print("depend on the small-sample correction. Saved outputs/tables/table6d_wild_bootstrap.csv")
