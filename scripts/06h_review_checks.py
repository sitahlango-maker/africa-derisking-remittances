"""
06h_review_checks.py
Three additional checks requested in review, in one script.

  PART 1  Pre-trend sensitivity (stacked event study, Section 5.3)
          Prices in countries about to be listed were already drifting before the listing.
          If that drift simply continued afterwards, could it be hiding a real effect?
          We extend the pre-listing line forward and measure the change in prices relative to
          it (linear extrapolation: the M = 0 case of the smoothness restriction in Rambachan
          and Roth, 2023). We also report a simple bound: what if the trend after the event
          differed by as much as the largest difference seen before it?

  PART 2  Provider exit (survivor bias)
          The price data only show services that are still quoted. If grey-listing or
          de-risking pushed providers out, the services that remain could make prices look
          unaffected. We test directly: is a service more likely to disappear from a corridor
          (while the corridor itself is still surveyed) when the receiving country is
          grey-listed, or when its banks' deposits abroad fall?

  PART 3  Grey-list dates confirmed from secondary sources
          Eight listing or removal dates inside the price window were confirmed from official
          or press reports rather than FATF's own statement. We re-run the main models
          (a) without the quarters within one year of those dates, where a dating error could
          matter, and (b) without those eight countries altogether.

Needs: scripts/fe.py, the panels from step 5, the grey-list file from step 2
Outputs: outputs/tables/table6e_pretrend.csv
         outputs/tables/table6f_exit.csv
         outputs/tables/table6g_secondary_dates.csv

Run from the repo folder:  python scripts/06h_review_checks.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts")
from fe import _demean, regress, show, verdict   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
HALVING = {"L1_log_dep": np.log(0.5)}
FE_C, FE_S = ["corridor", "period_q"], ["service_id", "period_q"]

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
ep = pd.read_csv("data/processed/fatf_greylist_africa_episodes.csv", parse_dates=["listed_from", "removed_at"])
for d in (cq, svc):
    d["qnum"] = d["year"] * 4 + d["quarter"]


def evidence(label, coef, se, G, n, extra=None):
    """The paper's three layers of evidence (Section 4.6) for one number."""
    dof = G - 1
    t95, t80 = stats.t.ppf(0.975, dof), stats.t.ppf(0.80, dof)
    r = {"model": label, "coef": coef, "se": se, "p": 2 * stats.t.sf(abs(coef / se), dof),
         "ci95_lo": coef - t95 * se, "ci95_hi": coef + t95 * se, "mde80": (t95 + t80) * se,
         "n_obs": n, "n_clusters": G}
    for th in (0.5, 1.0):
        r[f"tost_p_{th}"] = max(stats.t.sf((coef + th) / se, dof), stats.t.cdf((coef - th) / se, dof))
    r["verdict"] = verdict(r)
    r.update(extra or {})
    return r


def fe_vcov(df, y, xs, fixed_effects, cluster="iso3"):
    """Same estimator as fe.regress (same sample rules, CR1 clustering), but returns the full
    covariance matrix, which we need to combine several event-study coefficients."""
    d = df.dropna(subset=[y] + xs + fixed_effects + [cluster]).copy()
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
    g = pd.factorize(d[cluster])[0]
    G = g.max() + 1
    meat = np.zeros((len(xs), len(xs)))
    for k in range(G):
        s = X[g == k].T @ e[g == k]
        meat += np.outer(s, s)
    n = len(d)
    k_all = len(xs) + sum(d[f].nunique() for f in fixed_effects) - (len(fixed_effects) - 1)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / max(n - k_all, 1))
    return b, V, G, n


# =============================================================================
# PART 1. Pre-trend sensitivity, stacked event study (same design as step 7)
# =============================================================================
LO, HI = -3, 4
WIN_START, WIN_END = pd.Timestamp("2012-01-01"), pd.Timestamp("2025-09-30")
cq["q_end"] = pd.PeriodIndex.from_fields(year=cq["year"], quarter=cq["quarter"], freq="Q").end_time.normalize()
in_prices = set(cq["iso3"])
EVENTS = {
    "listing": ep.loc[ep["listed_from"].between(WIN_START, WIN_END) & ep["iso3"].isin(in_prices),
                      ["iso3", "listed_from"]].rename(columns={"listed_from": "date"}),
    "removal": ep.loc[ep["removed_at"].between(WIN_START, WIN_END) & ep["iso3"].isin(in_prices),
                      ["iso3", "removed_at"]].rename(columns={"removed_at": "date"}),
}


def stack(kind):
    blocks = []
    for n, ev in enumerate(EVENTS[kind].itertuples()):
        start, end = ev.date + pd.DateOffset(years=LO), ev.date + pd.DateOffset(years=HI + 1)
        w = cq[(cq["q_end"] >= start) & (cq["q_end"] < end)].copy()
        if w.empty:
            continue
        on_list = w.groupby("iso3")["grey_qend"].max()
        clean = set(on_list[on_list == 0].index)
        w = w[(w["iso3"] == ev.iso3) | w["iso3"].isin(clean)]
        w["treated"] = (w["iso3"] == ev.iso3).astype(int)
        w["k"] = np.floor((w["q_end"] - ev.date).dt.days / 365.25).clip(LO, HI)
        w["event_id"] = n
        blocks.append(w)
    s = pd.concat(blocks, ignore_index=True)
    s["corridor_event"] = s["corridor"] + "|" + s["event_id"].astype(str)
    s["quarter_event"] = s["period_q"] + "|" + s["event_id"].astype(str)
    return s


ks = [k for k in range(LO, HI + 1) if k != -1]
post = [k for k in ks if k >= 0]
pre1 = []
for kind in ["listing", "removal"]:
    s = stack(kind)
    xs = []
    for k in ks:
        s[f"k{k}"] = ((s["k"] == k) & (s["treated"] == 1)).astype(int)
        xs.append(f"k{k}")
    b, V, G, n = fe_vcov(s, "cost", xs, ["corridor_event", "quarter_event"])
    idx = {k: i for i, k in enumerate(ks)}
    pre_vals = {-3: b[idx[-3]], -2: b[idx[-2]], -1: 0.0}

    # (a) no adjustment: simple average of the post-event years
    a0 = np.zeros(len(ks))
    for k in post:
        a0[idx[k]] = 1 / len(post)
    est0, se0 = a0 @ b, np.sqrt(a0 @ V @ a0)

    # (b) linear extrapolation of the pre-event line (fitted through years -3, -2 and -1 = 0)
    #     slope = -beta(-3)/2, line passes through the pre-period mean at year -2
    #     adjusted effect in year k = beta(k) - [mean_pre + slope * (k + 2)]
    a1 = a0.copy()
    for k in post:
        w_line = np.zeros(len(ks))
        w_line[idx[-3]] += 1 / 3 - 0.5 * (k + 2)   # mean_pre contributes 1/3; slope contributes -(k+2)/2
        w_line[idx[-2]] += 1 / 3
        a1 -= w_line / len(post)
    est1, se1 = a1 @ b, np.sqrt(a1 @ V @ a1)
    slope = -b[idx[-3]] / 2

    r0 = evidence(f"{kind}: stacked, no adjustment", est0, se0, G, n)
    r1 = evidence(f"{kind}: stacked, relative to the pre-event trend", est1, se1, G, n,
                  {"pre_trend_slope_per_year": slope})
    # (c) simple bound: post-event trend could differ by as much as the largest pre-event gap
    m = max(abs(pre_vals[-3]), abs(pre_vals[-2]))
    r0["bound_if_post_gap_as_large_as_pre"] = f"{r0['ci95_lo'] - m:+.2f} to {r0['ci95_hi'] + m:+.2f}"
    r0["largest_pre_gap"] = m
    r0["pre_k-3"], r0["pre_k-2"] = pre_vals[-3], pre_vals[-2]
    pre1 += [r0, r1]

t_pre = pd.DataFrame(pre1)
t_pre.to_csv("outputs/tables/table6e_pretrend.csv", index=False)

print("=" * 100)
print("PART 1. PRE-TREND SENSITIVITY: average change in total cost (points) in the years after the event")
print("  'no adjustment' = average of the post-event years, as in the event study")
print("  'relative to the pre-event trend' = the same, after extending the pre-event line forward")
print("=" * 100)
show(t_pre, cols=("model", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
for r in pre1:
    if "largest_pre_gap" in r and not pd.isna(r.get("largest_pre_gap")):
        print(f"  {r['model'].split(':')[0]}: pre-event estimates k-3 = {r['pre_k-3']:+.2f}, k-2 = {r['pre_k-2']:+.2f}; "
              f"if the post-event trend differed by as much as {r['largest_pre_gap']:.2f}, the range widens to "
              f"{r['bound_if_post_gap_as_large_as_pre']}")
    if "pre_trend_slope_per_year" in r and not pd.isna(r.get("pre_trend_slope_per_year")):
        print(f"  {r['model'].split(':')[0]}: pre-event trend = {r['pre_trend_slope_per_year']:+.2f} points per year")

# =============================================================================
# PART 2. Provider exit
# =============================================================================
e = svc.copy()
last_service = e.groupby("service_id")["qnum"].transform("max")
last_corridor = e.groupby("corridor")["qnum"].transform("max")
# a service "exits" in its last quarter in the data, but only if its corridor is surveyed later
e["exit"] = np.where(e["qnum"] < last_corridor, (e["qnum"] == last_service).astype(float) * 100, np.nan)
base = e["exit"].mean()

t_exit = pd.concat([
    regress(e, "exit", ["grey_qend"], FE_C, label="E1: grey-listing"),
    regress(e, "exit", ["grey_qend", "L1_log_dep"] + CONTROLS, FE_C, label="E2: + de-risking and controls",
            scale=HALVING),
])
t_exit = t_exit[t_exit["variable"].isin(["grey_qend", "L1_log_dep"])]
t_exit["baseline_exit_rate_pct"] = base
t_exit.to_csv("outputs/tables/table6f_exit.csv", index=False)

print("\n" + "=" * 100)
print("PART 2. PROVIDER EXIT: change in the chance (percentage points) that a service is quoted for the last time")
print(f"  average chance in any quarter: {base:.1f}%   (corridor + quarter fixed effects; clustered by country)")
print("  L1_log_dep = if banks' deposits abroad HALVED. Verdict thresholds (0.5 / 1 point) are the paper's price")
print("  thresholds and do not apply here: read the estimate against the average chance instead.")
print("=" * 100)
show(t_exit, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80"))

# =============================================================================
# PART 3. Grey-list dates confirmed from secondary sources
# =============================================================================
# dates inside the price window (to 2025 Q3) confirmed from secondary sources (see 02_fatf_greylist.py)
SECONDARY = [("TZA", "2022-10-21"), ("NGA", "2023-02-24"), ("BWA", "2021-10-21"), ("MUS", "2021-10-21"),
             ("MLI", "2021-10-21"), ("COD", "2022-10-21"), ("MOZ", "2022-10-21"), ("ZAF", "2023-02-24")]


def near_secondary(df):
    """True for quarters within one year (4 quarters) of a secondary-source date in that country."""
    flag = np.zeros(len(df), dtype=bool)
    for iso, date in SECONDARY:
        dt = pd.Timestamp(date)
        qd = dt.year * 4 + dt.quarter
        flag |= (df["iso3"].to_numpy() == iso) & (np.abs(df["qnum"].to_numpy() - qd) <= 4)
    return flag


countries = {iso for iso, _ in SECONDARY}
C_XS = ["grey_qend", "L1_log_dep"] + CONTROLS + ["log_n_services"]
rows = []
for name, cq_s, svc_s in [
    ("main sample", cq, svc),
    ("(a) without quarters within a year of the 8 dates", cq[~near_secondary(cq)], svc[~near_secondary(svc)]),
    ("(b) without the 8 countries", cq[~cq["iso3"].isin(countries)], svc[~svc["iso3"].isin(countries)]),
]:
    rows += [
        regress(cq_s, "cost", ["grey_qend"], FE_C, label=f"A: grey only | {name}"),
        regress(cq_s, "cost", C_XS, FE_C, label=f"C: + de-risking | {name}", scale=HALVING),
        regress(svc_s, "cost_pct_200", ["grey_qend"], FE_S, label=f"S1: all services | {name}"),
    ]
t_sec = pd.concat(rows)
t_sec = t_sec[t_sec["variable"].isin(["grey_qend", "L1_log_dep"])]
t_sec.to_csv("outputs/tables/table6g_secondary_dates.csv", index=False)

print("\n" + "=" * 100)
print("PART 3. SECONDARY-SOURCE GREY-LIST DATES: main models re-estimated")
print("=" * 100)
show(t_sec, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("\nCountries with a secondary-source date in the price window:", ", ".join(sorted(countries)))
print("Saved outputs/tables/table6e_pretrend.csv, table6f_exit.csv, table6g_secondary_dates.csv")
