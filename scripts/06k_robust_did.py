"""
06k_robust_did.py
Two analyses requested in review.

  PART 1  A grey-listing estimate that is robust to effects differing across countries and over time
          The main models (two-way fixed effects) can give misleading averages when countries are
          listed at different dates and leave the list again, because already-listed countries end
          up serving as comparisons. Following de Chaisemartin and D'Haultfoeuille (2020; 2024), we
          compare each country that changes status with countries whose status has NOT yet changed:
            - "switchers": the corridors (or services) of a country in the year its grey-list status
              first changes within the data (added to the list, or removed if it starts on it);
            - "controls": corridors of countries that started with the same status and have not
              changed it by the year being compared.
          For each year l = 0, 1, 2, 3 after the change, the effect is
              [change in price for switchers since the year before] - [same change for controls].
          Removals are counted with the opposite sign, so every number reads as "the effect of
          being on the list". Placebos compare the same groups BEFORE the change: they should be
          close to zero. Annual data: price = the year's average, status = on the list at 31 December.
          Uncertainty: leave-one-country-out jackknife, t distribution with G-1 degrees of freedom.

  PART 2  Event study at service level (the headline estimate's own level)
          The stacked event study of step 7, re-run with services instead of corridors
          (service x event and quarter x event fixed effects).

Needs: scripts/fe.py, the panels from step 5, data/processed/fatf_greylist_africa_yearly.csv
Outputs: outputs/tables/table6j_robust_did.csv, outputs/tables/table6k_service_event_study.csv

Run from the repo folder:  python scripts/06k_robust_did.py   (part 2 takes a few minutes)
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts")
from fe import regress, show, verdict   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

L_MAX, P_MAX = 3, 2            # effects for years 0..3 after the change; placebos 1..2 years before

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
gy = pd.read_csv("data/processed/fatf_greylist_africa_yearly.csv")


# =============================================================================
# PART 1. Heterogeneity-robust difference-in-differences (first change of status)
# =============================================================================
def annual(df, unit, y):
    a = df.groupby([unit, "iso3", "year"])[y].mean().rename("Y").reset_index()
    return a.rename(columns={unit: "unit"})


def status_table(countries, years):
    g = gy.set_index(["iso3", "year"])["grey_31dec"]
    rows = [(c, t, int(g.get((c, t), 0))) for c in countries for t in years]
    st = pd.DataFrame(rows, columns=["iso3", "year", "D"])
    first = {}
    for c, d in st.groupby("iso3"):
        d = d.sort_values("year")
        ch = d["year"][d["D"].diff().fillna(0) != 0]
        first[c] = (int(ch.iloc[0]) if len(ch) else None, int(d["D"].iloc[0]))
    return st, first


def dcdh(a, first):
    """Returns {('effect', l) or ('placebo', p): estimate}, plus counts."""
    Y = a.set_index(["unit", "year"])["Y"]
    unit_c = a.drop_duplicates("unit").set_index("unit")["iso3"]
    years = sorted(a["year"].unique())
    out, counts = {}, {}
    for kind, lags in (("effect", range(0, L_MAX + 1)), ("placebo", range(1, P_MAX + 1))):
        for l in lags:
            num, den = 0.0, 0
            for c, (F, b) in first.items():
                if F is None:
                    continue
                base = F - 1
                target = F + l if kind == "effect" else F - 1 - l
                if base not in years or target not in years:
                    continue
                sw = unit_c.index[unit_c == c]
                # controls: same starting status, status unchanged through max(base, target)
                last = max(base, target)
                ctrl_c = [k for k, (Fk, bk) in first.items() if bk == b and (Fk is None or Fk > last) and k != c]
                ctrl = unit_c.index[unit_c.isin(ctrl_c)]

                def dy(units):
                    idx_t = pd.MultiIndex.from_product([units, [target]])
                    idx_b = pd.MultiIndex.from_product([units, [base]])
                    yt = Y.reindex(idx_t).to_numpy()
                    yb = Y.reindex(idx_b).to_numpy()
                    return (yt - yb)[~np.isnan(yt - yb)]
                d_sw, d_ct = dy(sw), dy(ctrl)
                if len(d_sw) == 0 or len(d_ct) == 0:
                    continue
                sign = 1 if b == 0 else -1          # removal counts with the opposite sign
                num += sign * (d_sw.mean() - d_ct.mean()) * len(d_sw)
                den += len(d_sw)
            out[(kind, l)] = num / den if den else np.nan
            counts[(kind, l)] = den
    eff = [out[("effect", l)] for l in range(L_MAX + 1)]
    out[("average", "0-3")] = np.nanmean(eff)
    counts[("average", "0-3")] = None
    return out, counts


def run_dcdh(df, unit, y, label):
    a = annual(df, unit, y)
    countries = sorted(a["iso3"].unique())
    st, first = status_table(countries, sorted(a["year"].unique()))
    full, counts = dcdh(a, first)
    # leave-one-country-out jackknife
    reps = []
    for c in countries:
        sub = a[a["iso3"] != c]
        f2 = {k: v for k, v in first.items() if k != c}
        reps.append(dcdh(sub, f2)[0])
    G = len(countries)
    rows = []
    for key, est in full.items():
        r = np.array([rp[key] for rp in reps], dtype=float)
        r = r[~np.isnan(r)]
        se = np.sqrt((len(r) - 1) / len(r) * ((r - r.mean()) ** 2).sum()) if len(r) > 1 else np.nan
        dof = G - 1
        t95, t80 = stats.t.ppf(0.975, dof), stats.t.ppf(0.80, dof)
        row = {"model": label, "what": key[0], "years": key[1], "coef": est, "se": se,
               "p": 2 * stats.t.sf(abs(est / se), dof), "ci95_lo": est - t95 * se, "ci95_hi": est + t95 * se,
               "mde80": (t95 + t80) * se, "n_switcher_units": counts[key], "n_clusters": G}
        for th in (0.5, 1.0):
            row[f"tost_p_{th}"] = max(stats.t.sf((est + th) / se, dof), stats.t.cdf((est - th) / se, dof))
        row["verdict"] = verdict(row)
        rows.append(row)
    n_sw = sum(1 for F, b in first.values() if F is not None)
    n_in = sum(1 for F, b in first.values() if F is not None and b == 0)
    print(f"  {label}: {n_sw} countries change status in the data ({n_in} added to the list, {n_sw - n_in} removed)")
    return pd.DataFrame(rows)


print("=" * 100)
print("PART 1. GREY-LISTING, ROBUST TO EFFECTS THAT DIFFER ACROSS COUNTRIES AND OVER TIME")
print("  (de Chaisemartin-D'Haultfoeuille style: first change of status vs countries not yet changed)")
print("=" * 100)
r1 = pd.concat([
    run_dcdh(cq, "corridor", "cost", "corridor level"),
    run_dcdh(svc, "service_id", "cost_pct_200", "service level"),
])
r1.to_csv("outputs/tables/table6j_robust_did.csv", index=False)
r1["model"] = r1["model"] + " | " + r1["what"] + " " + r1["years"].astype(str)
show(r1, cols=("model", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("  effect l = years after the change (0 = the year of the change); placebo p = p years BEFORE (should be ~0)")
print("  average 0-3 = simple average of the four yearly effects (the headline robust estimate)")

# =============================================================================
# PART 2. Stacked event study at service level
# =============================================================================
LO, HI = -3, 4
WIN_START, WIN_END = pd.Timestamp("2012-01-01"), pd.Timestamp("2025-09-30")
ep = pd.read_csv("data/processed/fatf_greylist_africa_episodes.csv", parse_dates=["listed_from", "removed_at"])
svc["q_end"] = pd.PeriodIndex.from_fields(year=svc["year"], quarter=svc["quarter"], freq="Q").end_time.normalize()
in_prices = set(svc["iso3"])
EVENTS = {
    "listing": ep.loc[ep["listed_from"].between(WIN_START, WIN_END) & ep["iso3"].isin(in_prices),
                      ["iso3", "listed_from"]].rename(columns={"listed_from": "date"}),
    "removal": ep.loc[ep["removed_at"].between(WIN_START, WIN_END) & ep["iso3"].isin(in_prices),
                      ["iso3", "removed_at"]].rename(columns={"removed_at": "date"}),
}
keep = ["service_id", "iso3", "period_q", "q_end", "grey_qend", "cost_pct_200"]
res2 = []
print("\n" + "=" * 100)
print("PART 2. STACKED EVENT STUDY AT SERVICE LEVEL (change in total cost vs the year before the event)")
print("=" * 100)
for kind in ["listing", "removal"]:
    blocks = []
    for n, ev in enumerate(EVENTS[kind].itertuples()):
        start, end = ev.date + pd.DateOffset(years=LO), ev.date + pd.DateOffset(years=HI + 1)
        w = svc.loc[(svc["q_end"] >= start) & (svc["q_end"] < end), keep].copy()
        if w.empty:
            continue
        on_list = w.groupby("iso3")["grey_qend"].max()
        clean = set(on_list[on_list == 0].index)
        w = w[(w["iso3"] == ev.iso3) | w["iso3"].isin(clean)]
        w["treated"] = (w["iso3"] == ev.iso3).astype(int)
        w["k"] = np.floor((w["q_end"] - ev.date).dt.days / 365.25).clip(LO, HI)
        w["unit_event"] = w["service_id"] + "|" + str(n)
        w["quarter_event"] = w["period_q"] + "|" + str(n)
        blocks.append(w)
    s = pd.concat(blocks, ignore_index=True)
    xs = []
    for j in range(LO, HI + 1):
        if j == -1:
            continue
        s[f"k{j}"] = ((s["k"] == j) & (s["treated"] == 1)).astype(int)
        xs.append(f"k{j}")
    print(f"  {kind}: {len(blocks)} events, {len(s):,} stacked observations; estimating...")
    r = regress(s, "cost_pct_200", xs, ["unit_event", "quarter_event"], label=f"service stacked | {kind}")
    s["after"] = ((s["k"] >= 0) & (s["treated"] == 1)).astype(int)
    a = regress(s, "cost_pct_200", ["after"], ["unit_event", "quarter_event"], label=f"service stacked average | {kind}")
    res2 += [r, a]
r2 = pd.concat(res2)
r2.to_csv("outputs/tables/table6k_service_event_study.csv", index=False)
show(r2, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("  k-3 = 3+ years before ... k4 = 4+ years after; the year before (k-1) is the reference")
print("\nSaved outputs/tables/table6j_robust_did.csv and table6k_service_event_study.csv")
