"""
06m_listing_removal.py
Second-round review: separate the effect of LISTING from the effect of REMOVAL, and test whether
the effect really differs across countries.

Same design as 06k (annual data; each country's first change of grey-list status compared with
countries whose status has not yet changed), but:

  PART 1  Listings only, and removals only (removals now reported with their natural sign:
          a negative number = prices fell after removal). Corridor and service level.
          Uncertainty: leave-one-country-out jackknife over all receiving countries; t distribution
          with (number of switching countries - 1) degrees of freedom, which is conservative.

  PART 2  Do the effects really differ across countries? (service level, one year after listing)
          a) Cochran's Q test and I-squared, using each country's own standard error. These
             treat services within a country as independent, so they are GENEROUS to finding
             differences.
          b) A stricter test: country-specific shocks also make country estimates differ, even
             with identical effects. So we compare the spread of the country estimates AFTER the
             change with their spread BEFORE it (placebo, one year before). If the spread after
             is no larger than before, the "variation" is no more than ordinary country noise.
          c) 95% intervals for each country's own one-year estimate.

  PART 3  The 2013-2014 removals: who are they compared with?
          a) The comparison countries (those still listed) for each removal.
          b) Removals only, leaving Zimbabwe out of the comparison group.
          c) Removals compared instead with countries NOT on the list in the same years (as in the
             stacked event study). If this gives ~0 while (a) gives ~-2, the large falls come from
             the few still-listed comparison countries, not from the removals themselves.

Needs: the panels from step 5 and data/processed/fatf_greylist_africa_yearly.csv
Output: outputs/tables/table6m_listing_removal.csv, table6m_heterogeneity.csv

Run from the repo folder:  python scripts/06m_listing_removal.py   (a few minutes)
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

os.makedirs("outputs/tables", exist_ok=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)   # single-service cells give empty variances; they are skipped
pd.set_option("display.width", 220)
L_MAX, P_MAX = 3, 2
THRESH = 0.5

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
gy = pd.read_csv("data/processed/fatf_greylist_africa_yearly.csv")
NAMES = {"GHA": "Ghana", "NGA": "Nigeria", "ZAF": "South Africa", "UGA": "Uganda", "MAR": "Morocco",
         "KEN": "Kenya", "BWA": "Botswana", "TUN": "Tunisia", "SSD": "South Sudan", "TZA": "Tanzania",
         "CIV": "Cote d'Ivoire", "ETH": "Ethiopia", "CMR": "Cameroon", "SEN": "Senegal", "MLI": "Mali",
         "COD": "Congo, Dem. Rep.", "MOZ": "Mozambique", "ZWE": "Zimbabwe", "AGO": "Angola",
         "NAM": "Namibia", "DZA": "Algeria", "SDN": "Sudan", "STP": "Sao Tome and Principe",
         "BFA": "Burkina Faso", "MUS": "Mauritius"}


class Panel:
    def __init__(self, df, unit, y):
        a = df.groupby([unit, "iso3", "year"])[y].mean().rename("Y").reset_index().rename(columns={unit: "unit"})
        self.years = sorted(a["year"].unique())
        self.countries = sorted(a["iso3"].unique())
        g = gy.set_index(["iso3", "year"])["grey_31dec"]
        self.D = {c: pd.Series([int(g.get((c, t), 0)) for t in self.years], index=self.years) for c in self.countries}
        self.first = {}
        for c, d in self.D.items():
            ch = d.index[d.diff().fillna(0) != 0]
            self.first[c] = (int(ch[0]) if len(ch) else None, int(d.iloc[0]))
        self.Y = a.set_index(["unit", "year"])["Y"]
        self.unit_c = a.drop_duplicates("unit").set_index("unit")["iso3"]

    def dy(self, countries, base, target):
        units = self.unit_c.index[self.unit_c.isin(countries)]
        yt = self.Y.reindex(pd.MultiIndex.from_product([units, [target]])).to_numpy()
        yb = self.Y.reindex(pd.MultiIndex.from_product([units, [base]])).to_numpy()
        d = yt - yb
        return d[~np.isnan(d)]

    def controls(self, c, b, last, drop, mode):
        if mode == "not yet changed":      # de Chaisemartin-D'Haultfoeuille: same starting status, unchanged so far
            return [k for k, (Fk, bk) in self.first.items()
                    if bk == b and (Fk is None or Fk > last) and k != c and k not in drop]
        # "never listed in window": status 0 in every year from base to last

        return [k for k in self.countries if k != c and k not in drop
                and all(self.D[k].get(t, 0) == 0 for t in range(self.window[0], last + 1))]

    def cell(self, c, l, kind, drop=(), mode="not yet changed"):
        """Country c's own estimate (natural sign) at horizon l, its SE, and service counts."""
        F, b = self.first[c]
        base = F - 1
        target = F + l if kind == "effect" else F - 1 - l
        if base not in self.years or target not in self.years:
            return None
        last = max(base, target)
        self.window = (min(base, target), last)
        ctrl = self.controls(c, b, last, drop, mode)
        d_sw, d_ct = self.dy([c], base, target), self.dy(ctrl, base, target)
        if len(d_sw) == 0 or len(d_ct) == 0:
            return None
        est = d_sw.mean() - d_ct.mean()
        se = np.sqrt(d_sw.var(ddof=1) / len(d_sw) + d_ct.var(ddof=1) / len(d_ct)) if len(d_sw) > 1 else np.nan
        return {"iso3": c, "b": b, "F": F, "l": l, "kind": kind, "est": est, "se": se,
                "n_sw": len(d_sw), "controls": ctrl}

    def pooled(self, which, kind, l, drop=(), mode="not yet changed"):
        num = den = 0.0
        for c, (F, b) in self.first.items():
            if F is None or c in drop or (which == "listing" and b != 0) or (which == "removal" and b != 1):
                continue
            r = self.cell(c, l, kind, drop, mode)
            if r is None:
                continue
            num += r["est"] * r["n_sw"]
            den += r["n_sw"]
        return num / den if den else np.nan

    def estimate(self, which, label, drop=(), mode="not yet changed"):
        keys = [("effect", l) for l in range(L_MAX + 1)] + [("placebo", p) for p in range(1, P_MAX + 1)]

        def all_est(dr):
            out = {k: self.pooled(which, k[0], k[1], dr, mode) for k in keys}
            out[("average", "0-3")] = np.nanmean([out[("effect", l)] for l in range(L_MAX + 1)])
            return out
        full = all_est(tuple(drop))
        reps = [all_est(tuple(drop) + (c,)) for c in self.countries if c not in drop]
        n_sw = sum(1 for c, (F, b) in self.first.items() if F is not None and c not in drop
                   and ((which == "listing" and b == 0) or (which == "removal" and b == 1)))
        dof = max(n_sw - 1, 1)
        rows = []
        for k, est in full.items():
            r = np.array([rp[k] for rp in reps], dtype=float)
            r = r[~np.isnan(r)]
            se = np.sqrt((len(r) - 1) / len(r) * ((r - r.mean()) ** 2).sum()) if len(r) > 1 else np.nan
            t95 = stats.t.ppf(0.975, dof)
            p = 2 * stats.t.sf(abs(est / se), dof) if se and se > 0 else np.nan
            tost = max(stats.t.sf((est + THRESH) / se, dof), stats.t.cdf((est - THRESH) / se, dof)) if se and se > 0 else np.nan
            rows.append({"model": label, "what": k[0], "years": k[1], "coef": est, "se": se, "p": p,
                         "ci95_lo": est - t95 * se, "ci95_hi": est + t95 * se, "tost_p_0.5": tost,
                         "switching_countries": n_sw, "dof": dof})
        return pd.DataFrame(rows)


def show(df):
    d = df.copy()
    for c in ("coef", "ci95_lo", "ci95_hi"):
        d[c] = d[c].map(lambda v: f"{v:+.2f}")
    d["p"] = d["p"].map(lambda v: f"{v:.3f}")
    d["tost_p_0.5"] = d["tost_p_0.5"].map(lambda v: f"{v:.3f}")
    d["reading"] = [("significant" if float(p) < 0.05 else "not significant") + ("; within +/-0.5" if float(t) < 0.05 else "")
                    for p, t in zip(d["p"], d["tost_p_0.5"])]
    print(d[["model", "what", "years", "coef", "ci95_lo", "ci95_hi", "p", "tost_p_0.5", "reading", "switching_countries"]]
          .to_string(index=False))


S = Panel(svc, "service_id", "cost_pct_200")
C = Panel(cq, "corridor", "cost")

# =============================================================================
print("=" * 100)
print("PART 1. LISTINGS ONLY AND REMOVALS ONLY (removals with natural sign: negative = prices fell)")
print("=" * 100)
p1 = pd.concat([
    S.estimate("listing", "service | listings only"),
    S.estimate("removal", "service | removals only"),
    C.estimate("listing", "corridor | listings only"),
    C.estimate("removal", "corridor | removals only"),
])
show(p1)
for P, lev in ((S, "service"), (C, "corridor")):
    rem = [c for c, (F, b) in P.first.items() if F is not None and b == 1]
    used = [c for c in rem if P.cell(c, 0, "effect") is not None]
    print(f"  {lev}: removals in the data {len(rem)}; with an estimable comparison {len(used)}; "
          f"not estimable: {', '.join(NAMES.get(c, c) for c in rem if c not in used) or 'none'}")

# =============================================================================
print("\n" + "=" * 100)
print("PART 2. DO EFFECTS REALLY DIFFER ACROSS COUNTRIES? (service level, listings)")
print("=" * 100)
het_rows = []
for which, bval in (("listings", 0), ("all changes, removals sign-reversed", None)):
    post, pre = [], []
    for c, (F, b) in S.first.items():
        if F is None or (bval is not None and b != bval):
            continue
        sign = 1 if b == 0 else -1
        r1 = S.cell(c, 1, "effect")
        r0 = S.cell(c, 1, "placebo")
        if r1 is not None and r1["n_sw"] > 1:
            post.append((c, sign * r1["est"], r1["se"], r1["n_sw"]))
        if r0 is not None and r0["n_sw"] > 1:
            pre.append((c, sign * r0["est"], r0["se"], r0["n_sw"]))
    est = np.array([x[1] for x in post]); se = np.array([x[2] for x in post])
    w = 1 / se ** 2
    mu = (w * est).sum() / w.sum()
    Q = (w * (est - mu) ** 2).sum(); k = len(est)
    pQ = stats.chi2.sf(Q, k - 1); I2 = max(0, (Q - (k - 1)) / Q) if Q > 0 else 0
    sd_post = est.std(ddof=1)
    pre_est = np.array([x[1] for x in pre]); sd_pre = pre_est.std(ddof=1)
    F_ratio = sd_post ** 2 / sd_pre ** 2
    pF = stats.f.sf(F_ratio, k - 1, len(pre_est) - 1)
    lev = stats.levene(est, pre_est)
    print(f"\n--- {which}: {k} countries with a one-year estimate ---")
    print(f"  a) Cochran's Q = {Q:.1f} on {k - 1} df, p = {pQ:.3f}; I-squared = {100 * I2:.0f}%  "
          f"(generous to finding differences: ignores country-specific shocks)")
    print(f"  b) Spread of country estimates AFTER (one year) = {sd_post:.2f} points; BEFORE (placebo) = {sd_pre:.2f} "
          f"points ({len(pre_est)} countries).")
    print(f"     Variance ratio after/before = {F_ratio:.2f}, one-sided p = {pF:.3f}; Levene test p = {lev.pvalue:.3f}")
    print("     If the spread after is not clearly larger than before, the variation is no more than country noise.")
    het_rows.append({"group": which, "countries": k, "Q": Q, "Q_p": pQ, "I2": I2, "sd_after": sd_post,
                     "sd_before": sd_pre, "var_ratio": F_ratio, "var_ratio_p": pF, "levene_p": lev.pvalue})
    if bval == 0:
        print("  c) Each listed country's own one-year estimate (95% interval treats its services as independent,")
        print("     so it is too narrow if country-specific shocks matter):")
        for c, e, s, n in sorted(post, key=lambda x: -x[1]):
            print(f"     {NAMES.get(c, c):<18} {e:+.2f}  ({e - 1.96 * s:+.2f} to {e + 1.96 * s:+.2f}); services {n}")
pd.DataFrame(het_rows).to_csv("outputs/tables/table6m_heterogeneity.csv", index=False)

# =============================================================================
print("\n" + "=" * 100)
print("PART 3. THE 2013-2014 REMOVALS: WHO ARE THEY COMPARED WITH? (service level)")
print("=" * 100)
for c, (F, b) in sorted(S.first.items(), key=lambda x: (x[1][0] or 0)):
    if F is None or b != 1:
        continue
    r = S.cell(c, 1, "effect")
    if r is None:
        print(f"  {NAMES.get(c, c):<12} removed {F}: no comparison country still listed one year later")
        continue
    print(f"  {NAMES.get(c, c):<12} removed {F}: one-year change vs still-listed countries {r['est']:+.2f}; "
          f"compared with: {', '.join(NAMES.get(k, k) for k in r['controls'])}")
p3 = pd.concat([
    S.estimate("removal", "service | removals, Zimbabwe left out", drop=("ZWE",)),
    S.estimate("removal", "service | removals vs countries NOT listed", mode="never listed in window"),
])
show(p3)
pd.concat([p1, p3]).to_csv("outputs/tables/table6m_listing_removal.csv", index=False)
print("\nSaved outputs/tables/table6m_listing_removal.csv and table6m_heterogeneity.csv")
