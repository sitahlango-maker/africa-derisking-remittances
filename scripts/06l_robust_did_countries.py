"""
06l_robust_did_countries.py
Diagnostic for step 06k: which countries drive the robust grey-listing estimates?

For every country whose grey-list status changes in the data, at service level:
  - its own estimate for each year after the change (its services' price change since the year
    before the change, minus the same change for comparison countries; removals sign-flipped so
    every number reads as "the effect of being on the list");
  - how many of its services enter each comparison;
  - the overall year-1 and average (years 0-3) estimates when that country is left out.

If one or two countries account for the positive year-1 estimate, that is reported in the paper.

Needs: the panels from step 5 and data/processed/fatf_greylist_africa_yearly.csv
Output: outputs/tables/table6l_robust_did_countries.csv

Run from the repo folder:  python scripts/06l_robust_did_countries.py
"""

import os

import numpy as np
import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)
L_MAX = 3

svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
gy = pd.read_csv("data/processed/fatf_greylist_africa_yearly.csv")

a = (svc.groupby(["service_id", "iso3", "year"])["cost_pct_200"].mean().rename("Y").reset_index()
     .rename(columns={"service_id": "unit"}))
years = sorted(a["year"].unique())
countries = sorted(a["iso3"].unique())
g = gy.set_index(["iso3", "year"])["grey_31dec"]
first = {}
for c in countries:
    d = pd.Series([int(g.get((c, t), 0)) for t in years], index=years)
    ch = d.index[d.diff().fillna(0) != 0]
    first[c] = (int(ch[0]) if len(ch) else None, int(d.iloc[0]))
Y = a.set_index(["unit", "year"])["Y"]
unit_c = a.drop_duplicates("unit").set_index("unit")["iso3"]


def dy(units, base, target):
    yt = Y.reindex(pd.MultiIndex.from_product([units, [target]])).to_numpy()
    yb = Y.reindex(pd.MultiIndex.from_product([units, [base]])).to_numpy()
    d = yt - yb
    return d[~np.isnan(d)]


def by_country(drop=None):
    """Each switching country's own estimate for years 0..L_MAX (and its number of services)."""
    rows = []
    for c, (F, b) in first.items():
        if F is None or c == drop:
            continue
        for l in range(L_MAX + 1):
            base, target = F - 1, F + l
            if base not in years or target not in years:
                continue
            ctrl_c = [k for k, (Fk, bk) in first.items()
                      if bk == b and (Fk is None or Fk > target) and k not in (c, drop)]
            d_sw = dy(unit_c.index[unit_c == c], base, target)
            d_ct = dy(unit_c.index[unit_c.isin(ctrl_c)], base, target)
            if len(d_sw) == 0 or len(d_ct) == 0:
                continue
            sign = 1 if b == 0 else -1
            rows.append({"iso3": c, "change": "added" if b == 0 else "removed", "year_of_change": F,
                         "l": l, "effect": sign * (d_sw.mean() - d_ct.mean()), "services": len(d_sw)})
    return pd.DataFrame(rows)


def pooled(df):
    """Service-weighted average across countries, as in 06k."""
    out = {}
    for l in range(L_MAX + 1):
        d = df[df["l"] == l]
        out[l] = np.average(d["effect"], weights=d["services"]) if len(d) else np.nan
    out["avg"] = np.nanmean([out[l] for l in range(L_MAX + 1)])
    return out


full = by_country()
overall = pooled(full)
loo = []
for c in sorted(full["iso3"].unique()):
    p = pooled(by_country(drop=c))
    loo.append({"left_out": c, "year1_without": p[1], "avg_without": p["avg"]})
loo = pd.DataFrame(loo)

wide = full.pivot_table(index=["iso3", "change", "year_of_change"], columns="l", values="effect").round(2)
wide.columns = [f"effect_y{l}" for l in wide.columns]
n1 = full[full["l"] == 1].set_index("iso3")["services"].rename("services_y1")
share = (full[full["l"] == 1].assign(w=lambda d: d["effect"] * d["services"])
         .set_index("iso3")["w"] / full.loc[full["l"] == 1, "services"].sum()).rename("contribution_to_y1")
tab = (wide.reset_index().merge(n1, on="iso3", how="left").merge(share.round(2), on="iso3", how="left")
       .merge(loo.rename(columns={"left_out": "iso3"}).round(2), on="iso3", how="left")
       .sort_values("contribution_to_y1", ascending=False))
tab.to_csv("outputs/tables/table6l_robust_did_countries.csv", index=False)

print("=" * 100)
print("WHICH COUNTRIES DRIVE THE ROBUST GREY-LISTING ESTIMATES? (service level, total cost, points)")
print("=" * 100)
print(f"Overall: year 0 {overall[0]:+.2f} | year 1 {overall[1]:+.2f} | year 2 {overall[2]:+.2f} | "
      f"year 3 {overall[3]:+.2f} | average {overall['avg']:+.2f}   (should match 06k)")
print("\neffect_yL = the country's own estimate L years after its change (removals sign-flipped)")
print("contribution_to_y1 = its share of the overall year-1 estimate (sums to the overall year-1 number)")
print("year1_without / avg_without = overall estimates if that country is left out\n")
print(tab.to_string(index=False))
print(f"\nYear-1 estimate when each country is left out: from {loo['year1_without'].min():+.2f} "
      f"to {loo['year1_without'].max():+.2f}")
print("Saved outputs/tables/table6l_robust_did_countries.csv")
