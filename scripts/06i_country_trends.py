"""
06i_country_trends.py
Additional check (requested in review): do the headline results survive a separate price trend
for each receiving country?

Why: the event study shows prices in countries about to be grey-listed were already falling
before the listing. If such country-specific drifts continued, the main models could mistake
them for (the absence of) a grey-listing effect. Adding a straight-line time trend for each
receiving country lets every country's prices drift at their own pace; grey-listing and
de-risking are then judged only on departures from that country's own trend.

Models (same as the paper, plus one linear trend per receiving country):
  A   corridor level, grey-listing only
  C   corridor level, grey-listing + de-risking + controls
  S1  service level, grey-listing

Needs: scripts/fe.py, the panels from step 5
Output: outputs/tables/table6h_country_trends.csv

Run from the repo folder:  python scripts/06i_country_trends.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fe import regress, show   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
HALVING = {"L1_log_dep": np.log(0.5)}
FE_C, FE_S = ["corridor", "period_q"], ["service_id", "period_q"]

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)


def add_trends(df):
    """One column per receiving country: years since 2011 Q1 for that country's rows, 0 elsewhere."""
    d = df.copy()
    t = ((d["year"] - 2011) * 4 + d["quarter"] - 1) / 4
    names = []
    # one country is left out as the reference: together the trends would repeat the quarter effects
    for iso in sorted(d["iso3"].dropna().unique())[1:]:
        name = f"trend_{iso}"
        d[name] = np.where(d["iso3"] == iso, t, 0.0)
        names.append(name)
    return d, names


cq_t, tr_c = add_trends(cq)
svc_t, tr_s = add_trends(svc)
C_XS = ["grey_qend", "L1_log_dep"] + CONTROLS + ["log_n_services"]

res = pd.concat([
    regress(cq, "cost", ["grey_qend"], FE_C, label="A: grey only | paper"),
    regress(cq_t, "cost", ["grey_qend"] + tr_c, FE_C, label="A: grey only | + country trends"),
    regress(cq, "cost", C_XS, FE_C, label="C: + de-risking | paper", scale=HALVING),
    regress(cq_t, "cost", C_XS + tr_c, FE_C, label="C: + de-risking | + country trends", scale=HALVING),
    regress(svc, "cost_pct_200", ["grey_qend"], FE_S, label="S1: all services | paper"),
    regress(svc_t, "cost_pct_200", ["grey_qend"] + tr_s, FE_S, label="S1: all services | + country trends"),
])
res = res[res["variable"].isin(["grey_qend", "L1_log_dep"])]
res.to_csv("outputs/tables/table6h_country_trends.csv", index=False)

print("=" * 100)
print("COUNTRY-SPECIFIC LINEAR TRENDS: headline estimates with and without (total cost, percentage points)")
print("  'paper' rows should match Tables 2 and 3; L1_log_dep = if banks' deposits abroad HALVED")
print("=" * 100)
show(res, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print(f"\nCountry trends added: {len(tr_c)} (corridor level), {len(tr_s)} (service level)")
print("Saved outputs/tables/table6h_country_trends.csv")
