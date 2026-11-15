"""
06e_services_count.py
Additional check (NOT pre-specified; prompted by Section 7.2).

The question:
  Prices did not move. But de-risking might work through ACCESS instead of price: a provider
  that loses its bank account leaves the corridor rather than raising its price. RPW only
  records services that are still offered, so an exit would be invisible in the price models.
  Does grey-listing (or a fall in deposits abroad) reduce the NUMBER of services quoted in a
  corridor, or change the MIX (the share of services offered by banks)?

The model:
  Same as the main model (Table 2, model C: corridor + quarter fixed effects, same controls),
  but the outcome is:
    1. log number of services in the corridor-quarter   -> reported as a % change
    2. share of services offered by banks (0 to 1)      -> reported in percentage points
  The number of services is dropped from the controls, since it is now the outcome.

Caution:
  The World Bank chooses which services to survey in each corridor, so the count reflects
  survey choices as well as the market. A change in the count is suggestive, not proof of exit.
  The +/-0.5 decision rule (Section 4.6) is in price points, so it does not apply here; read
  the estimate and its 95% confidence interval.

Needs: scripts/fe.py and data/processed/panel_corridor_quarter.csv
Output: outputs/tables/table9_services_count.csv

Run from the repo folder:  python scripts/06e_services_count.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fe import regress   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 200)

HALF = np.log(0.5)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]   # no log_n_services
FE_C = ["corridor", "period_q"]

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
xs = ["grey_qend", "L1_log_dep"] + CONTROLS

rows = []
for y, label in [("log_n_services", "Number of services (log)"), ("share_bank", "Share of services offered by banks")]:
    r = regress(cq, y, xs, FE_C, label=label, scale={"L1_log_dep": HALF})
    rows.append(r[r["variable"].isin(["grey_qend", "L1_log_dep"])])
res = pd.concat(rows)
res["variable"] = res["variable"].map({"grey_qend": "grey-listed", "L1_log_dep": "deposits abroad halve"})


def readable(row, col):
    """Logs -> % change; shares -> percentage points."""
    v = row[col]
    return (np.exp(v) - 1) * 100 if row["outcome"] == "log_n_services" else v * 100


for col in ("coef", "ci95_lo", "ci95_hi"):
    res[col + "_readable"] = res.apply(lambda r: readable(r, col), axis=1)
res.to_csv("outputs/tables/table9_services_count.csv", index=False)

print("=" * 110)
print("ADDITIONAL CHECK (not pre-specified): does grey-listing or de-risking change the number or mix of services?")
print("  Number of services: % change | Share of bank services: change in percentage points")
print("=" * 110)
for _, r in res.iterrows():
    unit = "%" if r["outcome"] == "log_n_services" else " pts"
    print(f"{r['model']:36s} {r['variable']:22s} {r['coef_readable']:+6.1f}{unit}  "
          f"(95% CI {r['ci95_lo_readable']:+.1f} to {r['ci95_hi_readable']:+.1f})  p = {r['p']:.3f}  "
          f"n = {int(r['n_obs']):,}, countries = {int(r['n_clusters'])}")
print(f"\nFor scale: the average corridor-quarter has {cq['n_services'].mean():.1f} services "
      f"(median {cq['n_services'].median():.0f}); banks offer {100 * cq['share_bank'].mean():.0f}% of services on average.")