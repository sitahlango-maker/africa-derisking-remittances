"""
06d_sample_coverage.py
Additional check (NOT pre-specified; prompted by the 06c observation counts).

The question:
  The main model (Table 2, model C) uses 29 of the 38 receiving countries, because a row is
  dropped if any input is missing. Five of the nine dropped countries (Benin, Cote d'Ivoire,
  Mali, Senegal, Togo) are WAEMU members: their reserves are pooled at the regional central
  bank (BCEAO), so there is no country-level reserves figure. Does leaving them out change
  the result?

The check:
  1. List the countries dropped from the main model and the input each one is missing.
  2. Re-run the main model WITHOUT the reserves control, which brings the WAEMU countries back.

Needs: scripts/fe.py and data/processed/panel_corridor_quarter.csv
Output: outputs/tables/table6b_sample_coverage.csv

Run from the repo folder:  python scripts/06d_sample_coverage.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fe import regress, show   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

HALF = np.log(0.5)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs", "log_n_services"]
FE_C = ["corridor", "period_q"]
WAEMU = ["BEN", "BFA", "CIV", "GNB", "MLI", "NER", "SEN", "TGO"]   # BCEAO members

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")

# 1. Who is dropped from the main model, and why
need = ["L1_log_dep"] + CONTROLS
used = set(cq.dropna(subset=need)["iso3"])
print("=" * 100)
print("COUNTRIES IN THE PRICE DATA BUT NOT IN THE MAIN MODEL")
print("=" * 100)
for c in sorted(set(cq["iso3"]) - used):
    d = cq[cq["iso3"] == c]
    missing = [v for v in need if d[v].isna().all()]
    tag = " (WAEMU)" if c in WAEMU else ""
    print(f"  {c} {d['destination_name'].iloc[0]}{tag} | missing for the whole period: {missing or 'none (never all at once)'}")

# 2. Main model, with and without the reserves control
xs_main = ["grey_qend", "L1_log_dep"] + CONTROLS
xs_9 = ["grey_qend", "L1_log_dep"] + [c for c in CONTROLS if c != "L1_log_reserves"]
main_rows = cq.dropna(subset=need)            # exactly the rows the main model uses
rows = []
for label, data, xs in [("0. MAIN MODEL (Table 2, model C)", cq, xs_main),
                        ("9. Without reserves control", cq, xs_9),
                        ("9b. Without reserves control, main-model rows only", main_rows, xs_9)]:
    r = regress(data, "cost", xs, FE_C, label=label, scale={"L1_log_dep": HALF})
    rows.append(r[r["variable"].isin(["grey_qend", "L1_log_dep"])])
res = pd.concat(rows)
res["variable"] = res["variable"].map({"grey_qend": "grey-listed", "L1_log_dep": "deposits abroad halve"})
res.to_csv("outputs/tables/table6b_sample_coverage.csv", index=False)

print("\n" + "=" * 100)
print("MAIN MODEL WITH AND WITHOUT THE RESERVES CONTROL")
print("=" * 100)
show(res, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict", "n_obs", "n_clusters"))

# Which WAEMU countries does check 9 bring back?
back = sorted(set(cq.dropna(subset=["L1_log_dep"] + [c for c in CONTROLS if c != "L1_log_reserves"])["iso3"]) - used)
print(f"\nCountries brought back by dropping the reserves control: {', '.join(back) if back else 'none'}")

# 3. Check 9, leaving out one WAEMU country at a time.
#    The BCEAO (headquartered in Dakar) is listed by the BIS as an official monetary authority;
#    if its deposits abroad are recorded under Senegal, dropping Senegal should tighten the range.
loo = []
for c in back:
    r = regress(cq[cq["iso3"] != c], "cost", xs_9, FE_C, label=f"9c. Check 9 without {c}", scale={"L1_log_dep": HALF})
    loo.append(r[r["variable"] == "L1_log_dep"])
loo = pd.concat(loo)
loo["variable"] = "deposits abroad halve"
loo.to_csv("outputs/tables/table6c_waemu_leave_one_out.csv", index=False)
print("\n" + "=" * 100)
print("CHECK 9, LEAVING OUT ONE WAEMU COUNTRY AT A TIME (de-risking estimate only)")
print("=" * 100)
show(loo, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict", "n_obs", "n_clusters"))