"""
06f_services_robustness.py
Robustness of the services result (06e). Additional, NOT pre-specified.

06e found that when deposits abroad halve, corridors have about 7% fewer services.
Here we test it the same way the main price result was tested (Table 6): change ONE thing
at a time and see whether the result survives.

  0.  As in 06e (main model, outcome = log number of services)
  2a. Deposits lagged four quarters
  2b. Deposits with unrelated banks only
  2c. Commercial banks' foreign assets (IMF) instead of BIS
  3.  Excluding countries whose BIS figure is more than twice the IMF figure
  9.  Without the reserves control (adds the WAEMU countries)
  10. Distinct FIRMS instead of services (a firm dropping one payout option is not an exit)
  11. Placebo: deposits FOUR QUARTERS AHEAD added to the model. Future deposits cannot cause
      today's exits; if they "predict" them, the result may reflect a common trend or survey
      changes rather than de-risking. The lead should be close to zero.

Needs: scripts/fe.py and the step 5 panels
Output: outputs/tables/table9b_services_robustness.csv

Run from the repo folder:  python scripts/06f_services_robustness.py
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
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
FE_C = ["corridor", "period_q"]
P = "data/processed/"

cq = pd.read_csv(P + "panel_corridor_quarter.csv")
svc = pd.read_csv(P + "panel_services.csv", low_memory=False)

# Distinct firms per corridor-quarter (for check 10)
firms = svc.groupby(["corridor", "year", "quarter"])["firm"].nunique().rename("n_firms").reset_index()
cq = cq.merge(firms, on=["corridor", "year", "quarter"], how="left")
cq["log_n_firms"] = np.log(cq["n_firms"])

# Deposits four quarters ahead (for check 11): L1 at quarter t+5 = deposits at t+4
cc = cq[["iso3", "year", "quarter", "L1_log_dep"]].drop_duplicates(["iso3", "year", "quarter"]).copy()
cc["qnum"] = cc["year"] * 4 + cc["quarter"]
ahead = cc[["iso3", "qnum", "L1_log_dep"]].rename(columns={"L1_log_dep": "F4_log_dep"})
ahead["qnum"] -= 5
cc = cc.merge(ahead, on=["iso3", "qnum"], how="left")
cq = cq.merge(cc[["iso3", "year", "quarter", "F4_log_dep"]], on=["iso3", "year", "quarter"], how="left")

# Countries where BIS is more than twice IMF (as in 06c, check 3)
cy = pd.read_csv(P + "panel_country_year.csv")
ratio = (cy["deposits_abroad_banks"] / cy["odc_foreign_assets_usdmn"]).groupby(cy["iso3"]).median()
cb_dominated = sorted(ratio[ratio > 2].index)


def run(data, label, dep="L1_log_dep", y="log_n_services", controls=CONTROLS, extra=()):
    xs = ["grey_qend", dep] + list(extra) + controls
    scale = {dep: HALF, **{e: HALF for e in extra}}
    r = regress(data, y, xs, FE_C, label=label, scale=scale)
    return r[r["variable"].isin(["grey_qend", dep] + list(extra))]


no_res = [c for c in CONTROLS if c != "L1_log_reserves"]
rows = [
    run(cq, "0. As in 06e"),
    run(cq, "2a. Deposits lagged 4 quarters", dep="L4_log_dep"),
    run(cq, "2b. Deposits with unrelated banks only", dep="L1_log_dep_unrel"),
    run(cq, "2c. IMF commercial-bank foreign assets", dep="L1_log_odc_assets"),
    run(cq[~cq["iso3"].isin(cb_dominated)], "3. Excl. countries with BIS > 2x IMF"),
    run(cq, "9. Without reserves control (adds WAEMU)", controls=no_res),
    run(cq, "10. Distinct firms instead of services", y="log_n_firms"),
    run(cq, "11. Placebo: deposits 4 quarters ahead", extra=["F4_log_dep"]),
]
res = pd.concat(rows)
names = {"grey_qend": "grey-listed", "F4_log_dep": "deposits AHEAD halve (placebo)"}
res["variable"] = res["variable"].map(lambda v: names.get(v, "deposits abroad halve"))
for col in ("coef", "ci95_lo", "ci95_hi"):
    res[col + "_pct"] = (np.exp(res[col]) - 1) * 100
res.to_csv("outputs/tables/table9b_services_robustness.csv", index=False)

print("=" * 120)
print("SERVICES RESULT, ROBUSTNESS (additional, not pre-specified): % change in the number of services")
print("=" * 120)
for _, r in res.iterrows():
    star = " *" if r["p"] < 0.05 else ""
    print(f"{r['model']:42s} {r['variable']:32s} {r['coef_pct']:+6.1f}%  "
          f"(95% CI {r['ci95_lo_pct']:+6.1f} to {r['ci95_hi_pct']:+6.1f})  p = {r['p']:.3f}{star}  "
          f"n = {int(r['n_obs']):,}, countries = {int(r['n_clusters'])}")
print("\n* = statistically different from zero at the 5% level")
print(f"Check 3 excluded: {', '.join(cb_dominated)}")