"""
06c_robustness.py
The pre-specified robustness checks (Section 4.7). Each check changes ONE thing about the
main model (Table 2, model C) and asks: does the conclusion still hold?

  Main model (for comparison): corridor + quarter fixed effects, grey-listed at quarter end,
  deposits abroad (BIS, previous quarter, reported as "deposits abroad halve"), and controls.

  1. Grey-listing measured on 31 December / as the share of the year listed
  2. De-risking measured differently: lagged four quarters; deposits with UNRELATED banks only;
     commercial banks' foreign assets (IMF) instead of the BIS measure
  3. ADAPTED: the plan said "exclude Libya", but Libya is not in the price data. Instead, exclude
     receiving countries whose BIS figure is more than twice their commercial banks' (IMF) figure,
     i.e. those most likely to be dominated by central bank deposits
  4. Excluding quotes with a negative exchange-rate margin
  5. Excluding the five receiving countries with the most negative exchange-rate margins
  6. Digital services only (service level)
  7. USD 500 instead of USD 200 (H6)

Needs: scripts/fe.py and the step 5 panels (re-run steps 1 and 5 first so the USD 500 cost is there).
Output: outputs/tables/table6_robustness.csv

Run from the repo folder:  python scripts/06c_robustness.py
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
P = "data/processed/"

cq = pd.read_csv(P + "panel_corridor_quarter.csv")
svc = pd.read_csv(P + "panel_services.csv", low_memory=False)
grey_y = pd.read_csv(P + "fatf_greylist_africa_yearly.csv")[["iso3", "year", "grey_31dec", "grey_share_of_year"]]
cq = cq.merge(grey_y, on=["iso3", "year"], how="left")
cq[["grey_31dec", "grey_share_of_year"]] = cq[["grey_31dec", "grey_share_of_year"]].fillna(0)


def main(data, grey="grey_qend", dep="L1_log_dep", y="cost", label=""):
    """Model C with one thing changed; returns the grey-listing and de-risking rows."""
    xs = [grey] + ([dep] if dep else []) + CONTROLS
    r = regress(data, y, xs, FE_C, label=label, scale={dep: HALF} if dep else None)
    return r[r["variable"].isin([grey, dep])]


def rebuild_corridor(s):
    """Average a (filtered) service panel back up to corridor x quarter, keeping country conditions."""
    keep = ["corridor", "iso3", "year", "quarter", "period_q", "grey_qend", "L1_log_dep",
            "L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
    g = s.groupby(keep, dropna=False).agg(cost=("cost_pct_200", "mean"), n=("cost_pct_200", "size")).reset_index()
    g["log_n_services"] = np.log(g["n"])
    return g


rows = [main(cq, label="0. MAIN MODEL (Table 2, model C)")]

# 1. Alternative grey-list measures
rows.append(main(cq, grey="grey_31dec", label="1a. Grey-listed on 31 December"))
rows.append(main(cq, grey="grey_share_of_year", label="1b. Share of year grey-listed"))

# 2. Alternative de-risking measures
rows.append(main(cq, dep="L4_log_dep", label="2a. Deposits lagged 4 quarters"))
rows.append(main(cq, dep="L1_log_dep_unrel", label="2b. Deposits with unrelated banks only"))
rows.append(main(cq, dep="L1_log_odc_assets", label="2c. IMF commercial-bank foreign assets"))

# 3. ADAPTED: exclude countries where BIS deposits are more than twice IMF commercial-bank assets
cy = pd.read_csv(P + "panel_country_year.csv")
ratio = (cy["deposits_abroad_banks"] / cy["odc_foreign_assets_usdmn"]).groupby(cy["iso3"]).median()
cb_dominated = sorted(ratio[ratio > 2].index)
rows.append(main(cq[~cq["iso3"].isin(cb_dominated)], label="3. Excl. countries with BIS > 2x IMF (adapted)"))

# 4. Excluding quotes with a negative FX margin
rows.append(main(rebuild_corridor(svc[svc["neg_fx_margin"] == 0]), label="4. Excl. negative-FX-margin quotes"))

# 5. Excluding the five receiving countries with the most negative FX margins
top5 = svc[svc["neg_fx_margin"] == 1]["iso3"].value_counts().head(5).index.tolist()
rows.append(main(cq[~cq["iso3"].isin(top5)], label="5. Excl. top-5 negative-FX countries"))

# 6. Digital services only (service level, service + quarter fixed effects)
dig = svc[svc["digital"] == 1]
r6 = regress(dig, "cost_pct_200", ["grey_qend", "L1_log_dep"], ["service_id", "period_q"],
             label="6. Digital services only (service level)", scale={"L1_log_dep": HALF})
rows.append(r6)

# 7. USD 500 instead of USD 200 (H6: effects should be smaller at USD 500)
rows.append(main(cq, y="cost500", label="7. USD 500 instead of USD 200"))

# 8. Outliers put back in (total cost <= 0% or > 50%, removed in step 5)
rpw = pd.read_csv(P + "rpw_africa_services_clean.csv", low_memory=False).rename(columns={"destination_code": "iso3"})
n_out = int(rpw["outlier"].sum())
cq8 = (rpw.groupby(["corridor", "iso3", "year", "quarter"])
       .agg(cost=("cost_pct_200", "mean"), n=("cost_pct_200", "size")).reset_index())
cq8["log_n_services"] = np.log(cq8["n"])
cq8["period_q"] = cq8["year"].astype(str) + "Q" + cq8["quarter"].astype(str)
country = cq[["iso3", "year", "quarter", "grey_qend", "L1_log_dep",       # country conditions,
              "L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]].drop_duplicates(["iso3", "year", "quarter"])
cq8 = cq8.merge(country, on=["iso3", "year", "quarter"], how="left")      # same for every corridor
rows.append(main(cq8, label=f"8. Outliers put back in ({n_out:,} quotes)"))
res = pd.concat(rows)
res["variable"] = res["variable"].map(lambda v: "grey-listed" if v.startswith("grey") else "deposits abroad halve")
res.to_csv("outputs/tables/table6_robustness.csv", index=False)

print("=" * 110)
print("ROBUSTNESS CHECKS (Section 4.7): does the main conclusion hold when one thing changes?")
print("  coef = change in total cost (points); 'deposits abroad halve' = effect of a 50% fall")
print("=" * 110)
show(res, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print(f"\nCheck 3 excluded (BIS > 2x IMF): {', '.join(cb_dominated) if cb_dominated else 'none'}")
print(f"Check 5 excluded (most negative FX margins): {', '.join(top5)}")
print("Libya is not among the receiving countries in the price data, so the planned 'exclude Libya' check does not apply.")
