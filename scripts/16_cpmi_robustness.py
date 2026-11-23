"""
16_cpmi_robustness.py
Robustness of the direct de-risking test (script 15). Additional analysis, not planned.

  1. Lag: prices one year AFTER the CPMI count (2012, 2019, 2020, 2021, 2023), so the loss of
     correspondents is measured before the prices it may affect.
  2. Service level with the country controls of Table 1, model C.
  3. Leave one country out: re-estimate the service-level model 38 times, dropping one country each time.
  4. Long difference: change in each corridor's average cost from 2011 to 2022 against the change in
     its country's correspondent relationships (one observation per corridor).

Run from the repo root, after script 15:  python scripts/16_cpmi_robustness.py
Writes outputs/tables/table16_cpmi_robustness.csv
"""

import numpy as np
import pandas as pd
from fe import regress, show

HALVE = np.log(0.5)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
SC = {"log_cbr": HALVE}

d = pd.read_csv("data/processed/panel_services.csv")
cb = pd.read_csv("data/processed/cpmi_cbr_africa.csv")
agg = {"cost_pct_200": "mean", "iso3": "first", "corridor": "first", **{c: "mean" for c in CONTROLS}}
res = []

# 1. Lag: CPMI at year Y, prices at Y+1
cl = cb.copy(); cl["year"] = cl["year"] + 1
lag_years = sorted(cl["year"].unique())
s = d[d["year"].isin(lag_years)]
sv = s.groupby(["service_id", "year"]).agg(agg).reset_index().merge(cl, on=["iso3", "year"])
cv = s.groupby(["corridor", "year"]).agg({k: v for k, v in agg.items() if k != "corridor"}).reset_index().merge(cl, on=["iso3", "year"])
res.append(regress(cv, "cost_pct_200", ["log_cbr"], ["corridor", "year"], label="1 lag | corridor", scale=SC))
res.append(regress(sv, "cost_pct_200", ["log_cbr"], ["service_id", "year"], label="1 lag | service", scale=SC))

# 2. Same years as script 15, service level with controls
YEARS = [2011, 2018, 2019, 2020, 2022]
s = d[d["year"].isin(YEARS)]
sv0 = s.groupby(["service_id", "year"]).agg(agg).reset_index().merge(cb, on=["iso3", "year"])
r = regress(sv0, "cost_pct_200", ["log_cbr"] + CONTROLS, ["service_id", "year"], label="2 service + controls", scale=SC)
res.append(r[r["variable"] == "log_cbr"])

# 3. Leave one country out (service level, no controls)
loo = []
for c in sorted(sv0["iso3"].unique()):
    r = regress(sv0[sv0["iso3"] != c], "cost_pct_200", ["log_cbr"], ["service_id", "year"], scale=SC)
    loo.append({"dropped": c, "coef": r["coef"].iloc[0], "lo": r["ci95_lo"].iloc[0], "hi": r["ci95_hi"].iloc[0], "p": r["p"].iloc[0]})
loo = pd.DataFrame(loo)
print("Leave one country out (service level): coefficient range "
      f"{loo['coef'].min():+.2f} to {loo['coef'].max():+.2f}; significant at 5% in {int((loo['p'] < 0.05).sum())} of {len(loo)} runs")
print("Most influential countries:\n", loo.reindex(loo["coef"].sub(loo["coef"].median()).abs().sort_values(ascending=False).index).head(5).round(2).to_string(index=False), "\n")

# 4. Long difference 2011 -> 2022, one observation per corridor
c11 = s[s["year"] == 2011].groupby("corridor").agg(cost11=("cost_pct_200", "mean"), iso3=("iso3", "first"))
c22 = s[s["year"] == 2022].groupby("corridor").agg(cost22=("cost_pct_200", "mean"))
ld = c11.join(c22, how="inner").reset_index().merge(cb[cb["year"] == 2022][["iso3", "log_cbr"]], on="iso3")
ld["d_cost"] = ld["cost22"] - ld["cost11"]
ld["one"] = 1
r = regress(ld, "d_cost", ["log_cbr"], ["one"], label="4 long difference 2011-2022 | corridor", scale=SC)
res.append(r)
print(f"Long difference: {len(ld)} corridors in {ld['iso3'].nunique()} countries\n")

out = pd.concat(res, ignore_index=True)
out.to_csv("outputs/tables/table16_cpmi_robustness.csv", index=False)
loo.to_csv("outputs/tables/table16_cpmi_leave_one_out.csv", index=False)
pd.set_option("display.width", 200)
show(out)
