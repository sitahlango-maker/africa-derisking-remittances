"""
15_cpmi_prices.py
De-risking measured directly: the number of correspondent banking relationships of each African
country (CPMI, from payment messages), relative to 2011, at end-2018, 2019, 2020 and 2022.
Additional analysis, not planned, prompted by the failed validation of the BIS measure (script 14).

Design: each corridor (and each service) is compared only with itself across the years
2011, 2018, 2019, 2020 and 2022, with year effects; standard errors clustered by receiving country.
Effect reported as the change in price if a country's correspondent relationships fell by half.

Run from the repo root:  python scripts/15_cpmi_prices.py
Writes outputs/tables/table15_cpmi_prices.csv and data/processed/cpmi_cbr_africa.csv
"""

import unicodedata
import numpy as np
import pandas as pd
from fe import regress, show

HALVE = np.log(0.5)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())

ALIAS = {"capeverde": "caboverde", "swaziland": "eswatini", "gambia": "thegambia", "thegambia": "thegambia",
         "egyptarabrep": "egypt", "unitedrepublicoftanzania": "tanzania", "tanzaniaunitedrepublicof": "tanzania",
         "libyanarabjamahiriya": "libya", "gambiathe": "thegambia",
         "congodemrep": "drcongo", "repdemocratiqueducongo": "drcongo",
         "somaliafederalrepublicof": "somalia"}

d = pd.read_csv("data/processed/panel_services.csv")
types = pd.read_csv("outputs/tables/table12_firm_types.csv")[["firm", "ptype"]]
d = d.merge(types, on="firm")
names = d[["iso3", "destination_name"]].drop_duplicates()
names["key"] = names["destination_name"].map(norm).replace(ALIAS)

# ---------------------------------------------------------------------------
# 1. CPMI correspondent relationships, index 2011 = 100
# ---------------------------------------------------------------------------
VINTAGES = {"data-1905": 2018, "data-2008-2-0": 2019, "data-2112": 2020, "data-2305": 2022}
allnames = set()
rows = []
for fname, yr in VINTAGES.items():
    t = pd.read_excel(f"data/raw/cpmi/{fname}.xlsx", sheet_name="T11")
    allnames |= set(t["country"])
    t["key"] = t["country"].map(norm).replace(ALIAS)
    for _, r in t.merge(names, on="key").iterrows():
        rows.append({"iso3": r["iso3"], "year": yr, "cbr_index": 100 * (1 + r["CBR"] / 100)})
cb = pd.DataFrame(rows)
cb = pd.concat([cb, pd.DataFrame({"iso3": cb["iso3"].unique(), "year": 2011, "cbr_index": 100.0})])
cb["log_cbr"] = np.log(cb["cbr_index"])
cb.to_csv("data/processed/cpmi_cbr_africa.csv", index=False)

missing = sorted(set(names["destination_name"]) - set(names[names["iso3"].isin(cb["iso3"])]["destination_name"]))
print("Receiving countries in the price data without CPMI match:", missing)
print("CPMI names that may be them:", sorted(n for n in allnames if any(k in norm(n) for k in
      ["congo", "zaire", "somal", "tome", "gambia", "ivoire", "tanzan", "egypt", "swazi", "eswat", "libya", "sudan"])), "\n")
print("Correspondent relationships, index 2011 = 100 (median across matched countries):")
print(cb.groupby("year")["cbr_index"].median().round(1).to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. Prices in the same years
# ---------------------------------------------------------------------------
YEARS = [2011, 2018, 2019, 2020, 2022]
s = d[d["year"].isin(YEARS)].copy()
agg = {"cost_pct_200": "mean", "fee_pct": "mean", "fx_margin_pct": "mean", "L1_log_dep": "mean",
       "iso3": "first", "ptype": "first", "corridor": "first", **{c: "mean" for c in CONTROLS}}
sv = s.groupby(["service_id", "year"]).agg(agg).reset_index().merge(cb, on=["iso3", "year"])
cv = s.groupby(["corridor", "year"]).agg({k: v for k, v in agg.items() if k not in ("ptype", "corridor")}).reset_index().merge(cb, on=["iso3", "year"])
sv["bank"] = (sv["ptype"] == "bank").astype(float)
sv["cbr_x_bank"] = sv["log_cbr"] * sv["bank"]
sv["cbr_x_nonbank"] = sv["log_cbr"] * (1 - sv["bank"])
sv["type_year"] = sv["ptype"] + "_" + sv["year"].astype(str)
print(f"Corridor-years: {len(cv)} ({cv['iso3'].nunique()} countries); service-years: {len(sv)}\n")

res = []
SC = {"log_cbr": HALVE, "cbr_x_bank": HALVE, "cbr_x_nonbank": HALVE, "L1_log_dep": HALVE}
for y in ["cost_pct_200", "fee_pct", "fx_margin_pct"]:
    res.append(regress(cv, y, ["log_cbr"], ["corridor", "year"], label="corridor | relationships halve", scale=SC))
    res.append(regress(cv, y, ["log_cbr"] + CONTROLS, ["corridor", "year"], label="corridor + controls | relationships halve", scale=SC))
    res.append(regress(sv, y, ["log_cbr"], ["service_id", "year"], label="service | relationships halve", scale=SC))
res.append(regress(sv, "cost_pct_200", ["cbr_x_bank", "cbr_x_nonbank"], ["service_id", "type_year"],
                   label="service, banks vs non-banks | relationships halve", scale=SC))
# the BIS measure on the same years, for comparison
res.append(regress(cv, "cost_pct_200", ["L1_log_dep"] + CONTROLS, ["corridor", "year"], label="corridor + controls | BIS deposits halve (same years)", scale=SC))

out = pd.concat(res, ignore_index=True)
out = out[~out["variable"].isin(CONTROLS)]
out.to_csv("outputs/tables/table15_cpmi_prices.csv", index=False)
pd.set_option("display.width", 200)
show(out)
