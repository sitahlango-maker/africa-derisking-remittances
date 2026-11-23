"""
13_business_models.py
Business-model analysis (additional, not planned), building on 12_firm_strategy.py.

  1. Same corridor, same quarter: do banks, digital-first and other operators respond differently
     from the global networks? Corridor x quarter fixed effects remove every shock shared by all
     providers in a corridor (exchange rate, demand, the listing itself), so only the gap between
     business models is left.
  2. Competition as a shock absorber: is the effect larger in corridors with fewer providers,
     more banks or fewer digital-first firms (measured a year earlier)?
  3. Speed and transparency: do services become slower or less transparent instead of dearer?

Run from the repo root, after 12_firm_strategy.py:  python scripts/13_business_models.py
Writes outputs/tables/table13_business_models.csv
"""

import numpy as np
import pandas as pd
from fe import regress, show

HALVE = np.log(0.5)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
TYPES = ["bank", "global_mto", "digital", "other_mto"]
res = []

d = pd.read_csv("data/processed/panel_services.csv")
types = pd.read_csv("outputs/tables/table12_firm_types.csv")[["firm", "ptype"]]
d = d.merge(types, on="firm")
d["t"] = d["year"] * 4 + d["quarter"]
d["type_q"] = d["ptype"] + "_" + d["period_q"]
d["corr_q"] = d["corridor"] + "_" + d["period_q"]

# ---------------------------------------------------------------------------
# 1. Same corridor, same quarter (reference: global networks)
# ---------------------------------------------------------------------------
for k in ["bank", "digital", "other_mto"]:
    d[f"grey_x_{k}"] = d["grey_qend"] * (d["ptype"] == k)
    d[f"dep_x_{k}"] = d["L1_log_dep"] * (d["ptype"] == k)
GX = [f"grey_x_{k}" for k in ["bank", "digital", "other_mto"]]
DX = [f"dep_x_{k}" for k in ["bank", "digital", "other_mto"]]
FE1 = ["service_id", "corr_q", "type_q"]
for y in ["cost_pct_200", "fee_pct", "fx_margin_pct"]:
    res.append(regress(d, y, GX, FE1, label="1 gap vs global networks, grey | service + corridor-quarter + type-quarter"))
    res.append(regress(d, y, DX, FE1, label="1 gap vs global networks, deposits | same FE", scale={v: HALVE for v in DX}))

# ---------------------------------------------------------------------------
# 2. Competition: corridor structure four quarters earlier
# ---------------------------------------------------------------------------
cq = d.groupby(["corridor", "t"])
s = pd.DataFrame({
    "n_firms": cq["firm"].nunique(),
    "bank_share": cq.apply(lambda g: 100 * (g["ptype"] == "bank").mean(), include_groups=False),
    "digital_share": cq.apply(lambda g: 100 * (g["ptype"] == "digital").mean(), include_groups=False),
    "hhi": cq.apply(lambda g: 10000 * ((g["firm"].value_counts(normalize=True)) ** 2).sum(), include_groups=False),
}).reset_index()
s["log_n_firms"] = np.log(s["n_firms"])
s["t"] = s["t"] + 4                                   # structure a year earlier, before any shock
d = d.merge(s.drop(columns="n_firms").add_prefix("L4_").rename(columns={"L4_corridor": "corridor", "L4_t": "t"}),
            on=["corridor", "t"], how="left")
print("Corridor structure a year earlier (means):")
print(d[["L4_log_n_firms", "L4_bank_share", "L4_digital_share", "L4_hhi"]].describe().round(2).to_string(), "\n")

FE2 = ["service_id", "type_q"]
for z, unit in [("L4_log_n_firms", "per 1 log point more firms"), ("L4_bank_share", "per 10 pts more bank services"),
                ("L4_digital_share", "per 10 pts more digital-first services"), ("L4_hhi", "per 1,000 HHI points")]:
    e = d.dropna(subset=[z]).copy()
    e["zc"] = e[z] - e[z].mean()
    mult = {"L4_log_n_firms": 1, "L4_bank_share": 10, "L4_digital_share": 10, "L4_hhi": 1000}[z]
    e["grey_x_z"] = e["grey_qend"] * e["zc"]
    e["dep_x_z"] = e["L1_log_dep"] * e["zc"]
    r = regress(e, "cost_pct_200", ["grey_qend", "grey_x_z", "zc"], FE2,
                label=f"2 competition, grey x {z} ({unit})", scale={"grey_x_z": mult, "zc": mult})
    res.append(r[r["variable"] != "zc"])
    r = regress(e, "cost_pct_200", ["L1_log_dep", "dep_x_z", "zc"] + CONTROLS, FE2,
                label=f"2 competition, deposits x {z} ({unit}; deposits halve)",
                scale={"L1_log_dep": HALVE, "dep_x_z": HALVE * mult, "zc": mult})
    res.append(r[r["variable"].isin(["L1_log_dep", "dep_x_z"])])

# ---------------------------------------------------------------------------
# 3. Speed and transparency (from the cleaned quote file)
# ---------------------------------------------------------------------------
q = pd.read_csv("data/processed/rpw_africa_services_clean.csv")
q = q[~q["outlier"].astype(str).str.lower().eq("true")].copy()
print("Speed categories in RPW:\n", q["speed"].value_counts(dropna=False).to_string(), "\n")
print("Transparency values:\n", q["transparent"].value_counts(dropna=False).to_string(), "\n")
sp = q["speed"].astype(str).str.lower()
q["within_hour"] = np.where(sp.str.contains("not available|nan"), np.nan,
                            100 * sp.str.contains("less than one hour|minute").astype(float))
q["two_days_plus"] = np.where(sp.str.contains("not available|nan"), np.nan,
                              100 * sp.str.contains(r"2 days|3-5 days|6 days").astype(float))
tr = q["transparent"].astype(str).str.lower()
q["transparent_pct"] = np.where(tr.isin(["yes", "no"]), 100 * (tr == "yes"), np.nan)
q["service_id2"] = (q["corridor"] + "|" + q["firm"] + "|" + q["payment_instrument"].fillna("?").astype(str)
                    + "|" + q["pickup_method"].fillna("?").astype(str))
q["period_q"] = q["year"].astype(str) + "Q" + q["quarter"].astype(str)
cvars = d[["destination_name", "year", "quarter", "grey_qend", "L1_log_dep"] + CONTROLS].drop_duplicates(
    ["destination_name", "year", "quarter"])
q = q.merge(cvars, on=["destination_name", "year", "quarter"], how="inner").merge(types, on="firm")
q["type_q"] = q["ptype"] + "_" + q["period_q"]
q["iso3"] = q["destination_code"]
for k in TYPES:
    q[f"grey_x_{k}"] = q["grey_qend"] * (q["ptype"] == k)
GT = [f"grey_x_{k}" for k in TYPES]
print("Quotes with a usable speed:", q["within_hour"].notna().sum(), "| transparency:", q["transparent_pct"].notna().sum(), "\n")
for y in ["within_hour", "two_days_plus", "transparent_pct"]:
    res.append(regress(q, y, GT, ["service_id2", "type_q"], label=f"3 {y} (% of quotes) | service + type-quarter"))

# ---------------------------------------------------------------------------
out = pd.concat(res, ignore_index=True)
out = out[~out["variable"].isin(CONTROLS)]
out.to_csv("outputs/tables/table13_business_models.csv", index=False)
pd.set_option("display.width", 220)
show(out)
print("\nNote: the +/-0.5 verdicts apply only to prices in percentage points (parts 1 and 2). "
      "For speed and transparency (part 3), outcomes are % of quotes; read the confidence intervals.")
