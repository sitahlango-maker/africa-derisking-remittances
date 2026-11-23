"""
14_referee_checks.py
Checks requested by the second review (additional, not planned).

  1. CPMI validation: do countries that lost correspondent relationships (CPMI, change since 2011,
     to end-2018, 2019, 2020 and 2022) also lose deposits abroad (BIS) over the same years?
  2. Classification sensitivity: Table 3 price results with (a) agents of the global networks moved
     to "other operators", (b) mobile-money operators moved to "other operators".
  3. Counts: how many receiving countries change grey-list status behind each provider type's
     estimate, and how many corridors have banks and global networks side by side.

Run from the repo root, after scripts 12 and 13:  python scripts/14_referee_checks.py
Writes outputs/tables/table14_cpmi_validation.csv, table14_sensitivity.csv, table14_counts.csv
"""

import unicodedata
import numpy as np
import pandas as pd
from scipy import stats
from fe import regress, show

pd.set_option("display.width", 200)

# ---------------------------------------------------------------------------
# 1. CPMI validation
# ---------------------------------------------------------------------------
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())

ALIAS = {"capeverde": "caboverde", "swaziland": "eswatini", "congo": "republicofthecongo",
         "congodemrep": "democraticrepublicofthecongo", "drcongo": "democraticrepublicofthecongo",
         "congodemocraticrepublic": "democraticrepublicofthecongo", "gambia": "thegambia",
         "egyptarabrep": "egypt", "tanzaniaunitedrepublicof": "tanzania", "unitedrepublicoftanzania": "tanzania",
         "libyanarabjamahiriya": "libya", "saotomeandprincipe": "saotomeandprincipe"}

b = pd.read_csv("data/processed/bis_lbs_africa_yearend.csv")
b["key"] = b["country"].map(norm).replace(ALIAS)
bis_keys = set(b["key"])

VINTAGES = {"data-1905": 2018, "data-2008-2-0": 2019, "data-2112": 2020, "data-2305": 2022}
rows = []
for fname, yr in VINTAGES.items():
    t = pd.read_excel(f"data/raw/cpmi/{fname}.xlsx", sheet_name="T11")
    t["key"] = t["country"].map(norm).replace(ALIAS)
    t = t[t["key"].isin(bis_keys)]
    for _, r in t.iterrows():
        g = b[b["key"] == r["key"]].set_index("year")
        row = {"country": r["country"], "vintage": yr, "cpmi_cbr_change": r["CBR"]}
        for col in ["deposits_abroad_banks", "deposits_abroad_unrelated", "deposits_abroad_all"]:
            v0, v1 = g[col].get(2011, np.nan), g[col].get(yr, np.nan)
            row[col + "_change"] = 100 * (v1 / v0 - 1) if v0 and v0 > 0 else np.nan
        rows.append(row)
v = pd.DataFrame(rows)
print("African countries matched to CPMI, by vintage:")
print(v.groupby("vintage")["country"].nunique().to_string())
unmatched = sorted(set(b["country"]) - set(b[b["key"].isin(set(v["country"].map(norm).replace(ALIAS)))]["country"]))
print("BIS African countries not found in CPMI (check names):", unmatched, "\n")

out1 = []
for col in ["deposits_abroad_banks_change", "deposits_abroad_unrelated_change", "deposits_abroad_all_change"]:
    for yr in sorted(v["vintage"].unique()):
        s = v[v["vintage"] == yr].dropna(subset=[col, "cpmi_cbr_change"])
        rho, p = stats.spearmanr(s[col], s["cpmi_cbr_change"])
        same = np.mean(np.sign(s[col]) == np.sign(s["cpmi_cbr_change"]))
        out1.append({"measure": col, "vintage": yr, "countries": len(s), "spearman": rho, "p": p,
                     "same_direction_pct": 100 * same})
out1 = pd.DataFrame(out1).round(3)
out1.to_csv("outputs/tables/table14_cpmi_validation.csv", index=False)
print("Change since 2011: BIS deposits abroad against CPMI correspondent relationships (rank correlation)")
print(out1.to_string(index=False), "\n")

# Pooled: regress the CPMI change on the BIS change, with vintage effects (log changes)
vv = v.dropna(subset=["deposits_abroad_banks_change", "cpmi_cbr_change"]).copy()
vv["d_bis"] = np.log1p(vv["deposits_abroad_banks_change"] / 100)
vv["d_cpmi"] = np.log1p(vv["cpmi_cbr_change"] / 100)
X = pd.get_dummies(vv["vintage"], prefix="v", drop_first=True).astype(float)
X.insert(0, "d_bis", vv["d_bis"]); X.insert(0, "const", 1.0)
beta, *_ = np.linalg.lstsq(X.to_numpy(), vv["d_cpmi"].to_numpy(), rcond=None)
print(f"Pooled elasticity of correspondent relationships to deposits abroad (log changes, vintage effects): "
      f"{beta[1]:.3f} on {len(vv)} country-vintages\n")

# ---------------------------------------------------------------------------
# 2. Classification sensitivity
# ---------------------------------------------------------------------------
HALVE = np.log(0.5)
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
TYPES = ["bank", "global_mto", "digital", "other_mto"]
MOBILE = ["orange money", "wave", "mtn mobile money", "mtn", "econet wireless", "m-pesa"]

d = pd.read_csv("data/processed/panel_services.csv")
types = pd.read_csv("outputs/tables/table12_firm_types.csv")[["firm", "ptype"]]
d = d.merge(types, on="firm")
lower = d["firm"].str.strip().str.lower()
variants = {
    "main": d["ptype"],
    "agents_to_other": np.where(lower.str.contains(" via ") & (d["ptype"] == "global_mto"), "other_mto", d["ptype"]),
    "mobile_to_other": np.where(lower.isin(MOBILE), "other_mto", d["ptype"]),
}
res = []
for name, pt in variants.items():
    e = d.copy(); e["pt"] = pt; e["type_q"] = e["pt"] + "_" + e["period_q"]
    for k in TYPES:
        e[f"grey_x_{k}"] = e["grey_qend"] * (e["pt"] == k)
        e[f"dep_x_{k}"] = e["L1_log_dep"] * (e["pt"] == k)
    G = [f"grey_x_{k}" for k in TYPES]; D = [f"dep_x_{k}" for k in TYPES]
    for y in ["cost_pct_200", "fx_margin_pct"]:
        res.append(regress(e, y, G, ["service_id", "type_q"], label=f"{name} grey"))
        r = regress(e, y, D + CONTROLS, ["service_id", "type_q"], label=f"{name} deposits", scale={x: HALVE for x in D})
        res.append(r[~r["variable"].isin(CONTROLS)])
out2 = pd.concat(res, ignore_index=True)
out2.to_csv("outputs/tables/table14_sensitivity.csv", index=False)
print("Classification sensitivity (service + type-specific quarter fixed effects):")
show(out2)

# ---------------------------------------------------------------------------
# 3. Counts behind the estimates
# ---------------------------------------------------------------------------
c = []
sw = d.groupby("iso3")["grey_qend"].agg(lambda x: x.nunique() > 1)
for k in TYPES:
    s = d[d["ptype"] == k]
    c.append({"type": k, "receiving_countries": s["iso3"].nunique(),
              "countries_changing_status": int(sw.reindex(s["iso3"].unique()).sum()),
              "services": s["service_id"].nunique(), "quotes": len(s)})
cq = d.groupby(["corridor", "period_q"])["ptype"].agg(lambda x: {"bank", "global_mto"} <= set(x))
both = cq[cq].reset_index()
c.append({"type": "corridor-quarters with banks and global networks", "receiving_countries":
          d[d["corridor"].isin(both["corridor"])]["iso3"].nunique(), "countries_changing_status": None,
          "services": both["corridor"].nunique(), "quotes": len(both)})
out3 = pd.DataFrame(c)
out3.to_csv("outputs/tables/table14_counts.csv", index=False)
print("\nCounts (last row: 'services' = corridors, 'quotes' = corridor-quarters):")
print(out3.to_string(index=False))
