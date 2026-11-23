"""
12_firm_strategy.py
Firm-level analysis requested by the referee (concern 1): how different kinds of provider respond
to grey-listing and to a fall in banks' deposits abroad. Additional analysis, not planned.

Three questions, all at service level (each service compared only with itself over time):
  A. Pass-through by provider type: does the total cost respond differently for banks,
     global money transfer operators, digital-first operators and other (regional/local) operators?
  B. Fee or margin: do providers move cost out of the visible fee and into the less visible
     exchange-rate margin ("shrouded" pricing, Gabaix and Laibson 2006)?
  C. Exit by provider type: which kinds of provider stop being quoted after a listing or a fall in deposits?

Uses the same estimator and decision rules as the paper (fe.py).
Run from the repo root:  python scripts/12_firm_strategy.py
Writes: outputs/tables/table12_firm_strategy.csv, table12_descriptives.csv, table12_firm_types.csv
"""

import numpy as np
import pandas as pd
from fe import regress, show

HALVE = np.log(0.5)                       # report deposits as "effect if deposits abroad halve"
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]

d = pd.read_csv("data/processed/panel_services.csv")
d["t"] = d["year"] * 4 + d["quarter"]     # running quarter number, for ordering

# ---------------------------------------------------------------------------
# 1. Classify each firm into one provider type (fixed over time)
# ---------------------------------------------------------------------------
GLOBAL_MTO = ["Western Union", "MoneyGram", "Ria"]   # the three global networks
# Firms are classified by business model, not by RPW's per-quote "digital" flag (which requires
# a digital payout too, so a WorldRemit cash pickup counts as non-digital).
DIGITAL_FIRST = ["wise", "transferwise", "worldremit", "remitly", "azimo", "xoom", "sendwave",
                 "taptap send", "xendpay", "exchange4free.com", "skrill", "equity direct", "flouss",
                 "rewire", "rocket remit", "transfergo", "tranzfers.com", "paysend", "chippercash",
                 "lebara money", "use remit", "eversend", "mhits remit"]          # online-only senders
MOBILE_MONEY = ["orange money", "wave", "mtn mobile money", "mtn", "econet wireless", "m-pesa"]
NOT_BANK = ["walmart2world"]                       # coded as a bank in RPW, but a retailer's service

firm = d.groupby("firm").agg(quotes=("cost_pct_200", "size"),
                             bank=("bank", "max"),
                             digital_share=("digital", "mean"))   # digital is recorded from 2016 Q2

def ptype(row):
    name = row.name.strip().lower()
    if any(name == g.lower() or name.startswith(g.lower() + " ") or f"via {g.lower()}" in name
           for g in GLOBAL_MTO):
        return "global_mto"                      # the networks and agents selling their service
    if name in DIGITAL_FIRST or name in MOBILE_MONEY or "via wise" in name:
        return "digital"                         # digital-first and mobile-money operators
    if row["bank"] == 1 and name not in NOT_BANK:
        return "bank"
    if row["digital_share"] >= 0.8:              # firms not named above: at least 80% digital quotes
        return "digital"
    return "other_mto"                           # regional and local operators, post offices

firm["ptype"] = firm.apply(ptype, axis=1)
firm.sort_values(["ptype", "quotes"], ascending=[True, False]).to_csv(
    "outputs/tables/table12_firm_types.csv")
d = d.merge(firm[["ptype"]], left_on="firm", right_index=True)

TYPES = ["bank", "global_mto", "digital", "other_mto"]
print("Provider types (check the firm lists in table12_firm_types.csv):")
print(d.groupby("ptype").agg(firms=("firm", "nunique"), services=("service_id", "nunique"),
                             quotes=("service_id", "size")).to_string(), "\n")

# ---------------------------------------------------------------------------
# 2. Variables
# ---------------------------------------------------------------------------
d["margin_share"] = np.where(d["cost_pct_200"] > 0, 100 * d["fx_margin_pct"] / d["cost_pct_200"], np.nan)
d["type_q"] = d["ptype"] + "_" + d["period_q"]          # separate quarter effects for each provider type
for k in TYPES:
    d[f"grey_x_{k}"] = d["grey_qend"] * (d["ptype"] == k)
    d[f"dep_x_{k}"] = d["L1_log_dep"] * (d["ptype"] == k)
GREY = [f"grey_x_{k}" for k in TYPES]
DEP = [f"dep_x_{k}" for k in TYPES]
SC = {v: HALVE for v in DEP}

# Descriptives: average cost, fee and margin share by type and year (for a figure)
desc = d.groupby(["ptype", "year"]).agg(quotes=("cost_pct_200", "size"),
                                        cost=("cost_pct_200", "mean"),
                                        fee=("fee_pct", "mean"),
                                        margin=("fx_margin_pct", "mean"),
                                        margin_share=("margin_share", "mean")).round(2)
desc.to_csv("outputs/tables/table12_descriptives.csv")

res = []

# ---------------------------------------------------------------------------
# A and B. Pass-through and fee/margin split, by provider type
#   F1: service + quarter fixed effects
#   F2: service + type-specific quarter fixed effects (removes trends that differ by type,
#       the lesson of the channel check in Section 5.5) - preferred
# ---------------------------------------------------------------------------
for fes, tag in [(["service_id", "period_q"], "F1 common quarters"),
                 (["service_id", "type_q"], "F2 type-specific quarters")]:
    for y in ["cost_pct_200", "fee_pct", "fx_margin_pct", "margin_share"]:
        res.append(regress(d, y, GREY, fes, label=f"A/B grey | {tag}"))
        res.append(regress(d, y, DEP + CONTROLS, fes, label=f"A/B deposits | {tag}", scale=SC))

# Differences between types (reference: global money transfer operators), preferred F2
for y in ["cost_pct_200", "fx_margin_pct"]:
    xs = ["grey_qend"] + [f"grey_x_{k}" for k in ["bank", "digital", "other_mto"]]
    res.append(regress(d, y, xs, ["service_id", "type_q"], label="A/B grey difference vs global MTO | F2"))

# ---------------------------------------------------------------------------
# C. Exit by provider type
#   A service "exits" in the last quarter it is quoted, if its corridor is still surveyed afterwards.
#   Linear probability model, in percentage points, corridor + type-specific quarter fixed effects.
# ---------------------------------------------------------------------------
last_service = d.groupby("service_id")["t"].transform("max")
last_corridor = d.groupby("corridor")["t"].transform("max")
d["exit"] = 100 * ((d["t"] == last_service) & (last_corridor > d["t"])).astype(float)
e = d[d["t"] < d["t"].max()]                       # the last quarter cannot show an exit
print("Average exit rate by type (% of services per quarter):")
print(e.groupby("ptype")["exit"].mean().round(2).to_string(), "\n")
res.append(regress(e, "exit", GREY, ["corridor", "type_q"], label="C exit grey | corridor + type quarters"))
res.append(regress(e, "exit", DEP + CONTROLS, ["corridor", "type_q"], label="C exit deposits | corridor + type quarters", scale=SC))

# ---------------------------------------------------------------------------
# 3. Save and print
# ---------------------------------------------------------------------------
out = pd.concat(res, ignore_index=True)
out = out[~out["variable"].isin(CONTROLS)]
out.to_csv("outputs/tables/table12_firm_strategy.csv", index=False)
pd.set_option("display.width", 200)
show(out)
print("\nNote: margin_share and exit are not in percentage points of the amount sent, so the "
      "+/-0.5 verdicts do not apply to them; read their confidence intervals instead.")
