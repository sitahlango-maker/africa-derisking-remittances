"""
06n_final_checks.py
Third-round review: the remaining tests, in one script.

  PART 1  Do listing effects differ across countries? Redone at MATCHED horizons.
          (06m compared a two-year change with a one-year placebo, which is tilted towards finding
          differences.) Here like is compared with like:
            year of listing  (change over one year)  vs  placebo one year before (one year)
            one year after   (change over two years) vs  placebo two years before (two years)
          with an F test of equal variances and the more robust Levene and Brown-Forsythe tests.

  PART 2  Removals with correct degrees of freedom: t distribution with (number of ESTIMABLE
          removals - 1) degrees of freedom, using the standard errors saved by 06m.

  PART 3  H6: is the effect on a USD 200 transfer larger than on USD 500? Direct test: the same
          corridor model with (cost of USD 200 - cost of USD 500) as the outcome.

  PART 4  H3 and the mobile money interaction:
          a) grey-listing and de-risking interacted with competition (log number of services, centred);
          b) de-risking interacted with mobile money ownership (per 10 points and per standard deviation);
          c) the grey-listing x mobile money interaction (Table 4, M1) also per standard deviation.

  PART 5  Break-even analysis for unrecorded charges (2024 corridor-payout pairs, as in step 9):
          how large must charges not recorded by RPW be before the share at or below 5% halves?

Needs: scripts/fe.py, scripts/06m_listing_removal.py, outputs/tables/table6m_listing_removal.csv,
       the panels from step 5 and data/processed/rpw_africa_services_clean.csv
Output: outputs/tables/table6n_final_checks.csv, table6n_breakeven.csv

Run from the repo folder:  python scripts/06n_final_checks.py   (a few minutes)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "scripts")
from fe import regress   # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

# reuse the Panel class from 06m without re-running 06m
src = open("scripts/06m_listing_removal.py", encoding="utf-8").read()
exec(src.split("S = Panel(")[0])          # defines Panel, NAMES, data (cq, svc, gy)
from fe import show as fe_show   # noqa: E402  (06m defines its own show)

CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
HALF = np.log(0.5)
FE_C = ["corridor", "period_q"]
rows = []

# =============================================================================
print("=" * 100)
print("PART 1. DO LISTING EFFECTS DIFFER ACROSS COUNTRIES? MATCHED HORIZONS (service level)")
print("=" * 100)
S = Panel(svc, "service_id", "cost_pct_200")
listed = [c for c, (F, b) in S.first.items() if F is not None and b == 0]


def country_estimates(kind, l):
    out = {}
    for c in listed:
        r = S.cell(c, l, kind)
        if r is not None and r["n_sw"] > 1:
            out[c] = r["est"]
    return out


for (k_a, l_a, lab_a), (k_b, l_b, lab_b), span in [
        (("effect", 0, "year of listing"), ("placebo", 1, "one year before"), "one-year changes"),
        (("effect", 1, "one year after"), ("placebo", 2, "two years before"), "two-year changes")]:
    a, b = country_estimates(k_a, l_a), country_estimates(k_b, l_b)
    xa, xb = np.array(list(a.values())), np.array(list(b.values()))
    F_ratio = xa.var(ddof=1) / xb.var(ddof=1)
    pF = stats.f.sf(F_ratio, len(xa) - 1, len(xb) - 1)
    lev = stats.levene(xa, xb, center="mean").pvalue
    bf = stats.levene(xa, xb, center="median").pvalue
    print(f"\n--- {span}: {lab_a} ({len(xa)} countries) vs {lab_b} ({len(xb)} countries) ---")
    print(f"  standard deviation: {xa.std(ddof=1):.2f} vs {xb.std(ddof=1):.2f}; variance ratio {F_ratio:.2f}")
    print(f"  F test (one-sided) p = {pF:.3f} | Levene p = {lev:.3f} | Brown-Forsythe p = {bf:.3f}")
    rows.append(pd.DataFrame([{"part": "1 heterogeneity", "model": f"{span}: {lab_a} vs {lab_b}", "variable": "variance ratio",
                 "coef": F_ratio, "p": pF, "levene_p": lev, "brown_forsythe_p": bf,
                 "sd_after": xa.std(ddof=1), "sd_before": xb.std(ddof=1), "n_after": len(xa), "n_before": len(xb)}]))
print("\n  Reading: differences across countries are supported only if the spread after listing is clearly")
print("  larger than the spread before it over the SAME length of change, in all three tests.")

# =============================================================================
print("\n" + "=" * 100)
print("PART 2. REMOVALS WITH CORRECT DEGREES OF FREEDOM (estimable removals - 1)")
print("=" * 100)
t6m = pd.read_csv("outputs/tables/table6m_listing_removal.csv")
C = Panel(cq, "corridor", "cost")


def n_estimable(P, drop=(), mode="not yet changed"):
    n = 0
    for c, (F, b) in P.first.items():
        if F is None or b != 1 or c in drop:
            continue
        if any(P.cell(c, l, "effect", drop, mode) is not None for l in range(L_MAX + 1)):
            n += 1
    return n


spec = {"service | removals only": (S, (), "not yet changed"),
        "corridor | removals only": (C, (), "not yet changed"),
        "service | removals, Zimbabwe left out": (S, ("ZWE",), "not yet changed"),
        "service | removals vs countries NOT listed": (S, (), "never listed in window")}
fix = []
for model, (P, drop, mode) in spec.items():
    n = n_estimable(P, drop, mode)
    dof = max(n - 1, 1)
    d = t6m[t6m["model"] == model].copy()
    t95 = stats.t.ppf(0.975, dof)
    d["dof"] = dof
    d["ci95_lo"] = d["coef"] - t95 * d["se"]
    d["ci95_hi"] = d["coef"] + t95 * d["se"]
    d["p"] = 2 * stats.t.sf(np.abs(d["coef"] / d["se"]), dof)
    d["tost_p_0.5"] = np.maximum(stats.t.sf((d["coef"] + 0.5) / d["se"], dof), stats.t.cdf((d["coef"] - 0.5) / d["se"], dof))
    d["estimable_removals"] = n
    fix.append(d)
fix = pd.concat(fix)
out = fix[["model", "what", "years", "coef", "ci95_lo", "ci95_hi", "p", "tost_p_0.5", "estimable_removals", "dof"]].copy()
for c in ("coef", "ci95_lo", "ci95_hi"):
    out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
for c in ("p", "tost_p_0.5"):
    out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
print(out.to_string(index=False))
fix.insert(0, "part", "2 removals dof")
rows.append(fix)

# =============================================================================
print("\n" + "=" * 100)
print("PART 3. H6: IS THE EFFECT LARGER FOR USD 200 THAN FOR USD 500? (corridor level, model C)")
print("=" * 100)
c6 = cq.copy()
c6["diff_200_500"] = c6["cost"] - c6["cost500"]
C_XS = ["grey_qend", "L1_log_dep"] + CONTROLS + ["log_n_services"]
h6 = regress(c6, "diff_200_500", C_XS, FE_C, label="H6: cost(200) - cost(500)", scale={"L1_log_dep": HALF})
h6 = h6[h6["variable"].isin(["grey_qend", "L1_log_dep"])]
fe_show(h6, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("  A positive estimate would mean a larger effect on USD 200 than on USD 500 (what H6 predicts).")
h6.insert(0, "part", "3 H6")
rows.append(h6)

# =============================================================================
print("\n" + "=" * 100)
print("PART 4. H3 (COMPETITION) AND MOBILE MONEY INTERACTIONS (corridor level)")
print("=" * 100)
c4 = cq.dropna(subset=["log_n_services"]).copy()
c4["ns_c"] = c4["log_n_services"] - c4["log_n_services"].mean()
c4["grey_x_comp"] = c4["grey_qend"] * c4["ns_c"]
c4["dep_x_comp"] = c4["L1_log_dep"] * c4["ns_c"]
h3 = regress(c4, "cost", ["grey_qend", "L1_log_dep", "grey_x_comp", "dep_x_comp"] + CONTROLS + ["log_n_services"], FE_C,
             label="H3: x competition (log services, centred)",
             scale={"L1_log_dep": HALF, "dep_x_comp": HALF})
h3 = h3[h3["variable"].isin(["grey_qend", "L1_log_dep", "grey_x_comp", "dep_x_comp"])]
fe_show(h3, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("  grey_x_comp / dep_x_comp = how the effect changes when the corridor has 1 log point (about 2.7 times) more services;")
print("  H3 predicts NEGATIVE values (more competition weakens the effect).")

mm = cq.dropna(subset=["L1_mobile_money_pct"]).copy()
sd_mm = mm["L1_mobile_money_pct"].std()
res_mm = []
for unit, div in (("per 10 points", 10.0), (f"per standard deviation ({sd_mm:.1f} points)", sd_mm)):
    mm["mm_c"] = (mm["L1_mobile_money_pct"] - mm["L1_mobile_money_pct"].mean()) / div
    mm["grey_x_mm"] = mm["grey_qend"] * mm["mm_c"]
    mm["dep_x_mm"] = mm["L1_log_dep"] * mm["mm_c"]
    r = regress(mm, "cost", ["grey_qend", "L1_log_dep", "grey_x_mm", "dep_x_mm"] + CONTROLS + ["log_n_services"], FE_C,
                label=f"M2: x mobile money, {unit}", scale={"L1_log_dep": HALF, "dep_x_mm": HALF})
    res_mm.append(r[r["variable"].isin(["grey_x_mm", "dep_x_mm"])])
res_mm = pd.concat(res_mm)
fe_show(res_mm, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("  grey_x_mm / dep_x_mm = how the effect changes with mobile money ownership (H5/H3 predict NEGATIVE values).")
for d, part in ((h3, "4 H3"), (res_mm, "4 mobile money")):
    d.insert(0, "part", part)
    rows.append(d)

# =============================================================================
print("\n" + "=" * 100)
print("PART 5. BREAK-EVEN: HOW LARGE MUST UNRECORDED CHARGES BE? (2024 corridor-payout pairs)")
print("=" * 100)
rpw = pd.read_csv("data/processed/rpw_africa_services_clean.csv", low_memory=False)
PAYOUT = {"Cash": "Cash pickup", "Bank account": "Bank account",
          "Own/partner bank account": "Bank account", "Mobile wallet": "Mobile wallet"}
r24 = rpw[(~rpw["outlier"]) & (rpw["year"] == 2024)].copy()
r24["payout"] = r24["pickup_method"].map(PAYOUT)
cells = (r24.dropna(subset=["payout"]).groupby(["destination_code", "corridor", "payout"])["cost_pct_200"]
         .mean().rename("quoted").reset_index())
bank = cells["payout"] == "Bank account"
grid = np.round(np.arange(0, 5.21, 0.1), 2)
be = []
for x in grid:
    be.append({"added_points": x,
               "all_pairs_le5_if_added_to_all": 100 * ((cells["quoted"] + x) <= 5).mean(),
               "bank_pairs_le5_if_added_to_bank": 100 * ((cells.loc[bank, "quoted"] + x) <= 5).mean()})
be = pd.DataFrame(be)
be.to_csv("outputs/tables/table6n_breakeven.csv", index=False)
base_all, base_bank = be.iloc[0, 1], be.iloc[0, 2]
half_all = be.loc[be["all_pairs_le5_if_added_to_all"] <= base_all / 2, "added_points"].min()
half_bank = be.loc[be["bank_pairs_le5_if_added_to_bank"] <= base_bank / 2, "added_points"].min()
zero_bank = be.loc[be["bank_pairs_le5_if_added_to_bank"] == 0, "added_points"].min()
print(f"Pairs: {len(cells)} (bank-account payouts: {bank.sum()}). As quoted: {base_all:.1f}% of all pairs and "
      f"{base_bank:.1f}% of bank-account pairs cost 5% or less.")
print(f"  Charges added to EVERY pair halve the share of all pairs at or below 5% once they reach {half_all:.1f} points.")
print(f"  Charges added to bank-account pairs halve their share once they reach {half_bank:.1f} points, "
      f"and leave none at or below 5% from {zero_bank:.1f} points.")
print(f"  For comparison, the FSB ranges (recipient fees 0.1-1.8%, intermediary charges 0.1-3.4%) sum to 0.2 to 5.2 points.")
print(be[be["added_points"].isin([0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5.2])].round(1).to_string(index=False))

pd.concat(rows, ignore_index=True).to_csv("outputs/tables/table6n_final_checks.csv", index=False)
print("\nSaved outputs/tables/table6n_final_checks.csv and table6n_breakeven.csv")
