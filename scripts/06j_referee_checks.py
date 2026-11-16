"""
06j_referee_checks.py
Four checks requested in review, in one script.

  PART 1  Without the number-of-services control
          Models B and C control for the (log) number of services in the corridor. If de-risking
          raises prices by thinning competition, that control would hide the effect. We re-run
          the main models without it.

  PART 2  Sending-country shocks: sender x quarter fixed effects
          Shocks on the sending side (a sending country's own grey-listing, UK banks closing
          money transfer operators' accounts in 2013-14, and so on) are absorbed by giving each
          sending country its own effect in every quarter. Grey-listing and de-risking are then
          compared only across corridors from the SAME sending country in the SAME quarter.

  PART 3  How much do banks' deposits abroad actually move?
          In the main-model sample: the within-country spread of log deposits, and how many
          countries ever see their deposits halve (within four quarters, and from a peak).

  PART 4  From 2016 only
          Before 2016 the grey-list record combines FATF's two public lists (the "public
          statement" and the "on-going process" list). Re-estimating from 2016 Q1 onwards uses
          only the single "jurisdictions under increased monitoring" list.

Needs: scripts/fe.py and the panels from step 5
Output: outputs/tables/table6i_referee_checks.csv

Run from the repo folder:  python scripts/06j_referee_checks.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fe import regress, show   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
HALVING = {"L1_log_dep": np.log(0.5)}
FE_C, FE_S = ["corridor", "period_q"], ["service_id", "period_q"]

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
cq["sender_q"] = cq["source_code"].astype(str) + "|" + cq["period_q"].astype(str)
svc["sender_q"] = svc["source_code"].astype(str) + "|" + svc["period_q"].astype(str)

B_XS = ["grey_qend"] + CONTROLS
C_XS = ["grey_qend", "L1_log_dep"] + CONTROLS
NS = ["log_n_services"]

rows = []


def add(res, part):
    res = res[res["variable"].isin(["grey_qend", "L1_log_dep"])].copy()
    res.insert(0, "part", part)
    rows.append(res)


# --- reference: the paper's models (should match Table 2 / Table 3)
add(regress(cq, "cost", B_XS + NS, FE_C, label="B: paper (with services control)"), "0 reference")
add(regress(cq, "cost", C_XS + NS, FE_C, label="C: paper (with services control)", scale=HALVING), "0 reference")
add(regress(svc, "cost_pct_200", ["grey_qend"], FE_S, label="S1: paper"), "0 reference")

# --- PART 1: without the number-of-services control
add(regress(cq, "cost", B_XS, FE_C, label="B: without services control"), "1 no services control")
add(regress(cq, "cost", C_XS, FE_C, label="C: without services control", scale=HALVING), "1 no services control")

# --- PART 2: sender x quarter fixed effects
FE_CS = ["corridor", "sender_q"]
FE_SS = ["service_id", "sender_q"]
add(regress(cq, "cost", ["grey_qend"], FE_CS, label="A: sender x quarter FE"), "2 sender x quarter")
add(regress(cq, "cost", C_XS, FE_CS, label="C: sender x quarter FE (no services control)", scale=HALVING),
    "2 sender x quarter")
add(regress(svc, "cost_pct_200", ["grey_qend"], FE_SS, label="S1: sender x quarter FE"), "2 sender x quarter")

# --- PART 4: from 2016 only
c16, s16 = cq[cq["year"] >= 2016], svc[svc["year"] >= 2016]
add(regress(c16, "cost", ["grey_qend"], FE_C, label="A: 2016 onwards"), "4 from 2016")
add(regress(c16, "cost", C_XS, FE_C, label="C: 2016 onwards (no services control)", scale=HALVING), "4 from 2016")
add(regress(s16, "cost_pct_200", ["grey_qend"], FE_S, label="S1: 2016 onwards"), "4 from 2016")

res = pd.concat(rows)
res.to_csv("outputs/tables/table6i_referee_checks.csv", index=False)

print("=" * 100)
print("PARTS 1, 2 AND 4: total cost of sending USD 200 (percentage points)")
print("  grey_qend = while grey-listed; L1_log_dep = if banks' deposits abroad HALVED")
print("  The 'reference' rows should match Tables 2 and 3.")
print("=" * 100)
for part, d in res.groupby("part", sort=True):
    print(f"\n--- {part.upper()} ---")
    show(d, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
    print("    observations / receiving countries:",
          "; ".join(f"{m.split(':')[0]} {n:,}/{g}" for m, n, g in
                    d.drop_duplicates("model")[["model", "n_obs", "n_clusters"]].itertuples(index=False)))

# --- PART 3: how much deposits move, in the main-model (model C) sample
need = ["cost", "grey_qend", "L1_log_dep"] + CONTROLS + NS
main = cq.dropna(subset=need)
cy = main.groupby(["iso3", "year", "quarter"])["L1_log_dep"].first().reset_index()
cy["within"] = cy["L1_log_dep"] - cy.groupby("iso3")["L1_log_dep"].transform("mean")
cy = cy.sort_values(["iso3", "year", "quarter"])
# change over four calendar quarters (RPW is not surveyed every quarter, so match on the date, not the row)
cy["qnum"] = cy["year"] * 4 + cy["quarter"]
prev = cy[["iso3", "qnum", "L1_log_dep"]].assign(qnum=lambda d: d["qnum"] + 4).rename(columns={"L1_log_dep": "dep_4q_before"})
cy = cy.merge(prev, on=["iso3", "qnum"], how="left")
cy["chg4"] = cy["L1_log_dep"] - cy["dep_4q_before"]
cy["from_peak"] = cy["L1_log_dep"] - cy.groupby("iso3")["L1_log_dep"].cummax()  # fall from the highest level so far
per = cy.groupby("iso3").agg(sd_within=("within", "std"),
                             worst_4q=("chg4", "min"),
                             worst_from_peak=("from_peak", "min"))
half = np.log(0.5)

print("\n" + "=" * 100)
print("PART 3. HOW MUCH DO BANKS' DEPOSITS ABROAD MOVE? (main-model sample)")
print("=" * 100)
print(f"Receiving countries: {len(per)}")
print(f"Within-country standard deviation of log deposits: {cy['within'].std():.2f} "
      f"(i.e. a typical deviation from a country's own average of about {100 * (np.exp(cy['within'].std()) - 1):.0f}%)")
print(f"  by country, median {per['sd_within'].median():.2f}, range {per['sd_within'].min():.2f} to {per['sd_within'].max():.2f}")
print(f"Countries whose deposits ever HALVED within four quarters: {(per['worst_4q'] <= half).sum()}")
print(f"Countries whose deposits ever fell to HALF their earlier peak: {(per['worst_from_peak'] <= half).sum()}")
print("\nLargest falls from peak (log points; -0.69 = halved):")
print(per.sort_values("worst_from_peak").head(10).round(2).to_string())
print("\nSaved outputs/tables/table6i_referee_checks.csv")
