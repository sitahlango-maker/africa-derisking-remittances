"""
06_models.py
Step 6 of the paper: the main results, exactly as fixed in Section 4 (Method).

  Table 2  H1 grey-listing and H2 de-risking, corridor level (main analysis)
  Table 3  H3/H5 channels: banks, digital, zero-fee services (service level)
  Table 4  H4 UK corridors, and mobile money (corridor level)

For every estimate it reports the three layers of evidence in Section 4.6: the 95% range,
equivalence tests at +/-0.5 and +/-1 point, and the minimum detectable effect (MDE).

How to read a de-risking result: it is reported as the change in price (percentage points)
if the receiving country's banks' deposits abroad HALVED. A positive number = halving
deposits raises the price (what de-risking would predict).

Needs: scripts/fe.py (the shared helper), and the panels from step 5.
Outputs: outputs/tables/table2_main.csv, table3_channels.csv, table4_uk_mobile.csv

Run from the repo folder:  python scripts/06_models.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fe import regress, show   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

HALVING = {"L1_log_dep": np.log(0.5)}     # report de-risking as "deposits abroad halve"
CONTROLS = ["L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
svc = pd.read_csv("data/processed/panel_services.csv", low_memory=False)

# ---------------------------------------------------------------------------
# Table 2: grey-listing (H1) and de-risking (H2), corridor x quarter, corridor + quarter fixed effects
# ---------------------------------------------------------------------------
FE_C = ["corridor", "period_q"]
t2 = []
for y in ["cost", "fee", "fx"]:
    t2.append(regress(cq, y, ["grey_qend"], FE_C, label="A: grey only"))
    t2.append(regress(cq, y, ["grey_qend"] + CONTROLS + ["log_n_services"], FE_C, label="B: + controls"))
    t2.append(regress(cq, y, ["grey_qend", "L1_log_dep"] + CONTROLS + ["log_n_services"], FE_C,
                      label="C: + de-risking", scale=HALVING))
t2 = pd.concat(t2)
t2.to_csv("outputs/tables/table2_main.csv", index=False)

print("=" * 100)
print("TABLE 2. Grey-listing (H1) and de-risking (H2): corridor level")
print("  grey_qend  = change in price (points) while the receiving country is grey-listed")
print("  L1_log_dep = change in price (points) if banks' deposits abroad HALVED (previous quarter)")
print("=" * 100)
show(t2[t2["variable"].isin(["grey_qend", "L1_log_dep"])])
print(f"\nObservations: {t2['n_obs'].min():,}-{t2['n_obs'].max():,} corridor-quarters; "
      f"clusters (receiving countries): {t2['n_clusters'].max()}")

# ---------------------------------------------------------------------------
# Table 3: channels (H3/H5), service level, service + quarter fixed effects, total cost
# ---------------------------------------------------------------------------
FE_S = ["service_id", "period_q"]
s = svc.copy()
s["grey_x_bank"] = s["grey_qend"] * s["bank"]
s["grey_x_nonbank"] = s["grey_qend"] * (1 - s["bank"])
s["grey_x_zero_fee"] = s["grey_qend"] * s["zero_fee"]
s["grey_x_with_fee"] = s["grey_qend"] * (1 - s["zero_fee"])
s["dep_x_bank"] = s["L1_log_dep"] * s["bank"]
s["dep_x_nonbank"] = s["L1_log_dep"] * (1 - s["bank"])
d16 = s[s["year"] >= 2016].copy()                       # digital flag exists from 2016 Q2
d16["grey_x_digital"] = d16["grey_qend"] * d16["digital"]
d16["grey_x_nondigital"] = d16["grey_qend"] * (1 - d16["digital"])
d16["dep_x_digital"] = d16["L1_log_dep"] * d16["digital"]
d16["dep_x_nondigital"] = d16["L1_log_dep"] * (1 - d16["digital"])
HALF_INT = {v: np.log(0.5) for v in ["dep_x_bank", "dep_x_nonbank", "dep_x_digital", "dep_x_nondigital"]}

t3 = pd.concat([
    regress(s, "cost_pct_200", ["grey_qend"], FE_S, label="S1: all services"),
    regress(s, "cost_pct_200", ["grey_x_bank", "grey_x_nonbank"], FE_S, label="S2: bank vs non-bank"),
    regress(d16, "cost_pct_200", ["grey_x_digital", "grey_x_nondigital"], FE_S, label="S3: digital vs not (2016+)"),
    regress(s, "cost_pct_200", ["grey_x_zero_fee", "grey_x_with_fee"], FE_S, label="S4: zero-fee vs fee"),
    regress(s, "cost_pct_200", ["dep_x_bank", "dep_x_nonbank"], FE_S, label="S5: de-risking, bank vs non-bank",
            scale=HALF_INT),
    regress(d16, "cost_pct_200", ["dep_x_digital", "dep_x_nondigital"], FE_S,
            label="S6: de-risking, digital vs not", scale=HALF_INT),
])
t3.to_csv("outputs/tables/table3_channels.csv", index=False)

print("\n" + "=" * 100)
print("TABLE 3. Channels (H3, H5): service level, total cost")
print("  each row = the effect for that type of service; de-risking rows = deposits abroad halving")
print("=" * 100)
show(t3)

# ---------------------------------------------------------------------------
# Table 4: UK corridors (H4) and mobile money, corridor level
# ---------------------------------------------------------------------------
c = cq.copy()
c["uk"] = (c["source_code"] == "GBR").astype(int)
c["grey_x_uk"] = c["grey_qend"] * c["uk"]
c["grey_x_other"] = c["grey_qend"] * (1 - c["uk"])
c["dep_x_uk"] = c["L1_log_dep"] * c["uk"]
c["dep_x_other"] = c["L1_log_dep"] * (1 - c["uk"])
mm = c.dropna(subset=["L1_mobile_money_pct"]).copy()
mm["mm_c"] = (mm["L1_mobile_money_pct"] - mm["L1_mobile_money_pct"].mean()) / 10   # per 10 points, centred
mm["grey_x_mm"] = mm["grey_qend"] * mm["mm_c"]

t4 = pd.concat([
    regress(c, "cost", ["grey_x_uk", "grey_x_other"] + CONTROLS + ["log_n_services"], FE_C,
            label="U1: UK vs other senders"),
    regress(c, "cost", ["dep_x_uk", "dep_x_other"] + CONTROLS + ["log_n_services"], FE_C,
            label="U2: de-risking, UK vs other", scale={"dep_x_uk": np.log(0.5), "dep_x_other": np.log(0.5)}),
    regress(mm, "cost", ["grey_qend", "grey_x_mm"] + CONTROLS + ["log_n_services"], FE_C,
            label="M1: grey x mobile money (per +10 pts)"),
])
t4 = t4[t4["variable"].str.contains("grey|dep")]
t4.to_csv("outputs/tables/table4_uk_mobile.csv", index=False)

print("\n" + "=" * 100)
print("TABLE 4. UK corridors (H4) and mobile money: corridor level, total cost, with controls")
print("  grey_x_mm = how the grey-listing effect changes for each +10 points of adults with mobile money")
print("=" * 100)
show(t4)
print(f"\nUK corridors in the data: {c.loc[c['uk'] == 1, 'corridor'].nunique()}; "
      f"of which with a receiving country that changed grey-list status: "
      f"{(c[c['uk'] == 1].groupby('corridor')['grey_qend'].nunique() > 1).sum()}")

print("\nVerdicts follow Section 4.6: 'no meaningful effect' only if the equivalence test at +/-0.5 passes;")
print("'inconclusive' if the estimate is neither significant nor provably below 0.5.")
