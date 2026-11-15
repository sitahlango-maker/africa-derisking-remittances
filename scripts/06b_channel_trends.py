"""
06b_channel_trends.py
Additional check (NOT pre-specified in Section 4; prompted by the step 6 results).

The question:
  Step 6 found that digital and zero-fee services became cheaper in grey-listed countries.
  But most listings happened in 2021-2025, when digital and zero-fee prices were falling
  everywhere. The quarter fixed effects in step 6 remove trends common to ALL services,
  not trends specific to digital (or zero-fee, or bank) services. So the "effect" might be
  a digital-specific trend that happens to coincide with the listings.

The check:
  Re-run models S2-S4 from step 6, but give each type of service ITS OWN quarter effects
  (e.g. one set for digital services, another for non-digital). Grey-listing is then judged
  only against services of the same type in the same quarter.
    - If the digital / zero-fee effects stay large  -> the surprise is likely real.
    - If they shrink towards zero                   -> it was the type-specific trend.

Needs: scripts/fe.py and data/processed/panel_services.csv
Output: outputs/tables/table3b_channel_trends.csv

Run from the repo folder:  python scripts/06b_channel_trends.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from fe import regress, show   # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 220)

s = pd.read_csv("data/processed/panel_services.csv", low_memory=False)
s["grey_x_bank"] = s["grey_qend"] * s["bank"]
s["grey_x_nonbank"] = s["grey_qend"] * (1 - s["bank"])
s["grey_x_zero_fee"] = s["grey_qend"] * s["zero_fee"]
s["grey_x_with_fee"] = s["grey_qend"] * (1 - s["zero_fee"])

d16 = s[s["digital"].notna()].copy()            # digital label exists from 2016 Q2 (new RPW format)
d16["grey_x_digital"] = d16["grey_qend"] * d16["digital"]
d16["grey_x_nondigital"] = d16["grey_qend"] * (1 - d16["digital"])

# "Own quarter effects": a separate quarter effect for each type of service
s["quarter_by_bank"] = s["period_q"] + "_bank" + s["bank"].astype(str)
s["quarter_by_zero_fee"] = s["period_q"] + "_zf" + s["zero_fee"].astype(str)
d16["quarter_by_digital"] = d16["period_q"] + "_dig" + d16["digital"].astype(int).astype(str)

rows = []
for label, data, xs, qfe in [
    ("S2 bank vs non-bank", s, ["grey_x_bank", "grey_x_nonbank"], "quarter_by_bank"),
    ("S3 digital vs not (2016+)", d16, ["grey_x_digital", "grey_x_nondigital"], "quarter_by_digital"),
    ("S4 zero-fee vs fee", s, ["grey_x_zero_fee", "grey_x_with_fee"], "quarter_by_zero_fee"),
]:
    rows.append(regress(data, "cost_pct_200", xs, ["service_id", "period_q"], label=label + " | common quarters (step 6)"))
    rows.append(regress(data, "cost_pct_200", xs, ["service_id", qfe], label=label + " | own quarters (check)"))
res = pd.concat(rows)
res.to_csv("outputs/tables/table3b_channel_trends.csv", index=False)

print("=" * 110)
print("ADDITIONAL CHECK (not pre-specified): do the channel effects survive type-specific time trends?")
print("  'common quarters' = step 6 model; 'own quarters' = each type of service gets its own quarter effects")
print("=" * 110)
show(res)

# How many zero-fee and digital services are actually observed both before and during a listing?
print("\nServices that contribute to the estimate (observed both on and off the list):")
for name, data, flag in [("zero-fee", s, "zero_fee"), ("digital", d16, "digital")]:
    x = data[data[flag] == 1].groupby("service_id")["grey_qend"].nunique()
    print(f"  {name:9s}: {int((x > 1).sum())} services")
