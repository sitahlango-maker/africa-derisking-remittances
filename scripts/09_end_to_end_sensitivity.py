"""
09_end_to_end_sensitivity.py
Step 9 of the paper (Section 5.7): beyond the sender's price.

The question:
  RPW records only what the SENDER is quoted (fee + exchange-rate margin). Charges taken by
  intermediaries along the chain, or paid by the recipient, are not recorded (Section 2.5).
  If we add published estimates of those charges, how many corridors still meet the
  SDG 10.c benchmarks (3%, and no corridor above 5%)?

How:
  1. Take 2024, the last full year, and average the quoted total cost for each corridor and
     payout method (cash, bank account, mobile wallet).
  2. Add published estimates of the unrecorded charges, by payout method. These are ASSUMPTIONS,
     taken from public sources and not measured for these corridors:
       - Bank account: recipient-side fees 0.1% to 1.8% plus intermediary charges 0.1% to 3.4%
         (FSB, 2024; measured for retail cross-border payments in general, not for Africa)
         -> low +0.2, high +5.2 points
       - Mobile wallet: cashing out raised the cost of sending USD 200 from 3.73% to 5.12%
         (GSMA, Q3 2022) -> low +0 (money stays in the wallet), high +1.39 points
       - Cash pickup: no published estimate -> +0 in both scenarios
  3. Count the share of corridor-payout pairs at or below 5% and 3%: as quoted, low, high.

This is an illustration with clearly labelled assumptions, not an estimate.

Needs: data/processed/rpw_africa_services_clean.csv
Output: outputs/tables/table7_end_to_end.csv

Run from the repo folder:  python scripts/09_end_to_end_sensitivity.py
"""

import os

import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 200)

YEAR = 2024
PAYOUT = {"Cash": "Cash pickup", "Bank account": "Bank account",
          "Own/partner bank account": "Bank account", "Mobile wallet": "Mobile wallet"}
ADD_ON = {   # (low, high) extra points of the amount sent, by payout method
    "Bank account": (0.1 + 0.1, 1.8 + 3.4),
    "Mobile wallet": (0.0, 5.12 - 3.73),
    "Cash pickup": (0.0, 0.0),
}

df = pd.read_csv("data/processed/rpw_africa_services_clean.csv", low_memory=False)
df = df[(~df["outlier"]) & (df["year"] == YEAR)].copy()
df["payout"] = df["pickup_method"].map(PAYOUT)
print(f"{YEAR} quotes: {len(df):,}; with a known payout method: {df['payout'].notna().sum():,}")
df = df[df["payout"].notna()]

# 1. One number per corridor and payout method: the average quoted total cost in the year
cells = df.groupby(["corridor", "payout"])["cost_pct_200"].agg(quoted="mean", quotes="size").reset_index()

# 2. Add the low and high estimates of unrecorded charges
cells["low"] = cells["quoted"] + cells["payout"].map(lambda p: ADD_ON[p][0])
cells["high"] = cells["quoted"] + cells["payout"].map(lambda p: ADD_ON[p][1])

# 3. Share of corridor-payout pairs meeting each benchmark
rows = []
for name, g in list(cells.groupby("payout")) + [("All payout methods", cells)]:
    row = {"payout": name, "pairs": len(g), "median_quoted": g["quoted"].median()}
    for target in (5, 3):
        for scen in ("quoted", "low", "high"):
            row[f"<= {target}% | {scen}"] = 100 * (g[scen] <= target).mean()
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv("outputs/tables/table7_end_to_end.csv", index=False)

print("=" * 110)
print(f"SHARE OF CORRIDORS (corridor x payout method, {YEAR}) MEETING THE SDG 10.c BENCHMARKS, %")
print("  quoted = as recorded by RPW; low / high = adding published estimates of unrecorded charges")
print("=" * 110)
print(table.round(1).to_string(index=False))
print("\nAdd-ons used (points of the amount sent, low / high):")
for p, (lo, hi) in ADD_ON.items():
    print(f"  {p:14s}: +{lo:.2f} / +{hi:.2f}")/africa-derisking-remitta