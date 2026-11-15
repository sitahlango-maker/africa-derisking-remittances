"""
01_clean_rpw.py
Step 1 of the paper: clean the World Bank Remittance Prices Worldwide (RPW) file.

What it does, in plain words:
  1. Reads the two price sheets (old format up to Q1 2016, new format from Q2 2016).
  2. Gives the columns the same names so the two sheets can be stacked into one table.
  3. Keeps only services that send money TO Africa (Sub-Saharan Africa + North Africa).
  4. Flags suspicious prices (below 0% or above 50%) instead of deleting them.
  5. Adds three labels we need later: fee vs FX margin, "digital" service, and firm type.
  6. Saves one clean file and prints checks so you can see what happened.

Input : data/raw/rpw_dataset_2011_2025_q3.xlsx
Output: data/processed/rpw_africa_services_clean.csv

Run from the repo folder:  python scripts/01_clean_rpw.py
"""

import numpy as np
import pandas as pd

RAW = "data/raw/rpw_dataset_2011_2025_q3.xlsx"
OUT = "data/processed/rpw_africa_services_clean.csv"

# ---------------------------------------------------------------------------
# 1. Read the two price sheets (this takes a minute: the file is large)
# ---------------------------------------------------------------------------
print("Reading the RPW file...")
old = pd.read_excel(RAW, sheet_name="Dataset (up to Q1 2016)")
new = pd.read_excel(RAW, sheet_name="Dataset (from Q2 2016)")
print(f"  old sheet: {len(old):,} rows | new sheet: {len(new):,} rows")

# ---------------------------------------------------------------------------
# 2. Same column names in both sheets, then stack them
# ---------------------------------------------------------------------------
old = old.rename(columns={"sending location": "access_point",
                          "pick-up method": "pickup_method"})
new = new.rename(columns={"access point": "access_point",
                          "pickup method": "pickup_method",
                          "payment instrument": "payment_instrument"})
old["payment_instrument"] = np.nan  # the old format did not record this

keep = ["id", "period", "source_code", "source_name", "destination_code",
        "destination_name", "firm", "firm_type", "access_point",
        "payment_instrument", "pickup_method", "speed actual",
        "cc1 total cost %", "cc1 fx margin", "cc1 lcu fee", "transparent",
        "cc2 total cost %", "cc2 fx margin"]   # cc2 = the USD 500 send amount (for H6)

df = pd.concat([old[keep].assign(dataset="up_to_2016Q1"),
                new[keep].assign(dataset="from_2016Q2")], ignore_index=True)

# "cc1" = the $200 send amount (cc2 is $500). We use $200, the G20/SDG benchmark.
df = df.rename(columns={"cc1 total cost %": "cost_pct_200",
                        "cc1 fx margin": "fx_margin_pct",
                        "cc1 lcu fee": "fee_lcu",
                        "speed actual": "speed",
                        "cc2 total cost %": "cost_pct_500",
                        "cc2 fx margin": "fx_margin_pct_500"})

df["year"] = df["period"].str[:4].astype(int)      # "2016_2Q" -> 2016
df["quarter"] = df["period"].str[5].astype(int)    # "2016_2Q" -> 2
df["corridor"] = df["source_code"] + "->" + df["destination_code"]

# ---------------------------------------------------------------------------
# 3. Keep only services sending money TO Africa
#    Region comes from the file's own "Countries" sheet (World Bank groups, Q3 2025)
# ---------------------------------------------------------------------------
countries = pd.read_excel(RAW, sheet_name="Countries", header=1).iloc[:, :3]
countries.columns = ["code", "name", "region"]
region_of = dict(zip(countries["code"], countries["region"]))

NORTH_AFRICA = ["DZA", "EGY", "LBY", "MAR", "TUN"]  # grouped with the Middle East by the World Bank
df["dest_region"] = df["destination_code"].map(region_of)
is_africa = (df["dest_region"] == "Sub-Saharan Africa") | df["destination_code"].isin(NORTH_AFRICA)
df = df[is_africa].copy()
df["subregion"] = np.where(df["destination_code"].isin(NORTH_AFRICA),
                           "North Africa", "Sub-Saharan Africa")

# Before 2016 some rows are coded COD (DR Congo) but named "Congo, Rep.":
# we cannot tell which country they are, so we drop them and say so in the paper.
ambiguous = (df["destination_code"] == "COD") & (df["dataset"] == "up_to_2016Q1")
print(f"  dropped {ambiguous.sum():,} ambiguous pre-2016 COD rows")
df = df[~ambiguous].copy()

# Use the latest name for each country (e.g. Swaziland -> Eswatini)
latest_name = df.sort_values("period").groupby("destination_code")["destination_name"].last()
df["destination_name"] = df["destination_code"].map(latest_name)

# ---------------------------------------------------------------------------
# 4. Flag suspicious prices instead of deleting them
# ---------------------------------------------------------------------------
for col in ["cost_pct_200", "fx_margin_pct", "cost_pct_500", "fx_margin_pct_500"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")  
# Outlier (excluded from main analysis): total cost of 0% or less, or above 50%.
# Negative FX margin: kept but flagged for a robustness check. These cluster in
# FX-stressed markets (e.g. Nigeria, Egypt) and in 2021-2025, so dropping them
# would remove the very cases the paper is about.
df["outlier"] = (df["cost_pct_200"] <= 0) | (df["cost_pct_200"] > 50)
df["neg_fx_margin"] = df["fx_margin_pct"] < 0
# ---------------------------------------------------------------------------
# 5. Labels we need later
# ---------------------------------------------------------------------------
# Total cost = fee + FX margin, so the fee share is the difference
df["fee_pct"] = df["cost_pct_200"] - df["fx_margin_pct"]
df["zero_fee"] = df["fee_pct"].abs() < 0.01

# Digital = sent online or by phone, and neither paid in nor picked up in cash.
# Only the new format (from Q2 2016) records enough detail to tell.
low = lambda s: s.fillna("").astype(str).str.lower()
ap, pi, pm = low(df["access_point"]), low(df["payment_instrument"]), low(df["pickup_method"])
digital = (ap.str.contains("internet|mobile phone") & ~pi.str.contains("cash") & (pi != "")
           & ~pm.str.contains("cash") & (pm != ""))
df["digital"] = np.where(df["dataset"] == "from_2016Q2", digital.astype(float), np.nan)

df["firm_group"] = np.select(
    [df["firm_type"].str.contains("Mobile", na=False),
     df["firm_type"].str.strip().eq("Bank"),
     df["firm_type"].str.contains("Money Transfer", na=False)],
    ["Mobile operator", "Bank", "Money transfer operator"], "Other")

# ---------------------------------------------------------------------------
# 6. Save and print checks
# ---------------------------------------------------------------------------
df.to_csv(OUT, index=False)
print(f"\nSaved {OUT}")

print("\n--- CHECKS ---")
print(f"Rows (price quotes) kept : {len(df):,}")
print(f"Flagged as outliers      : {int(df['outlier'].sum()):,}")
print(f"Negative FX margin (kept): {int(df['neg_fx_margin'].sum()):,}")
print(f"Receiving countries      : {df['destination_code'].nunique()}")
print(f"Sending countries        : {df['source_code'].nunique()}")
print(f"Corridors                : {df['corridor'].nunique()}")
print(f"Years                    : {df['year'].min()} to {df['year'].max()}")
print("\nQuotes per year:")
print(df.groupby("year").size().to_string())
print("\nFirst 5 rows (a few columns):")
print(df[["period", "corridor", "firm", "firm_group", "cost_pct_200",
          "fx_margin_pct", "fee_pct", "digital"]].head().to_string(index=False))
