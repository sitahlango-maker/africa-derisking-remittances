"""
03_bis.py
Step 3 of the paper: build the de-risking measure from the BIS Locational Banking Statistics.

The idea in plain words:
  To send money abroad, an African bank keeps accounts with banks in other countries
  ("correspondent" or "nostro" accounts). The money it holds in those accounts shows up in the
  BIS data as what international banks OWE to banks in that African country.
  If global banks cut ties (de-risking), those balances shrink.
  So: "deposits of African banks held with banks abroad" = our de-risking proxy.

  We also keep the opposite direction, what international banks LEND to African banks
  ("claims"), as a second view.

What the script does:
  1. Reads the BIS bulk file line by line (it is ~540 MB, too big to open in one go).
  2. Keeps only the series we need:
       - quarterly, amounts outstanding, in US dollars, all currencies, all instruments
       - positions of banks in ALL BIS reporting countries (code 5A)
       - against counterparties in each African country
       - counterparty sector: all sectors (A) and banks only (B)
       - both directions: claims (C) and liabilities (L)
  3. Saves a quarterly table and a year-end (Q4) table, in USD millions.
  4. Prints checks.

Input : data/raw/WS_LBS_D_PUB_csv_col.csv   (unzipped BIS bulk file)
Output: data/processed/bis_lbs_africa_quarterly.csv
        data/processed/bis_lbs_africa_yearend.csv

Run from the repo folder:  python scripts/03_bis.py
"""

import csv
import sys

import pandas as pd

RAW = "data/raw/WS_LBS_D_PUB_csv_col.csv"
OUT = "data/processed/"

# The 54 African countries, by the two-letter codes the BIS uses
AFRICA = {"AO", "BJ", "BW", "BF", "BI", "CV", "CM", "CF", "TD", "KM", "CG", "CD", "CI", "DJ",
          "GQ", "ER", "SZ", "ET", "GA", "GM", "GH", "GN", "GW", "KE", "LS", "LR", "MG", "MW",
          "ML", "MR", "MU", "MZ", "NA", "NE", "NG", "RW", "ST", "SN", "SC", "SL", "SO", "ZA",
          "SS", "SD", "TZ", "TG", "UG", "ZM", "ZW", "DZ", "EG", "LY", "MA", "TN"}

# Filters that pick out exactly the series described above (BIS column codes)
WANT = {"FREQ": "Q",               # quarterly
        "L_MEASURE": "S",          # amounts outstanding (stocks)
        "L_INSTR": "A",            # all instruments
        "L_DENOM": "TO1",          # all currencies, converted to US dollars
        "L_CURR_TYPE": "A",        # all currency types
        "L_PARENT_CTY": "5J",      # banks of all nationalities
        "L_REP_BANK_TYPE": "A",    # all reporting bank types
        "L_REP_CTY": "5A",         # all BIS reporting countries
        "L_POS_TYPE": "N"}         # cross-border positions

# ---------------------------------------------------------------------------
# 1-2. Read the big file one row at a time and keep only what we need
# ---------------------------------------------------------------------------
print("Reading the BIS file (a few minutes, it is large)...")
rows = []
with open(RAW, newline="", encoding="utf-8-sig") as fh:
    reader = csv.reader(fh)
    header = next(reader)
    col = {name: i for i, name in enumerate(header)}

    needed = list(WANT) + ["L_POSITION", "L_CP_SECTOR", "L_CP_COUNTRY", "Counterparty country"]
    missing = [c for c in needed if c not in col]
    if missing:
        sys.exit(f"Stopped: the file does not have these expected columns: {missing}")

    quarters = [c for c in header if len(c) == 7 and c[4:6] == "-Q"]   # e.g. "2011-Q4"

    for row in reader:
        if any(row[col[k]] != v for k, v in WANT.items()):
            continue
        if row[col["L_POSITION"]] not in ("C", "L") or row[col["L_CP_SECTOR"]] not in ("A", "B", "I"):
            continue
        iso2 = row[col["L_CP_COUNTRY"]]
        if iso2 not in AFRICA:
            continue
        for q in quarters:
            value = row[col[q]]
            if value not in ("", "NaN"):
                rows.append((iso2, row[col["Counterparty country"]], row[col["L_POSITION"]],
                             row[col["L_CP_SECTOR"]], q, float(value)))

df = pd.DataFrame(rows, columns=["iso2", "country", "position", "sector", "period", "usd_mn"])

# Give the four series plain names
df["series"] = (df["position"].map({"C": "claims", "L": "liabilities"}) + "_"
                + df["sector"].map({"A": "all", "B": "banks", "I": "related"}))
NAMES = {"liabilities_banks": "deposits_abroad_banks",   # MAIN MEASURE: African banks' deposits held abroad
         "liabilities_all": "deposits_abroad_all",       # all African sectors' deposits held abroad
         "claims_banks": "lending_to_banks",             # international banks' lending to African banks
         "claims_all": "lending_to_all", "liabilities_related": "deposits_abroad_related", "claims_related": "lending_to_related"}                 # international banks' lending to all African sectors
df["series"] = df["series"].map(NAMES)

# ---------------------------------------------------------------------------
# 3. One row per country and quarter; then year-end (Q4) values
# ---------------------------------------------------------------------------
wide = df.pivot_table(index=["iso2", "country", "period"], columns="series",
                      values="usd_mn").reset_index()
wide["deposits_abroad_unrelated"] = wide["deposits_abroad_banks"] - wide["deposits_abroad_related"]
wide["year"] = wide["period"].str[:4].astype(int)
wide["quarter"] = wide["period"].str[-1].astype(int)
wide.to_csv(OUT + "bis_lbs_africa_quarterly.csv", index=False)

yearend = wide[wide["quarter"] == 4].drop(columns=["period", "quarter"])
yearend.to_csv(OUT + "bis_lbs_africa_yearend.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Checks
# ---------------------------------------------------------------------------
print("\n--- CHECKS ---")
print(f"African countries found  : {wide['iso2'].nunique()} of {len(AFRICA)}")
print(f"Quarters covered         : {wide['period'].min()} to {wide['period'].max()}")
absent = sorted(AFRICA - set(wide["iso2"]))
print(f"Countries with no data   : {', '.join(absent) if absent else 'none'}")

last = int(yearend["year"].max())
tot = yearend[yearend["year"].isin([2011, last])].groupby("year")[
    ["deposits_abroad_banks", "lending_to_banks"]].sum().round(0)
print(f"\nAll Africa, year-end totals (USD millions), 2011 vs {last}:")
print(tot.to_string())

dep = yearend.pivot_table(index="country", columns="year", values="deposits_abroad_banks")
if 2011 in dep and last in dep:
    both = dep[[2011, last]].dropna()
    fell = (both[last] < both[2011]).sum()
    print(f"\nCountries whose banks' deposits abroad FELL, 2011 to {last}: {fell} of {len(both)}")
    print("\nThe 10 largest falls (%):")
    print(((both[last] / both[2011] - 1) * 100).sort_values().head(10).round(0).to_string())
