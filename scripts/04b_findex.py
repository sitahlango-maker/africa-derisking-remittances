"""
04b_findex.py
Step 4b of the paper: financial access from the World Bank's Global Findex Database 2025.

What it does, in plain words:
  1. Reads the Findex country file (survey waves: 2011, 2014, 2017, 2021/2022, 2024).
  2. Keeps African countries and four measures (shares of adults aged 15+):
       account_pct            - has any account (bank, similar institution or mobile money)
       fi_account_pct         - has a bank or similar financial-institution account
       mobile_money_pct       - has a mobile money account
       dom_remit_into_acct    - of people who RECEIVED money from family elsewhere in the country,
                                the share who got it into an account (not cash).
                                Findex 2025 does not ask about international remittances,
                                so this is the closest measure of "receiving digitally".
  3. Fills the years between surveys by drawing a straight line between two survey values
     (linear interpolation). It does NOT guess beyond the last survey (no extrapolation).
  4. Prints checks.

Input : data/raw/GlobalFindexDatabase2025.xlsx
Output: data/processed/findex_africa.csv

Run from the repo folder:  python scripts/04b_findex.py
"""

import pandas as pd

RAW = "data/raw/GlobalFindexDatabase2025.xlsx"
OUT = "data/processed/findex_africa.csv"

AFRICA = ["AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "COD",
          "CIV", "DJI", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN",
          "LSO", "LBR", "MDG", "MWI", "MLI", "MRT", "MUS", "MOZ", "NAM", "NER", "NGA", "RWA",
          "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "UGA", "ZMB",
          "ZWE", "DZA", "EGY", "LBY", "MAR", "TUN"]

COLS = {"account.t.d": "account_pct",
        "fiaccount.t.d": "fi_account_pct",
        "mobileaccount.t.d": "mobile_money_pct",
        "fin29.s": "dom_remit_into_acct"}

# 1-2. Read the 'Data' sheet. Row 2 of the sheet holds long labels, so we skip it.
print("Reading the Findex file...")
fx = pd.read_excel(RAW, sheet_name="Data", usecols=["codewb", "countrynewwb", "year", "group"] + list(COLS),
                   skiprows=[1])
fx = fx[(fx["group"] == "all") & (fx["codewb"].isin(AFRICA))]        # whole adult population only
fx = fx.rename(columns={"codewb": "iso3", "countrynewwb": "country", **COLS})
for c in COLS.values():
    fx[c] = pd.to_numeric(fx[c], errors="coerce") * 100              # shares -> percentages
fx["year"] = pd.to_numeric(fx["year"], errors="coerce").astype(int)

# 3. Fill the years between survey waves, country by country
out = []
for iso3, g in fx.groupby("iso3"):
    g = g.set_index("year").sort_index()
    years = range(int(g.index.min()), int(g.index.max()) + 1)
    filled = g[list(COLS.values())].reindex(years).interpolate(limit_area="inside")
    filled["findex_survey_year"] = [int(y in g.index) for y in years]   # 1 = real survey, 0 = filled in
    filled["iso3"] = iso3
    filled["country"] = g["country"].iloc[0]
    out.append(filled.reset_index().rename(columns={"index": "year"}))
findex = pd.concat(out)[["iso3", "country", "year", *COLS.values(), "findex_survey_year"]]
findex.to_csv(OUT, index=False)

# 4. Checks
print(f"\nSaved {OUT}")
print("\n--- CHECKS ---")
print(f"African countries in Findex : {findex['iso3'].nunique()} of {len(AFRICA)}")
print(f"Survey years                : {sorted(fx['year'].unique())}")
print(f"Years after filling         : {findex['year'].min()} to {findex['year'].max()}")

latest = fx.sort_values("year").groupby("iso3").tail(1)
print(f"\nLatest survey, African average: account {latest['account_pct'].mean():.0f}%, "
      f"mobile money {latest['mobile_money_pct'].mean():.0f}%")
print("\nTop 5 by mobile money account (latest survey):")
print(latest.nlargest(5, "mobile_money_pct")[["country", "year", "mobile_money_pct", "fi_account_pct"]]
      .round(0).to_string(index=False))
