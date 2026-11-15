"""
04a_wdi.py
Step 4a of the paper: country-level controls from the World Bank's World Development Indicators (WDI).

What it does, in plain words:
  1. Opens the six WDI zip files you downloaded into data/raw/wdi/ (one per indicator).
  2. Keeps the 54 African countries and the years 2010-2025.
  3. Puts all six indicators side by side: one row per country and year.
  4. Prints checks.

The six indicators and why we need them:
  BX.TRF.PWKR.CD.DT   Personal remittances received (current US$)   -> size of the remittance market
  BX.TRF.PWKR.DT.GD.ZS Personal remittances received (% of GDP)      -> how much the country depends on them
  FI.RES.XGLD.CD      Total reserves minus gold (current US$)       -> dollar shortages / central bank reserves
  FI.RES.TOTL.MO      Total reserves in months of imports           -> the same, scaled to the economy
  NY.GDP.PCAP.CD      GDP per capita (current US$)                  -> income level
  PA.NUS.FCRF         Official exchange rate (local currency per US$) -> to convert IMF data in step 4d

Input : data/raw/wdi/<indicator>.zip   (six files)
Output: data/processed/wdi_africa.csv

Run from the repo folder:  python scripts/04a_wdi.py
"""

import io
import zipfile

import pandas as pd

RAW = "data/raw/wdi/"
OUT = "data/processed/wdi_africa.csv"

INDICATORS = {
    "BX.TRF.PWKR.CD.DT": "remit_in_usd",
    "BX.TRF.PWKR.DT.GD.ZS": "remit_in_pct_gdp",
    "FI.RES.XGLD.CD": "reserves_ex_gold_usd",
    "FI.RES.TOTL.MO": "reserves_months_imports",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "PA.NUS.FCRF": "fx_official_lcu_per_usd",
}

# The 54 African countries (three-letter codes, as in the price data)
AFRICA = ["AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "COD",
          "CIV", "DJI", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN",
          "LSO", "LBR", "MDG", "MWI", "MLI", "MRT", "MUS", "MOZ", "NAM", "NER", "NGA", "RWA",
          "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "UGA", "ZMB",
          "ZWE", "DZA", "EGY", "LBY", "MAR", "TUN"]
YEARS = [str(y) for y in range(2010, 2026)]

frames = []
for code, name in INDICATORS.items():
    path = RAW + code + ".zip"
    with zipfile.ZipFile(path) as z:
        # the data file inside the zip starts with "API_"; the others are metadata
        data_file = [f for f in z.namelist() if f.startswith("API_")][0]
        raw = pd.read_csv(io.BytesIO(z.read(data_file)), skiprows=4)  # first 4 lines are titles
    raw = raw[raw["Country Code"].isin(AFRICA)]
    long = raw.melt(id_vars=["Country Code", "Country Name"], value_vars=[y for y in YEARS if y in raw],
                    var_name="year", value_name=name)
    frames.append(long.rename(columns={"Country Code": "iso3", "Country Name": "country"}))
    print(f"  read {code:22s} -> {name}")

wdi = frames[0]
for f in frames[1:]:
    wdi = wdi.merge(f, on=["iso3", "country", "year"], how="outer")
wdi["year"] = wdi["year"].astype(int)
wdi = wdi.sort_values(["iso3", "year"])
wdi.to_csv(OUT, index=False)

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
print(f"\nSaved {OUT}")
print("\n--- CHECKS ---")
print(f"Countries : {wdi['iso3'].nunique()} of {len(AFRICA)}")
print(f"Years     : {wdi['year'].min()} to {wdi['year'].max()}")
print("\nLatest year with data, per indicator (number of countries):")
for name in INDICATORS.values():
    have = wdi.dropna(subset=[name])
    last = int(have["year"].max())
    print(f"  {name:26s} latest {last}, countries {have[have['year'] == last]['iso3'].nunique()}")

y = 2024
top = wdi[wdi["year"] == y].nlargest(5, "remit_in_usd")[["country", "remit_in_usd", "remit_in_pct_gdp"]]
top["remit_in_usd"] = (top["remit_in_usd"] / 1e9).round(1)
print(f"\nTop 5 remittance receivers in {y} (US$ billions, % of GDP):")
print(top.to_string(index=False))
