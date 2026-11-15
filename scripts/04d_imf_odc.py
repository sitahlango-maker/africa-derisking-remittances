"""
04d_imf_odc.py
Step 4d of the paper: commercial banks' foreign assets and liabilities, from the IMF
(Monetary and Financial Statistics, "Other Depository Corporations" = banks and similar,
EXCLUDING the central bank).

Why we need it:
  The BIS measure in step 3 counts central banks as "banks", so central bank reserves are mixed in.
  This IMF series covers commercial banks only, so it is a cleaner cross-check of correspondent access.

What it does, in plain words:
  1. Asks the IMF data API, country by country, for two yearly series in US dollars:
       ODCORP_A_IL_ACO_NRES_ODCS  - commercial banks' claims on nonresidents (money they hold/lend abroad)
       ODCORP_L_IL_LT_NRES_ODCS   - commercial banks' liabilities to nonresidents (money they owe abroad)
  2. Saves every raw answer (so the download can be checked and repeated).
  3. Reads the numbers out of the answers and converts them to US$ millions.
  4. Prints checks, including a side-by-side with the BIS measure from step 3.

Input : the IMF API (internet)
Output: data/raw/imf_odc/<country>_<series>.xml   (raw answers)
        data/processed/imf_odc_africa.csv

Run from the repo folder:  python scripts/04d_imf_odc.py
"""

import os
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

RAW = "data/raw/imf_odc/"
OUT = "data/processed/imf_odc_africa.csv"
API = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,MFS_ODC/{country}.{series}.USD.A?startPeriod=2010"

SERIES = {"ODCORP_A_IL_ACO_NRES_ODCS": "odc_foreign_assets_usdmn",
          "ODCORP_L_IL_LT_NRES_ODCS": "odc_foreign_liabilities_usdmn"}

AFRICA = ["AGO", "BEN", "BWA", "BFA", "BDI", "CPV", "CMR", "CAF", "TCD", "COM", "COG", "COD",
          "CIV", "DJI", "GNQ", "ERI", "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN", "GNB", "KEN",
          "LSO", "LBR", "MDG", "MWI", "MLI", "MRT", "MUS", "MOZ", "NAM", "NER", "NGA", "RWA",
          "STP", "SEN", "SYC", "SLE", "SOM", "ZAF", "SSD", "SDN", "TZA", "TGO", "UGA", "ZMB",
          "ZWE", "DZA", "EGY", "LBY", "MAR", "TUN"]

os.makedirs(RAW, exist_ok=True)

# ---------------------------------------------------------------------------
# 1-2. Download (skips files already downloaded, so a re-run is quick)
# ---------------------------------------------------------------------------
print("Downloading from the IMF API (about 2 minutes)...")
for country in AFRICA:
    for series in SERIES:
        path = f"{RAW}{country}_{series}.xml"
        if os.path.exists(path):
            continue
        try:
            with urllib.request.urlopen(API.format(country=country, series=series), timeout=60) as r:
                open(path, "wb").write(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 404:                     # 404 = the IMF has no such series for this country
                print(f"  {country} {series}: HTTP {e.code}")
            open(path, "wb").write(b"")           # remember "no data" so we don't ask again
        time.sleep(0.3)                           # be polite to the IMF server

# ---------------------------------------------------------------------------
# 3. Read the numbers out of the answers
#    Each answer holds <Series COUNTRY=.. INDICATOR=..> with <Obs TIME_PERIOD=.. OBS_VALUE=..>
# ---------------------------------------------------------------------------
rows = []
for name in sorted(os.listdir(RAW)):
    path = RAW + name
    if os.path.getsize(path) == 0:
        continue
    for series_el in ET.parse(path).getroot().iter():
        if not series_el.tag.endswith("Series"):
            continue
        for obs in series_el:
            if obs.get("OBS_VALUE") not in (None, "", "NaN"):
                rows.append({"iso3": series_el.get("COUNTRY"),
                             "series": SERIES[series_el.get("INDICATOR")],
                             "year": int(obs.get("TIME_PERIOD")),
                             "value": float(obs.get("OBS_VALUE")) / 1e6})   # US$ -> US$ millions

odc = (pd.DataFrame(rows)
       .pivot_table(index=["iso3", "year"], columns="series", values="value")
       .reset_index())
odc.to_csv(OUT, index=False)

# ---------------------------------------------------------------------------
# 4. Checks
# ---------------------------------------------------------------------------
print(f"\nSaved {OUT}")
print("\n--- CHECKS ---")
print(f"Countries with data : {odc['iso3'].nunique()} of {len(AFRICA)}")
print(f"Years               : {odc['year'].min()} to {odc['year'].max()}")
missing = sorted(set(AFRICA) - set(odc["iso3"]))
print(f"No IMF data for     : {', '.join(missing) if missing else 'none'}")

# Side by side with step 3 (BIS), 2024: BIS includes central banks, IMF does not
bis = pd.read_csv("data/processed/bis_lbs_africa_yearend.csv")
iso2_to_3 = {"EG": "EGY", "NG": "NGA", "LY": "LBY", "ZA": "ZAF", "MU": "MUS", "KE": "KEN"}
bis = bis[bis["iso2"].isin(iso2_to_3) & (bis["year"] == 2024)].assign(iso3=lambda d: d["iso2"].map(iso2_to_3))
side = bis[["iso3", "deposits_abroad_banks"]].merge(
    odc[odc["year"] == 2024][["iso3", "odc_foreign_assets_usdmn"]], on="iso3", how="left")
side.columns = ["country", "BIS: all banks incl. central bank", "IMF: commercial banks only"]
print("\n2024, US$ millions: BIS measure vs IMF commercial-bank measure")
print(side.round(0).to_string(index=False))
