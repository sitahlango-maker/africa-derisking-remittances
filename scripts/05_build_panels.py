"""
05_build_panels.py
Step 5 of the paper: join all the cleaned data into the tables the analysis uses,
and produce Table 1 (summary statistics).

What it does, in plain words:
  1. Loads the seven cleaned tables from steps 1-4.
  2. Builds three analysis tables ("panels"):
       a. SERVICE panel   - one row per price quote (a firm's service in a corridor in a quarter).
                            Used for the bank / digital / zero-fee comparisons (H5).
       b. CORRIDOR panel  - one row per corridor per quarter (e.g. UK->Kenya, 2024 Q2),
                            the average of its services. The MAIN analysis table.
       c. COUNTRY panel   - one row per African receiving country per year. Used for Table 1
                            and for describing the data.
  3. Attaches to every row the receiving country's conditions:
       - grey-listed at the END of that quarter (same period: a listing is announced, then prices may react)
       - deposits abroad (BIS) from the PREVIOUS quarter                  -> "lagged"
       - reserves, remittances, income, Findex, IMF bank data from the PREVIOUS year -> "lagged"
       - fragility list of the same year
     Lagging means conditions are measured BEFORE the price they may affect.
  4. Excludes outliers (total cost <= 0% or > 50%) from the panels; keeps the negative-FX flag.
  5. Saves the panels, saves Table 1, and prints checks.

Inputs : data/processed/*.csv from steps 1-4
Outputs: data/processed/panel_services.csv
         data/processed/panel_corridor_quarter.csv
         data/processed/panel_country_year.csv
         outputs/tables/table1_summary_statistics.csv

Run from the repo folder:  python scripts/05_build_panels.py
"""

import os

import numpy as np
import pandas as pd

P = "data/processed/"
os.makedirs("outputs/tables", exist_ok=True)

# BIS uses two-letter country codes; everything else uses three-letter codes
ISO2_TO_3 = {"AO": "AGO", "BJ": "BEN", "BW": "BWA", "BF": "BFA", "BI": "BDI", "CV": "CPV", "CM": "CMR",
             "CF": "CAF", "TD": "TCD", "KM": "COM", "CG": "COG", "CD": "COD", "CI": "CIV", "DJ": "DJI",
             "GQ": "GNQ", "ER": "ERI", "SZ": "SWZ", "ET": "ETH", "GA": "GAB", "GM": "GMB", "GH": "GHA",
             "GN": "GIN", "GW": "GNB", "KE": "KEN", "LS": "LSO", "LR": "LBR", "MG": "MDG", "MW": "MWI",
             "ML": "MLI", "MR": "MRT", "MU": "MUS", "MZ": "MOZ", "NA": "NAM", "NE": "NER", "NG": "NGA",
             "RW": "RWA", "ST": "STP", "SN": "SEN", "SC": "SYC", "SL": "SLE", "SO": "SOM", "ZA": "ZAF",
             "SS": "SSD", "SD": "SDN", "TZ": "TZA", "TG": "TGO", "UG": "UGA", "ZM": "ZMB", "ZW": "ZWE",
             "DZ": "DZA", "EG": "EGY", "LY": "LBY", "MA": "MAR", "TN": "TUN"}

# ---------------------------------------------------------------------------
# 1. Load the cleaned tables
# ---------------------------------------------------------------------------
print("Loading cleaned data...")
rpw = pd.read_csv(P + "rpw_africa_services_clean.csv", low_memory=False)
grey_q = pd.read_csv(P + "fatf_greylist_africa_quarterly.csv")
grey_y = pd.read_csv(P + "fatf_greylist_africa_yearly.csv")
bis_q = pd.read_csv(P + "bis_lbs_africa_quarterly.csv")
wdi = pd.read_csv(P + "wdi_africa.csv")
findex = pd.read_csv(P + "findex_africa.csv")
fcs = pd.read_csv(P + "fcs_africa.csv")
odc = pd.read_csv(P + "imf_odc_africa.csv")

# ---------------------------------------------------------------------------
# 2. Country conditions, ready to attach
# ---------------------------------------------------------------------------
# 2a. BIS, quarterly: log of deposits abroad, lagged one quarter and four quarters
bis_q["iso3"] = bis_q["iso2"].map(ISO2_TO_3)
bis_q = bis_q.sort_values(["iso3", "year", "quarter"])
# South Sudan: 2011 (independence year) is an unusual base, so start in 2012 (see paper, Section 3.3)
bis_q.loc[(bis_q["iso3"] == "SSD") & (bis_q["year"] < 2012),
          ["deposits_abroad_banks", "deposits_abroad_unrelated", "lending_to_banks"]] = np.nan
for col, short in [("deposits_abroad_banks", "dep"), ("deposits_abroad_unrelated", "dep_unrel"),
                   ("lending_to_banks", "lend")]:
    bis_q[f"log_{short}"] = np.log(bis_q[col].where(bis_q[col] > 0))
    bis_q[f"L1_log_{short}"] = bis_q.groupby("iso3")[f"log_{short}"].shift(1)   # previous quarter
    bis_q[f"L4_log_{short}"] = bis_q.groupby("iso3")[f"log_{short}"].shift(4)   # same quarter last year
bis_keep = bis_q[["iso3", "year", "quarter", "L1_log_dep", "L4_log_dep", "L1_log_dep_unrel", "L1_log_lend"]]

# 2b. Yearly conditions (World Bank, Findex, IMF), shifted forward one year = lagged
yearly = wdi.merge(findex.drop(columns=["country"]), on=["iso3", "year"], how="left")
yearly = yearly.merge(odc, on=["iso3", "year"], how="left")
yearly["log_reserves"] = np.log(yearly["reserves_ex_gold_usd"].where(yearly["reserves_ex_gold_usd"] > 0))
yearly["log_remit_in"] = np.log(yearly["remit_in_usd"].where(yearly["remit_in_usd"] > 0))
yearly["log_gdp_pc"] = np.log(yearly["gdp_per_capita_usd"])
yearly["log_odc_assets"] = np.log(yearly["odc_foreign_assets_usdmn"].where(yearly["odc_foreign_assets_usdmn"] > 0))
lag_vars = ["log_reserves", "reserves_months_imports", "log_remit_in", "remit_in_pct_gdp", "log_gdp_pc",
            "account_pct", "mobile_money_pct", "log_odc_assets"]
lagged = yearly[["iso3", "year"] + lag_vars].copy()
lagged["year"] = lagged["year"] + 1                       # last year's value sits on this year's row
lagged = lagged.rename(columns={v: "L1_" + v for v in lag_vars})

# 2c. Fragility (same year) and yearly grey-list measures
fcs_keep = fcs[["iso3", "year", "fcs"]]
grey_y_keep = grey_y[["iso3", "year", "grey_any", "grey_31dec", "grey_share_of_year"]]


def attach(df):
    """Add all country conditions to a table that has iso3 / year / quarter columns."""
    df = df.merge(grey_q[["iso3", "year", "quarter", "grey_qend"]], on=["iso3", "year", "quarter"], how="left")
    df["grey_qend"] = df["grey_qend"].fillna(0).astype(int)  # countries never listed are not in the grey file
    df = df.merge(bis_keep, on=["iso3", "year", "quarter"], how="left")
    df = df.merge(lagged, on=["iso3", "year"], how="left")
    df = df.merge(fcs_keep, on=["iso3", "year"], how="left")
    return df


# ---------------------------------------------------------------------------
# 3a. SERVICE panel (outliers excluded)
# ---------------------------------------------------------------------------
svc = rpw[~rpw["outlier"]].copy()
svc = svc.rename(columns={"destination_code": "iso3"})
svc["bank"] = (svc["firm_group"] == "Bank").astype(int)
svc["zero_fee"] = svc["zero_fee"].astype(int)
svc["neg_fx_margin"] = svc["neg_fx_margin"].astype(int)
# a "service" = the same firm, corridor, payment method and payout method, followed over time
svc["service_id"] = (svc["corridor"] + "|" + svc["firm"].fillna("?") + "|"
                     + svc["payment_instrument"].fillna("?").astype(str) + "|"
                     + svc["pickup_method"].fillna("?").astype(str))
svc["period_q"] = svc["year"].astype(str) + "Q" + svc["quarter"].astype(str)
svc = attach(svc)
svc_cols = ["service_id", "corridor", "source_code", "iso3", "destination_name", "subregion", "firm",
            "firm_group", "bank", "digital", "zero_fee", "neg_fx_margin", "year", "quarter", "period_q",
            "cost_pct_200", "fee_pct", "fx_margin_pct", "cost_pct_500"] + [c for c in svc.columns if c.startswith(("grey", "L1_", "L4_"))] + ["fcs"]
svc[svc_cols].to_csv(P + "panel_services.csv", index=False)

# ---------------------------------------------------------------------------
# 3b. CORRIDOR x QUARTER panel: average of the services in each corridor and quarter
# ---------------------------------------------------------------------------
cq = (svc.groupby(["corridor", "source_code", "iso3", "destination_name", "subregion", "year", "quarter", "period_q"])
      .agg(cost=("cost_pct_200", "mean"), fee=("fee_pct", "mean"), fx=("fx_margin_pct", "mean"),
           cost500=("cost_pct_500", "mean"), n_services=("cost_pct_200", "size"), share_digital=("digital", "mean"),
           share_zero_fee=("zero_fee", "mean"), share_bank=("bank", "mean"))
      .reset_index())
cq["log_n_services"] = np.log(cq["n_services"])
cq = attach(cq)
cq.to_csv(P + "panel_corridor_quarter.csv", index=False)

# ---------------------------------------------------------------------------
# 3c. COUNTRY x YEAR panel (for Table 1 and description)
# ---------------------------------------------------------------------------
cy = (svc.groupby(["iso3", "destination_name", "year"])
      .agg(cost=("cost_pct_200", "mean"), fee=("fee_pct", "mean"), fx=("fx_margin_pct", "mean"),
           n_quotes=("cost_pct_200", "size"), share_digital=("digital", "mean"),
           share_zero_fee=("zero_fee", "mean"), share_bank=("bank", "mean"))
      .reset_index())
bis_y = bis_q[bis_q["quarter"] == 4][["iso3", "year", "deposits_abroad_banks", "lending_to_banks"]]
cy = (cy.merge(grey_y_keep, on=["iso3", "year"], how="left")
        .merge(bis_y, on=["iso3", "year"], how="left")
        .merge(yearly[["iso3", "year", "reserves_months_imports", "remit_in_pct_gdp", "gdp_per_capita_usd",
                       "account_pct", "mobile_money_pct", "odc_foreign_assets_usdmn"]], on=["iso3", "year"], how="left")
        .merge(fcs_keep, on=["iso3", "year"], how="left"))
for c in ["grey_any", "grey_31dec", "grey_share_of_year"]:
    cy[c] = cy[c].fillna(0)
cy.to_csv(P + "panel_country_year.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Table 1: summary statistics (country-year panel, 2011-2025)
# ---------------------------------------------------------------------------
labels = {
    "cost": "Total cost of sending USD 200 (%)",
    "fee": "Fee (% of USD 200)",
    "fx": "Exchange-rate margin (%)",
    "share_digital": "Share of digital services (from 2016)",
    "share_zero_fee": "Share of zero-fee services",
    "share_bank": "Share of bank services",
    "grey_any": "Grey-listed at some point in the year (0/1)",
    "deposits_abroad_banks": "Banks' deposits abroad, BIS (USD million)",
    "lending_to_banks": "International lending to banks, BIS (USD million)",
    "odc_foreign_assets_usdmn": "Commercial banks' foreign assets, IMF (USD million)",
    "reserves_months_imports": "Reserves (months of imports)",
    "remit_in_pct_gdp": "Remittances received (% of GDP)",
    "gdp_per_capita_usd": "GDP per capita (USD)",
    "account_pct": "Adults with an account (%)",
    "mobile_money_pct": "Adults with mobile money (%)",
    "fcs": "Fragile or conflict-affected (0/1)",
}
t1 = cy[list(labels)].describe().T[["count", "mean", "std", "min", "max"]]
t1.index = [labels[i] for i in t1.index]
t1 = t1.round(2)
t1["count"] = t1["count"].astype(int)
t1.to_csv("outputs/tables/table1_summary_statistics.csv")

# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------
print("\n--- CHECKS ---")
print(f"Service panel  : {len(svc):,} quotes, {svc['service_id'].nunique():,} services")
print(f"Corridor panel : {len(cq):,} corridor-quarters, {cq['corridor'].nunique()} corridors, "
      f"{cq['iso3'].nunique()} receiving countries")
print(f"Country panel  : {len(cy):,} country-years, {cy['iso3'].nunique()} countries")

switch = cq.groupby("corridor")["grey_qend"].agg(["min", "max"])
print(f"\nCorridors whose receiving country changed grey-list status during the period: "
      f"{((switch['min'] == 0) & (switch['max'] == 1)).sum()} of {len(switch)}")
print(f"Share of corridor-quarters with the receiving country grey-listed: {cq['grey_qend'].mean():.0%}")

print("\nShare of corridor-quarters with each condition available (after lagging):")
for c in ["L1_log_dep", "L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "L1_mobile_money_pct",
          "L1_log_odc_assets", "fcs"]:
    print(f"  {c:22s} {cq[c].notna().mean():.0%}")

print("\nTable 1 (saved to outputs/tables/table1_summary_statistics.csv):")
pd.set_option("display.width", 200)
print(t1.to_string())
