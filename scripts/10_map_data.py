"""
10_map_data.py
Exports, for each African country, the numbers shown on the interactive map, read only
from the files the earlier steps produced. Nothing is estimated here and nothing is typed in.

For each country:
  - grey-list episodes (verified dates and where each was confirmed)
  - whether it is in the price data, and whether it is in the main model (and if not, why)
  - yearly price of sending USD 200: total cost, fee, exchange-rate margin, number of quotes
  - 2024 average total cost by type of provider
  - banks' deposits abroad (BIS) and commercial banks' foreign assets (IMF) by year
  - latest remittances (% of GDP), account and mobile money ownership
  - 2024 share of corridor-payout pairs at or below 5% and 3%, as quoted and with
    published estimates of unrecorded charges (low / high), as in step 9

The map states the paper's POOLED results for all countries; it does not show
country-specific effects, because the paper does not estimate them.

Needs: the step 1, 2 and 5 outputs and the result tables (steps 6-7)
Output: docs/map_data.json  (the "docs" folder is what GitHub Pages publishes)

Run from the repo folder:  python scripts/10_map_data.py
"""

import json
import os

import numpy as np
import pandas as pd

P = "data/processed/"
os.makedirs("docs", exist_ok=True)

ep = pd.read_csv(P + "fatf_greylist_africa_episodes.csv")
cy = pd.read_csv(P + "panel_country_year.csv")
cq = pd.read_csv(P + "panel_corridor_quarter.csv")
svc = pd.read_csv(P + "panel_services.csv", low_memory=False)
rpw = pd.read_csv(P + "rpw_africa_services_clean.csv", low_memory=False)

# --- Main-model coverage (same rule as 06d): every input available
NEED = ["L1_log_dep", "L1_log_reserves", "L1_log_remit_in", "L1_log_gdp_pc", "fcs"]
LABEL = {"L1_log_dep": "BIS deposits abroad", "L1_log_reserves": "reserves",
         "L1_log_remit_in": "remittance inflows", "L1_log_gdp_pc": "GDP per capita", "fcs": "fragility status"}
used = set(cq.dropna(subset=NEED)["iso3"])


def why_missing(c):
    d = cq[cq["iso3"] == c]
    miss = [LABEL[v] for v in NEED if d[v].isna().all()]
    return ("no " + " or ".join(miss) + " data") if miss else "inputs never all available in the same year"


# --- Step 9 shares by country (2024, corridor x payout method)
PAYOUT = {"Cash": "Cash pickup", "Bank account": "Bank account",
          "Own/partner bank account": "Bank account", "Mobile wallet": "Mobile wallet"}
ADD = {"Bank account": (0.2, 5.2), "Mobile wallet": (0.0, 1.39), "Cash pickup": (0.0, 0.0)}
r24 = rpw[(~rpw["outlier"]) & (rpw["year"] == 2024)].copy()
r24["payout"] = r24["pickup_method"].map(PAYOUT)
cells = (r24.dropna(subset=["payout"]).groupby(["destination_code", "corridor", "payout"])["cost_pct_200"]
         .mean().rename("quoted").reset_index())
cells["low"] = cells["quoted"] + cells["payout"].map(lambda p: ADD[p][0])
cells["high"] = cells["quoted"] + cells["payout"].map(lambda p: ADD[p][1])

# --- 2024 cost by provider type
prov = (svc[svc["year"] == 2024].groupby(["iso3", "firm_group"])["cost_pct_200"]
        .agg(["mean", "size"]).reset_index())


def num(x, nd=2):
    return None if pd.isna(x) else round(float(x), nd)


countries = sorted(set(cy["iso3"]) | set(ep["iso3"]))
out = {}
for c in countries:
    e = ep[ep["iso3"] == c]
    d = cy[cy["iso3"] == c].sort_values("year")
    name = (d["destination_name"].iloc[0] if len(d) else e["country"].iloc[0])
    rec = {
        "name": name,
        "in_price_data": bool(len(d)),
        "grey_episodes": [{"listed": r.listed_from, "removed": (None if pd.isna(r.removed_at) else r.removed_at),
                           "checked": r.checked, "source": r.source} for r in e.itertuples()],
    }
    if len(d):
        rec["in_main_model"] = c in used
        rec["why_not_in_main_model"] = None if c in used else why_missing(c)
        rec["corridors"] = int(cq.loc[cq["iso3"] == c, "corridor"].nunique())
        rec["years"] = [int(y) for y in d["year"]]
        for col in ["cost", "fee", "fx"]:
            rec[col] = [num(v) for v in d[col]]
        rec["n_quotes"] = [int(v) for v in d["n_quotes"]]
        rec["deposits_abroad_bis_usdmn"] = [num(v, 0) for v in d["deposits_abroad_banks"]]
        rec["foreign_assets_imf_usdmn"] = [num(v, 0) for v in d["odc_foreign_assets_usdmn"]]
        last = lambda col: (lambda s: None if s.empty else {"year": int(s.index[-1]), "value": num(s.iloc[-1], 1)})(
            d.set_index("year")[col].dropna())
        rec["remit_pct_gdp_latest"] = last("remit_in_pct_gdp")
        rec["account_pct_latest"] = last("account_pct")
        rec["mobile_money_pct_latest"] = last("mobile_money_pct")
        p = prov[prov["iso3"] == c]
        rec["cost_2024_by_provider"] = {r.firm_group: {"cost": num(r["mean"]), "quotes": int(r["size"])}
                                        for _, r in p.iterrows() if r["size"] >= 5}
        k = cells[cells["destination_code"] == c]
        rec["targets_2024"] = None if k.empty else {
            "pairs": int(len(k)),
            **{f"le{t}_{s}": num(100 * (k[s] <= t).mean(), 0) for t in (5, 3) for s in ("quoted", "low", "high")}}
    out[c] = rec

T = "outputs/tables/"
t2, t3, t5 = (pd.read_csv(T + f) for f in ("table2_main.csv", "table3_channels.csv", "table5_event_study.csv"))


def fmt(df, model_start, variable, outcome=None):
    r = df[df["model"].str.startswith(model_start) & (df["variable"] == variable)]
    if outcome is not None:
        r = r[r["outcome"] == outcome]
    r = r.iloc[0]
    return f"{r['coef']:+.2f} percentage points (95% CI {r['ci95_lo']:+.2f} to {r['ci95_hi']:+.2f})"


# robust grey-listing estimate (step 06m): listings only, average of the four years after listing
t6m = pd.read_csv(T + "table6m_listing_removal.csv")


def robust(level):
    r = t6m[(t6m["model"] == f"{level} | listings only") & (t6m["what"] == "average")].iloc[0]
    return f"{r['coef']:+.2f} percentage points (95% CI {r['ci95_lo']:+.2f} to {r['ci95_hi']:+.2f})"


meta = {
    "title": "De-risking, grey-listing and the cost of remittances to Africa, 2011-2025",
    "author": "Winnie Sitah Lang'o",
    "pooled_findings": {   # read from the saved result tables
        "grey_listing_service_level": fmt(t3, "S1", "grey_qend"),
        "grey_listing_corridor_level": fmt(t2, "C", "grey_qend", "cost"),
        "deposits_abroad_halve": fmt(t2, "C", "L1_log_dep", "cost"),
        "years_after_removal": fmt(t5, "stacked average | removal", "after"),
        "grey_listing_robust_service": robust("service"),
        "grey_listing_robust_corridor": robust("corridor"),
    },
    "sources": "World Bank RPW 2011-2025 Q3; FATF statements; BIS LBS; IMF MFS; World Bank WDI, Findex 2025, FCS lists",
}
with open("docs/map_data.json", "w", encoding="utf-8") as f:
    json.dump({"meta": meta, "countries": out}, f, ensure_ascii=False, indent=1)

print(f"Countries exported: {len(out)} (in price data: {sum(v['in_price_data'] for v in out.values())}; "
      f"in main model: {sum(v.get('in_main_model', False) for v in out.values())})")
print("Pooled findings:", json.dumps(meta["pooled_findings"], ensure_ascii=False, indent=1))
print("Example (Kenya):")
k = out.get("KEN", {})
print("  grey episodes:", [(g['listed'], g['removed']) for g in k.get('grey_episodes', [])])
print("  cost 2011 / last year:", k.get("cost", [None])[0], "/", k.get("cost", [None])[-1])
print("  2024 targets:", k.get("targets_2024"))
print(f"\nSaved docs/map_data.json ({os.path.getsize('docs/map_data.json') / 1024:.0f} KB)")
~/africa-de