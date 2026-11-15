"""
11_map_figure.py
Map 1 for Section 3: the receiving countries covered, shaded by grey-list history,
with countries outside the main model outlined with a dashed line.

Draws straight from two files, with no mapping library needed:
  docs/map_data.json                        (step 10: each country's data)
  docs/ne_110m_admin_0_countries.geojson   (Natural Earth country boundaries, public domain)

Output: outputs/figures/map1_coverage.png

Run from the repo folder:  python scripts/11_map_figure.py
"""

import json

import matplotlib
matplotlib.use("Agg")                      # draw to a file (Cloud Shell has no screen)
import matplotlib.pyplot as plt           # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402

COLOURS = {"current": "#eb6834", "earlier": "#2a78d6", "never": "#cfe0f5", "nodata": "#efeeea"}
LABELS = {"current": "On the grey list (November 2026)", "earlier": "Grey-listed earlier, 2011–2026",
          "never": "In the price data, not grey-listed", "nodata": "No remittance price data"}
FIX = {"SDS": "SSD", "SAH": "ESH", "SOL": "SOM"}   # Natural Earth codes -> ISO3 used in the data

data = json.load(open("docs/map_data.json", encoding="utf-8"))["countries"]
world = json.load(open("docs/ne_110m_admin_0_countries.geojson", encoding="utf-8"))


def status(c):
    if c is None:
        return "nodata"
    ep = c["grey_episodes"]
    if any(e["removed"] is None for e in ep):
        return "current"
    if ep:
        return "earlier" if c["in_price_data"] else "nodata"
    return "never" if c["in_price_data"] else "nodata"


fig, ax = plt.subplots(figsize=(7, 7.4))
counts = {k: 0 for k in COLOURS}
for f in world["features"]:
    p = f["properties"]
    if p.get("CONTINENT") != "Africa":
        continue
    iso = p["ISO_A3"] if p.get("ISO_A3") not in (None, "-99") else p.get("ADM0_A3")
    iso = FIX.get(iso, iso)
    c = data.get(iso)
    s = status(c)
    if c and c["in_price_data"]:
        counts[s] += 1
    outside = bool(c and c["in_price_data"] and not c.get("in_main_model", True))
    g = f["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        ax.add_patch(Polygon(poly[0], closed=True, facecolor=COLOURS[s], edgecolor="white", linewidth=0.5))
        if outside:
            ax.add_patch(Polygon(poly[0], closed=True, facecolor="none", edgecolor="#52514e",
                                 linewidth=0.9, linestyle=(0, (2.5, 2))))

ax.set_xlim(-19, 53)
ax.set_ylim(-36, 38.5)
ax.set_aspect(1.05)
ax.axis("off")
handles = [Patch(facecolor=COLOURS[k], edgecolor="none", label=LABELS[k]) for k in ("current", "earlier", "never", "nodata")]
handles.append(Patch(facecolor="none", edgecolor="#52514e", linestyle=(0, (2.5, 2)), label="Outside the main model (missing data)"))
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig("outputs/figures/map1_coverage.png", dpi=200)

in_data = [k for k, v in data.items() if v["in_price_data"]]
print(f"Countries in the price data: {len(in_data)}; drawn on the map: {sum(counts[k] for k in ('current', 'earlier', 'never'))}")
print("  on the grey list now:", counts["current"], "| grey-listed earlier:", counts["earlier"], "| never listed:", counts["never"])
print("Saved outputs/figures/map1_coverage.png")
