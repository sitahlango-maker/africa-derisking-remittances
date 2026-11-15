"""
08_figures.py
Step 8 of the paper: Figure 1 (Section 5.1). How did the price of sending USD 200 to Africa
move between 2011 and 2025, by channel?

Three small panels, same vertical scale so they can be compared:
  (a) by type of provider: bank, money transfer operator, mobile operator
  (b) digital vs non-digital services (the digital label exists from 2016 Q2)
  (c) zero-fee vs fee-charging services
Each line is the simple average total cost of all quotes in that year (outliers excluded, as
in the analysis). Dashed lines mark the SDG 10.c targets: 3% on average, no corridor above 5%.

Important: these are raw averages. The services and corridors in the data change over time
(Section 3.1), so the figure DESCRIBES prices; it does not estimate effects. The models in
Section 5 compare each corridor or service only with itself.

Needs: data/processed/rpw_africa_services_clean.csv
Output: outputs/figures/figure1_cost_by_channel.png
        outputs/tables/figure1_data.csv  (the numbers behind the figure)

Run from the repo folder:  python scripts/08_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")                      # draw to a file (Cloud Shell has no screen)
import matplotlib.pyplot as plt           # noqa: E402
import pandas as pd                        # noqa: E402

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
pd.set_option("display.width", 200)

MIN_QUOTES = 30                            # a yearly average needs at least 30 quotes to be shown
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # colourblind-checked palette

df = pd.read_csv("data/processed/rpw_africa_services_clean.csv", low_memory=False)
df = df[~df["outlier"]].copy()
df["zero_fee"] = df["zero_fee"].astype(bool)


def yearly(data, group_col, labels):
    """Average total cost per year for each group; years with too few quotes are left blank."""
    g = data.groupby(["year", group_col])["cost_pct_200"].agg(["mean", "size"]).reset_index()
    g = g[g[group_col].isin(labels)]
    g.loc[g["size"] < MIN_QUOTES, "mean"] = float("nan")
    g["group"] = g[group_col].map(labels)
    return g[["year", "group", "mean", "size"]]


panels = [
    ("(a) By type of provider", yearly(df, "firm_group",
        {"Bank": "Banks", "Money transfer operator": "Money transfer operators", "Mobile operator": "Mobile operators"})),
    ("(b) Digital and non-digital (from 2016)", yearly(df[df["digital"].notna()], "digital",
        {1.0: "Digital", 0.0: "Non-digital"})),
    ("(c) Zero-fee and fee-charging", yearly(df, "zero_fee",
        {True: "Zero fee", False: "With a fee"})),
]
colours = [BLUE, ORANGE, AQUA]
markers = ["o", "s", "^"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
for ax, (title, g) in zip(axes, panels):
    for i, name in enumerate(g["group"].unique()):
        s = g[g["group"] == name].sort_values("year")
        ax.plot(s["year"], s["mean"], color=colours[i], marker=markers[i], ms=4.5, lw=2, label=name)
    for target in (3, 5):
        ax.axhline(target, color="#52514e", lw=0.9, ls="--")
    ax.text(2011, 3, "3% target", va="bottom", fontsize=8, color="#52514e")
    ax.text(2011, 5, "5% target", va="bottom", fontsize=8, color="#52514e")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_xlim(2010.5, 2025.5)
    ax.set_xticks([2011, 2013, 2015, 2017, 2019, 2021, 2023, 2025])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", color="#e6e5e0", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
axes[0].set_ylabel("Average total cost (% of USD 200)")
axes[0].set_ylim(0, None)
fig.tight_layout()
fig.savefig("outputs/figures/figure1_cost_by_channel.png", dpi=200)

data = pd.concat([g.assign(panel=t) for t, g in panels])
data.to_csv("outputs/tables/figure1_data.csv", index=False)

# Print the numbers behind the figure (first year, 2016, 2019, last full year 2024, 2025)
print("=" * 90)
print("FIGURE 1: average total cost (% of USD 200) by channel; blank = fewer than 30 quotes")
print("=" * 90)
wide = data.pivot_table(index=["panel", "group"], columns="year", values="mean")
print(wide[[c for c in [2011, 2016, 2019, 2024, 2025] if c in wide.columns]].round(2).to_string())
print("\nQuotes behind each line (all years):")
print(data.groupby(["panel", "group"])["size"].sum().to_string())
print("\nSaved outputs/figures/figure1_cost_by_channel.png")