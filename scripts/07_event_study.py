"""
07_event_study.py
Step 7 of the paper: timing. What happens to prices year by year around a grey-listing,
and around a removal from the list? (Section 4.5)

Two methods:
  1. STANDARD event study: corridor + quarter fixed effects, with one indicator for each year
     relative to the event (3+ years before ... 4+ years after). The year before the event is
     the reference (set to zero), so every estimate reads "compared with the year before".
  2. STACKED event study: for EACH listing (or removal), build a separate mini-experiment:
     the listed country's corridors versus corridors into countries NOT on the list at any point
     in the same window (3 years before to 4 years after). Stack the mini-experiments and
     estimate them together. This avoids comparing a newly listed country with one listed
     years earlier, which can distort the standard method when listings happen at different dates.
     It also gives one summary number: the average change in price after the event.

What to look for:
  - Years BEFORE the event should be close to zero (no "pre-trend"); otherwise the event is
    not the likely cause.
  - Years AFTER the event show whether prices move, and how fast.

Needs: scripts/fe.py, data/processed/panel_corridor_quarter.csv,
       data/processed/fatf_greylist_africa_episodes.csv
Outputs: outputs/tables/table5_event_study.csv
         outputs/figures/figure2_event_study.png

Run from the repo folder:  python scripts/07_event_study.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")                       # draw to a file (Cloud Shell has no screen)
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

sys.path.insert(0, "scripts")
from fe import regress, show                # noqa: E402

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)
pd.set_option("display.width", 220)

LO, HI = -3, 4                              # years relative to the event; ends are binned (<=-3, >=4)
WIN_START, WIN_END = pd.Timestamp("2012-01-01"), pd.Timestamp("2025-09-30")

cq = pd.read_csv("data/processed/panel_corridor_quarter.csv")
cq["q_end"] = pd.PeriodIndex.from_fields(year=cq["year"], quarter=cq["quarter"], freq="Q").end_time.normalize()
ep = pd.read_csv("data/processed/fatf_greylist_africa_episodes.csv", parse_dates=["listed_from", "removed_at"])
in_prices = set(cq["iso3"])

# Events inside the price data window, for countries that appear in the price data
EVENTS = {
    "listing": ep.loc[ep["listed_from"].between(WIN_START, WIN_END) & ep["iso3"].isin(in_prices),
                      ["iso3", "listed_from"]].rename(columns={"listed_from": "date"}),
    "removal": ep.loc[ep["removed_at"].between(WIN_START, WIN_END) & ep["iso3"].isin(in_prices),
                      ["iso3", "removed_at"]].rename(columns={"removed_at": "date"}),
}


def years_since(q_end, date):
    """Whole years from the event to the end of the quarter (0 = the event's first year)."""
    return np.floor((q_end - date).dt.days / 365.25)


def standard(kind):
    """Standard event study: one event per country (its first in the window)."""
    first = EVENTS[kind].sort_values("date").groupby("iso3")["date"].first()
    d = cq.copy()
    d["event_date"] = d["iso3"].map(first)
    k = years_since(d["q_end"], d["event_date"]).clip(LO, HI)
    xs = []
    for j in range(LO, HI + 1):
        if j == -1:
            continue                          # reference year
        name = f"k{j}"
        d[name] = ((k == j) & d["event_date"].notna()).astype(int)
        xs.append(name)
    r = regress(d, "cost", xs, ["corridor", "period_q"], label=f"standard | {kind}")
    r["k"] = [int(v[1:]) for v in r["variable"]]
    return r


def stacked(kind):
    """Stacked event study: one clean mini-experiment per event."""
    blocks = []
    for n, ev in enumerate(EVENTS[kind].itertuples()):
        start = ev.date + pd.DateOffset(years=LO)
        end = ev.date + pd.DateOffset(years=HI + 1)
        w = cq[(cq["q_end"] >= start) & (cq["q_end"] < end)].copy()
        if w.empty:
            continue
        # clean comparison countries: never on the list during this window
        on_list = w.groupby("iso3")["grey_qend"].max()
        clean = set(on_list[on_list == 0].index)
        w = w[(w["iso3"] == ev.iso3) | w["iso3"].isin(clean)]
        w["treated"] = (w["iso3"] == ev.iso3).astype(int)
        w["k"] = years_since(w["q_end"], ev.date).clip(LO, HI)
        w["event_id"] = n
        blocks.append(w)
    s = pd.concat(blocks, ignore_index=True)
    s["corridor_event"] = s["corridor"] + "|" + s["event_id"].astype(str)
    s["quarter_event"] = s["period_q"] + "|" + s["event_id"].astype(str)
    xs = []
    for j in range(LO, HI + 1):
        if j == -1:
            continue
        name = f"k{j}"
        s[name] = ((s["k"] == j) & (s["treated"] == 1)).astype(int)
        xs.append(name)
    r = regress(s, "cost", xs, ["corridor_event", "quarter_event"], label=f"stacked | {kind}")
    r["k"] = [int(v[1:]) for v in r["variable"]]
    # one summary number: average change after the event (years 0..4+) vs before
    s["after"] = ((s["k"] >= 0) & (s["treated"] == 1)).astype(int)
    summ = regress(s, "cost", ["after"], ["corridor_event", "quarter_event"], label=f"stacked average | {kind}")
    return r, summ, s["event_id"].nunique()


results, summaries = [], []
for kind in ["listing", "removal"]:
    results.append(standard(kind))
    r, summ, n_ev = stacked(kind)
    results.append(r)
    summ["n_events"] = n_ev
    summaries.append(summ)
res = pd.concat(results)
summ = pd.concat(summaries)
pd.concat([res, summ]).to_csv("outputs/tables/table5_event_study.csv", index=False)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print("=" * 100)
print("EVENT STUDY: change in total cost (points) by year relative to the event, vs the year before")
print("  k-3 = 3 or more years before; k0 = the event's first year; k4 = 4 or more years after")
print("=" * 100)
for kind in ["listing", "removal"]:
    print(f"\n--- {kind.upper()}: number of events in the price data = {len(EVENTS[kind])} ---")
    show(res[res["model"].str.endswith(kind)].sort_values(["model", "k"]),
         cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))

print("\n" + "=" * 100)
print("STACKED SUMMARY: average change in total cost in the years after the event (one number per event type)")
print("=" * 100)
show(summ, cols=("model", "variable", "coef", "ci95_lo", "ci95_hi", "p", "mde80", "verdict"))
print("Events used in the stacked design:", dict(zip(summ["model"], summ["n_events"])))

# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, kind, title in zip(axes, ["listing", "removal"], ["Added to the grey list", "Removed from the grey list"]):
    for model, shift, colour, name in [("standard", -0.12, "#8a8a8a", "Standard"), ("stacked", 0.12, "#1f5aa6", "Stacked")]:
        r = res[res["model"] == f"{model} | {kind}"].sort_values("k")
        ks = list(r["k"]) + [-1]
        cs = list(r["coef"]) + [0.0]
        lo = list(r["ci95_lo"]) + [0.0]
        hi = list(r["ci95_hi"]) + [0.0]
        order = np.argsort(ks)
        ks, cs, lo, hi = [np.array(v)[order] for v in (ks, cs, lo, hi)]
        ax.errorbar(ks + shift, cs, yerr=[cs - lo, hi - cs], fmt="o", color=colour, capsize=2, ms=4, label=name)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhspan(-0.5, 0.5, color="#1f5aa6", alpha=0.07, lw=0)
    ax.axvline(-0.5, color="#555555", lw=0.8, ls="--")
    ax.set_xticks(range(LO, HI + 1))
    ax.set_xticklabels(["≤−3", "−2", "−1", "0", "1", "2", "3", "≥4"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Years relative to the event (−1 = reference year)")
axes[0].set_ylabel("Change in total cost (percentage points)")
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("outputs/figures/figure2_event_study.png", dpi=200)
print("\nSaved outputs/figures/figure2_event_study.png")
