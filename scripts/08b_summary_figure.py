"""
08b_summary_figure.py
The key-findings figure for the end of the Introduction: the paper's story in one image.

  Left  - "What moves the quoted price?": the main estimates as dots with 95% confidence
          intervals, against the +/-0.5-point band (Section 4.6). Grey-listing is shown with
          the standard estimates and with the estimator robust to heterogeneous effects
          (listings only, step 06m); deposits abroad with the BIS and IMF measures.
  Right - "What the quoted price misses": share of corridor-payout pairs meeting the 5%
          benchmark in 2024, as quoted and in the illustrative high scenario (Table 5).

Every number is read from the tables the earlier steps saved, so nothing is typed by hand.

Needs: outputs/tables/table2_main.csv, table3_channels.csv, table6_robustness.csv,
       table6m_listing_removal.csv, table7_end_to_end.csv
Output: outputs/figures/figure0_key_findings.png

Run from the repo folder:  python scripts/08b_summary_figure.py
"""

import matplotlib
matplotlib.use("Agg")                      # draw to a file (Cloud Shell has no screen)
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402

T = "outputs/tables/"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

t2 = pd.read_csv(T + "table2_main.csv")
t3 = pd.read_csv(T + "table3_channels.csv")
t6m = pd.read_csv(T + "table6m_listing_removal.csv")
t6 = pd.read_csv(T + "table6_robustness.csv")
t7 = pd.read_csv(T + "table7_end_to_end.csv")


def pick(df, model_start, variable, outcome=None):
    r = df[df["model"].str.startswith(model_start) & (df["variable"] == variable)]
    if outcome is not None:
        r = r[r["outcome"] == outcome]
    r = r.iloc[0]
    return r["coef"], r["ci95_lo"], r["ci95_hi"]




def robust(level):
    r = t6m[(t6m["model"] == f"{level} | listings only") & (t6m["what"] == "average")].iloc[0]
    return r["coef"], r["ci95_lo"], r["ci95_hi"]


rows = [  # (label, (coef, lo, hi))
    ("Grey-listed: corridor level", pick(t2, "C", "grey_qend", "cost")),
    ("Grey-listed: service level", pick(t3, "S1", "grey_qend")),
    ("Listing, robust: corridor level", robust("corridor")),
    ("Listing, robust: service level", robust("service")),
    ("Deposits abroad halve: BIS", pick(t2, "C", "L1_log_dep", "cost")),
    ("Deposits abroad halve: IMF", pick(t6, "2c", "deposits abroad halve")),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

# ---- Left: estimates against the +/-0.5 band
y = np.arange(len(rows))[::-1]
ax1.axvspan(-0.5, 0.5, color=BLUE, alpha=0.10, lw=0)
ax1.axvline(0, color=INK, lw=0.8)
for yi, (label, (c, lo, hi)) in zip(y, rows):
    ax1.plot([lo, hi], [yi, yi], color=BLUE, lw=2)
    ax1.plot(c, yi, "o", color=BLUE, ms=7, mec="white", mew=1.5)
    ax1.text(hi + 0.08, yi, f"{c:+.2f}", va="center", fontsize=8.5, color=INK)
ax1.set_yticks(y)
ax1.set_yticklabels([r[0] for r in rows], fontsize=9)
ax1.set_xlim(-2.0, 1.6)
ax1.set_xlabel("Change in total cost of sending USD 200 (percentage points)", fontsize=9)
ax1.text(0, len(rows) - 0.35, "±0.5-point band", ha="center", fontsize=8, color=MUTED)
ax1.set_ylim(-0.7, len(rows) + 0.3)
ax1.set_title("(a) What moves the quoted price?", loc="left", fontsize=10.5)
for side in ("top", "right"):
    ax1.spines[side].set_visible(False)

# ---- Right: share of corridors meeting 5%, quoted vs with unrecorded charges (high scenario)
order = ["Mobile wallet", "Bank account", "Cash pickup", "All payout methods"]
t7 = t7.set_index("payout").loc[order]
x = np.arange(len(order))
w = 0.36
quoted = t7["<= 5% | quoted"].values
high = t7["<= 5% | high"].values
b1 = ax2.bar(x - w / 2 - 0.01, quoted, w, color=BLUE, label="As quoted to the sender")
b2 = ax2.bar(x + w / 2 + 0.01, high, w, color=ORANGE, label="Illustrative: with published estimates\nof unrecorded charges (high)")
for bars in (b1, b2):
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.0f}%",
                 ha="center", fontsize=8.5, color=INK)
ax2.set_xticks(x)
ax2.set_xticklabels(["Mobile\nwallet", "Bank\naccount", "Cash\npickup", "All payout\nmethods"], fontsize=9)
ax2.set_ylabel("Corridor–payout pairs at or below 5%, 2024 (%)", fontsize=9)
ax2.set_ylim(0, 70)
ax2.grid(axis="y", color="#e6e5e0", lw=0.7)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, fontsize=8.5, loc="upper right")
ax2.set_title("(b) What the quoted price misses", loc="left", fontsize=10.5)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("outputs/figures/figure0_key_findings.png", dpi=200)

print("Numbers in the figure (read from the saved tables):")
for label, (c, lo, hi) in rows:
    print(f"  {label:32s} {c:+.2f}  (95% CI {lo:+.2f} to {hi:+.2f})")
print(t7[["<= 5% | quoted", "<= 5% | high"]].round(1).to_string())
print("\nSaved outputs/figures/figure0_key_findings.png")
