"""
02_fatf_greylist.py
Step 2 of the paper: build the record of African countries on the FATF grey list.

What it does, in plain words:
  1. Holds the list of grey-list "episodes" (country, date added, date removed),
     assembled by hand from FATF plenary statements, 2010-2026.
  2. Saves that list with 'checked' and 'source' columns, recording where each
     date was confirmed (verified against FATF statements on 14 Nov 2026).
  3. Turns the episodes into two tables the analysis needs:
       - yearly : was the country listed at any point in the year, on 31 Dec,
                  and for what share of the year
       - quarterly: was the country listed at the end of each quarter
  4. Prints checks so you can see the result.

How dates are recorded:
  - Dates are FATF plenary dates (when the decision was announced).
  - 2010-12-31 means "already listed before 2011" (exact start not needed,
    because the price data start in 2011).
  - An empty 'removed_at' means the country is still listed.
  - Before 2016 the FATF used two lists (the Public Statement and the
    "Improving global AML/CFT compliance" process). Both count as grey-listed here.

Outputs:
  data/processed/fatf_greylist_africa_episodes.csv
  data/processed/fatf_greylist_africa_yearly.csv
  data/processed/fatf_greylist_africa_quarterly.csv

Run from the repo folder:  python scripts/02_fatf_greylist.py
"""

import pandas as pd

OUT = "data/processed/"
AS_OF = pd.Timestamp("2026-11-13")   # the date this list was last brought up to date

# ---------------------------------------------------------------------------
# 1. The episodes (iso3 code, country, listed_from, removed_at, source)
#    Verified on 14 Nov 2026 against FATF statements. 'checked' says where each
#    date was confirmed: "FATF" = FATF's own statement; "secondary" = the FATF page
#    did not load, so the date was confirmed from an official or press report.
# ---------------------------------------------------------------------------
EPISODES = [
    ("DZA", "Algeria",               "2011-10-28", "2016-02-19", "FATF",      "listed 28 Oct 2011; removed 19 Feb 2016"),
    ("DZA", "Algeria",               "2024-10-25", "2026-06-19", "FATF",      "listed 25 Oct 2024; removed 19 Jun 2026"),
    ("AGO", "Angola",                "2010-12-31", "2016-02-19", "FATF",      "on Feb 2011 list (commitment Feb 2010); removed 19 Feb 2016"),
    ("AGO", "Angola",                "2024-10-25", None,         "FATF",      "listed 25 Oct 2024; still listed"),
    ("MAR", "Morocco",               "2010-12-31", "2013-10-18", "FATF",      "on Feb 2011 list; removed 18 Oct 2013"),
    ("MAR", "Morocco",               "2021-02-25", "2023-02-24", "FATF",      "listed Feb 2021; removed 24 Feb 2023 (date: RBI)"),
    ("NAM", "Namibia",               "2011-06-24", "2015-02-27", "FATF",      "CORRECTED: first listed 24 Jun 2011, not on Feb 2011 list; removed 27 Feb 2015"),
    ("NAM", "Namibia",               "2024-02-23", "2026-06-19", "FATF",      "listed 23 Feb 2024; removed 19 Jun 2026"),
    ("SDN", "Sudan",                 "2010-12-31", "2015-10-23", "FATF",      "on Feb 2011 list; removed 23 Oct 2015"),
    ("ZWE", "Zimbabwe",              "2011-06-24", "2015-02-27", "FATF",      "CORRECTED: first listed 24 Jun 2011, not on Feb 2011 list; removed 27 Feb 2015"),
    ("ZWE", "Zimbabwe",              "2019-10-18", "2022-03-04", "FATF",      "listed 18 Oct 2019; removed Mar 2022 (date 4 Mar 2022: RBI)"),
    ("GHA", "Ghana",                 "2010-12-31", "2013-02-22", "FATF",      "on Feb 2011 list (commitment Oct 2010); removed 22 Feb 2013"),
    ("GHA", "Ghana",                 "2018-10-19", "2021-06-25", "FATF",      "listed 19 Oct 2018; removed Jun 2021"),
    ("TZA", "Tanzania",              "2010-12-31", "2014-06-27", "FATF",      "on Feb 2011 list; removed 27 Jun 2014"),
    ("TZA", "Tanzania",              "2022-10-21", "2025-06-13", "secondary", "listed 21 Oct 2022 (FATF page did not load); removed 13 Jun 2025 (FATF)"),
    ("NGA", "Nigeria",               "2010-12-31", "2013-10-18", "FATF",      "on Feb 2011 list; removed 18 Oct 2013"),
    ("NGA", "Nigeria",               "2023-02-24", "2025-10-24", "secondary", "listed 24 Feb 2023 (RBI); removed 24 Oct 2025 (FATF)"),
    ("ETH", "Ethiopia",              "2010-12-31", "2014-10-24", "FATF",      "on Feb 2011 list (identified Jun 2010); removed 24 Oct 2014"),
    ("ETH", "Ethiopia",              "2017-02-24", "2019-10-18", "FATF",      "commitment Feb 2017 (Jun 2017 statement); removed 18 Oct 2019"),
    ("KEN", "Kenya",                 "2010-12-31", "2014-06-27", "FATF",      "on Feb 2011 list; removed 27 Jun 2014"),
    ("KEN", "Kenya",                 "2024-02-23", None,         "FATF",      "listed 23 Feb 2024; still listed"),
    ("STP", "Sao Tome and Principe", "2010-12-31", "2013-10-18", "FATF",      "on Feb 2011 list; 18 Oct 2013: to continue work with GIABA (left FATF process)"),
    ("UGA", "Uganda",                "2014-02-14", "2017-11-03", "FATF",      "listed 14 Feb 2014; removed 3 Nov 2017"),
    ("UGA", "Uganda",                "2020-02-21", "2024-02-23", "FATF",      "listed 21 Feb 2020; removed 23 Feb 2024"),
    ("TUN", "Tunisia",               "2017-11-03", "2019-10-18", "FATF",      "listed 3 Nov 2017; removed 18 Oct 2019"),
    ("BWA", "Botswana",              "2018-10-19", "2021-10-21", "secondary", "listed 19 Oct 2018 (FATF); removed 21 Oct 2021 (IFC Review)"),
    ("MUS", "Mauritius",             "2020-02-21", "2021-10-21", "secondary", "listed 21 Feb 2020 (FATF); removed 21 Oct 2021 (IFC Review)"),
    ("BFA", "Burkina Faso",          "2021-02-25", "2025-10-24", "FATF",      "listed Feb 2021; removed 24 Oct 2025"),
    ("SEN", "Senegal",               "2021-02-25", "2024-10-25", "FATF",      "listed Feb 2021; removed 25 Oct 2024"),
    ("SSD", "South Sudan",           "2021-06-25", None,         "FATF",      "listed Jun 2021; still listed"),
    ("MLI", "Mali",                  "2021-10-21", "2025-06-13", "secondary", "listed 21 Oct 2021 (IFC Review); removed 13 Jun 2025 (FATF)"),
    ("COD", "Congo, Dem. Rep.",      "2022-10-21", "2026-10-30", "secondary", "listed 21 Oct 2022; removed 30 Oct 2026 (Sumsub; FATF page did not load)"),
    ("MOZ", "Mozambique",            "2022-10-21", "2025-10-24", "secondary", "listed 21 Oct 2022 (FATF page did not load); removed 24 Oct 2025 (FATF)"),
    ("ZAF", "South Africa",          "2023-02-24", "2025-10-24", "secondary", "listed 24 Feb 2023 (RBI); removed 24 Oct 2025 (FATF)"),
    ("CMR", "Cameroon",              "2023-06-23", None,         "FATF",      "listed 23 Jun 2023; still listed"),
    ("CIV", "Cote d'Ivoire",         "2024-10-25", "2026-10-30", "secondary", "listed 25 Oct 2024 (FATF); removed 30 Oct 2026 (Sumsub; FATF page did not load)"),
]

ep = pd.DataFrame(EPISODES, columns=["iso3", "country", "listed_from", "removed_at", "checked", "source"])
ep["listed_from"] = pd.to_datetime(ep["listed_from"])
ep["removed_at"] = pd.to_datetime(ep["removed_at"])
ep.to_csv(OUT + "fatf_greylist_africa_episodes.csv", index=False)

# The end of an open episode is "today" for counting purposes
ep["end"] = ep["removed_at"].fillna(AS_OF + pd.Timedelta(days=1))


def listed_on(iso3, date):
    """1 if the country was on the list on this date, else 0."""
    e = ep[ep["iso3"] == iso3]
    return int(((e["listed_from"] <= date) & (e["end"] > date)).any())


countries = ep.drop_duplicates("iso3").set_index("iso3")["country"]

# ---------------------------------------------------------------------------
# 2. Yearly table, 2010-2026
# ---------------------------------------------------------------------------
rows = []
for iso3 in countries.index:
    for year in range(2010, 2027):
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        e = ep[ep["iso3"] == iso3]
        # days of the year spent on the list (overlap of each episode with the year)
        days = sum(max(0, (min(r.end, end) - max(r.listed_from, start)).days) for r in e.itertuples())
        rows.append({"iso3": iso3, "country": countries[iso3], "year": year,
                     "grey_any": int(days > 0),
                     "grey_31dec": listed_on(iso3, end),
                     "grey_share_of_year": round(min(days / 365, 1), 2)})
yearly = pd.DataFrame(rows)
yearly.to_csv(OUT + "fatf_greylist_africa_yearly.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Quarterly table, 2011 Q1 - 2026 Q3 (status on the last day of each quarter)
# ---------------------------------------------------------------------------
quarter_ends = pd.date_range("2011-03-31", "2026-09-30", freq="QE")
quarterly = pd.DataFrame([{"iso3": iso3, "country": countries[iso3],
                           "year": q.year, "quarter": q.quarter,
                           "grey_qend": listed_on(iso3, q)}
                          for iso3 in countries.index for q in quarter_ends])
quarterly.to_csv(OUT + "fatf_greylist_africa_quarterly.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Checks
# ---------------------------------------------------------------------------
print("--- CHECKS ---")
print(f"Episodes                 : {len(ep)}")
print(f"Countries ever listed    : {ep['iso3'].nunique()}")
print(f"Countries listed twice   : {(ep.groupby('iso3').size() > 1).sum()}")
now = sorted(c for c in countries.index if listed_on(c, AS_OF))
print(f"Listed on {AS_OF.date()}    : {', '.join(countries[c] for c in now)}")
print(f"Checked against FATF    : {(ep['checked'] == 'FATF').sum()} | secondary source: {(ep['checked'] == 'secondary').sum()}")

print("\nCountries on the list at any point in each year:")
print(yearly[yearly["year"].between(2011, 2026)].groupby("year")["grey_any"].sum().to_string())

print("\nTimeline (1 = on the list at some point that year):")
grid = yearly.pivot(index="country", columns="year", values="grey_any").loc[:, 2011:2026]
grid.columns = [str(c)[2:] for c in grid.columns]   # 2011 -> '11' to keep it narrow
print(grid.to_string())