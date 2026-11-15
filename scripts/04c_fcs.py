"""
04c_fcs.py
Step 4c of the paper: which African countries were "fragile or conflict-affected" each year,
from the World Bank's historical FCS lists (FY2006-FY2026, one PDF).

What it does, in plain words:
  1. Turns the PDF into plain text.
  2. Walks down the text. Each time it meets a heading for a new fiscal year's list
     (e.g. "FY14 List"), it starts collecting the countries named under it.
  3. Matches country names to three-letter codes. The PDF spells some names differently
     over the years (e.g. "Congo, Dem. Rep." / "DRC", "Swaziland" / "Eswatini"), so each
     country has a list of accepted spellings.
  4. Builds a table: country x fiscal year, fcs = 1 if on the list, 0 if not.
  5. Prints each year's list so you can check it against the PDF by eye.

About fiscal years: a World Bank fiscal year runs July to June and is named after the year
it ends in (FY2014 = July 2013 to June 2014). Here FY2014 is matched to calendar 2014.
Since FY2027 the World Bank publishes new lists instead of the FCS list, so 2026 is the last year.

Input : data/raw/C2-FCSList-FY06toFY26.pdf
Output: data/processed/fcs_africa.csv

Run from the repo folder:  python scripts/04c_fcs.py
"""

import re
import shutil
import subprocess
import unicodedata

import pandas as pd

RAW = "data/raw/C2-FCSList-FY06toFY26.pdf"
OUT = "data/processed/fcs_africa.csv"

# ---------------------------------------------------------------------------
# 1. PDF -> text with the 'pdftotext' tool, keeping the table layout (-layout).
#    If it is missing, install it in Cloud Shell with: sudo apt-get install -y poppler-utils
# ---------------------------------------------------------------------------
if not shutil.which("pdftotext"):
    raise SystemExit("Stopped: 'pdftotext' is not installed. Run: sudo apt-get install -y poppler-utils")
text = subprocess.run(["pdftotext", "-layout", RAW, "-"], capture_output=True, text=True).stdout


def norm(s):
    """Lower-case, drop accents and anything that is not a letter, so spellings compare cleanly."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


# ---------------------------------------------------------------------------
# 3. Accepted spellings for each African country
# ---------------------------------------------------------------------------
SPELLINGS = {
    "AGO": ["Angola"], "BEN": ["Benin"], "BWA": ["Botswana"], "BFA": ["Burkina Faso"], "BDI": ["Burundi"],
    "CPV": ["Cabo Verde", "Cape Verde"], "CMR": ["Cameroon"],
    "CAF": ["Central African Republic", "CAR", "Central African Rep.", "Central Afr. Rep."], "TCD": ["Chad"],
    "COM": ["Comoros"],
    "COG": ["Congo, Rep", "Congo, Rep.", "Congo, Republic of", "Republic of Congo", "Congo Republic"],
    "COD": ["Congo, Dem. Rep.", "Congo, Dem. Rep", "Congo, Democratic Republic of",
            "Democratic Republic of Congo", "Congo DR", "DRC", "Dem. Rep. of Congo", "Congo, DR"],
    "CIV": ["Cote d'Ivoire", "Côte d'Ivoire", "Côte dIvoire"], "DJI": ["Djibouti"],
    "GNQ": ["Equatorial Guinea"], "ERI": ["Eritrea"], "SWZ": ["Eswatini", "Swaziland"], "ETH": ["Ethiopia"],
    "GAB": ["Gabon"], "GMB": ["Gambia, The", "The Gambia", "Gambia"], "GHA": ["Ghana"], "GIN": ["Guinea"],
    "GNB": ["Guinea-Bissau", "Guinea Bissau"], "KEN": ["Kenya"], "LSO": ["Lesotho"], "LBR": ["Liberia"],
    "MDG": ["Madagascar"], "MWI": ["Malawi"], "MLI": ["Mali"], "MRT": ["Mauritania"], "MUS": ["Mauritius"],
    "MOZ": ["Mozambique"], "NAM": ["Namibia"], "NER": ["Niger"], "NGA": ["Nigeria"], "RWA": ["Rwanda"],
    "STP": ["São Tomé and Príncipe", "Sao Tome and Principe", "Sao Tome and Pr", "Sao Tome & Principe"],
    "SEN": ["Senegal"], "SYC": ["Seychelles"], "SLE": ["Sierra Leone"], "SOM": ["Somalia"],
    "ZAF": ["South Africa"], "SSD": ["South Sudan"], "SDN": ["Sudan"], "TZA": ["Tanzania"], "TGO": ["Togo"],
    "UGA": ["Uganda"], "ZMB": ["Zambia"], "ZWE": ["Zimbabwe"], "DZA": ["Algeria"],
    "EGY": ["Egypt", "Egypt, Arab Rep."], "LBY": ["Libya"], "MAR": ["Morocco"], "TUN": ["Tunisia"],
}
lookup = {norm(name): iso3 for iso3, names in SPELLINGS.items() for name in names}

# ---------------------------------------------------------------------------
# 2. Walk down the text, year by year
# ---------------------------------------------------------------------------
found, fy = [], None
for line in text.splitlines():
    heading = re.search(r"FY ?(\d{2})", line)
    # a list heading names a fiscal year AND says list / LICUS / fragile states (not the "premium" tables)
    if heading and re.search(r"(?i)list|licus|fragile states", line) and not re.search(r"(?i)premium", line):
        fy = 2000 + int(heading.group(1))
    if fy is None:
        continue
    for cell in re.split(r"\s{2,}", line.strip()):          # table columns are separated by 2+ spaces
        cell = re.sub(r"\s+\d[\d.]*.*$", "", cell)          # drop trailing numbers (scores)
        cell = re.sub(r"\(.*?\)|\b[a-e]/", "", cell).strip()  # drop notes in brackets and footnote marks
        iso3 = lookup.get(norm(cell))
        if iso3:
            found.append({"iso3": iso3, "fiscal_year": fy})

listed = pd.DataFrame(found).drop_duplicates()

# ---------------------------------------------------------------------------
# 4. Full table: every African country x every fiscal year
# ---------------------------------------------------------------------------
years = sorted(listed["fiscal_year"].unique())
grid = pd.MultiIndex.from_product([sorted(SPELLINGS), years], names=["iso3", "fiscal_year"]).to_frame(index=False)
grid = grid.merge(listed.assign(fcs=1), how="left").fillna({"fcs": 0})
grid["fcs"] = grid["fcs"].astype(int)
grid["year"] = grid["fiscal_year"]          # FY2014 -> calendar 2014 (see note at top)
grid.to_csv(OUT, index=False)

# ---------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------
print(f"Saved {OUT}")
print("\n--- CHECKS ---")
print(f"Fiscal years found : {years[0]} to {years[-1]} ({len(years)} lists)")
print(f"African countries ever on the list: {listed['iso3'].nunique()}")
print("\nAfrican countries on each year's list (check a few against the PDF):")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 250)
summary = listed.groupby("fiscal_year")["iso3"].apply(lambda s: f"{len(s):2d}  " + " ".join(sorted(s)))
print(summary.to_string())
