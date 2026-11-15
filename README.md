# De-risking, Grey-Listing and the Cost of Remittances to Africa: Corridor-Level Evidence, 2011–2025

**Winnie Sitah Lang'o** · Independent researcher, Nairobi · ORCID [0009-0006-3017-9437](https://orcid.org/0009-0006-3017-9437)

This repository contains the code, cleaned data and results behind the paper of the same title (manuscript under preparation for journal submission). The research was carried out independently, in a personal capacity, using public data only. The views are the author's own.

**Interactive map:** https://sitahlango-maker.github.io/africa-derisking-remittances/ — click any African country to see its grey-list history, remittance prices, banks' balances abroad and progress towards the SDG 10.c cost benchmarks.

## The question and the answer in brief

When African countries are placed on the FATF grey list, or their banks lose correspondent banking depth, does the cost of sending remittances to them rise?

- Following individual services over time, grey-listing changes the quoted cost of sending USD 200 by −0.02 percentage points (95% CI −0.23 to +0.18). A halving of banks' deposits abroad changes it by +0.04 points (95% CI −0.25 to +0.34). Both are statistically smaller than half a percentage point, and removal from the list is not followed by lower prices.
- The quoted price is not the whole cost. Adding published estimates of charges taken by intermediaries or paid by recipients can halve the share of corridors that meet the 5% SDG benchmark, from 34% to 17%.

## Data (all public)

| Source | Used for | Where to get it |
| --- | --- | --- |
| World Bank, Remittance Prices Worldwide (2011–2025 Q3) | Prices of sending USD 200 and USD 500 | [Data catalog](https://datacatalog.worldbank.org/search/dataset/0037898/remittance-prices-worldwide) (file `rpw_dataset_2011_2025_q3.xlsx`) |
| FATF statements, 2010–2026 | Grey-list episodes (hand-built, each date checked; sources recorded in `scripts/02_fatf_greylist.py`) | [FATF](https://www.fatf-gafi.org/en/publications/High-risk-and-other-monitored-jurisdictions.html) |
| BIS Locational Banking Statistics | Banks' deposits abroad (correspondent depth) | [BIS bulk download](https://data.bis.org/bulkdownload) (`WS_LBS_D_PUB_csv_col.zip`) |
| IMF Monetary and Financial Statistics (other depository corporations) | Commercial banks' foreign assets | IMF data API (see `scripts/04d_imf_odc.py`) |
| World Bank World Development Indicators | Remittances, reserves, GDP per capita | WDI API (see `scripts/04a_wdi.py`) |
| World Bank Global Findex 2025 | Account and mobile money ownership | [Findex](https://www.worldbank.org/en/publication/globalfindex/download-data) |
| World Bank FCS lists, FY2006–FY2026 | Fragility status | [FCS lists](https://thedocs.worldbank.org/en/doc/aa5bc1b55f2c7d72c157de331ab71f30-0090082026/original/C2-FCSList-FY06toFY26.pdf) |
| Natural Earth (1:110m) | Country boundaries for the map | [Natural Earth](https://www.naturalearthdata.com/) (public domain) |

Raw downloads go in `data/raw/` and are not included here; each source keeps its own terms of use. The cleaned and processed files produced by the scripts are in `data/processed/`.

## How to reproduce

Python 3 with pandas, numpy, matplotlib and openpyxl. Run the scripts in order from the repository folder:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `01_clean_rpw.py` | Cleans the RPW prices; keeps African receiving countries; flags outliers |
| 2 | `02_fatf_greylist.py` | Builds the grey-list record (36 episodes, verified) |
| 3 | `03_bis.py` | Extracts African banks' positions from the BIS statistics |
| 4a–4d | `04a_wdi.py`, `04b_findex.py`, `04c_fcs.py`, `04d_imf_odc.py` | Controls and the IMF cross-check |
| 5 | `05_build_panels.py` | Builds the service, corridor-quarter and country-year panels; Table 1 |
| 6 | `06_models.py` (+ `fe.py`) | Main results: Tables 2–4 |
| 6b–6f | `06b_channel_trends.py`, `06c_robustness.py`, `06d_sample_coverage.py`, `06e_services_count.py`, `06f_services_robustness.py` | Additional checks, robustness (Table 6) and the number of services (Table 9) |
| 7 | `07_event_study.py` | Event studies around listing and removal (Figure 3) |
| 8 | `08_figures.py`, `08b_summary_figure.py` | Prices by channel (Figure 2) and key findings (Figure 1) |
| 9 | `09_end_to_end_sensitivity.py` | Corridors meeting 3% and 5% with unrecorded charges (Table 7) |
| 10–11 | `10_map_data.py`, `11_map_figure.py` | Data for the interactive map; Map 1 |

Results are written to `outputs/tables/` and `outputs/figures/`; the console output of each step is kept in `outputs/logs/`. The decision rules (equivalence tests at ±0.5 percentage points, minimum detectable effects) are set out in Section 4.6 of the paper; analyses added after the method was fixed are labelled as additional in the code and the paper.

## Citation

Lang'o, W.S. (2026). *De-risking, Grey-Listing and the Cost of Remittances to Africa: Corridor-Level Evidence, 2011–2025*. Working paper. https://github.com/sitahlango-maker/africa-derisking-remittances
