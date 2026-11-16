# De-risking, Grey-Listing and the Cost of Remittances to Africa: Corridor-Level Evidence, 2011–2025

**Winnie Sitah Lang'o** · Independent researcher, Nairobi · ORCID [0009-0006-3017-9437](https://orcid.org/0009-0006-3017-9437)

This repository contains the code, cleaned data and results behind the paper of the same title (manuscript under preparation for journal submission). The research was carried out independently, in a personal capacity, using public data only. The views are the author's own.

**Interactive map:** https://sitahlango-maker.github.io/africa-derisking-remittances/ — click any African country to see its grey-list history, remittance prices, banks' balances abroad and progress towards the SDG 10.c cost benchmarks.

## The question and the answer in brief

When African countries are placed on the FATF grey list, or their banks lose correspondent banking depth, does the cost of sending remittances to them rise?

- **Banks' deposits abroad:** a halving changes the quoted cost of sending USD 200 by +0.04 percentage points (95% CI −0.25 to +0.34), an effect statistically smaller than half a percentage point in the main sample.
- **Grey-listing has no consistent effect.** The standard estimate, following individual services over time, is −0.02 points (95% CI −0.23 to +0.18). Estimators that allow the effect to differ across countries and over time are much less precise (+0.24, 95% CI −0.45 to +0.93, averaged over the four years after a change of status) and show wide variation: prices rose after some listings, such as South Africa's and Uganda's, and fell after others, such as Mozambique's and the Democratic Republic of Congo's. An average increase of up to about one percentage point cannot be ruled out, and there is no clear evidence that removal from the list lowers prices.
- **The quoted price is not the whole cost.** In an illustrative calculation adding published global estimates of charges taken by intermediaries or paid by recipients, the share of corridor–payout pairs that meet the 5% SDG benchmark halves, from 34% to 17%.

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
| 6b–6f | `06b_channel_trends.py`, `06c_robustness.py`, `06d_sample_coverage.py`, `06e_services_count.py`, `06f_services_robustness.py` | Additional checks, robustness (Table 6) and the number of services (Online Appendix B) |
| 6g–6j | `06g_wild_bootstrap.py`, `06h_review_checks.py`, `06i_country_trends.py`, `06j_referee_checks.py` | Wild cluster bootstrap, pre-trend sensitivity, provider exit, secondary-source dates, country trends, sender × quarter effects (Table 6, checks 10–15) |
| 6k–6l | `06k_robust_did.py`, `06l_robust_did_countries.py` | Grey-listing estimates robust to heterogeneous effects, service-level event study, and results by country (Section 5.3; Online Appendix C) |
| 7 | `07_event_study.py` | Event studies around listing and removal (Figure 4) |
| 8 | `08_figures.py`, `08b_summary_figure.py` | Prices by channel (Figure 3) and key findings (Figure 1) |
| 9 | `09_end_to_end_sensitivity.py` | Corridor–payout pairs meeting 3% and 5% with unrecorded charges (Table 5) |
| 10–11 | `10_map_data.py`, `11_map_figure.py` | Data for the interactive map; map of countries covered (Figure 2) |

Results are written to `outputs/tables/` and `outputs/figures/`; the console output of each step is kept in `outputs/logs/`. The decision rules (equivalence tests at ±0.5 percentage points, minimum detectable effects) are set out in Section 4.6 of the paper; analyses added after the method was fixed are labelled as additional (not planned) in the code and the paper.

## Citation

Lang'o, W.S. (2026). *De-risking, Grey-Listing and the Cost of Remittances to Africa: Corridor-Level Evidence, 2011–2025*. Working paper. https://github.com/sitahlango-maker/africa-derisking-remittances
