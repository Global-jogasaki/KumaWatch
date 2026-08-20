# KumaWatch 🐻

**Benchmarking Wildlife Encounter Prediction for Municipal Decision Support in Northern Japan**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ACM SIGSPATIAL 2026](https://img.shields.io/badge/ACM%20SIGSPATIAL-2026-red.svg)](https://sigspatial.acm.org/)
[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://todalaba.github.io/KumaWatch/)

---

## 🌐 Live Demo

> **View the interactive map in your browser — no server or installation required**

| Link | Description |
|------|-------------|
| [**🗺️ KumaWatch Landing Page**](https://todalaba.github.io/KumaWatch/) | System overview and links to the interactive maps |
| [**▶ Multi-Method Benchmark Map (2025)**](https://todalaba.github.io/KumaWatch/maps/kumawatch_primary_layer.html) | Switch between GLM-Logit / HierBayes / TTM / Extra Trees layers. 365-day date slider, click any cell for a detailed stats panel |

To open locally, open `maps/kumawatch_primary_layer.html` directly in any modern browser (single self-contained HTML file, no external dependencies).

---

## Overview

Human–bear conflicts in northern Japan have escalated sharply, with **publicly available Yamagata Prefecture records reporting 2,655 Asiatic black bear sighting records in 2025**. Municipalities face a daily resource-constrained question: which twenty grid cells should patrols visit today?

**KumaWatch** is an open benchmark and browser-based decision-support prototype comparing eleven wildlife encounter prediction methods under a fixed municipal patrol budget. The central finding is negative and procurement-relevant: a foundation model requiring ~4 hours of API inference is significantly worse than a static prior costing milliseconds, and a 30-minute MCMC pipeline yields no ranking improvement over a sub-30-second logistic regression.

---

## Abstract

We present **KumaWatch**, a cost-annotated top-K benchmark of eleven wildlife encounter prediction methods on two Japanese prefectures (Yamagata 144 cells, Akita 260 cells) over a 365-day held-out year (2025), with measured computational cost reported alongside each predictive-performance estimate.

**Central finding (negative and procurement-relevant):** IBM Granite TTM requires ~4 hours of API inference per day and is significantly *worse* than a static per-cell prior costing milliseconds (Δ = −0.041, *p* = 0.0004). A 30-minute MCMC pipeline (HierBayes) yields no Recall@20 improvement over a sub-30-second logistic regression on Yamagata (*p* = 0.624) and is significantly worse on Akita (*p* = 0.003). On Yamagata — the primary evaluation setting — GLM-Logit's margin over the static prior B1 does not approach significance (+0.014, *p* = 0.155).

The benchmark uses Bonferroni-corrected permutation tests (α = 0.0038 over 13 comparisons, P = 5,000). We additionally release a browser-based decision-support map (no server-side computation, no external API) and argue that cost-annotated top-K benchmarking should precede model selection in municipal geospatial alerting.

---

## Key Contributions

1. **Cost-annotated top-K benchmark** — 11 methods (naive baselines B0–B5, Poisson-GLM, GLM-Logit, HierBayes, Extra Trees, TTM) evaluated on identical 365-day held-out windows across two prefectures, with measured daily computational cost (training, refitting, and inference) beside every Recall@K / Precision@K figure. Bonferroni-corrected permutation tests (α = 0.0038, 13 comparisons).

2. **Negative result with operational reading** — Neither foundation-model inference (~4 h/day) nor MCMC (~30 min/day) buys top-K accuracy over far cheaper alternatives. On Yamagata, GLM-Logit's lead over the static prior is not significant (*p* = 0.155). Extra Trees is strongly miscalibrated (BSS = −1.63). These are null results about ranking only; downstream uses of HierBayes posterior variance and ET environmental covariates are untested rather than refuted.

3. **Browser-based decision-support prototype** — A single-file Leaflet map (no server, no API) serving pre-computed GLM-Logit predictions for all 144 Yamagata cells × 365 days, with four risk tiers and a user-adjustable threshold. Architecture follows directly from Table 1's cost column.

4. **Open benchmark release** — Complete codebase, benchmark data (Yamagata 144 cells + Akita 260 cells), pre-computed score files (GLM-Logit, HierBayes, ET, TTM), and evaluation notebooks under Apache 2.0 and CC-BY 4.0 licenses.

---

## Results Summary

### Yamagata Prefecture (144 cells, 10 km × 10 km)

| Method | Recall@10 | Recall@20 | Recall@30 | Significance vs GLM-Logit (Recall@20) |
|--------|:---------:|:---------:|:---------:|---------------------------------------|
| **GLM-Logit** (best-ranked) | 0.345 | **0.547** | 0.692 | — |
| HierBayes | 0.329 | 0.542 | 0.697 | ns (p = 0.624) |
| B5: Recent MA + Seasonality | 0.333 | 0.534 | 0.660 | ns (p = 0.310) |
| B1: Static Prior | 0.286 | 0.533 | 0.659 | ns (p = 0.155) |
| B4: Static Prior + Seasonality | 0.320 | 0.517 | 0.644 | — |
| **TTM** (IBM Granite 1536-96-R2) | 0.291 | 0.492 | 0.620 | sig. (p < 0.001) |
| B2: Recent Moving Average | 0.311 | 0.486 | 0.607 | — |
| B3: DoY Seasonality | 0.305 | 0.475 | 0.587 | — |
| **Extra Trees** | 0.293 | 0.474 | 0.607 | sig. (p < 0.001) |
| B0: Random | 0.060 | 0.126 | 0.186 | — |
| Poisson-GLM | 0.020 | 0.027 | 0.029 | — |

### Akita Prefecture (260 cells, 10 km × 10 km)

| Method | Recall@10 | Recall@20 | Recall@30 | Significance vs GLM-Logit (Recall@20) |
|--------|:---------:|:---------:|:---------:|---------------------------------------|
| **GLM-Logit** (best-ranked) | 0.259 | **0.454** | 0.587 | — |
| HierBayes | 0.262 | 0.431 | 0.577 | sig. (p = 0.003) |
| B5: Recent MA + Seasonality | 0.261 | 0.427 | 0.568 | ns (p = 0.043, above the Bonferroni-corrected α = 0.0038) |
| B2: Recent Moving Average | 0.265 | 0.418 | 0.538 | — |
| B4: Static Prior + Seasonality | 0.240 | 0.418 | 0.541 | — |
| B1: Static Prior | 0.251 | 0.405 | 0.530 | sig. (p < 0.001) |
| **TTM** (IBM Granite 512-96-R2) | 0.227 | 0.395 | 0.516 | sig. (p < 0.001) |
| B3: DoY Seasonality | 0.215 | 0.352 | 0.451 | — |
| **Extra Trees** | 0.183 | 0.326 | 0.470 | sig. (p < 0.001) |
| B0: Random | 0.047 | 0.080 | 0.114 | — |
| Poisson-GLM | 0.003 | 0.003 | 0.003 | — |

*Bonferroni-corrected permutation tests, α = 0.0038 (0.05 / 13 comparisons, P = 5,000 permutations). "sig." = Bonferroni-significant (p < 0.0038); ns = not significant. Significance tests are computed on Recall@20. On Yamagata (primary setting), GLM-Logit's margin over the static prior B1 is not significant (+0.014, p = 0.155); a method requiring no model, no features and no daily computation is indistinguishable from the best method tested. TTM is significantly worse than B1 (Δ = −0.041, p = 0.0004). GLM-Logit significantly outperforms TTM and Extra Trees on both prefectures.*

*Each row reports a single run per method; no row mixes results from different runs. GLM-Logit and HierBayes Recall@K are recomputed from the released score files in `data/scores/` (`yamagata_glm_logit_scores_2025.npy`, SHA-256 `999e119c167fc22d…`; `yamagata_hier_mean_scores_2025.npy`, SHA-256 `af7b177712a1fa89…`); the released HierBayes file reproduces the paper's Recall@20 = 0.542 exactly, so no MCMC run-to-run mixing remains. Baselines B0–B5 are regenerated deterministically from `notebooks/kumawatch_benchmark.ipynb` Cell 5 (`RAND_SEED = 42`) over the documented training windows (Yamagata from 2018-10-01, Akita from 2022-04-01). TTM Recall@K is recomputed from the released TTM score CSVs. Extra Trees and Poisson-GLM Recall@K are carried over from the archived benchmark run in `notebooks/kumawatch_benchmark_table3_colab.ipynb` (saved cell outputs). One residual discrepancy: the released GLM-Logit score file yields Recall@20 = 0.546 (Yamagata), while the paper reports 0.547; the table keeps the paper's value, and the ±0.001 gap is not resolved by tie-breaking or float64 promotion.*

### Confidence-Filtered Recall@20 (Yamagata — days ranked by prediction confidence)

| Method | Top 25% days | Top 50% days | Top 75% days | All days |
|--------|:------------:|:------------:|:------------:|:--------:|
| HierBayes | 0.875 | **0.639** | 0.555 | 0.542 |
| GLM-Logit | **0.889** | 0.619 | 0.548 | 0.547 |
| TTM | 0.630 | 0.523 | 0.495 | 0.492 |
| B5 | 0.542 | 0.541 | 0.531 | 0.534 |
| B1 | 1.000 | 0.592 | 0.544 | 0.533 |
| B2 | 0.150 | 0.416 | 0.458 | 0.486 |

*The columns other than "All days" are computed on selected subsets of days and are **not** comparable with the all-days figures in the main results tables above; this table is reported separately for that reason. HierBayes exceeds GLM-Logit at the top-50% confidence subset (0.639 vs 0.619). This exploratory analysis concerns subsets of days ranked by prediction confidence and does not validate cell-level uncertainty for graduated alerts; our benchmark scores ranking under a fixed patrol budget only, and does not evaluate those downstream uses. B1's top-25% = 1.000 is a degenerate artifact: constant confidence scores cause argsort tie-breaking to select winter days, which have few valid sighting days — not evidence of genuine uncertainty quantification.*

### Calibration Metrics

| Method | YGT Brier ↓ | YGT BSS ↑ | AKT Brier ↓ | AKT BSS ↑ |
|--------|:-----------:|:---------:|:-----------:|:---------:|
| **GLM-Logit** | 0.034 | 0.08 | 0.041 | 0.28 |
| HierBayes | 0.034 | 0.08 | 0.041 | 0.30 |
| TTM | 0.036 | 0.02 | 0.055 | 0.04 |
| Extra Trees | 0.097 | −1.63 | 0.126 | −1.18 |
| B2: Recent MA | **0.031** | **0.15** | **0.039** | **0.32** |

*BSS (Brier Skill Score) > 0 indicates better calibration than the climatological baseline. B2 achieves the best Brier Skill Score of any method on both prefectures. ET is strongly miscalibrated (BSS = −1.63 on Yamagata), consistent with known behaviour of tree ensembles on probability estimation tasks. HierBayes and GLM-Logit are well-calibrated and similar (BSS ≈ 0.08 YGT, 0.28–0.30 AKT).*

### Cross-Layer Divergence Analysis (Yamagata, Jaccard@20)

| Layer Pair | Jaccard@20 | Interpretation |
|-----------|:----------:|----------------|
| GLM-Logit vs HierBayes | 0.95 | Near-redundant rankings; HierBayes value is in posterior uncertainty (untested here) |
| GLM-Logit vs TTM | 0.55 | Moderate disagreement; does not by itself establish useful complementarity |
| HierBayes vs TTM | 0.50 | Similar level of disagreement |
| GLM-Logit vs Extra Trees | 0.30 | Substantial disagreement; does not establish that ET provides useful complementary information |

*Low Jaccard agreement indicates different rankings, but disagreement alone is not evidence of useful complementarity. HierBayes' posterior variance as a confidence signal for graduated alerts, and ET's environmental covariates as an independent check on a recency-driven prediction, are untested rather than refuted by this benchmark.*

---

## System Architecture

The roles below are **candidate** operational roles represented in the prototype; their downstream effectiveness has not been validated by this ranking benchmark.

```
KumaWatch — Prototype Architecture and Candidate Method Roles:

  BEST-RANKED       GLM-Logit
  METHOD            L2-regularized logistic regression (C=1.0, max_iter=2000)
                    Features: cell fixed effects + rolling30 + log(recent365+1) + sin/cos(DOY) + year_idx
                    Output: daily probability scores per cell → Precision@K / Recall@K ranking

  CANDIDATE         HierBayes
  UNCERTAINTY       Hierarchical Bayesian Poisson (PyMC + NumPyro NUTS)
  SIGNAL            2 chains × 1500 draws (500 tune + 1000 retain)
                    target_accept=0.9, random_seed=42, R̂ < 1.01, zero divergent transitions
                    Output: posterior predictive distributions → candidate confidence
                            signal for future study

  ALTERNATIVE       TTM (IBM Granite Tiny Time Mixers)  +  Extra Trees
  METHODS           TTM 1536-96-R2 (Yamagata) / 512-96-R2 (Akita), zero-shot in-context learning
                    Extra Trees: Nakamoto & Fukazawa [2025] reimplementation
                    Output: alternative risk rankings → disagreement analysis;
                            useful complementarity not established

  VISUALIZATION     Interactive web map (Leaflet.js)
                    10 km grid overlay, 365-day playback, per-method toggle
```

**Grid Definition**: Each prefecture is partitioned into a grid of ~10 km × 10 km cells:
- Yamagata: 9 × 16 = **144 cells**
- Akita: 13 × 20 = **260 cells**

---

## Repository Structure

```
KumaWatch/
├── index.html                             # GitHub Pages landing page (live demo entry point)
├── notebooks/
│   ├── kumawatch_benchmark.ipynb          # Full 11-method benchmark (GLM-Logit, HierBayes, ET, TTM, B0-B5)
│   ├── kumawatch_benchmark_table3_colab.ipynb  # Confidence-filtered Recall@20 (Table 3) + permutation test
│   ├── ttm_yamagata.ipynb                 # TTM inference — Yamagata 144 cells (IBM Granite TTM 1536-96-R2)
│   ├── ttm_akita.ipynb                    # TTM inference — Akita 260 cells (IBM Granite TTM 512-96-R2)
│   └── et_akita.ipynb                     # Extra Trees baseline — Akita (Colab)
├── scripts/
│   ├── et_benchmark_yamagata.py           # Extra Trees benchmark — Yamagata
│   ├── et_benchmark_akita.py              # Extra Trees benchmark — Akita
│   ├── generate_kumawatch_webmap.py       # Three-layer map generator (GLM-Logit, HierBayes, TTM, ET)
│   ├── generate_glm_webmap.py             # GLM-Logit single-layer map generator
│   └── calibration_validation.py          # Post-hoc Platt/Isotonic calibration validation
├── maps/
│   ├── kumawatch_primary_layer.html       # Three-layer interactive web map (2025) — self-contained HTML
│   └── kumawatch_complementary_layer.html # Complementary-layer focused map view
├── data/
│   ├── yamagata_10km_daily_timeseries.csv # 144 cells × daily sightings (Apr 2018–Dec 2025; training from Oct 2018)
│   ├── akita_10km_daily_timeseries.csv    # 260 cells × daily sightings (Apr 2020–Dec 2025; training from Apr 2022)
│   ├── yamagata_10km_grid_coords.csv      # Grid cell coordinates and IDs
│   ├── akita_10km_grid_coords.csv         # Grid cell coordinates and IDs
│   ├── scores/                            # Pre-computed 2025 test-period scores
│   │   ├── yamagata_glm_logit_scores_2025.npy   # GLM-Logit (365 × 144, float32)
│   │   ├── yamagata_hier_mean_scores_2025.npy   # HierBayes posterior mean (365 × 144, float64)
│   │   ├── yamagata_hier_std_scores_2025.npy    # HierBayes posterior std / confidence (365 × 144, float64)
│   │   ├── yamagata_et_scores_2025.csv
│   │   ├── yamagata_ttm_scores_2025.csv
│   │   ├── yamagata_ttm_scores.npy        # NumPy binary format (365 × 144, float32)
│   │   ├── akita_et_scores_2025.csv
│   │   ├── akita_ttm_scores_2025.csv
│   │   └── akita_ttm_scores.npy           # NumPy binary format (365 × 260, float32)
│   └── README.md                          # Data description and provenance
├── README.md
└── LICENSE
```

---

## Dataset

| Dataset | Region | Training Period | Evaluation Period | Cells | Granularity |
|---------|--------|----------------|------------------|-------|-------------|
| Yamagata bear sightings | Yamagata, Japan | 2018-10-01 – 2024-12-31 | 2025-01-01 – 2025-12-31 | 144 | Daily |
| Akita bear sightings | Akita, Japan | 2022-04-01 – 2024-12-31 | 2025-01-01 – 2025-12-31 | 260 | Daily |

Strict temporal separation for model fitting: **all model fitting uses data through 2024-12-31 only**, and all 365 days of 2025 are held out as the test set.

Evaluation is **sequential one-day-ahead forecasting**. For a forecast date *d* in 2025, recency features (`rolling30`, `log(recent365+1)`, and the moving averages behind B2/B5) are computed from sightings observed strictly before *d*, which from late January 2025 onward consist entirely of test-period observations. No observation on or after the forecast date is ever used, and no model coefficients are refit on test-period data. This is the standard sequential-forecasting setup, not label leakage — but note that it does mean test-period observations enter feature computation, which affects GLM-Logit, Poisson-GLM, HierBayes, B2 and B5. B0, B1, B3 and B4 use training-period data only.

Data source: Yamagata and Akita prefectural wildlife observation databases (publicly available).

---

## Evaluation Framework

Metrics are computed under the **Global formulation**: for each day *t*, let S_t be the top-K cells by predicted score. Then:

- **Precision@K** = |S_t ∩ A_t| / K
- **Recall@K** = |S_t ∩ A_t| / |A_t|

where A_t is the set of cells with actual sightings on day *t*. Results are averaged over the 365-day evaluation window (2025).

Statistical significance: **permutation tests** with Bonferroni correction over 13 comparisons (α = 0.0038). Paired bootstrap (B = 5,000) and cell-level bootstrap are used for confidence intervals.

Calibration metrics (per-cell, averaged over cells with ≥1 sighting in evaluation period): **Brier score**, **Brier Skill Score (BSS)**, ECE, MAE, RMSE.

Ranking metrics: per-cell **ROC-AUC**, **PR-AUC**.

Confidence-filtered evaluation: Recall@K computed on the top-N% of days ranked by mean prediction confidence (posterior standard deviation for HierBayes; |score − 0.5| for point-prediction methods).

---

## Getting Started

### Dependencies

```bash
# Core
pip install scikit-learn pandas numpy scipy

# Bayesian layer
pip install pymc numpyro

# Visualization
# Maps are self-contained HTML (no server required)
```

### View the web maps

Open the live demo in your browser — no installation needed:

```
https://todalaba.github.io/KumaWatch/maps/kumawatch_primary_layer.html
```

Or open `maps/kumawatch_primary_layer.html` locally in any modern browser.

### Prerequisite: pre-computed score files

The benchmark notebook (`notebooks/kumawatch_benchmark.ipynb`) requires pre-computed ET and TTM daily scores for 2025. These are included in `data/scores/`:

| File | Model |
|------|-------|
| `data/scores/yamagata_et_scores_2025.csv` | Extra Trees — Yamagata |
| `data/scores/yamagata_ttm_scores_2025.csv` | IBM Granite TTM — Yamagata |
| `data/scores/akita_et_scores_2025.csv` | Extra Trees — Akita |
| `data/scores/akita_ttm_scores_2025.csv` | IBM Granite TTM — Akita |

To **regenerate** these scores from scratch:
- ET scores: run `scripts/et_benchmark_yamagata.py` and `scripts/et_benchmark_akita.py` (see external data requirements below)
- TTM scores: run `notebooks/ttm_yamagata.ipynb` / `notebooks/ttm_akita.ipynb` on Colab (requires IBM watsonx.ai API key)

### Extra Trees: External Data Requirements

The Extra Trees reimplementation (following [Nakamoto & Fukazawa 2025](https://doi.org/10.1007/s41060-025-00866-0)) uses the following external datasets, which are **not included** in this repository and must be obtained separately:

| Dataset | Source | Features Used |
|---------|--------|--------------|
| Land cover | [JAXA ALOS Land Cover](https://www.eorc.jaxa.jp/ALOS/en/dataset/lc_e.htm) | Forest/agricultural/residential area ratio per cell |
| Census data | [Statistics Bureau of Japan](https://www.stat.go.jp/english/data/kokusei/) | Population density, elderly population ratio |
| Weather data | [JMA (Japan Meteorological Agency)](https://www.data.jma.go.jp/gmd/risk/obsdl/) | Daily temperature, precipitation, snow depth |
| Mast indices | [Forestry Agency of Japan](https://www.rinya.maff.go.jp/j/hogo/higai/dounami.html) | Annual acorn abundance (Quercus, Fagus) per region |

Configure the paths to these external files in the header block of `scripts/et_benchmark_yamagata.py` and `scripts/et_benchmark_akita.py` before running. Without these files, only the temporal features (past sightings) will be used, and results will differ from those reported in the paper.

### TTM Model Versions

| Prefecture | Model | Context Length | Forecast Horizon | Notes |
|-----------|-------|:-------------:|:----------------:|-------|
| Yamagata | IBM Granite **TTM 1536-96-R2** | 1,536 days | 96 days | Uses most recent 1,536 days of training data as context (full training span: ~2,284 days) |
| Akita | IBM Granite **TTM 512-96-R2** | 512 days | 96 days | Akita training data < 1,536 days |

Predictions cover the 365-day evaluation period via four overlapping 96-day inference windows. Model weights are available on [Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2). Inference requires an IBM watsonx.ai API key (free tier available).

### Run the benchmark

Open `notebooks/kumawatch_benchmark.ipynb` in Jupyter or Google Colab. Set the score file paths in **Cell 2** (the `★ USER EDIT HERE` block) to point to the pre-computed score files, then run all cells. The notebook includes all 11 methods, evaluation metrics, permutation tests, and calibration analysis.

### Run calibration validation

```bash
python scripts/calibration_validation.py --prefecture yamagata
python scripts/calibration_validation.py --prefecture akita
```

### Generate the web map

```bash
python scripts/generate_kumawatch_webmap.py
```

Outputs `kumawatch_map_2025.html` (self-contained, ~1.5 MB). Requires pre-computed TTM score CSV at the path configured in the script header.

---

## Citation

```bibtex
@inproceedings{jogasaki2026kumawatch,
  author    = {Hiroshi Jogasaki},
  title     = {{KumaWatch}: Benchmarking Wildlife Encounter Prediction for
               Municipal Decision Support in Northern Japan},
  booktitle = {Proceedings of the 34th ACM SIGSPATIAL International Conference on
               Advances in Geographic Information Systems (SIGSPATIAL '26)},
  year      = {2026},
  address   = {Riverside, CA, USA},
  month     = {November}
}
```

---

## License

- **Code**: [Apache License 2.0](LICENSE)
- **Data / Benchmark**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgments

This research was supported by the **Yonezawa City Research Promotion Subsidy** through the Yamagata University Industrial Research Institute (FY2025 Young Researcher Encouragement Grant).

The author thanks the **Yamagata Prefecture Wildlife Management Division** and participating municipalities for cooperation in data preparation and informal operational consultation.

The author gratefully acknowledges **Prof. Yusuke Fukazawa and Mr. Shin Nakamoto** (Sophia University) for their pioneering work on bear encounter prediction, which served as the foundation for the Extra Trees reimplementation in this study.

**IBM Granite TTM** is developed by IBM Research; computational resources were provided through the IBM watsonx.ai free tier. Bear sighting data is provided by Yamagata and Akita prefectural governments (publicly available wildlife observation databases).

---

*Accepted as a short paper in the Applications Track of ACM SIGSPATIAL 2026; camera-ready version in preparation.*
