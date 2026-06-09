# KumaWatch 🐻

**A Multi-Method Wildlife Encounter Alert System toward Operational Municipal Deployment in Northern Japan**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ACM SIGSPATIAL 2026](https://img.shields.io/badge/ACM%20SIGSPATIAL-2026-red.svg)](https://sigspatial.acm.org/)
[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://global-jogasaki.github.io/KumaWatch/)

---

## 🌐 Live Demo

> **View the interactive map in your browser — no server or installation required**

| Link | Description |
|------|-------------|
| [**🗺️ KumaWatch Landing Page**](https://global-jogasaki.github.io/KumaWatch/) | System overview and links to the interactive maps |
| [**▶ Three-Layer Prediction Map (2025)**](https://global-jogasaki.github.io/KumaWatch/maps/kumawatch_primary_layer.html) | Switch between GLM-Logit / HierBayes / TTM / Extra Trees layers. 365-day date slider, click any cell for a detailed stats panel |

To open locally, open `maps/kumawatch_primary_layer.html` directly in any modern browser (single self-contained HTML file, no external dependencies).

---

## Overview

Human–bear conflicts in northern Japan have escalated sharply, with **publicly available Yamagata Prefecture records reporting 2,655 Asiatic black bear sighting records in 2025**. Municipalities face the challenge of allocating limited patrol resources across large geographic areas under high daily uncertainty.

**KumaWatch** is a deployable web-based decision-support system combining three complementary modeling layers to predict daily bear encounter risk across grid cells in Yamagata and Akita Prefectures, Japan.

---

## Abstract

We present **KumaWatch**, a multi-method wildlife encounter alert system designed for operational municipal deployment in northern Japan. The system integrates three complementary modeling layers:

1. **Primary Layer — GLM-Logit**: L2-regularized logistic regression combining cell-level fixed effects with temporal dynamics (rolling 30-day, log-annual, and seasonal harmonics). Evaluated on 365 days in 2025 across 144 cells (Yamagata) and 260 cells (Akita).

2. **Uncertainty Layer — HierBayes**: Hierarchical Bayesian Poisson model (PyMC + NumPyro) quantifying predictive uncertainty across grid cells. Enables a *proposed* graduated alert protocol: restricting alerts to the top-50% confidence subset raises Recall@20 from 0.542 to **0.639** on Yamagata. *The operational effectiveness of this protocol has not yet been validated in live deployment.*

3. **Complementary Layer — TTM + Extra Trees**: IBM Granite Tiny Time Mixers (zero-shot in-context learning) and Extra Trees (following Nakamoto & Fukazawa 2025) provide independent signal sources for cross-validation and operational auditability.

We benchmark **11 methods** in total (6 naive baselines B0–B5, Poisson-GLM, GLM-Logit, HierBayes, Extra Trees, TTM) using permutation tests with Bonferroni correction (α = 0.0038 over 13 comparisons). GLM-Logit achieves **Recall@20 = 0.547** (Yamagata) and **0.454** (Akita), statistically significantly outperforming IBM Granite TTM and Extra Trees (*p* < 0.001). Naive baselines (Static Prior, Recent Moving Average, and their seasonal augmentations) are competitive on Recall@K but lack uncertainty quantification for graduated response and lack independent signal sources for cross-validation auditability.

Cross-layer analysis (Jaccard@20: Primary vs TTM = 0.55, Primary vs ET = 0.30) confirms that TTM and Extra Trees capture partially distinct spatial patterns, motivating their retention as complementary audit layers.

We release the complete benchmark codebase, multi-layer web map, and dataset under permissive licenses to support reproducibility and future municipal deployments.

---

## Key Contributions

1. **Three-layer operational architecture** — GLM-Logit (primary precision), HierBayes (uncertainty quantification + graduated alerts), TTM + Extra Trees (complementary audit layers), integrated into a Leaflet.js web decision-support map.

2. **Rigorous 11-method benchmark** — Head-to-head comparison of statistical, Bayesian, tree-ensemble, and time series foundation models on identical 365-day evaluation windows across two prefectures, using resource-aware metrics (Precision@K, Recall@K) and Bonferroni-corrected permutation tests.

3. **Proposed graduated alert protocol** — HierBayes uncertainty quantification enables dynamic confidence-based filtering, raising Recall@20 from 0.542 (all-days) to **0.639** (top-50% confidence days). *Not yet validated in live operational deployment.*

4. **Cross-layer divergence analysis** — Jaccard@20 decomposition reveals that GLM-Logit, TTM, and Extra Trees capture partially non-overlapping spatial risk signals, supporting a multi-method ensemble rather than single-model deployment.

5. **Open benchmark release** — Complete codebase, benchmark data (Yamagata 144 cells + Akita 260 cells), interactive web maps, and evaluation scripts under Apache 2.0 and CC-BY 4.0 licenses.

---

## Results Summary

### Yamagata Prefecture (144 cells, 10 km × 10 km)

| Method | Recall@20 | Significance vs GLM-Logit |
|--------|:---------:|--------------------------|
| **GLM-Logit** (Primary) | **0.547** | — |
| HierBayes (top-50% conf. days) | **0.639** | — (uncertainty layer; different metric) |
| HierBayes (all days) | 0.542 | ns (p = 0.624) |
| B5: Recent MA + Seasonality | 0.534 | ns (p = 0.310) |
| B1: Static Prior | 0.533 | ns (p = 0.155) |
| **TTM** (IBM Granite 1536-96-R2) | 0.492 | sig. (p < 0.001) |
| B2: Recent Moving Average | 0.486 | — |
| **Extra Trees** | 0.474 | sig. (p < 0.001) |
| Poisson-GLM | 0.027 | — |

### Akita Prefecture (260 cells, 10 km × 10 km)

| Method | Recall@20 | Significance vs GLM-Logit |
|--------|:---------:|--------------------------|
| **GLM-Logit** (Primary) | **0.454** | — |
| HierBayes (all days) | 0.431 | sig. (p = 0.003) |
| B5: Recent MA + Seasonality | 0.427 | ns (p = 0.043, below Bonferroni α) |
| B2: Recent Moving Average | 0.418 | — |
| B1: Static Prior | 0.405 | sig. (p < 0.001) |
| **TTM** (IBM Granite 512-96-R2) | 0.395 | sig. (p < 0.001) |
| **Extra Trees** | 0.326 | sig. (p < 0.001) |
| Poisson-GLM | 0.003 | — |

*Bonferroni-corrected permutation tests, α = 0.0038 (0.05 / 13 comparisons, P = 5,000 permutations). "sig." = Bonferroni-significant (p < 0.0038); ns = not significant. GLM-Logit significantly outperforms TTM and Extra Trees on both prefectures. Comparisons against naive baselines (B1, B5) are significant on Akita but not on Yamagata — a key motivation for KumaWatch's multi-layer architecture.*

### Confidence-Filtered Recall@20 (Yamagata — days ranked by prediction confidence)

| Method | Top 25% days | Top 50% days | Top 75% days | All days |
|--------|:------------:|:------------:|:------------:|:--------:|
| HierBayes | 0.875 | **0.639** | 0.555 | 0.542 |
| GLM-Logit | **0.889** | 0.619 | 0.548 | 0.547 |
| TTM | 0.630 | 0.523 | 0.495 | 0.492 |
| B5 | 0.542 | 0.541 | 0.531 | 0.534 |
| B1 | 1.000 | 0.592 | 0.544 | 0.533 |
| B2 | 0.150 | 0.416 | 0.458 | 0.486 |

*HierBayes is the only method that exceeds GLM-Logit at the operationally critical top-50% confidence subset (0.639 vs 0.619), validating its role as the uncertainty quantification layer for graduated alert protocols. B1's top-25% = 1.000 reflects concentration of correct predictions on high-frequency cells, not genuine uncertainty quantification.*

### Calibration Metrics

| Method | YGT Brier ↓ | YGT BSS ↑ | AKT Brier ↓ | AKT BSS ↑ |
|--------|:-----------:|:---------:|:-----------:|:---------:|
| **GLM-Logit** | 0.034 | 0.08 | 0.041 | 0.28 |
| HierBayes | 0.034 | 0.08 | 0.041 | 0.30 |
| TTM | 0.036 | 0.02 | 0.055 | 0.04 |
| Extra Trees | 0.097 | −1.63 | 0.126 | −1.18 |
| B2: Recent MA | **0.031** | **0.15** | **0.039** | **0.32** |

*BSS (Brier Skill Score) > 0 indicates better calibration than climatological baseline. B2 achieves the best Brier score by construction (predicted probability ≈ recent observed rate), but produces only point estimates with no uncertainty quantification — the key reason HierBayes is chosen as the uncertainty layer despite a similar Brier score to GLM-Logit.*

### Cross-Layer Divergence Analysis (Yamagata, Jaccard@20)

| Layer Pair | Jaccard@20 | Interpretation |
|-----------|:----------:|----------------|
| GLM-Logit vs HierBayes | 0.95 | Near-redundant rankings; HierBayes value is in uncertainty quantification |
| GLM-Logit vs TTM | 0.55 | Moderate disagreement; useful cross-validation |
| HierBayes vs TTM | 0.50 | Similar to primary vs TTM |
| GLM-Logit vs Extra Trees | 0.30 | Substantial disagreement; ET captures spatial features primary layer ignores |

---

## System Architecture

```
KumaWatch — Three-Layer Alert System:

  PRIMARY LAYER     GLM-Logit
  ─────────────     L2-regularized logistic regression (C=1.0, max_iter=2000)
                    Features: cell fixed effects + rolling30 + log(recent365+1) + sin/cos(DOY) + year_idx
                    Output: daily probability scores per cell → Precision@K / Recall@K alerts

  UNCERTAINTY       HierBayes
  LAYER             Hierarchical Bayesian Poisson (PyMC + NumPyro NUTS)
                    2 chains × 1500 draws (500 tune + 1000 retain)
                    target_accept=0.9, random_seed=42, R̂ < 1.01, zero divergent transitions
                    Output: posterior predictive distributions → graduated alert strategy

  COMPLEMENTARY     TTM (IBM Granite Tiny Time Mixers)  +  Extra Trees
  LAYER             TTM 1536-96-R2 (Yamagata) / 512-96-R2 (Akita), zero-shot in-context learning
                    Extra Trees: Nakamoto & Fukazawa [2025] reimplementation
                    Output: independent risk rankings → cross-validation / auditability

  VISUALIZATION     Interactive web map (Leaflet.js)
                    10 km grid overlay, 365-day playback, per-layer toggle
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
│   ├── yamagata_10km_daily_timeseries.csv # 144 cells × daily sightings (Apr 2018–2025)
│   ├── akita_10km_daily_timeseries.csv    # 260 cells × daily sightings (Apr 2020–2025)
│   ├── yamagata_10km_grid_coords.csv      # Grid cell coordinates and IDs
│   ├── akita_10km_grid_coords.csv         # Grid cell coordinates and IDs
│   ├── scores/                            # Pre-computed 2025 test-period scores
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

Strict temporal separation: all 365 days of 2025 are held out as the test set. No test-period data is used in model training or feature computation.

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
https://global-jogasaki.github.io/KumaWatch/maps/kumawatch_primary_layer.html
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
  title     = {{KumaWatch}: A Multi-Method Wildlife Encounter Alert System toward
               Operational Municipal Deployment in Northern Japan},
  booktitle = {Proceedings of the 34th ACM SIGSPATIAL International Conference on
               Advances in Geographic Information Systems (SIGSPATIAL '26)},
  year      = {2026},
  address   = {Riverside, CA, USA},
  month     = {November},
  note      = {[Applications]}
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

*Paper submitted to ACM SIGSPATIAL 2026 — Applications Track*
