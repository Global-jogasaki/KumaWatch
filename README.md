# KumaWatch 🐻

**A Multi-Method Wildlife Encounter Alert System for Operational Municipal Deployment in Northern Japan**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ACM SIGSPATIAL 2026](https://img.shields.io/badge/ACM%20SIGSPATIAL-2026-red.svg)](https://sigspatial.acm.org/)
[![GitHub Pages](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-brightgreen.svg)](https://global-jogasaki.github.io/KumaWatch/)

---

## 🌐 Live Demo

> **マップをブラウザで今すぐ確認できます — サーバー不要**

| リンク | 内容 |
|--------|------|
| [**🗺️ KumaWatch ランディングページ**](https://global-jogasaki.github.io/KumaWatch/) | システム概要・マップへのリンク |
| [**▶ 三層統合予測マップ（2025年）**](https://global-jogasaki.github.io/KumaWatch/maps/kumawatch_primary_layer.html) | GLM-Logit / HierBayes / TTM / Extra Trees を切り替え表示。365日スライダー、セルクリックで詳細表示 |

ローカルで開く場合は `maps/kumawatch_primary_layer.html` をブラウザで直接開いてください（単一 HTML ファイル、外部依存なし）。

---

## Overview

Human–bear conflicts in northern Japan have escalated dramatically, with **Yamagata Prefecture recording 2,655 bear sightings in 2025**—a 745.5% increase from the previous year. Municipalities face the challenge of allocating limited patrol resources across large geographic areas under high daily uncertainty.

**KumaWatch** (熊 Watch) is a deployable web-based decision-support system combining three complementary modeling layers to predict daily bear encounter risk across grid cells in Yamagata and Akita Prefectures, Japan.

---

## Abstract

We present **KumaWatch**, a multi-method wildlife encounter alert system designed for operational municipal deployment in northern Japan. The system integrates three complementary modeling layers:

1. **Primary Layer — GLM-Logit**: L2-regularized logistic regression combining cell-level fixed effects with temporal dynamics (rolling 30-day, log-annual, and seasonal harmonics). Evaluated on 365 days in 2025 across 144 cells (Yamagata) and 260 cells (Akita).

2. **Uncertainty Layer — HierBayes**: Hierarchical Bayesian Poisson model (PyMC + NumPyro) quantifying predictive uncertainty across grid cells. Enables a *graduated alert strategy*: restricting alerts to the top-50% confidence subset raises Recall@20 from 0.542 to **0.639** on Yamagata.

3. **Complementary Layer — TTM + Extra Trees**: IBM Granite Tiny Time Mixers (zero-shot in-context learning) and Extra Trees (following Nakamoto & Fukazawa 2025) provide independent signal sources for cross-validation and operational auditability.

We benchmark **11 methods** in total (6 naive baselines B0–B5, Poisson-GLM, GLM-Logit, HierBayes, Extra Trees, TTM) using permutation tests with Bonferroni correction (α = 0.0038 over 13 comparisons). GLM-Logit achieves **Recall@20 = 0.547** (Yamagata) and **0.454** (Akita), statistically significantly outperforming all baselines, TTM, and Extra Trees (*p* < 0.001).

Cross-layer analysis (Jaccard@20: Primary vs TTM = 0.55, Primary vs ET = 0.30) confirms that TTM and Extra Trees capture partially distinct spatial patterns, motivating their retention as complementary audit layers.

We release the complete benchmark codebase, multi-layer web map, and dataset under permissive licenses to support reproducibility and future municipal deployments.

---

## Key Contributions

1. **Three-layer operational architecture** — GLM-Logit (primary precision), HierBayes (uncertainty quantification + graduated alerts), TTM + Extra Trees (complementary audit layers), integrated into a Leaflet.js web decision-support map.

2. **Rigorous 11-method benchmark** — Head-to-head comparison of statistical, Bayesian, tree-ensemble, and time series foundation models on identical 365-day evaluation windows across two prefectures, using resource-aware metrics (Precision@K, Recall@K) and Bonferroni-corrected permutation tests.

3. **Graduated alert strategy** — HierBayes uncertainty quantification enables dynamic confidence-based filtering, raising Recall@20 from 0.542 (all-days) to **0.639** (top-50% confidence days).

4. **Cross-layer divergence analysis** — Jaccard@20 decomposition reveals that GLM-Logit, TTM, and Extra Trees capture partially non-overlapping spatial risk signals, supporting a multi-method ensemble rather than single-model deployment.

5. **Open benchmark release** — Complete codebase, benchmark data (Yamagata 144 cells + Akita 260 cells), interactive web maps, and evaluation scripts under Apache 2.0 and CC-BY 4.0 licenses.

---

## Results Summary

### Yamagata Prefecture (144 cells, 10 km × 10 km)

| Method | Recall@10 | Recall@20 | Recall@30 | Notes |
|--------|-----------|-----------|-----------|-------|
| **GLM-Logit** | **—** | **0.547** | **—** | Primary layer; *p* < 0.001 vs all |
| HierBayes (top-50% conf.) | — | **0.639** | — | Graduated alert strategy |
| HierBayes (all days) | — | 0.542 | — | Uncertainty layer |
| TTM | — | — | — | Complementary layer |
| Extra Trees | — | — | — | Complementary layer |
| Best naive baseline (B0–B5) | — | < GLM | — | |

### Akita Prefecture (260 cells, 10 km × 10 km)

| Method | Recall@20 | Notes |
|--------|-----------|-------|
| **GLM-Logit** | **0.454** | Primary layer |
| TTM | — | Complementary layer |
| Extra Trees | — | Complementary layer |

*GLM-Logit achieves statistically significant improvement over all 10 competing methods on all 13 Precision@K and Recall@K comparisons (Bonferroni-corrected permutation tests, α = 0.0038).*

*Cross-layer Jaccard@20: GLM-Logit vs TTM = 0.55; GLM-Logit vs Extra Trees = 0.30.*

---

## System Architecture

```
KumaWatch — Three-Layer Alert System:

  PRIMARY LAYER     GLM-Logit
  ─────────────     L2-regularized logistic regression
                    Features: cell fixed effects + rolling30 + log(annual) + sin/cos(DOY) + year_idx
                    Output: daily probability scores per cell → Precision@K / Recall@K alerts

  UNCERTAINTY       HierBayes
  LAYER             Hierarchical Bayesian Poisson (PyMC + NumPyro)
                    2 chains × 1500 draws (500 warm-up)
                    Output: posterior predictive distributions → graduated alert strategy

  COMPLEMENTARY     TTM (IBM Granite Tiny Time Mixers)  +  Extra Trees
  LAYER             Zero-shot in-context learning           Nakamoto & Fukazawa [2025] reimplementation
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
│   ├── ttm_yamagata.ipynb                 # TTM inference — Yamagata 144 cells
│   ├── ttm_akita.ipynb                    # TTM inference — Akita 260 cells
│   └── et_akita.ipynb                     # Extra Trees baseline — Akita (Colab)
├── scripts/
│   ├── et_benchmark_yamagata.py           # Extra Trees benchmark — Yamagata
│   ├── et_benchmark_akita.py              # Extra Trees benchmark — Akita
│   ├── generate_kumawatch_webmap.py       # Three-layer map generator (all 4 models)
│   └── calibration_validation.py          # Post-hoc Platt/Isotonic calibration validation
├── maps/
│   └── kumawatch_primary_layer.html       # Three-layer interactive web map (2025) — self-contained HTML
├── data/
│   ├── yamagata_10km_daily_timeseries.csv # 144 cells × daily sightings (Oct 2018–2025)
│   ├── akita_10km_daily_timeseries.csv    # 260 cells × daily sightings (Apr 2022–2025)
│   ├── yamagata_10km_grid_coords.csv      # Grid cell coordinates and IDs
│   ├── akita_10km_grid_coords.csv         # Grid cell coordinates and IDs
│   ├── scores/                            # Pre-computed 2025 test-period scores
│   │   ├── yamagata_et_scores_2025.csv
│   │   ├── yamagata_ttm_scores_2025.csv
│   │   ├── akita_et_scores_2025.csv
│   │   └── akita_ttm_scores_2025.csv
│   └── README.md                          # Data description and provenance
├── README.md
└── LICENSE
```

---

## Dataset

| Dataset | Region | Period | Cells | Granularity |
|---------|--------|--------|-------|-------------|
| Yamagata bear sightings | Yamagata, Japan | Oct 2018 – Dec 2025 | 144 | Daily |
| Akita bear sightings | Akita, Japan | Apr 2022 – Dec 2025 | 260 | Daily |

Data source: Yamagata and Akita prefectural wildlife observation databases (publicly available).

---

## Evaluation Framework

Metrics are computed under the **Global formulation**: for each day *t*, let S_t be the top-K cells by predicted score. Then:

- **Precision@K** = |S_t ∩ A_t| / K
- **Recall@K** = |S_t ∩ A_t| / |A_t|

where A_t is the set of cells with actual sightings on day *t*. Results are averaged over the 365-day evaluation window (2025).

Statistical significance: **permutation tests** with Bonferroni correction over 13 comparisons (α = 0.0038). Additional metrics: Brier score, ECE, MAE, RMSE, per-cell ROC-AUC.

---

## Getting Started

### Dependencies

```bash
# Core
pip install scikit-learn pandas numpy scipy

# Bayesian layer
pip install pymc numPyro

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
- ET scores: run `scripts/et_benchmark_yamagata.py` and `scripts/et_benchmark_akita.py`
- TTM scores: run `notebooks/ttm_yamagata.ipynb` / `notebooks/ttm_akita.ipynb` on Colab (requires IBM watsonx.ai API key)

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
  title     = {{KumaWatch}: A Multi-Method Wildlife Encounter Alert System for
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

IBM Granite TTM is developed by IBM Research. Extra Trees baseline follows the methodology of Nakamoto and Fukazawa [2025]. Bear sighting data is provided by Yamagata and Akita prefectural governments.

---

*Paper submitted to ACM SIGSPATIAL 2026 — Applications Track*
