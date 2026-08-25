# KumaWatch 🐻

**Benchmarking Wildlife Encounter Prediction for Municipal Decision Support in Northern Japan**

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
| [**▶ Multi-Method Benchmark Map (2025)**](https://global-jogasaki.github.io/KumaWatch/maps/kumawatch_primary_layer.html) | Switch between GLM-Logit / HierBayes / TTM / Extra Trees layers. 365-day date slider, click any cell for a detailed stats panel |

To open locally, open `maps/kumawatch_primary_layer.html` directly in any modern browser: a single HTML file requiring no server, no installation and no external prediction API. All predictions are embedded in the file. Leaflet and the OpenStreetMap base tiles are fetched from their public CDNs at load time, so an internet connection is needed for the map furniture — but never for a prediction.

---

## Overview

Human–bear conflicts in northern Japan have escalated sharply: **Yamagata Prefecture's official monthly tally reports 3,079 Asiatic black bear sightings in 2025** (as of August 16, 2026; excludes track-only reports and injury incidents), the highest annual count on record. The benchmark dataset in this repository is a snapshot of the prefectural sighting database extracted on January 17, 2026, containing 2,016 sightings for 2025 (records through mid-November; see Dataset section). Municipalities face a daily resource-constrained question: which twenty grid cells should patrols visit today?

**KumaWatch** is an open benchmark and browser-based decision-support prototype comparing eleven wildlife encounter prediction methods under a fixed municipal patrol budget. The central finding is negative and procurement-relevant: on Yamagata, a foundation model requiring ~4 hours of API inference is significantly worse than a static prior costing milliseconds, and a 30-minute MCMC pipeline yields no ranking improvement over a sub-30-second logistic regression.

---

## Abstract

We present **KumaWatch**, a cost-annotated top-K benchmark of eleven wildlife encounter prediction methods on two Japanese prefectures (Yamagata 144 cells, Akita 260 cells) over a 365-day held-out year (2025), with measured per-day computational cost — fitting, refitting or inference as applicable — reported alongside each predictive-performance estimate.

In the benchmark itself each trainable method is fitted once on data through 2024-12-31, while TTM is applied zero-shot; all methods then produce scores for the 365 evaluation days. The cost column states what one day of operation would cost a municipality that refreshed the model on that cadence, which is the quantity a procurement decision turns on.

**Central finding (negative and procurement-relevant):** IBM Granite TTM requires ~4 hours of API inference per day and, on Yamagata, is significantly *worse* than a static per-cell prior costing milliseconds (Δ = −0.041, *p* = 0.0004). On Akita it also trails the static prior, but not significantly (Δ = −0.010, *p* = 0.364). A 30-minute MCMC pipeline (HierBayes) yields no Recall@20 improvement over a sub-30-second logistic regression on Yamagata (*p* = 0.624) and is significantly worse on Akita (*p* = 0.003). On Yamagata — the primary evaluation setting — GLM-Logit's margin over the static prior B1 does not approach significance (+0.014, *p* = 0.155).

The benchmark uses Bonferroni-corrected permutation tests (α = 0.0038 over 13 comparisons, P = 5,000). We additionally release a browser-based decision-support map (no server-side prediction computation, no external prediction API) and argue that cost-annotated top-K benchmarking should precede model selection in municipal geospatial alerting.

---

## Key Contributions

1. **Cost-annotated top-K benchmark** — 11 methods (naive baselines B0–B5, Poisson-GLM, GLM-Logit, HierBayes, Extra Trees, TTM) evaluated on identical 365-day held-out windows across two prefectures, with measured per-day computational cost — fitting, refitting or inference as applicable — beside every Recall@K / Precision@K figure. Bonferroni-corrected permutation tests (α = 0.0038, 13 comparisons).

2. **Negative result with operational reading** — Neither foundation-model inference (~4 h/day) nor MCMC (~30 min/day) buys top-K accuracy over far cheaper alternatives. On Yamagata, GLM-Logit's lead over the static prior is not significant (*p* = 0.155). Extra Trees is strongly miscalibrated (BSS = −1.63). These are null results about ranking only; downstream uses of HierBayes posterior variance and ET environmental covariates are untested rather than refuted.

3. **Browser-based decision-support prototype** — A single-file Leaflet map (no server-side prediction computation, no external prediction API) serving pre-computed scores for GLM-Logit, HierBayes, TTM and Extra Trees across all 144 Yamagata cells × 365 days, with GLM-Logit as the default decision layer, four risk tiers and a slider for how many cells to show (default 20, the paper's patrol budget). Architecture follows directly from Table 1's cost column.

4. **Open benchmark release** — Complete codebase, benchmark data (Yamagata 144 cells + Akita 260 cells), pre-computed score files for all four learned methods on both prefectures (GLM-Logit, HierBayes, ET, TTM), and evaluation notebooks under Apache 2.0 and CC-BY 4.0 licenses.

---

## Results Summary

### Yamagata Prefecture (144 cells, 10 km × 10 km)

| Method | Recall@10 | Recall@20 | Recall@30 | Significance vs GLM-Logit (Recall@20) |
|--------|:---------:|:---------:|:---------:|---------------------------------------|
| **GLM-Logit** (best-ranked) | 0.345 | **0.547** | 0.690 | — |
| HierBayes | 0.328 | 0.542 | 0.692 | ns (p = 0.624) |
| B5: Recent MA + Seasonality | 0.333 | 0.534 | 0.660 | ns (p = 0.354) |
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
| HierBayes | 0.263 | 0.432 | 0.578 | sig. (p = 0.003) |
| B5: Recent MA + Seasonality | 0.261 | 0.427 | 0.568 | ns (p = 0.048, above the Bonferroni-corrected α = 0.0038) |
| B2: Recent Moving Average | 0.265 | 0.418 | 0.538 | — |
| B4: Static Prior + Seasonality | 0.240 | 0.418 | 0.541 | — |
| B1: Static Prior | 0.251 | 0.405 | 0.530 | sig. (p < 0.001) |
| **TTM** (IBM Granite 512-96-R2) | 0.227 | 0.395 | 0.516 | sig. (p < 0.001) |
| B3: DoY Seasonality | 0.215 | 0.352 | 0.451 | — |
| **Extra Trees** | 0.183 | 0.326 | 0.470 | sig. (p < 0.001) |
| B0: Random | 0.047 | 0.080 | 0.114 | — |
| Poisson-GLM | 0.003 | 0.003 | 0.003 | — |

*Bonferroni-corrected permutation tests, α = 0.0038 (0.05 / 13 comparisons, P = 5,000 permutations). "sig." = Bonferroni-significant (p < 0.0038); ns = not significant. Significance tests are computed on Recall@20. On Yamagata (primary setting), GLM-Logit's margin over the static prior B1 is not significant (+0.014, p = 0.155); a method requiring no model, no features and no daily computation is indistinguishable from the best method tested. On Yamagata TTM is significantly worse than B1 (Δ = −0.041, p = 0.0004); on Akita it trails B1 by a margin that is not significant (Δ = −0.010, p = 0.364), a comparison not tabulated in the paper. GLM-Logit significantly outperforms TTM and Extra Trees on both prefectures.*

*Each row reports a single run per method; no row mixes results from different runs. GLM-Logit and HierBayes Recall@K are recomputed from the released score files in `data/scores/` (`yamagata_glm_logit_scores_2025.npy`, SHA-256 `2de6593f4169b98e…`; `yamagata_hier_mean_scores_2025.npy`, SHA-256 `726fbeed366a7240…`). Both reproduce the paper's Recall@20 exactly — GLM-Logit 0.5470 and HierBayes 0.5425. The released GLM-Logit score matrix also reproduces Precision@20 = 0.2446, reported as 0.245 to three decimals in Section 6. No run-to-run mixing remains. All six baselines are regenerated deterministically from `notebooks/kumawatch_benchmark.ipynb` Cell 5 (`RAND_SEED = 42`) over the training windows documented above — Yamagata from 2018-10-01, Akita from 2022-04-01 — so every method in these tables shares one training period. (An earlier release computed B3 and B4 over the full data span instead, giving Yamagata B4 = 0.523 and Akita B4 = 0.417; those values are superseded.) TTM and Extra Trees Recall@K are recomputed from the released score CSVs. Poisson-GLM Recall@K is carried over from the archived benchmark run in `notebooks/kumawatch_benchmark_table3_colab.ipynb` (saved cell outputs).*

### Calibration Metrics

| Method | YGT Brier ↓ | YGT BSS ↑ | AKT Brier ↓ | AKT BSS ↑ |
|--------|:-----------:|:---------:|:-----------:|:---------:|
| **GLM-Logit** | 0.034 | 0.08 | 0.041 | 0.28 |
| HierBayes | 0.033 | 0.10 | 0.040 | 0.30 |
| TTM | 0.036 | 0.02 | 0.055 | 0.04 |
| Extra Trees | 0.097 | −1.63 | 0.126 | −1.18 |
| B2: Recent MA | **0.031** | **0.15** | **0.039** | **0.32** |

*BSS (Brier Skill Score) > 0 indicates better calibration than the climatological baseline. B2 achieves the best Brier Skill Score of any method on both prefectures. ET is strongly miscalibrated (BSS = −1.63 on Yamagata), consistent with known behaviour of tree ensembles on probability estimation tasks. HierBayes and GLM-Logit are both well calibrated (BSS 0.08–0.10 on Yamagata, 0.28–0.30 on Akita). The HierBayes row is recomputed from the released posterior-mean files; the paper's Table 1 reports its Yamagata BSS as 0.08, from the superseded run.*

### Cross-Method Top-K Agreement

Pairwise Jaccard@K between the released methods is not tabulated here; it is
computed on demand from the score files in `data/scores/`:

```bash
python scripts/crosslayer_jaccard.py --check
```

Rankings that disagree are not thereby complementary. HierBayes' posterior
variance as a confidence signal for graduated alerts, and ET's environmental
covariates as an independent check on a recency-driven prediction, are untested
rather than refuted by this benchmark, which scores ranking under a fixed patrol
budget only.

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
├── index.html                             # GitHub Pages landing page — supplementary diagnostics
├── notebooks/
│   ├── kumawatch_benchmark.ipynb          # Full 11-method benchmark (GLM-Logit, HierBayes, ET, TTM, B0–B5, Poisson-GLM)
│   ├── kumawatch_benchmark_table3_colab.ipynb  # Archived run; its confidence-filtered analysis is dropped from the final paper
│   ├── ttm_yamagata.ipynb                 # TTM inference — Yamagata 144 cells (IBM Granite TTM 1536-96-R2)
│   ├── ttm_akita.ipynb                    # TTM inference — Akita 260 cells (IBM Granite TTM 512-96-R2)
│   └── et_akita.ipynb                     # Extra Trees baseline — Akita (Colab)
├── scripts/
│   ├── table2_significance.py             # Table 2 of the paper; --all runs every pair among the 11 methods
│   ├── all_vs_static_prior.py             # Every method tested against the static prior B1
│   ├── daily_diagnostics.py               # Precision@K, three ROC-AUC definitions, per-day TP/FP/FN
│   ├── crosslayer_jaccard.py              # Cross-method Jaccard@K from released score files
│   ├── calibration_validation.py          # Post-hoc Platt/Isotonic calibration validation
│   ├── generate_kumawatch_webmap.py       # Builds maps/ from the released scores; aborts if they do not verify
│   ├── generate_glm_webmap.py             # GLM-Logit single-layer map generator
│   ├── et_benchmark_yamagata.py           # Extra Trees benchmark — Yamagata (needs external covariates)
│   └── et_benchmark_akita.py              # Extra Trees benchmark — Akita (needs external covariates)
├── results/                               # Fixed outputs, so a figure never has to be recomputed to be quoted
│   ├── all_pairwise_tests_{yamagata,akita}_2025.csv|.md   # 55 pairs per prefecture, sorted by p
│   ├── all_vs_static_prior_{yamagata,akita}_2025.csv|.md  # All 11 methods vs B1
│   └── daily_diagnostics_{yamagata,akita}_2025.csv        # 3,650 method-day rows each
├── maps/
│   ├── kumawatch_primary_layer.html       # Four-method interactive web map (2025) — all 144 cells × 365 days embedded
│   └── kumawatch_complementary_layer.html # Complementary-layer focused map view
├── data/
│   ├── yamagata_10km_daily_timeseries.csv # 144 cells × daily sightings, wide format (Apr 2018–Dec 2025; training from Oct 2018)
│   ├── akita_10km_daily_timeseries.csv    # 260 cells × daily sightings, wide format (Apr 2020–Dec 2025; training from Apr 2022)
│   ├── yamagata_10km_grid_coords.csv      # Grid cell coordinates and IDs
│   ├── akita_10km_grid_coords.csv         # Grid cell coordinates and IDs
│   ├── scores/                            # Pre-computed 2025 test-period scores
│   │   ├── yamagata_glm_logit_scores_2025.npy   # GLM-Logit (365 × 144, float32)
│   │   ├── yamagata_hier_mean_scores_2025.npy   # HierBayes posterior mean (365 × 144, float32)
│   │   ├── yamagata_hier_std_scores_2025.npy    # HierBayes posterior std — a different MCMC run; do not pair with the mean
│   │   ├── yamagata_et_scores_2025.csv          # Extra Trees (365 × 144)
│   │   ├── yamagata_ttm_scores_2025.csv         # IBM Granite TTM (365 × 144)
│   │   ├── yamagata_ttm_scores.npy              # Same, NumPy binary
│   │   ├── akita_glm_logit_scores_2025.npy      # GLM-Logit (365 × 260, float32)
│   │   ├── akita_hier_mean_scores_2025.npy      # HierBayes posterior mean (365 × 260, float32)
│   │   ├── akita_et_scores_2025.csv             # Extra Trees (365 × 260)
│   │   ├── akita_ttm_scores_2025.csv            # IBM Granite TTM (365 × 260)
│   │   └── akita_ttm_scores.npy                 # Same, NumPy binary
│   └── README.md                          # Score-file provenance, verified Recall@K and SHA-256
├── .claude/skills/                        # Task-entry guides (benchmark, scores, webmap, calibration, data reference)
├── CLAUDE.md                              # Index into the skills above
├── requirements-diagnostics.txt           # Versions the published diagnostics were computed under
├── .nojekyll                              # Serves the Pages site as plain files, without Jekyll
├── .gitignore
├── README.md
└── LICENSE
```

Everything in `results/` is regenerated by the scripts above; the daily
diagnostics and pairwise tests cover all eleven benchmarked methods, with B0–B5
and Poisson-GLM regenerated deterministically rather than read from a file.

---

## Dataset

| Dataset | Region | Training Period | Evaluation Period | Cells | Granularity |
|---------|--------|----------------|------------------|-------|-------------|
| Yamagata bear sightings | Yamagata, Japan | 2018-10-01 – 2024-12-31 | 2025-01-01 – 2025-12-31 | 144 | Daily |
| Akita bear sightings | Akita, Japan | 2022-04-01 – 2024-12-31 | 2025-01-01 – 2025-12-31 | 260 | Daily |

Strict temporal separation for model fitting: **all model fitting uses data through 2024-12-31 only**, and all 365 days of 2025 are held out as the test set.

**Snapshot provenance.** Both label sets are snapshots of the prefectures' public sighting databases. The Yamagata snapshot was extracted on January 17, 2026, at which point the prefectural database contained records only through mid-November 2025; the Akita data extend through December 2025. Prefectural tallies are consolidated retroactively as municipal reports arrive: against Yamagata's official monthly tally as of August 2026 (3,079 sightings for 2025), the snapshot records 2,016, with the gap concentrated in the autumn surge (October 581 vs. 870; November 309 vs. 612) and December absent entirely (0 vs. 128). Days with at least one recorded sighting — the days on which Recall@K is defined — number 213 of 365 (Yamagata) and 323 of 365 (Akita). All methods are trained and scored against the same snapshot, so between-method comparisons are internally consistent.

Evaluation is **sequential one-day-ahead forecasting**. For a forecast date *d* in 2025, recency features (`rolling30`, `log(recent365+1)`, and the moving averages behind B2/B5) are computed from sightings observed strictly before *d*, which from late January 2025 onward consist entirely of test-period observations. No observation on or after the forecast date is ever used, and no model coefficients are refit on test-period data. This is the standard sequential-forecasting setup, not label leakage — but note that it does mean test-period observations enter feature computation, which affects GLM-Logit, Poisson-GLM, HierBayes, B2 and B5. B0, B1, B3 and B4 use training-period data only.

Data source: Yamagata and Akita prefectural wildlife observation databases (publicly available).

---

## Evaluation Framework

Metrics are computed under the **Global formulation**: for each day *t*, let S_t be the top-K cells by predicted score. Then:

- **Precision@K** = |S_t ∩ A_t| / K
- **Recall@K** = |S_t ∩ A_t| / |A_t|

where A_t is the set of cells with actual sightings on day *t*. Results are averaged over the 365-day evaluation window (2025).

Statistical significance: day-level paired **permutation tests** (P = 5,000, sign-flip) with day-level paired bootstrap confidence intervals (B = 5,000), both seeded with 42. Bonferroni is applied per family, and each table states which family it belongs to:

| Family | Comparisons | α |
|--------|:-----------:|:-:|
| The paper's Table 2 | 13 | 0.0038 |
| Every method vs the static prior (`all_vs_static_prior.py`) | 10 | 0.005 |
| Every pair among the 11 methods (`table2_significance.py --all`) | 55 | 0.00091 |

Calibration metrics are computed **globally over all cell-days** in the evaluation window, against the climatological base rate: **Brier score**, **Brier Skill Score (BSS)**, ECE, MAE, RMSE.

Ranking metrics: per-cell **ROC-AUC**, **PR-AUC**.

Cross-method agreement: **Jaccard@K** between the top-K cell sets of two methods, averaged over days (`scripts/crosslayer_jaccard.py`).

---

## Getting Started

### Dependencies

```bash
# Reproducing the published tables — pinned versions the results were computed under
pip install -r requirements-diagnostics.txt

# Bayesian layer, only needed to refit HierBayes from scratch
pip install pymc numpyro
```

The maps need nothing installed: they are single HTML files with every prediction
embedded (Leaflet and the OpenStreetMap tiles load from their CDNs).

### View the web maps

Open the live demo in your browser — no installation needed:

```
https://global-jogasaki.github.io/KumaWatch/maps/kumawatch_primary_layer.html
```

Or open `maps/kumawatch_primary_layer.html` locally in any modern browser.

### Prerequisite: pre-computed score files

The benchmark notebook (`notebooks/kumawatch_benchmark.ipynb`) needs pre-computed
ET and TTM daily scores for 2025. Those, and the GLM-Logit and HierBayes matrices
the reproduction scripts read, are all in `data/scores/`:

| File | Model | Prefecture |
|------|-------|------------|
| `yamagata_glm_logit_scores_2025.npy` | GLM-Logit | Yamagata |
| `yamagata_hier_mean_scores_2025.npy` | HierBayes (posterior mean) | Yamagata |
| `yamagata_et_scores_2025.csv` | Extra Trees | Yamagata |
| `yamagata_ttm_scores_2025.csv` | IBM Granite TTM | Yamagata |
| `akita_glm_logit_scores_2025.npy` | GLM-Logit | Akita |
| `akita_hier_mean_scores_2025.npy` | HierBayes (posterior mean) | Akita |
| `akita_et_scores_2025.csv` | Extra Trees | Akita |
| `akita_ttm_scores_2025.csv` | IBM Granite TTM | Akita |

B0–B5 and Poisson-GLM are not shipped as files: they are regenerated
deterministically from the benchmark notebook (Cell 5 and Cell 10) wherever they
are needed. `data/scores/README.md` records the verified Recall@K and SHA-256 of
each released file.

To **regenerate** the ET and TTM scores from scratch:
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

### Supplementary diagnostics

```bash
pip install -r requirements-diagnostics.txt
python scripts/daily_diagnostics.py
```

Produces the Precision@K, ROC-AUC and per-day case tables published on the
landing page, plus `results/daily_diagnostics_<pref>_2025.csv` — 3,650 method-day
rows per prefecture giving, for each day and method, the positive-cell count,
Recall@20, Precision@20, the true-positive / false-positive / false-negative cell
ids, and the pairwise top-20 agreement with every other method.

Coverage is ten of the paper's eleven methods: the four learned methods from
released score matrices, and B0–B5 regenerated deterministically from Cell 5 of
the benchmark notebook. Poisson-GLM has no released score matrix.

Two conventions matter and are stated on the page as well. Precision@K is
reported twice — over sighting days (the set the paper reports, where Recall@K is
defined) and over all 365 days — and the two are never combined into one
aggregate. ROC-AUC is reported under three definitions (pooled cell-day, mean
daily cross-sectional, mean per-cell temporal) because they answer different
questions; it is a supplementary threshold-free diagnostic and does not replace
Recall@K or Precision@K, which correspond to the fixed daily patrol budget.

These analyses are retrospective comparisons against the held-out 2025 labels.
They are not a post-deployment evaluation: no patrol routes, deployment period or
control municipality exist in this dataset.

### Every pairwise comparison

```bash
python scripts/table2_significance.py --all
```

Runs the same test on **every unordered pair** among the eleven methods — 55
pairs per prefecture, 110 in total — and writes them to
`results/all_pairwise_tests_<pref>_2025.csv` and `.md`, sorted by *p*. This
answers pairwise questions the paper's six selected rows do not, such as TTM
versus Extra Trees.

Bonferroni for this family is α = 0.05 / 55 = 0.00091, stricter than Table 2's
0.0038 over its own thirteen; each table states the threshold it used.

The TTM–ET comparison is a good illustration of why the pair matters: on Yamagata
the two are indistinguishable (Δ = −0.018 in ET's favour direction, *p* = 0.186),
while on Akita ET is clearly worse (Δ = −0.069, *p* = 0.0002). Overall 34 of 55
pairs separate on Yamagata and 35 of 55 on Akita; GLM-Logit and HierBayes do not
separate on either.

### Every method against the static prior

```bash
python scripts/all_vs_static_prior.py
```

Table 2 of the paper reports six selected pairwise comparisons. This runs the
same test — day-level paired bootstrap (B = 5,000) and sign-flip permutation
(P = 5,000), seeded — for **all eleven methods against B1** on both prefectures,
and fixes the answer on disk as `results/all_vs_static_prior_<pref>_2025.csv`
and `.md`: method, Recall@20, Δ vs B1, 95% CI, permutation *p*, Bonferroni
verdict and the number of days scored.

Coverage is complete: the four learned methods come from released score matrices,
and B0–B5 and Poisson-GLM are regenerated deterministically from the benchmark
notebook (Cell 5 and Cell 10). The regenerated Poisson-GLM reproduces its
published Recall@20 — 0.0267 on Yamagata and 0.0033 on Akita.

Bonferroni here is α = 0.05 / 10 = 0.005 over this family; Table 2 uses
α = 0.0038 over its own family of thirteen. No comparison falls between the two
thresholds, so both give the same verdict on every row.

What the table shows: **on Yamagata no method beats the static prior
significantly** — GLM-Logit's +0.014 (*p* = 0.155), HierBayes' +0.009
(*p* = 0.439) and B5's +0.001 (*p* = 0.936) are the three that are ahead at all,
and none approaches the threshold. On Akita, GLM-Logit (+0.050) and HierBayes
(+0.027) do beat it significantly, which is the asymmetry Section 6 describes.

### Reproduce the significance table

```bash
python scripts/table2_significance.py
```

Recomputes the effect size, 95% confidence interval and permutation p-value for
each published pairwise comparison at K = 20, straight from `data/scores/` — no
model is retrained and no MCMC is rerun. Day-level paired bootstrap (B = 5,000)
and sign-flip permutation (P = 5,000), both seeded, with Bonferroni-corrected
α = 0.0038. Each row is printed beside its published value and flagged `[OK]` or
`[DIFFERS]`. Use `--all` for every available method pair.

### Verify cross-method top-K agreement

```bash
python scripts/crosslayer_jaccard.py --check
```

Recomputes Jaccard@K between every pair of released methods straight from
`data/scores/`, for both all days and sighting days. `--check` first verifies
that each score file reproduces its Recall@20 from the results table, so a
mismatched or misaligned score file is caught before the agreement figures are
reported. No model needs to be rerun.

### Generate the web map

```bash
python scripts/generate_kumawatch_webmap.py
```

The generator reads the four released score files and **retrains nothing**. Before
any HTML is written it checks Recall@20 against Table 1 for each method, plus day
count, cell count and column alignment, and aborts rather than embed scores that
do not match the paper. The published map reproduces the paper's top-20 ranking
on all 365 days for all four methods.

Earlier releases of this script fitted GLM-Logit and Extra Trees inline and
substituted a Beta-Binomial seasonal approximation for HierBayes, so the layers
labelled "Extra Trees" and "HierBayes" in the demo were not the methods the paper
evaluated. That path has been removed.

Outputs `maps/kumawatch_primary_layer.html` (~2.3 MB). Every one of the 144 cells
is stored for every one of the 365 days, so the daily rank runs over the full grid,
any cell can be clicked, and widening the slider always reveals more cells.

Cell values are the released scores themselves. How they are read differs by
method, because only two of the four are calibrated probabilities:

| Layer | Displayed as | Tiers cut on |
|-------|--------------|--------------|
| GLM-Logit, HierBayes | predicted probability (%) | the probability: ≥20% / 10–20% / 5–10% / below |
| TTM, Extra Trees | raw score (3 dp) | the cell's rank that day: top 5 / 10 / 20 / rest |

Extra Trees is strongly miscalibrated (BSS = −1.63) and its raw scores sit near
1.0 for tens of cells a day, so presenting them as a risk percentage would
overstate danger to a municipal user. TTM and ET therefore carry no probability
wording. Earlier releases divided every layer by its 95th percentile and labelled
the result as a percentage, which made a quiet January look as alarming as
November; that rescaling is gone.

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
