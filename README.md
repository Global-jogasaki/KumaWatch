# KumaWatch 🐻

**TTM-Bear: Time Series Foundation Models for Operational Wildlife Encounter Prediction**  
*A Resource-Aware Comparison with Feature-Engineered Baselines*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![ACM SIGSPATIAL 2026](https://img.shields.io/badge/ACM%20SIGSPATIAL-2026-red.svg)](https://sigspatial.acm.org/)

---

## Overview

Human–bear conflicts in northern Japan have escalated dramatically, with **Yamagata Prefecture recording 2,655 bear sightings in 2025**—a 745.5% increase from the previous year. Existing approaches based on engineered features achieve moderate predictive performance, but their suitability for resource-constrained municipal operations remains unexamined.

**KumaWatch** (熊 Watch) is a deployable web-based decision-support system applying **IBM Granite Tiny Time Mixers (TTM)** with in-context learning to predict daily bear encounter risk across **144 grid cells** (each 10 km × 10 km) in Yamagata Prefecture, Japan.

---

## Abstract

We present **TTM-Bear**, applying IBM Granite TTM with in-context learning to predict daily bear encounter risk across 144 grid cells (each 10 km × 10 km) in Yamagata Prefecture. We select Yamagata as the primary study region because its publicly available sighting record (since October 2018) is currently the only Tohoku prefectural dataset reaching the 1,536-day context window required by TTM 1536-96-R2.

To enable fair comparison, we implement an Extra Trees (ET) baseline following Nakamoto and Fukazawa [2025] on the same data, and validate it on Akita Prefecture (the original evaluation region) using TTM 512-96-R2, the shorter-context variant whose data requirements fit Akita's three-year publicly available record.

**Our central finding is the robust operational advantage of foundation models on resource-aware metrics**: in all 12 head-to-head comparisons of Precision@K and Recall@K (K = 10, 20, 30) across both prefectures, TTM substantially outperforms ET, achieving **1.4–2.0× advantages**:

- Yamagata TTM-1536: **Recall@20 = 0.492** vs ET 0.361
- Akita TTM-512: **Recall@20 = 0.395** vs ET 0.215

This Recall@K dominance is robust across regions and context lengths. Long-context TTM on Yamagata also achieves substantially better probability calibration than ET (**TTM Brier 0.058 vs ET 0.287**, a 5× advantage). Rigorous post-hoc calibration analysis confirms this advantage is irreducible: applying Platt scaling and Isotonic regression to ET reduces its Brier to 0.085–0.092, but TTM retains a **1.47× advantage**.

Most strikingly, the operational Recall@K advantage holds even when ROC-AUC favors ET or is near-random for TTM (Akita TTM-512 ROC-AUC = 0.506), **revealing that ROC-AUC fundamentally fails to capture operational utility for top-K alert systems**.

We release the complete codebase, baseline reimplementation, calibration validation scripts, and benchmark dataset under permissive licenses.

---

## Key Contributions

1. **TTM-Bear prototype** — A deployable web-based decision support system applying IBM Granite TTM to predict daily bear encounter risk across Yamagata Prefecture's 144 grid cells using in-context learning without fine-tuning.

2. **Resource-aware evaluation framework** — Precision@K, Recall@K, and probability calibration (Brier, MAE, ECE) reflecting operational constraints of municipal wildlife management; the first direct head-to-head comparison between time series foundation models and feature-engineered ensemble methods for wildlife encounter prediction.

3. **ROC-AUC vs. Recall@K divergence analysis** — Empirical attribution of the divergence between ROC-AUC and resource-aware metrics to distinct information sources (spatial features vs. temporal dynamics), with seasonal analysis and a dynamic alert operation strategy.

4. **Open benchmark release** — Complete codebase, Extra Trees reimplementation with cross-prefecture validation, web-based decision support map, and aggregated benchmark data under Apache 2.0 and CC-BY 4.0 licenses.

---

## Results Summary

| Model | Prefecture | Context | Recall@20 | Brier Score | ROC-AUC |
|-------|-----------|---------|-----------|-------------|---------|
| TTM-1536 | Yamagata | 1,536 days | **0.492** | **0.058** | — |
| Extra Trees | Yamagata | — | 0.361 | 0.287 | 0.677 |
| TTM-512 | Akita | 512 days | **0.395** | — | 0.506 |
| Extra Trees | Akita | — | 0.215 | — | 0.719 |

*TTM achieves 1.4–2.0× advantages on all 12 Precision@K and Recall@K comparisons (K = 10, 20, 30) across both prefectures.*

---

## System Architecture

```
Daily Inference Pipeline:
  (i)  Data ingestion  ← Yamagata Prefecture wildlife observation database
  (ii) Forecast        ← IBM Granite TTM via watsonx.ai API  +  Extra Trees (scikit-learn)
  (iii)Post-processing ← Grid-level probability scores, Precision@K / Recall@K evaluation
  (iv) Visualization   ← Interactive web map (Leaflet.js, 10 km grid overlay)
```

**Grid Definition**: Yamagata Prefecture is partitioned into a 9 × 16 grid of **144 cells**, each approximately 10 km × 10 km. The resolution balances data density (>99% of 1 km cells have zero sightings), patrol unit mobility (10 km matches typical daily patrol radius), and authority boundaries (aligns with municipal jurisdiction boundaries).

**Models**:
- **TTM 1536-96-R2** (primary): 1,536-day input context, 96-day forecast horizon, zero-shot / in-context learning
- **Extra Trees** (baseline): reimplementation of Nakamoto & Fukazawa [2025], L2-regularized GLM-Logit operational layer

---

## Dataset

| Dataset | Region | Period | Cells | Granularity |
|---------|--------|--------|-------|-------------|
| Yamagata bear sightings | Yamagata, Japan | Oct 2018 – present | 144 | Daily |
| Akita bear sightings | Akita, Japan | Apr 2022 – present | 260 | Daily |

Data source: prefectural wildlife observation databases (publicly available).

---

## Evaluation Framework

We define evaluation metrics under the **Global formulation**: for each day *t*, let S_t be the set of K grid cells with the highest predicted scores. Then:

- **Precision@K** = |S_t ∩ A_t| / K, where A_t is the set of cells with actual sightings
- **Recall@K** = |S_t ∩ A_t| / |A_t|

This formulation evaluates each model's ability to rank the **entire grid**, reflecting how alert systems are operated in practice.

Additional metrics: Brier score, Expected Calibration Error (ECE), MAE, RMSE (per-cell, on 90 daily-evaluable Yamagata cells).

---

## Citation

```bibtex
@inproceedings{jogasaki2026ttmbear,
  author    = {Hiroshi Jogasaki},
  title     = {{TTM-Bear}: Time Series Foundation Models for Operational Wildlife
               Encounter Prediction---A Resource-Aware Comparison with
               Feature-Engineered Baselines},
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
