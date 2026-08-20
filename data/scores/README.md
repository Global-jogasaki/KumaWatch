# Pre-computed Score Files

This directory contains pre-computed daily prediction scores for 2025 (the test period),
used as inputs to `notebooks/kumawatch_benchmark.ipynb`.

## File Format

All CSV files are in **wide format**:
- Row: one date (365 rows, 2025-01-01 to 2025-12-31)
- Columns: `Date`, then one column per grid cell (e.g., `0_0`, `1_0`, ..., `8_15`)
- Cell column names match exactly those in the corresponding sightings CSV

## Files

| File | Model | Prefecture | Shape |
|------|-------|-----------|-------|
| `yamagata_glm_logit_scores_2025.npy` | GLM-Logit | Yamagata | (365, 144) float32 |
| `yamagata_hier_mean_scores_2025.npy` | HierBayes (posterior mean) | Yamagata | (365, 144) float64 |
| `yamagata_hier_std_scores_2025.npy` | HierBayes (posterior std) ⚠️ different run — see below | Yamagata | (365, 144) float64 |
| `yamagata_et_scores_2025.csv` | Extra Trees | Yamagata | 365 × 144 |
| `yamagata_ttm_scores_2025.csv` | IBM Granite TTM | Yamagata | 365 × 144 |
| `yamagata_ttm_scores.npy` | IBM Granite TTM | Yamagata | (365, 144) float32 |
| `akita_glm_logit_scores_2025.npy` | GLM-Logit | Akita | (365, 260) float32 |
| `akita_hier_mean_scores_2025.npy` | HierBayes (posterior mean) | Akita | (365, 260) float32 |
| `akita_et_scores_2025.csv` | Extra Trees | Akita | 365 × 260 |
| `akita_ttm_scores_2025.csv` | IBM Granite TTM | Akita | 365 × 260 |
| `akita_ttm_scores.npy` | IBM Granite TTM | Akita | (365, 260) float32 |

### Verification

Recomputing global Recall@K from these files reproduces the values reported in the
paper and in the top-level README:

| File | Recall@10 | Recall@20 | Recall@30 | SHA-256 (first 16) |
|------|:---------:|:---------:|:---------:|--------------------|
| `yamagata_et_scores_2025.csv` | 0.2927 | 0.4739 | 0.6066 | `456fcbc3e01e8b48` |
| `akita_et_scores_2025.csv` | 0.1829 | 0.3258 | 0.4698 | `569cebe6df15c667` |
| `akita_glm_logit_scores_2025.npy` | 0.2590 | 0.4543 | 0.5868 | `a9b078c5217e4c20` |
| `akita_hier_mean_scores_2025.npy` | 0.2630 | 0.4316 | 0.5777 | `7dc7fe0b7837abdf` |
| `yamagata_glm_logit_scores_2025.npy` | 0.3448 | 0.5470 | 0.6904 | `2de6593f4169b98e` |
| `yamagata_hier_mean_scores_2025.npy` | 0.3280 | 0.5425 | 0.6922 | `726fbeed366a7240` |

The Extra Trees files also reproduce the reported calibration figures
(Yamagata Brier = 0.097, BSS = −1.63), and `yamagata_glm_logit_scores_2025.npy`
reproduces the Precision@20 = 0.244 quoted in Section 6 of the paper (0.2446).

### ⚠️ The HierBayes posterior standard deviation does not pair with the mean

`yamagata_hier_std_scores_2025.npy` comes from an **earlier MCMC run** than
`yamagata_hier_mean_scores_2025.npy`, which is the run the paper reports. The two
are not draws from the same posterior.

**Do not combine them.** A credible interval, a confidence signal or an
uncertainty-stratified metric built from one run's mean and another run's spread
is not a posterior summary of anything, and no such figure should be published.
The web map therefore shows no credible interval, and the confidence-filtered
analysis — which the 4-page final paper drops entirely — cannot be reproduced
from these files. No reported result depends on the std file.

To restore that analysis, re-emit the posterior mean and standard deviation
together from a single trace; replacing only one of the two recreates the same
problem.

Cell columns in the ET files are ordered `row_col` and must be aligned to the
sightings CSV **by column name**, not by position, since the two files use
different column orderings. The `.npy` files carry no labels: their rows are the
365 evaluation days in ascending date order and their columns follow the cell
order of the corresponding sightings CSV.

## How to Use

In `notebooks/kumawatch_benchmark.ipynb`, set `SCORE_FORMAT = 'CSV'` and point the
path variables to these files (or upload to Google Drive for Colab):

```python
SCORE_FORMAT = 'CSV'
YAMA_ET_SCORES_CSV  = 'data/scores/yamagata_et_scores_2025.csv'
YAMA_TTM_SCORES_CSV = 'data/scores/yamagata_ttm_scores_2025.csv'
AKITA_ET_SCORES_CSV = 'data/scores/akita_et_scores_2025.csv'
AKITA_TTM_SCORES_CSV = 'data/scores/akita_ttm_scores_2025.csv'
```

## Regenerating GLM-Logit and HierBayes Scores

GLM-Logit and HierBayes scores are computed directly in `notebooks/kumawatch_benchmark.ipynb`
(or `notebooks/kumawatch_benchmark_table3_colab.ipynb` for Table 3 confidence-filtered analysis).
HierBayes requires PyMC + NumPyro (JAX backend); on Windows set
`PYTENSOR_FLAGS=device=cpu,floatX=float64,optimizer=fast_compile,cxx=` to disable C++ compilation.

## Regenerating ET Scores

To regenerate ET scores from scratch, run the benchmark scripts:

```bash
python scripts/et_benchmark_yamagata.py  # → yamagata_et_scores_2025.csv
python scripts/et_benchmark_akita.py     # → akita_et_scores_2025.csv
```

## Regenerating TTM Scores

Open `notebooks/ttm_yamagata.ipynb` or `notebooks/ttm_akita.ipynb` on Google Colab
(requires IBM watsonx.ai API credentials) and run all cells.
