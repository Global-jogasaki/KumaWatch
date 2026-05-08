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
| `yamagata_et_scores_2025.csv` | Extra Trees | Yamagata | 365 × 144 |
| `yamagata_ttm_scores_2025.csv` | IBM Granite TTM | Yamagata | 365 × 144 |
| `yamagata_ttm_scores.npy` | IBM Granite TTM | Yamagata | (365, 144) float32 |
| `akita_et_scores_2025.csv` | Extra Trees | Akita | 365 × 260 |
| `akita_ttm_scores_2025.csv` | IBM Granite TTM | Akita | 365 × 260 |
| `akita_ttm_scores.npy` | IBM Granite TTM | Akita | (365, 260) float32 |

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

## Regenerating ET Scores

To regenerate ET scores from scratch, run the benchmark scripts:

```bash
python scripts/et_benchmark_yamagata.py  # → yamagata_et_scores_2025.csv
python scripts/et_benchmark_akita.py     # → akita_et_scores_2025.csv
```

## Regenerating TTM Scores

Open `notebooks/ttm_yamagata.ipynb` or `notebooks/ttm_akita.ipynb` on Google Colab
(requires IBM watsonx.ai API credentials) and run all cells.
