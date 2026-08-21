# KumaWatch Benchmark Data

This directory contains the benchmark datasets used in the KumaWatch study.

## License

All data files in this directory are released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
See: https://creativecommons.org/licenses/by/4.0/

Data source: Yamagata and Akita prefectural wildlife observation databases (publicly available).

---

## Files

### Daily Time Series

| File | Prefecture | Cells | Period | Size |
|------|-----------|-------|--------|------|
| `yamagata_10km_daily_timeseries.csv` | Yamagata | 144 | Apr 2018 – Dec 2025 | ~915 KB |
| `akita_10km_daily_timeseries.csv` | Akita | 260 | Apr 2020 – Dec 2025 | ~1.17 MB |

**Format (wide):** one row per date, one column per grid cell.

- `Date` — date (e.g., `2018/4/11`; parse with `pd.to_datetime(..., format='mixed')`)
- `Year`, `Month`, `Week`, `Weekday` — calendar metadata columns
- `{col}_{row}` — one column per grid cell (e.g., `0_0`, `1_0`, ..., `8_15` for Yamagata); value = number of bear sightings recorded in that cell on that date
- `Sum` — total sightings across all cells on that date

### Grid Coordinates

| File | Prefecture | Cells |
|------|-----------|-------|
| `yamagata_10km_grid_coords.csv` | Yamagata | 144 |
| `akita_10km_grid_coords.csv` | Akita | 260 |

**Columns:**
- `Grid_ID` — cell ID in `{col}_{row}` format (e.g., `0_0`), matches the cell column names in the daily time series CSVs
- `Grid_Row`, `Grid_Col` — 0-indexed row / column indices
- `Center_Latitude`, `Center_Longitude` — grid cell center (decimal degrees)
- `Min_Latitude`, `Max_Latitude`, `Min_Longitude`, `Max_Longitude` — cell bounding box (decimal degrees)

---

## Grid Definition

Each prefecture is partitioned into a regular grid of approximately 10 km × 10 km cells:

- **Yamagata**: 9 columns × 16 rows = **144 cells** (active cells vary by season)
- **Akita**: 13 columns × 20 rows = **260 cells**

The resolution was chosen to balance:
- Data density (>99% of 1 km cells have zero annual sightings)
- Patrol unit mobility (10 km matches typical daily patrol radius)
- Administrative boundaries (aligns with municipal jurisdiction limits)

---

## Evaluation Split

The 2025 calendar year (January 1 – December 31, 365 days) was used as the held-out test set for all models in the benchmark. Training data used all observations prior to January 1, 2025.
