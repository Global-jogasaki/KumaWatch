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
| `yamagata_10km_daily_timeseries.csv` | Yamagata | 144 | Oct 2018 – Dec 2025 | ~915 KB |
| `akita_10km_daily_timeseries.csv` | Akita | 260 | Apr 2022 – Dec 2025 | ~1.17 MB |

**Columns:**
- `date` — ISO 8601 date (YYYY-MM-DD)
- `grid_id` — Integer cell ID (0-indexed)
- `sightings` — Number of bear sightings recorded in that cell on that date

### Grid Coordinates

| File | Prefecture | Cells |
|------|-----------|-------|
| `yamagata_10km_grid_coords.csv` | Yamagata | 144 |
| `akita_10km_grid_coords.csv` | Akita | 260 |

**Columns:**
- `grid_id` — Integer cell ID (0-indexed), matches `grid_id` in daily time series
- `lat` — Latitude of grid cell center (decimal degrees)
- `lon` — Longitude of grid cell center (decimal degrees)
- `municipality` — Japanese municipality name (市区町村) overlapping the cell

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
