#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExtraTrees bear sighting benchmark — Akita Prefecture, 10km x 10km grid.

Same methodology as benchmark_akita.py (7km) but using the exact Akita
10km grid aligned with Yamagata_10km_Grid_0.csv:
  20 rows x 13 cols = 260 cells
  lat_step=0.090090°, lon_step=0.114326°
  GRID_LAT_MIN=38.839510, GRID_LON_MIN=139.663417
  Cell ID: {col}_{row}  (col-first, same as Yamagata)

Outputs:
  - Global P@K, R@K, ROC-AUC, seasonal breakdown
  - Per-cell metrics for TOP20 cells vs TTM-512
Train: 2022-2024  |  Test: 2025
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import math, pickle, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CACHE_DIR   = BASE_DIR / '.cache'
RESULTS_DIR = BASE_DIR / 'results_akita_10km'
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

DATA_CSV = BASE_DIR.parent / 'bear-sighting-data' / 'data' / 'akita' / '050008_kumadas.csv'

# ── 10km grid (aligned with Yamagata_10km_Grid_0.csv) ─────────────────────
LAT_STEP     = 0.090090
LON_STEP     = 0.114326
N_ROWS       = 20
N_COLS       = 13
N_CELLS      = N_ROWS * N_COLS   # 260
GRID_LAT_MIN = 38.839510
GRID_LON_MIN = 139.663417
GRID_LAT_MAX = GRID_LAT_MIN + N_ROWS * LAT_STEP
GRID_LON_MAX = GRID_LON_MIN + N_COLS * LON_STEP
REF_LAT      = 39.7

TRAIN_YEARS  = [2022, 2023, 2024]
TEST_YEAR    = 2025
K_VALUES     = [10, 20, 30]
DEM_ZOOM     = 11

SEASONS = {
    'spring': [4, 5],
    'summer': [6, 7, 8],
    'fall':   [9, 10, 11],
    'winter': [12, 1, 2, 3],
}

TOP20 = ['4_9','6_15','9_14','4_10','5_9','7_8','8_15','7_15',
         '3_11','3_10','6_14','9_15','5_8','5_15','9_16','6_7',
         '7_16','4_8','3_14','5_14']

# ── DEM ────────────────────────────────────────────────────────────────────

def _deg2tile(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) +
             1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def _tile2bbox(x, y, z):
    n = 2 ** z
    def merc(yy): return math.degrees(math.atan(math.sinh(math.pi * (1 - 2*yy/n))))
    return merc(y), merc(y+1), (x/n)*360-180, ((x+1)/n)*360-180

def _fetch_tile(z, x, y):
    cache = CACHE_DIR / f'dem_{z}_{x}_{y}.npy'
    if cache.exists():
        return np.load(str(cache))
    url = f'https://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt'
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        rows = [[float(v) if v != 'e' else np.nan for v in line.split(',')]
                for line in r.text.strip().split('\n')]
        data = np.array(rows, dtype=np.float32)
    except Exception:
        data = np.full((256, 256), np.nan, dtype=np.float32)
    np.save(str(cache), data)
    return data

def download_dem():
    cache_file = CACHE_DIR / 'dem_assembled_z11_akita10km.pkl'
    if cache_file.exists():
        print("  Using cached Akita 10km DEM...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    x_min, y_min = _deg2tile(GRID_LAT_MAX + 0.1, GRID_LON_MIN - 0.1, DEM_ZOOM)
    x_max, y_max = _deg2tile(GRID_LAT_MIN - 0.1, GRID_LON_MAX + 0.1, DEM_ZOOM)
    nx, ny = x_max - x_min + 1, y_max - y_min + 1
    print(f"  Downloading DEM: {nx}x{ny}={nx*ny} tiles...")
    assembled = np.full((ny*256, nx*256), np.nan, dtype=np.float32)
    done = 0
    for xi in range(x_min, x_max+1):
        for yi in range(y_min, y_max+1):
            tile = _fetch_tile(DEM_ZOOM, xi, yi)
            assembled[(yi-y_min)*256:(yi-y_min+1)*256,
                      (xi-x_min)*256:(xi-x_min+1)*256] = tile
            done += 1
            if done % 30 == 0:
                print(f"    {done}/{nx*ny}...")
            time.sleep(0.03)
    _, lat_max_r, lon_min_r, _ = _tile2bbox(x_min, y_min, DEM_ZOOM)
    lat_min_r, _, _, lon_max_r = _tile2bbox(x_max, y_max, DEM_ZOOM)
    info = {'data': assembled,
            'lat_min': lat_min_r, 'lat_max': lat_max_r,
            'lon_min': lon_min_r, 'lon_max': lon_max_r}
    with open(cache_file, 'wb') as f:
        pickle.dump(info, f)
    return info

# ── grid ───────────────────────────────────────────────────────────────────

def build_grid() -> pd.DataFrame:
    rows = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            lat0 = GRID_LAT_MIN + r * LAT_STEP
            lon0 = GRID_LON_MIN + c * LON_STEP
            rows.append({'cell_id':    f'{c}_{r}',
                         'row': r, 'col': c,
                         'lat_center': lat0 + LAT_STEP/2,
                         'lon_center': lon0 + LON_STEP/2,
                         'lat_min': lat0, 'lat_max': lat0 + LAT_STEP,
                         'lon_min': lon0, 'lon_max': lon0 + LON_STEP})
    return pd.DataFrame(rows)

# ── sightings ──────────────────────────────────────────────────────────────

def load_sightings(years=None) -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV, encoding='utf-8-sig')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[df['獣種'] == 'ツキノワグマ'].copy()
    df['event_date'] = pd.to_datetime(df['目撃日時'], format='%Y/%m/%d %H:%M', errors='coerce')
    df['latitude']   = pd.to_numeric(df['x(緯度)'], errors='coerce')
    df['longitude']  = pd.to_numeric(df['y(経度)'], errors='coerce')
    df = df.dropna(subset=['event_date', 'latitude', 'longitude'])
    df['year']  = df['event_date'].dt.year
    df['month'] = df['event_date'].dt.month
    if years:
        df = df[df['year'].isin(years)].copy()
    return df

def assign_to_grid(sightings: pd.DataFrame) -> pd.DataFrame:
    df = sightings.copy()
    df['col'] = np.floor((df['longitude'] - GRID_LON_MIN) / LON_STEP).astype(int)
    df['row'] = np.floor((df['latitude']  - GRID_LAT_MIN) / LAT_STEP).astype(int)
    in_grid   = df['col'].between(0, N_COLS-1) & df['row'].between(0, N_ROWS-1)
    df = df[in_grid].copy()
    df['cell_id']   = df['col'].astype(str) + '_' + df['row'].astype(str)
    df['date_only'] = df['event_date'].dt.normalize()
    return df

# ── static features ────────────────────────────────────────────────────────

def elevation_features(grid: pd.DataFrame, dem: dict) -> pd.DataFrame:
    arr  = dem['data']
    nrow, ncol = arr.shape
    dlat = dem['lat_max'] - dem['lat_min']
    dlon = dem['lon_max'] - dem['lon_min']
    recs = []
    for _, cell in grid.iterrows():
        r0 = int((dem['lat_max'] - cell['lat_max']) / dlat * nrow)
        r1 = int((dem['lat_max'] - cell['lat_min']) / dlat * nrow)
        c0 = int((cell['lon_min'] - dem['lon_min']) / dlon * ncol)
        c1 = int((cell['lon_max'] - dem['lon_min']) / dlon * ncol)
        r0 = max(0, min(r0, nrow-1)); r1 = max(r0+1, min(r1, nrow))
        c0 = max(0, min(c0, ncol-1)); c1 = max(c0+1, min(c1, ncol))
        px = arr[r0:r1, c0:c1].ravel()
        v  = px[~np.isnan(px)]
        recs.append({'cell_id':   cell['cell_id'],
                     'elev_mean': float(np.mean(v)) if len(v) else np.nan,
                     'elev_std':  float(np.std(v))  if len(v) else np.nan,
                     'elev_max':  float(np.max(v))  if len(v) else np.nan,
                     'elev_min':  float(np.min(v))  if len(v) else np.nan})
    return pd.DataFrame(recs)

_LULC_COLS = ['lulc_water','lulc_paddy','lulc_crop','lulc_urban',
              'lulc_grass','lulc_deciduous','lulc_mixed','lulc_conifer','lulc_bare']

def lulc_features(grid: pd.DataFrame, elev: pd.DataFrame) -> pd.DataFrame:
    df = grid[['cell_id']].merge(elev[['cell_id','elev_mean']], on='cell_id', how='left')
    def classify(e):
        zero = {c: 0.0 for c in _LULC_COLS}
        if pd.isna(e): return zero
        if e < 0:      return {**zero, 'lulc_water': 1.0}
        if e < 20:     return {**zero, 'lulc_paddy': 0.55, 'lulc_crop': 0.20,
                               'lulc_urban': 0.15, 'lulc_grass': 0.10}
        if e < 100:    return {**zero, 'lulc_crop': 0.40, 'lulc_urban': 0.20,
                               'lulc_grass': 0.20, 'lulc_deciduous': 0.20}
        if e < 400:    return {**zero, 'lulc_deciduous': 0.60, 'lulc_mixed': 0.25,
                               'lulc_grass': 0.10, 'lulc_crop': 0.05}
        if e < 800:    return {**zero, 'lulc_mixed': 0.50, 'lulc_deciduous': 0.30,
                               'lulc_conifer': 0.20}
        if e < 1500:   return {**zero, 'lulc_conifer': 0.60, 'lulc_mixed': 0.30,
                               'lulc_bare': 0.10}
        return             {**zero, 'lulc_bare': 0.55, 'lulc_conifer': 0.25, 'lulc_grass': 0.20}
    lulc = df['elev_mean'].apply(classify).apply(pd.Series)
    lulc.insert(0, 'cell_id', df['cell_id'].values)
    return lulc

_AKITA_CITIES = [
    (39.7192, 140.1025, 301000),
    (39.3147, 140.5618,  87000),
    (39.4625, 140.4849,  84000),
    (39.3906, 140.0490,  76000),
    (40.2797, 140.5640,  61000),
    (40.2092, 140.0313,  52000),
    (39.1621, 140.4964,  44000),
    (39.8883, 140.0189,  33000),
    (40.2231, 140.3773,  29000),
    (39.8806, 139.8494,  27000),
]

def pop_features(grid: pd.DataFrame) -> pd.DataFrame:
    lats = grid['lat_center'].values
    lons = grid['lon_center'].values
    pop_den  = np.zeros(len(grid), np.float32)
    min_dist = np.full(len(grid), np.inf, np.float32)
    for clat, clon, pop in _AKITA_CITIES:
        dy = (lats - clat) * 111.0
        dx = (lons - clon) * 111.0 * math.cos(math.radians(REF_LAT))
        d  = np.clip(np.sqrt(dy**2 + dx**2), 0.5, None)
        pop_den  += pop / d**2
        min_dist  = np.minimum(min_dist, d)
    scale = sum(p for _, _, p in _AKITA_CITIES) / 10.0
    return pd.DataFrame({'cell_id':          grid['cell_id'],
                         'pop_density':       (pop_den / scale).astype(np.float32),
                         'log_pop_density':   np.log1p(pop_den / scale).astype(np.float32),
                         'dist_nearest_city': min_dist})

# ── temporal features ──────────────────────────────────────────────────────

def temporal_features(year: int, month: int) -> dict:
    return {
        'month_sin':         math.sin(2 * math.pi * month / 12),
        'month_cos':         math.cos(2 * math.pi * month / 12),
        'season':            (month - 1) // 3,
        'active_season':     int(4 <= month <= 11),
        'pre_hibernation':   int(month in (9, 10, 11)),
        'post_hibernation':  int(month in (4, 5)),
        'years_since_start': year - min(TRAIN_YEARS),
    }

FEATURE_COLS = [
    'elev_mean','elev_std','elev_max','elev_min',
    *_LULC_COLS,
    'pop_density','log_pop_density','dist_nearest_city',
    'month_sin','month_cos','season','active_season',
    'pre_hibernation','post_hibernation','years_since_start',
    'hist_positive_rate',
]

# ── labels & feature matrix ────────────────────────────────────────────────

def make_train_labels(train_s: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    all_cells  = grid['cell_id'].tolist()
    all_months = list(range(1, 13))
    rows = [{'cell_id': c, 'year': y, 'month': m, 'label': 0}
            for y in TRAIN_YEARS for c in all_cells for m in all_months]
    labels = pd.DataFrame(rows)
    hits = (train_s.groupby(['cell_id','year','month']).size()
            .reset_index(name='cnt').assign(label=1))
    labels = labels.merge(hits[['cell_id','year','month','label']],
                          on=['cell_id','year','month'], how='left',
                          suffixes=('','_h'))
    labels['label'] = labels['label_h'].fillna(0).astype(int)
    return labels.drop(columns='label_h')

def build_features(label_df: pd.DataFrame, static: pd.DataFrame,
                   hist_rate: pd.Series) -> pd.DataFrame:
    df = label_df.merge(static, on='cell_id', how='left')
    df = df.merge(hist_rate.rename('hist_positive_rate'), on='cell_id', how='left')
    temporal = pd.DataFrame([temporal_features(r.year, r.month)
                              for r in df[['year','month']].itertuples()],
                             index=df.index)
    return pd.concat([df, temporal], axis=1)

# ── model ──────────────────────────────────────────────────────────────────

def train_et(X, y):
    pos = y.sum()
    print(f"  {len(y):,} samples, {pos} positive ({100*pos/len(y):.1f}%)")
    Xf = X.fillna(X.median())
    rus = RandomUnderSampler(random_state=42)
    Xr, yr = rus.fit_resample(Xf, y)
    print(f"  After undersampling: {len(yr):,}")
    clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(Xr, yr)
    return clf

# ── evaluation ─────────────────────────────────────────────────────────────

def evaluate(clf, test_s: pd.DataFrame, static: pd.DataFrame,
             hist_rate: pd.Series, grid: pd.DataFrame):
    all_cells = grid['cell_id'].tolist()

    # Monthly predictions for 2025
    monthly_rows = [{'cell_id': c, 'year': TEST_YEAR, 'month': m}
                    for m in range(1, 13) for c in all_cells]
    mdf = pd.DataFrame(monthly_rows)
    mdf = mdf.merge(static, on='cell_id', how='left')
    mdf = mdf.merge(hist_rate.rename('hist_positive_rate'), on='cell_id', how='left')
    temporal = pd.DataFrame([temporal_features(r.year, r.month)
                              for r in mdf[['year','month']].itertuples()],
                             index=mdf.index)
    mdf = pd.concat([mdf, temporal], axis=1)
    Xm = mdf[FEATURE_COLS].fillna(mdf[FEATURE_COLS].median())
    mdf['proba'] = clf.predict_proba(Xm)[:, 1]

    proba_by_month = {mo: mdf[mdf.month == mo].set_index('cell_id')['proba']
                      for mo in range(1, 13)}

    # Daily sightings lookup
    all_days = pd.date_range(f'{TEST_YEAR}-01-01', f'{TEST_YEAR}-12-31')
    day_sightings = {}
    for day in all_days:
        mask = test_s['event_date'].dt.date == day.date()
        day_sightings[day] = set(test_s[mask]['cell_id'].values)

    avg_daily_pos = np.mean([len(v) for v in day_sightings.values()])

    # Global P@K and R@K
    day_p = {k: [] for k in K_VALUES}
    day_r = {k: [] for k in K_VALUES}
    for day in all_days:
        proba   = proba_by_month[day.month]
        ranked  = proba.sort_values(ascending=False).index.tolist()
        present = day_sightings[day]
        n_pos   = len(present)
        for k in K_VALUES:
            topk = set(ranked[:k])
            hits = len(topk & present)
            day_p[k].append(hits / k)
            day_r[k].append(hits / n_pos if n_pos > 0 else np.nan)

    # Global ROC-AUC (monthly, per cell)
    test_monthly = (test_s.groupby(['cell_id','month']).size()
                    .reset_index(name='cnt').assign(label=1))
    cell_roc = {}
    for cid in all_cells:
        yt = []; yp = []
        for mo in range(1, 13):
            row = test_monthly[(test_monthly.cell_id == cid) & (test_monthly.month == mo)]
            yt.append(1 if len(row) > 0 else 0)
            yp.append(float(proba_by_month[mo].get(cid, 0.5)))
        if 0 < sum(yt) < 12:
            try:
                cell_roc[cid] = roc_auc_score(yt, yp)
            except Exception:
                pass

    res = {
        'avg_daily_pos':    avg_daily_pos,
        'n_cells_roc':      len(cell_roc),
        'roc_auc_avg':      np.mean(list(cell_roc.values())) if cell_roc else np.nan,
    }
    for k in K_VALUES:
        rv = [v for v in day_r[k] if not np.isnan(v)]
        res[f'precision_at_{k}'] = np.mean(day_p[k])
        res[f'recall_at_{k}']    = np.mean(rv) if rv else 0.0

    # Seasonal (K=20)
    for season, months in SEASONS.items():
        ps, rs = [], []
        for day in all_days:
            if day.month not in months:
                continue
            proba  = proba_by_month[day.month]
            ranked = proba.sort_values(ascending=False).index.tolist()
            top20  = set(ranked[:20])
            present= day_sightings[day]
            n_pos  = len(present)
            ps.append(len(top20 & present) / 20)
            if n_pos > 0:
                rs.append(len(top20 & present) / n_pos)
        res[f'precision_{season}_K20'] = np.mean(ps) if ps else 0.0
        res[f'recall_{season}_K20']    = np.mean(rs) if rs else 0.0

    return res, cell_roc, proba_by_month, day_sightings

# ── per-cell metrics (generalized: TOP20 or ALL-GRID) ──────────────────────

def ece_score(yt, yp, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece  = 0.0
    n    = len(yt)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (yp >= lo) & (yp < hi)
        if not mask.any():
            continue
        ece += mask.sum()/n * abs(yt[mask].mean() - yp[mask].mean())
    return float(ece)

def compute_percell_metrics(cell_list, proba_by_month, day_sightings):
    """Compute per-cell daily calibration metrics for any list of cells."""
    all_days = sorted(day_sightings.keys())
    n_days   = len(all_days)

    day_rankings = {d: proba_by_month[d.month].sort_values(ascending=False).index.tolist()
                    for d in all_days}

    records = []
    for cid in cell_list:
        yp    = np.array([float(proba_by_month[d.month].get(cid, 0.0)) for d in all_days])
        yt    = np.array([int(cid in day_sightings[d]) for d in all_days])
        n_pos = int(yt.sum())
        if n_pos == 0 or n_pos == n_days:
            continue
        roc  = roc_auc_score(yt, yp)
        pr   = average_precision_score(yt, yp)
        brier= brier_score_loss(yt, yp)
        mae  = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt)**2)))
        ece  = ece_score(yt, yp)

        rec = {'grid_id': cid,
               'PR_AUC': pr, 'ROC_AUC': roc,
               'Brier': brier, 'ECE': ece, 'MAE': mae, 'RMSE': rmse}
        for k in K_VALUES:
            in_topk = np.array([int(cid in day_rankings[d][:k]) for d in all_days])
            tp = int((in_topk & yt.astype(bool)).sum())
            p_k = tp / max(in_topk.sum(), 1)
            r_k = tp / max(n_pos, 1)
            rec[f'Precision_{k}'] = p_k
            rec[f'Recall_{k}']    = r_k
            if k == 10:
                rec['Hit_Top10'] = r_k
        records.append(rec)

    return pd.DataFrame(records).set_index('grid_id')

# ── print results ──────────────────────────────────────────────────────────

def print_global_results(res):
    avg_pos  = res['avg_daily_pos']
    rnd_prec = avg_pos / N_CELLS

    print('\n' + '='*74)
    print('AKITA — ExtraTrees 10km×10km  グローバル評価')
    print(f'Grid: {N_ROWS}x{N_COLS}={N_CELLS} cells | Test: {TEST_YEAR} (365 days)')
    print(f'Train: {TRAIN_YEARS} | 平均出没セル/日: {avg_pos:.1f} | ランダムP: {rnd_prec:.4f}')
    print('='*74)

    print(f'\n  {"指標":<30} {"ExtraTrees":<20} {"ランダム"}')
    print('  ' + '-'*65)
    print(f'  {"ROC-AUC (全活動セル平均)":<30} '
          f'{res["roc_auc_avg"]:.3f}  (n={res["n_cells_roc"]})      0.500')
    for k in K_VALUES:
        rnd_r = k / N_CELLS
        ep = res[f'precision_at_{k}']
        er = res[f'recall_at_{k}']
        print(f'  Precision@{k:<4} ({k/N_CELLS*100:.1f}% cells)        '
              f'{ep:.3f}  ({ep/rnd_prec:.1f}x rnd)        {rnd_prec:.4f}')
        print(f'  Recall@{k:<4}    ({k/N_CELLS*100:.1f}% cells)        '
              f'{er:.3f}  ({er/rnd_r:.1f}x rnd)        {rnd_r:.4f}')

    print('\n  季節別 K=20:')
    print(f'  {"季節":<22} {"Precision@20":<16} {"Recall@20"}')
    print('  ' + '-'*50)
    for season, label in [('spring','春 (4-5月)'), ('summer','夏 (6-8月)'),
                           ('fall','秋 (9-11月)'),  ('winter','冬 (12-3月)')]:
        ep = res[f'precision_{season}_K20']
        er = res[f'recall_{season}_K20']
        print(f'  {label:<22} {ep:.3f}            {er:.3f}')

def print_allgrid_summary(et_df: pd.DataFrame):
    """Print per-cell metric averages over all evaluable cells (260-cell pool)."""
    m = et_df.mean()
    metric_cols = ['PR_AUC','ROC_AUC','Brier','ECE','MAE','RMSE']
    print('\n' + '='*60)
    print(f'全グリッド ({len(et_df)} セル) per-cell 平均指標  [Test={TEST_YEAR}]')
    print('='*60)
    for col in metric_cols:
        print(f'  {col:<12}: {float(m[col]):.4f}')

def print_top20_comparison(et_df: pd.DataFrame):
    # TTM-512 hardcoded averages (from external evaluation)
    TTM_MEAN = {
        'PR_AUC':0.3451,'ROC_AUC':0.5465,'Brier':0.2535,'ECE':0.2098,
        'MAE':0.3410,'RMSE':0.5016,'Hit_Top10':0.1075,
        'Precision_10':0.3222,'Precision_20':0.3274,'Precision_30':0.3202,
        'Recall_10':0.1075,'Recall_20':0.2181,'Recall_30':0.3174,
    }
    et_mean = et_df.mean()

    metric_cols = ['PR_AUC','ROC_AUC','Brier','ECE','MAE','RMSE','Hit_Top10',
                   'Precision_10','Precision_20','Precision_30',
                   'Recall_10','Recall_20','Recall_30']

    print('\n' + '='*80)
    print(f'TOP20セル 平均比較: ExtraTrees vs TTM-512  (n={len(et_df)} cells, Test={TEST_YEAR})')
    print('='*80)
    better_markers = {'PR_AUC','ROC_AUC','Hit_Top10',
                      'Precision_10','Precision_20','Precision_30',
                      'Recall_10','Recall_20','Recall_30'}
    fmt = '  {:<22} {:>12} {:>12} {:>12} {:>8}'
    print(fmt.format('指標','ExtraTrees','TTM-512','差(ET-TTM)','優位'))
    print('  ' + '-'*72)
    for m in metric_cols:
        ev = float(et_mean.get(m, float('nan')))
        tv = TTM_MEAN.get(m, float('nan'))
        diff = ev - tv
        if m in better_markers:
            better = 'ET↑' if diff > 0 else 'TTM↑'
        else:
            better = 'ET↑' if diff < 0 else 'TTM↑'
        print(fmt.format(m, f'{ev:.4f}', f'{tv:.4f}', f'{diff:+.4f}', better))

    # Per-cell ROC-AUC detail
    print('\n  【Per-cell ROC-AUC】')
    print(f'  {"cell":>8} {"ET":>8} {"TTM":>8} {"diff":>8}')
    TTM_ROC = {'5_14':0.6352,'4_8':0.5838,'6_7':0.5545,'5_15':0.5000,'3_14':0.6690,
               '7_16':0.6549,'9_16':0.7172,'5_8':0.4313,'6_14':0.5675,'8_15':0.5224,
               '9_14':0.6928,'3_11':0.4206,'9_15':0.6878,'6_15':0.5530,'7_15':0.3565,
               '3_10':0.5055,'5_9':0.4630,'4_10':0.5541,'7_8':0.5358,'4_9':0.3257}
    for cid in TOP20:
        ev = float(et_df.loc[cid,'ROC_AUC']) if cid in et_df.index else float('nan')
        tv = TTM_ROC.get(cid, float('nan'))
        print(f'  {cid:>8} {ev:>8.3f} {tv:>8.3f} {ev-tv:>+8.3f}')

# ── plots ──────────────────────────────────────────────────────────────────

def save_plots(res, et_df: pd.DataFrame):
    avg_pos  = res['avg_daily_pos']
    rnd_prec = avg_pos / N_CELLS

    # 1. Global P@K / R@K
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Akita 10km ET — Global P@K & R@K (Test {TEST_YEAR})', fontweight='bold')
    x = np.arange(len(K_VALUES)); w = 0.35
    for ax, metric, title, rnd_fn in [
        (axes[0],'precision','Precision@K (daily avg)',lambda k: rnd_prec),
        (axes[1],'recall',   'Recall@K (daily avg)',   lambda k: k/N_CELLS),
    ]:
        ev  = [res[f'{metric}_at_{k}'] for k in K_VALUES]
        rv  = [rnd_fn(k) for k in K_VALUES]
        ax.bar(x-w/2, ev, w, label='ExtraTrees', color='steelblue', alpha=0.85)
        ax.bar(x+w/2, rv, w, label='Random',     color='gray',      alpha=0.55)
        for bars, vals in [(ax.containers[0],ev),(ax.containers[1],rv)]:
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f'K={k}' for k in K_VALUES])
        ax.set_title(title); ax.legend(); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(ev)*1.4+0.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/'akita10km_global_pk_rk.png', dpi=150)
    plt.close()

    # 2. TOP20 comparison bar
    TTM_MEAN = {'ROC_AUC':0.5465,'PR_AUC':0.3451,
                'Recall_10':0.1075,'Recall_20':0.2181,'Recall_30':0.3174,
                'Precision_10':0.3222,'Precision_20':0.3274,'Precision_30':0.3202}
    plot_m = ['ROC_AUC','PR_AUC','Recall_10','Recall_20','Recall_30',
              'Precision_10','Precision_20','Precision_30']
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(plot_m)); w = 0.35
    et_mean = et_df.mean()
    ev = [float(et_mean.get(m, 0)) for m in plot_m]
    tv = [TTM_MEAN.get(m, 0) for m in plot_m]
    ax.bar(x-w/2, ev, w, label='ExtraTrees', color='steelblue')
    ax.bar(x+w/2, tv, w, label='TTM-512',    color='coral')
    ax.set_xticks(x); ax.set_xticklabels(plot_m, rotation=30, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1); ax.legend()
    ax.set_title(f'Akita 10km TOP20平均: ExtraTrees vs TTM-512 (Test={TEST_YEAR})')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR/'akita10km_top20_vs_ttm512.png', dpi=150)
    plt.close()
    print(f'  Plots saved to {RESULTS_DIR}/')

# ── main ───────────────────────────────────────────────────────────────────

def main():
    print('='*74)
    print('Akita Prefecture — ExtraTrees 10km×10km Benchmark')
    print(f'Grid: {N_ROWS}x{N_COLS}={N_CELLS} cells | Train:{TRAIN_YEARS} | Test:{TEST_YEAR}')
    print('='*74)

    print('\n[1/7] グリッド構築...')
    grid = build_grid()
    print(f'  {len(grid)} cells  lat {GRID_LAT_MIN:.4f}-{GRID_LAT_MAX:.4f},'
          f' lon {GRID_LON_MIN:.4f}-{GRID_LON_MAX:.4f}')

    print('\n[2/7] 出没データ読込...')
    all_s   = load_sightings()
    train_s = assign_to_grid(load_sightings(TRAIN_YEARS))
    test_s  = assign_to_grid(load_sightings([TEST_YEAR]))
    print(f'  Train {TRAIN_YEARS}: {len(train_s):,}件  |  Test {TEST_YEAR}: {len(test_s):,}件')
    print(f'  2025出没セル数: {test_s.cell_id.nunique()} / {N_CELLS}')

    print('\n[3/7] 月次ラベル生成...')
    train_labels = make_train_labels(train_s, grid)
    pos = train_labels.label.sum()
    print(f'  {len(train_labels):,} cell×month, pos={pos} ({100*pos/len(train_labels):.1f}%)')
    hist_rate = train_labels.groupby('cell_id')['label'].mean()

    print('\n[4/7] 標高・土地被覆・人口特徴量...')
    dem  = download_dem()
    elev = elevation_features(grid, dem)
    lulc = lulc_features(grid, elev)
    pop  = pop_features(grid)
    static = (grid[['cell_id']]
              .merge(elev, on='cell_id', how='left')
              .merge(lulc, on='cell_id', how='left')
              .merge(pop,  on='cell_id', how='left'))
    print(f'  elev: {elev.elev_mean.min():.0f}-{elev.elev_mean.max():.0f} m')

    print('\n[5/7] 学習特徴量行列...')
    train_df = build_features(train_labels, static, hist_rate)
    X_train  = train_df[FEATURE_COLS]
    y_train  = train_df['label']
    print(f'  Shape: {X_train.shape}')

    print('\n[6/7] ExtraTreesClassifier + RandomUnderSampler...')
    clf = train_et(X_train, y_train)
    fi  = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print('  特徴量重要度 TOP5:', fi.head(5).to_dict())
    fi.to_csv(RESULTS_DIR / 'feature_importance_akita10km.csv', header=True)

    print(f'\n[7/7] 評価 (2025年 365日)...')
    res, cell_roc, proba_by_month, day_sightings = evaluate(
        clf, test_s, static, hist_rate, grid)

    # Global results
    print_global_results(res)

    # Per-cell metrics: TOP20 and all 260 cells
    all_cells   = grid['cell_id'].tolist()
    et_top20    = compute_percell_metrics(TOP20,     proba_by_month, day_sightings)
    et_allgrid  = compute_percell_metrics(all_cells, proba_by_month, day_sightings)
    print_allgrid_summary(et_allgrid)
    print_top20_comparison(et_top20)

    # Plots & CSV
    save_plots(res, et_top20)

    # Save per-cell ROC
    roc_df = (pd.DataFrame({'cell_id': list(cell_roc.keys()),
                            'roc_auc': list(cell_roc.values())})
              .sort_values('roc_auc', ascending=False))
    roc_df.to_csv(RESULTS_DIR/'per_cell_roc_akita10km.csv', index=False, float_format='%.4f')
    print(f'  ROC-AUC: mean={roc_df.roc_auc.mean():.3f}, '
          f'>0.6: {(roc_df.roc_auc>0.6).sum()}/{len(roc_df)}')

    # Save global summary
    rows = [{'metric':'roc_auc_avg', 'ET':f"{res['roc_auc_avg']:.4f}", 'random':'0.5000'}]
    rnd_p = res['avg_daily_pos'] / N_CELLS
    for k in K_VALUES:
        rows += [
            {'metric':f'precision_at_K{k}',
             'ET':f"{res[f'precision_at_{k}']:.4f}", 'random':f'{rnd_p:.4f}'},
            {'metric':f'recall_at_K{k}',
             'ET':f"{res[f'recall_at_{k}']:.4f}", 'random':f'{k/N_CELLS:.4f}'},
        ]
    for s in SEASONS:
        rows += [
            {'metric':f'precision_{s}_K20',
             'ET':f"{res[f'precision_{s}_K20']:.4f}", 'random':f'{rnd_p:.4f}'},
            {'metric':f'recall_{s}_K20',
             'ET':f"{res[f'recall_{s}_K20']:.4f}", 'random':f'{20/N_CELLS:.4f}'},
        ]
    pd.DataFrame(rows).to_csv(RESULTS_DIR/'akita10km_et_global.csv', index=False)
    et_top20.to_csv(RESULTS_DIR/'akita10km_top20_percell.csv',   float_format='%.6f')
    et_allgrid.to_csv(RESULTS_DIR/'akita10km_allgrid_percell.csv', float_format='%.6f')
    print(f'  CSVs saved to {RESULTS_DIR}/')

    print('\n' + '='*74)
    print('完了')
    print('='*74)


if __name__ == '__main__':
    main()
