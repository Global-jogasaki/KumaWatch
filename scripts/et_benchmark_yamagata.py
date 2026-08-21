#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExtraTrees baseline for DIRECT comparison with TTM paper
(城ヶ﨑 2025, 山形大学産業研究所 米沢市研究奨励研究)

Uses the EXACT Yamagata 10km grid (Yamagata_10km_Grid_0.csv):
  9 cols × 16 rows = 144 cells, each ~10km × 10km
  lat_min=37.758430, lon_min=139.549091
  lat_step=0.090090°, lon_step=0.114326°
  Grid_ID format: {col}_{row}  (same as TTM paper)

TTM paper reported results (表1・表2, 城ヶ﨑 2025):
  ROC-AUC avg (20 active cells):  0.634
  K=10: Precision=14.6%, Recall=30.4%
  K=20: Precision=13.8%, Recall=54.4%  ← 3.9× random (recommended)
  K=30: Precision= 9.4%, Recall=55.4%
  Seasonal K=20: Spring 4.6%/58.8%, Summer 20.3%/54.2%,
                 Fall 31.4%/48.1%, Winter 0.4%/96.3%
  Train: 2018-10 ─ 2024-12 (~2,284 days)
  Test:  2025-01 ─ 2025-12 (365 days)
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import warnings
from pathlib import Path
import math
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CACHE_DIR     = BASE_DIR / '.cache'
RESULTS_DIR   = BASE_DIR / 'results_10km'
RESULTS_DIR.mkdir(exist_ok=True)

SIGHTINGS_DIR = BASE_DIR.parent / 'bear-sighting-data' / 'data' / 'yamagata'
GRID_CSV      = Path(r'F:\SSD-PGU3\電動モビリティ専門職大学\山形大学申請\山形県データ\Yamagata_10km_Grid_0.csv')

# ── exact grid parameters from Yamagata_10km_Grid_0.csv ────────────────────
LAT_STEP     = 0.090090   # ≈ 10.0 km
LON_STEP     = 0.114326   # ≈ 10.0 km (ref 38°N)
GRID_LAT_MIN = 37.758430
GRID_LON_MIN = 139.549091
N_ROWS       = 16
N_COLS       = 9
N_CELLS      = N_ROWS * N_COLS   # 144

TRAIN_YEARS  = list(range(2018, 2025))   # 2018-2024
TEST_YEAR    = 2025
RANDOM_STATE = 42
K_VALUES     = [10, 20, 30]

# TTM paper results (城ヶ﨑 2025, 表1・表2)
TTM = {
    'roc_auc_avg20':     0.634,
    'precision_at_10':   0.146,
    'recall_at_10':      0.304,
    'precision_at_20':   0.138,
    'recall_at_20':      0.544,
    'precision_at_30':   0.094,
    'recall_at_30':      0.554,
    'precision_spring':  0.046,
    'recall_spring':     0.588,
    'precision_summer':  0.203,
    'recall_summer':     0.542,
    'precision_fall':    0.314,
    'recall_fall':       0.481,
    'precision_winter':  0.004,
    'recall_winter':     0.963,
}

SEASONS = {
    'spring': [4, 5],
    'summer': [6, 7, 8],
    'fall':   [9, 10, 11],
    'winter': [12, 1, 2, 3],
}


# ── 1. GRID (from CSV) ─────────────────────────────────────────────────────

def load_grid() -> pd.DataFrame:
    g = pd.read_csv(GRID_CSV)
    # Ensure consistent column names
    g = g.rename(columns={'Grid_ID': 'cell_id', 'Grid_Row': 'row',
                           'Grid_Col': 'col', 'Center_Latitude': 'lat_center',
                           'Center_Longitude': 'lon_center',
                           'Min_Latitude': 'lat_min', 'Max_Latitude': 'lat_max',
                           'Min_Longitude': 'lon_min', 'Max_Longitude': 'lon_max'})
    print(f"  Grid loaded: {len(g)} cells  ({N_ROWS} rows x {N_COLS} cols)")
    print(f"  lat {g.lat_min.min():.4f}-{g.lat_max.max():.4f}, "
          f"lon {g.lon_min.min():.4f}-{g.lon_max.max():.4f}")
    return g


# ── 2. SIGHTINGS ───────────────────────────────────────────────────────────

def load_and_assign(years: list, grid: pd.DataFrame) -> pd.DataFrame:
    dfs = []
    for yr in years:
        p = SIGHTINGS_DIR / f'sightings_{yr}.csv'
        if p.exists():
            df = pd.read_csv(p, parse_dates=['event_date'])
            df['year']  = df['event_date'].dt.year
            df['month'] = df['event_date'].dt.month
            df['day']   = df['event_date'].dt.day
            df['doy']   = df['event_date'].dt.dayofyear
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No sighting files for {years}")
    s = pd.concat(dfs, ignore_index=True).dropna(subset=['latitude', 'longitude', 'event_date'])

    # Assign using exact grid boundaries
    s['col'] = np.floor((s['longitude'] - GRID_LON_MIN) / LON_STEP).astype(int)
    s['row'] = np.floor((s['latitude']  - GRID_LAT_MIN) / LAT_STEP).astype(int)
    in_grid  = s['row'].between(0, N_ROWS-1) & s['col'].between(0, N_COLS-1)
    s = s[in_grid].copy()
    s['cell_id'] = s['col'].astype(str) + '_' + s['row'].astype(str)   # col_row
    return s


# ── 3. LABELS ──────────────────────────────────────────────────────────────

def make_train_labels(sightings: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in TRAIN_YEARS:
        for mo in range(1, 13):
            for _, cell in grid.iterrows():
                rows.append({'cell_id': cell['cell_id'], 'year': yr, 'month': mo})
    base = pd.DataFrame(rows)
    pos  = (sightings[sightings.year.isin(TRAIN_YEARS)]
            [['cell_id', 'year', 'month']]
            .drop_duplicates()
            .assign(label=1))
    df = base.merge(pos, on=['cell_id', 'year', 'month'], how='left')
    df['label'] = df['label'].fillna(0).astype(np.int8)
    return df


# ── 4. FEATURES ────────────────────────────────────────────────────────────

def elevation_features(grid: pd.DataFrame) -> pd.DataFrame:
    dem_cache = CACHE_DIR / 'dem_assembled_z11.pkl'
    if dem_cache.exists():
        with open(dem_cache, 'rb') as f:
            dem = pickle.load(f)
        arr  = dem['data']
        nrow, ncol = arr.shape
        dlat = dem['lat_max'] - dem['lat_min']
        dlon = dem['lon_max'] - dem['lon_min']
        records = []
        for _, cell in grid.iterrows():
            r0 = int((dem['lat_max'] - cell['lat_max']) / dlat * nrow)
            r1 = int((dem['lat_max'] - cell['lat_min']) / dlat * nrow)
            c0 = int((cell['lon_min'] - dem['lon_min']) / dlon * ncol)
            c1 = int((cell['lon_max'] - dem['lon_min']) / dlon * ncol)
            r0 = max(0, min(r0, nrow-1)); r1 = max(r0+1, min(r1, nrow))
            c0 = max(0, min(c0, ncol-1)); c1 = max(c0+1, min(c1, ncol))
            px = arr[r0:r1, c0:c1].ravel()
            v  = px[~np.isnan(px)]
            if len(v) > 0:
                records.append({'cell_id': cell['cell_id'],
                                'elev_mean': float(np.mean(v)), 'elev_std': float(np.std(v)),
                                'elev_max':  float(np.max(v)),  'elev_min': float(np.min(v))})
            else:
                records.append({'cell_id': cell['cell_id'],
                                'elev_mean': np.nan, 'elev_std': np.nan,
                                'elev_max': np.nan,  'elev_min': np.nan})
        return pd.DataFrame(records)

    # Proxy (no DEM cache)
    print("  (DEM cache not found; using elevation proxy)")
    lat = grid['lat_center'].values; lon = grid['lon_center'].values
    d_e = np.abs(lon - 140.45) * 80; d_w = np.abs(lon - 139.78) * 80
    elev = np.clip(np.minimum(d_e, d_w) * 1.5 + np.abs(lat-38.2)*300, 0, 2000)
    return pd.DataFrame({'cell_id': grid['cell_id'], 'elev_mean': elev,
                         'elev_std': elev*0.25, 'elev_max': elev*1.35, 'elev_min': elev*0.65})


_LULC_COLS = ['lulc_water','lulc_urban','lulc_paddy','lulc_crop',
              'lulc_grass','lulc_deciduous','lulc_mixed','lulc_conifer','lulc_bare']

def lulc_features(grid, elev):
    df = grid[['cell_id']].merge(elev[['cell_id','elev_mean']], on='cell_id')
    def classify(e):
        zero = {c: 0.0 for c in _LULC_COLS}
        if pd.isna(e):  return zero
        if e < 0:       return {**zero, 'lulc_water': 1.0}
        if e < 20:      return {**zero, 'lulc_paddy':0.55,'lulc_crop':0.20,'lulc_urban':0.15,'lulc_grass':0.10}
        if e < 100:     return {**zero, 'lulc_crop':0.40,'lulc_urban':0.20,'lulc_grass':0.20,'lulc_deciduous':0.20}
        if e < 400:     return {**zero, 'lulc_deciduous':0.60,'lulc_mixed':0.25,'lulc_grass':0.10,'lulc_crop':0.05}
        if e < 800:     return {**zero, 'lulc_mixed':0.50,'lulc_deciduous':0.30,'lulc_conifer':0.20}
        if e < 1500:    return {**zero, 'lulc_conifer':0.60,'lulc_mixed':0.30,'lulc_bare':0.10}
        return              {**zero, 'lulc_bare':0.55,'lulc_conifer':0.25,'lulc_grass':0.20}
    lulc = df['elev_mean'].apply(classify).apply(pd.Series)
    lulc.insert(0, 'cell_id', df['cell_id'].values)
    return lulc

# Yamagata major cities (lat, lon, population)
_CITIES = [(38.2404,140.3634,249000),(38.7267,139.8268,124000),
           (38.9141,139.8369,101000),(37.9222,140.1174, 82000),
           (38.3667,140.3667, 59000),(38.7736,140.3114, 37000),
           (38.4264,140.3638, 55000),(38.5500,140.4000, 25000)]

def pop_features(grid):
    lats = grid['lat_center'].values; lons = grid['lon_center'].values
    REF  = 38.5
    pop_den = np.zeros(len(grid), np.float32)
    min_dist = np.full(len(grid), np.inf, np.float32)
    for clat, clon, pop in _CITIES:
        dy = (lats-clat)*111; dx = (lons-clon)*111*math.cos(math.radians(REF))
        d = np.clip(np.sqrt(dy**2+dx**2), 0.5, None)
        pop_den += pop/d**2; min_dist = np.minimum(min_dist, d)
    scale = sum(p for _,_,p in _CITIES) / 10.0
    return pd.DataFrame({'cell_id': grid['cell_id'],
                         'pop_density':     (pop_den/scale).astype(np.float32),
                         'log_pop_density': np.log1p(pop_den/scale).astype(np.float32),
                         'dist_nearest_city': min_dist})

def temporal_feats(year, month):
    return {'month_sin': math.sin(2*math.pi*month/12),
            'month_cos': math.cos(2*math.pi*month/12),
            'season':    (month-1)//3,
            'active_season':    int(4<=month<=11),
            'pre_hibernation':  int(month in (9,10,11)),
            'post_hibernation': int(month in (4,5)),
            'years_since_start': year-2018}

FEATURE_COLS = ['elev_mean','elev_std','elev_max','elev_min',
                *_LULC_COLS,
                'pop_density','log_pop_density','dist_nearest_city',
                'month_sin','month_cos','season','active_season',
                'pre_hibernation','post_hibernation','years_since_start',
                'hist_positive_rate']

def build_features(label_df, static, hist_rate):
    df = label_df.merge(static, on='cell_id', how='left')
    df = df.merge(hist_rate.rename('hist_positive_rate'), on='cell_id', how='left')
    temp = pd.DataFrame([temporal_feats(r.year, r.month)
                         for r in df[['year','month']].itertuples()], index=df.index)
    return pd.concat([df, temp], axis=1)


# ── 5. MODEL ───────────────────────────────────────────────────────────────

def train_et(X, y):
    pos = y.sum()
    print(f"  Train: {len(y):,} samples, {pos} positive ({100*pos/len(y):.2f}%)")
    X_f = X.fillna(X.median())
    X_r, y_r = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(X_f, y)
    print(f"  After undersampling: {len(y_r):,} samples")
    clf = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_r, y_r)
    return clf


# ── 6. DAILY EVALUATION ────────────────────────────────────────────────────

def evaluate_daily(clf, test_s, static, hist_rate, grid):
    # Monthly predictions for 2025
    monthly_rows = [{'cell_id': c['cell_id'], 'year': TEST_YEAR, 'month': mo}
                    for mo in range(1,13) for _, c in grid.iterrows()]
    mdf = pd.DataFrame(monthly_rows)
    mdf = mdf.merge(static, on='cell_id', how='left')
    mdf = mdf.merge(hist_rate.rename('hist_positive_rate'), on='cell_id', how='left')
    temp = pd.DataFrame([temporal_feats(r.year, r.month)
                         for r in mdf[['year','month']].itertuples()], index=mdf.index)
    mdf = pd.concat([mdf, temp], axis=1)
    X_m  = mdf[FEATURE_COLS].fillna(mdf[FEATURE_COLS].median())
    mdf['proba'] = clf.predict_proba(X_m)[:, 1]
    proba_by_month = {mo: mdf[mdf.month==mo].set_index('cell_id')['proba']
                      for mo in range(1,13)}

    all_days = pd.date_range(f'{TEST_YEAR}-01-01', f'{TEST_YEAR}-12-31')
    day_sightings = {}
    for day in all_days:
        mask = test_s.event_date.dt.date == day.date()
        day_sightings[day.timetuple().tm_yday] = set(test_s[mask]['cell_id'].values)

    avg_daily_pos = np.mean([len(v) for v in day_sightings.values()])

    day_p = {k: [] for k in K_VALUES}
    day_r = {k: [] for k in K_VALUES}

    for day in all_days:
        ranked = proba_by_month[day.month].sort_values(ascending=False).index.tolist()
        cells  = day_sightings[day.timetuple().tm_yday]
        n_pos  = len(cells)
        for k in K_VALUES:
            top_k = set(ranked[:k])
            hits  = len(top_k & cells)
            day_p[k].append(hits / k)
            day_r[k].append(hits / n_pos if n_pos > 0 else np.nan)

    # ROC-AUC per cell
    test_monthly = (test_s.groupby(['cell_id','month']).size()
                    .reset_index(name='cnt').assign(label=1)[['cell_id','month','label']])
    cell_roc = {}
    for cid in grid['cell_id']:
        yt, yp = [], []
        for mo in range(1,13):
            lbl = 1 if len(test_monthly[(test_monthly.cell_id==cid)&(test_monthly.month==mo)])>0 else 0
            yt.append(lbl)
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

    # Top-20 cells ROC-AUC (matching paper's "データが充実している20地区")
    top20_by_activity = (test_s.groupby('cell_id')['event_date']
                         .count().sort_values(ascending=False).head(20).index.tolist())
    roc_top20 = [cell_roc[c] for c in top20_by_activity if c in cell_roc]
    res['roc_auc_top20']   = np.mean(roc_top20) if roc_top20 else np.nan
    res['n_roc_top20']     = len(roc_top20)

    for k in K_VALUES:
        pv = day_p[k]; rv = [v for v in day_r[k] if not np.isnan(v)]
        res[f'precision_at_{k}'] = np.mean(pv)
        res[f'recall_at_{k}']    = np.mean(rv) if rv else 0.0

    # Seasonal (K=20)
    for season, months in SEASONS.items():
        ps, rs = [], []
        for day in all_days:
            if day.month not in months: continue
            ranked = proba_by_month[day.month].sort_values(ascending=False).index.tolist()
            cells  = day_sightings[day.timetuple().tm_yday]
            n_pos  = len(cells)
            hits   = len(set(ranked[:20]) & cells)
            ps.append(hits/20)
            if n_pos > 0: rs.append(hits/n_pos)
        res[f'precision_{season}'] = np.mean(ps) if ps else 0.0
        res[f'recall_{season}']    = np.mean(rs) if rs else 0.0

    return res, cell_roc, proba_by_month


# ── 7. COMPARISON TABLE ────────────────────────────────────────────────────

def print_comparison(et: dict):
    avg_p   = et['avg_daily_pos']
    rnd_p   = avg_p / N_CELLS     # random precision baseline

    print('\n' + '='*76)
    print('直接比較: ExtraTrees vs TTM (城ヶ﨑 2025)')
    print(f'グリッド: {N_ROWS}x{N_COLS}={N_CELLS}セル, 約10km×10km | テスト: {TEST_YEAR} (365日)')
    print(f'平均出没セル数/日: {avg_p:.2f}  ランダムP={rnd_p:.4f} ({avg_p:.2f}/{N_CELLS})')
    print('='*76)

    w = [32, 22, 24, 20]
    fmt = '  '.join(f'{{:<{x}}}' for x in w)
    print(fmt.format('指標', 'ExtraTrees', 'TTM (Granite 1536-96)', 'ランダムベースライン'))
    print('  ' + '-'*(sum(w)+2*len(w)))

    # ROC-AUC
    print(fmt.format(
        'ROC-AUC (上位20地区平均)',
        f"{et['roc_auc_top20']:.3f}  (n={et['n_roc_top20']})",
        f"0.634  (n=20)",
        '0.500'))
    print(fmt.format(
        'ROC-AUC (全活動地区平均)',
        f"{et['roc_auc_avg']:.3f}  (n={et['n_cells_roc']})",
        '-', '0.500'))

    # P@K and R@K
    for k in K_VALUES:
        ep = et[f'precision_at_{k}']
        er = et[f'recall_at_{k}']
        tp = TTM[f'precision_at_{k}']
        tr = TTM[f'recall_at_{k}']
        rnd_r = k / N_CELLS
        print(fmt.format(
            f'Precision@{k}',
            f"{ep:.3f}  ({ep/rnd_p:.1f}x rnd)",
            f"{tp:.3f}  ({tp/rnd_p:.1f}x rnd)",
            f"{rnd_p:.3f}  ({avg_p:.2f}/{N_CELLS})"))
        print(fmt.format(
            f'Recall@{k}',
            f"{er:.3f}  ({er/rnd_r:.1f}x rnd)",
            f"{tr:.3f}  ({tr/rnd_r:.1f}x rnd)",
            f"{rnd_r:.3f}  ({k}/{N_CELLS})"))

    print('\n  季節別 K=20:')
    print(fmt.format('季節', 'ET Precision / Recall', 'TTM Precision / Recall', ''))
    seasons_labels = [('spring','春 (4-5月)'),('summer','夏 (6-8月)'),
                      ('fall','秋 (9-11月)'),('winter','冬 (12-3月)')]
    for s, label in seasons_labels:
        ep = et[f'precision_{s}']; er = et[f'recall_{s}']
        tp = TTM[f'precision_{s}']; tr = TTM[f'recall_{s}']
        print(fmt.format(label,
                         f"{ep:.3f} / {er:.3f}",
                         f"{tp:.3f} / {tr:.3f}", ''))
    print()


# ── 8. PLOTS ───────────────────────────────────────────────────────────────

def plot_comparison(et: dict, out_dir: Path):
    avg_p = et['avg_daily_pos']
    rnd_p = avg_p / N_CELLS
    x     = np.arange(len(K_VALUES))
    w     = 0.28
    C     = {'ET': '#2196F3', 'TTM': '#FF5722', 'Rnd': '#9E9E9E'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('ExtraTrees vs TTM Granite 1536-96-R2\n'
                 '(山形県 10km×10km 144セル | テスト2025年)',
                 fontsize=12, fontweight='bold')

    for ax, metric, title, rnd_fn in [
        (axes[0], 'precision', 'Precision@K (日次平均)', lambda k: rnd_p),
        (axes[1], 'recall',    'Recall@K (日次平均)',    lambda k: k/N_CELLS),
    ]:
        ev = [et[f'{metric}_at_{k}'] for k in K_VALUES]
        tv = [TTM[f'{metric}_at_{k}'] for k in K_VALUES]
        rv = [rnd_fn(k) for k in K_VALUES]

        for bars, vals, label, color in [
            (ax.bar(x-w,   ev, w, color=C['ET'],  alpha=0.85), ev, 'ExtraTrees', C['ET']),
            (ax.bar(x,     tv, w, color=C['TTM'], alpha=0.85), tv, 'TTM Granite', C['TTM']),
            (ax.bar(x+w,   rv, w, color=C['Rnd'], alpha=0.55), rv, 'Random', C['Rnd']),
        ]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h+0.003, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=7.5)

        ax.set_xticks(x); ax.set_xticklabels([f'K={k}' for k in K_VALUES])
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(max(ev), max(tv)) * 1.30 + 0.04)
        ax.legend(['ExtraTrees','TTM Granite','Random'], fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_10km_pk_rk.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Seasonal
    seasons_keys   = ['spring','summer','fall','winter']
    seasons_labels = ['Spring\n(Apr-May)','Summer\n(Jun-Aug)',
                      'Fall\n(Sep-Nov)','Winter\n(Dec-Mar)']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('季節別 P@20 / R@20 比較 (10kmグリッド | テスト2025)',
                 fontsize=12, fontweight='bold')
    for ax, metric, title in [
        (axes[0], 'precision', 'Precision@20 by season'),
        (axes[1], 'recall',    'Recall@20 by season'),
    ]:
        ev = [et[f'{metric}_{s}']  for s in seasons_keys]
        tv = [TTM[f'{metric}_{s}'] for s in seasons_keys]
        x  = np.arange(len(seasons_keys))
        ax.bar(x-0.2, ev, 0.38, label='ExtraTrees', color=C['ET'],  alpha=0.85)
        ax.bar(x+0.2, tv, 0.38, label='TTM',        color=C['TTM'], alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(seasons_labels, fontsize=9)
        ax.set_title(title, fontsize=11); ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(20/N_CELLS, ls='--', color='gray', lw=1)

    plt.tight_layout()
    plt.savefig(out_dir / 'seasonal_10km.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {out_dir}/")


def save_csv(et: dict, out_path: Path):
    avg_p = et['avg_daily_pos']; rnd_p = avg_p / N_CELLS
    rows = [
        {'metric':'roc_auc_top20_cells',
         'ET':f"{et['roc_auc_top20']:.4f}", 'TTM':'0.6340', 'random':'0.5000'},
        {'metric':'roc_auc_all_active',
         'ET':f"{et['roc_auc_avg']:.4f}",   'TTM':'-', 'random':'0.5000'},
    ]
    for k in K_VALUES:
        rows += [
            {'metric':f'precision_at_K{k}',
             'ET':f"{et[f'precision_at_{k}']:.4f}",
             'TTM':f"{TTM[f'precision_at_{k}']:.4f}",
             'random':f"{rnd_p:.4f}"},
            {'metric':f'recall_at_K{k}',
             'ET':f"{et[f'recall_at_{k}']:.4f}",
             'TTM':f"{TTM[f'recall_at_{k}']:.4f}",
             'random':f"{k/N_CELLS:.4f}"},
        ]
    for s in SEASONS:
        rows += [
            {'metric':f'precision_{s}_K20',
             'ET':f"{et[f'precision_{s}']:.4f}",
             'TTM':f"{TTM[f'precision_{s}']:.4f}",
             'random':f"{rnd_p:.4f}"},
            {'metric':f'recall_{s}_K20',
             'ET':f"{et[f'recall_{s}']:.4f}",
             'TTM':f"{TTM[f'recall_{s}']:.4f}",
             'random':f"{20/N_CELLS:.4f}"},
        ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  CSV saved: {out_path}")


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    print('='*76)
    print('山形県 10kmグリッド ベンチマーク: ET vs TTM (城ヶ﨑 2025) 直接比較')
    print('='*76)

    print('\n[1/7] グリッド読込 (Yamagata_10km_Grid_0.csv)...')
    grid = load_grid()

    print('\n[2/7] 出没データ読込 (2018-2025)...')
    train_s = load_and_assign(TRAIN_YEARS, grid)
    test_s  = load_and_assign([TEST_YEAR],  grid)
    print(f'  Train 2018-2024: {len(train_s):,} 件 (グリッド内)')
    print(f'  Test  2025:      {len(test_s):,} 件 (グリッド内)')
    print(f'  2025出没セル数: {test_s["cell_id"].nunique()} / {N_CELLS}')

    print('\n[3/7] 月次ラベル生成...')
    train_labels = make_train_labels(train_s, grid)
    pos = train_labels.label.sum()
    print(f'  {len(train_labels):,} セル×月, 陽性 {pos} ({100*pos/len(train_labels):.2f}%)')
    hist_rate = train_labels.groupby('cell_id')['label'].mean().rename('hist_positive_rate')

    print('\n[4/7] 特徴量生成 (標高・土地被覆・人口)...')
    elev   = elevation_features(grid)
    lulc   = lulc_features(grid, elev)
    pop    = pop_features(grid)
    static = (grid[['cell_id']]
              .merge(elev, on='cell_id', how='left')
              .merge(lulc, on='cell_id', how='left')
              .merge(pop,  on='cell_id', how='left'))
    print(f'  標高範囲: {elev.elev_mean.min():.0f}-{elev.elev_mean.max():.0f} m')

    print('\n[5/7] 訓練特徴量行列組立...')
    train_df = build_features(train_labels, static, hist_rate)
    X_train  = train_df[FEATURE_COLS]
    y_train  = train_df['label']
    print(f'  特徴量行列: {X_train.shape}')

    print('\n[6/7] ExtraTreesClassifier + RandomUnderSampler 訓練...')
    clf = train_et(X_train, y_train)

    fi = (pd.DataFrame({'feature': FEATURE_COLS, 'importance': clf.feature_importances_})
          .sort_values('importance', ascending=False))
    print('\n  特徴量重要度 TOP10:')
    print(fi.head(10).to_string(index=False, float_format='%.4f'))
    fi.to_csv(RESULTS_DIR / 'feature_importance_10km.csv', index=False)

    print('\n[7/7] 日次評価 (2025年 365日)...')
    et_res, cell_roc, _ = evaluate_daily(clf, test_s, static, hist_rate, grid)

    # 結果出力
    print_comparison(et_res)
    plot_comparison(et_res, RESULTS_DIR)
    save_csv(et_res, RESULTS_DIR / 'comparison_10km_et_vs_ttm.csv')

    roc_df = (pd.DataFrame({'cell_id': list(cell_roc.keys()),
                            'roc_auc': list(cell_roc.values())})
              .sort_values('roc_auc', ascending=False))
    roc_df.to_csv(RESULTS_DIR / 'per_cell_roc_10km.csv', index=False, float_format='%.4f')
    print(f'\n  全活動セルROC-AUC: mean={roc_df.roc_auc.mean():.3f}, '
          f'>0.6: {(roc_df.roc_auc>0.6).sum()} / {len(roc_df)}')

    print('\n' + '='*76)
    print(f'完了 - 結果は {RESULTS_DIR}/')
    print('='*76)


if __name__ == '__main__':
    main()
