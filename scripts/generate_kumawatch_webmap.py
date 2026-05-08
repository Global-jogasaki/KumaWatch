"""
KumaWatch Web Decision Support Map Generator
=============================================
三層アーキテクチャ対応の包括的な意思決定支援マップを生成する。

[主層]    GLM-Logit 予測確率 (0-1 probability)
[不確実性層] HierBayes Beta-Binomial seasonal 近似 (posterior mean + 95% CI)
[補完層]   TTM + Extra Trees (pre-computed scores, normalized to 0-1)

Paper: KumaWatch: A Multi-Method Wildlife Encounter Alert System for
       Operational Municipal Deployment in Northern Japan [Applications]
       ACM SIGSPATIAL 2026
"""

import sys, re, json, csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# ─── パス設定 ───────────────────────────────────────────────────────────────
BASE = r'F:\SSD-PGU3\電動モビリティ専門職大学\山形大学申請'
SIGHTINGS_CSV = BASE + r'\山形県データ\Yamagata_10km_AllGrid_144cells_Daily_TimeSeries.csv'
GRID_CSV      = BASE + r'\山形県データ\Yamagata_10km_Grid_0.csv'
TTM_SCORES_CSV = BASE + r'\山形県データ\yamagata_ttm_scores_2025.csv'
OUT_HTML       = BASE + r'\ACM-Application Track\kumawatch_map_2025.html'

TRAIN_START = '2018-10-01'; TRAIN_END = '2024-12-31'
TEST_START  = '2025-01-01'; TEST_END  = '2025-12-31'
GLM_C       = 1.0
RAND_SEED   = 42

# ─── 1. データ読み込み ───────────────────────────────────────────────────────
print('[1] データ読み込み...', flush=True)
df_raw = pd.read_csv(SIGHTINGS_CSV)
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='mixed')
grid_cols = [c for c in df_raw.columns if re.match(r'^\d+_\d+$', c)]
n_cells = len(grid_cols)
print(f'    グリッド数: {n_cells}')

mask_tr = (df_raw['Date'] >= TRAIN_START) & (df_raw['Date'] <= TRAIN_END)
mask_te = (df_raw['Date'] >= TEST_START)  & (df_raw['Date'] <= TEST_END)
df_tr = df_raw[mask_tr].copy(); df_te = df_raw[mask_te].copy()

train_L      = (df_tr[grid_cols].values > 0).astype(np.float32)
test_L       = (df_te[grid_cols].values > 0).astype(np.float32)
test_actual  = df_te[grid_cols].values.astype(np.int32)
train_dates  = pd.to_datetime(df_tr['Date'].values)
test_dates   = pd.to_datetime(df_te['Date'].values)
T_tr, T_te   = len(train_dates), len(test_dates)
print(f'    訓練: {train_dates[0].date()} - {train_dates[-1].date()} ({T_tr}日)')
print(f'    評価: {test_dates[0].date()} - {test_dates[-1].date()} ({T_te}日)')

# ─── 2. TTM スコア読み込み ───────────────────────────────────────────────────
print('[2] TTM スコア読み込み...', flush=True)

def load_score_csv(path, grid_cols, test_dates):
    df = pd.read_csv(path)
    df['_dt'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.set_index('_dt').sort_index()
    return df.reindex(test_dates)[grid_cols].fillna(0).values.astype(np.float32)

ttm_raw = load_score_csv(TTM_SCORES_CSV, grid_cols, test_dates)
print(f'    TTM raw scores: {ttm_raw.shape}, range [{ttm_raw.min():.4f}, {ttm_raw.max():.4f}]')

# 正規化 (95パーセンタイルを 1.0 に)
def normalize_to_unit(scores, pct=95):
    pos = scores[scores > 0]
    if len(pos) == 0: return scores
    p = float(np.percentile(pos, pct))
    if p == 0: return scores
    return np.clip(scores / p, 0.0, 1.0).astype(np.float32)

ttm_scores = normalize_to_unit(ttm_raw)
print(f'    TTM normalized: [{ttm_scores.min():.4f}, {ttm_scores.max():.4f}]')

# ─── 3. GLM-Logit 訓練 ──────────────────────────────────────────────────────
print('[3] GLM-Logit 訓練中...', flush=True)
all_L = np.concatenate([train_L, test_L], axis=0).astype(np.float64)
cs    = np.zeros((len(all_L) + 1, n_cells), dtype=np.float64)
np.cumsum(all_L, axis=0, out=cs[1:])

def rolling_sum(pos, window):
    return (cs[pos] - cs[max(0, pos - window)]).astype(np.float32)

base_year = train_dates[0].year

def make_block(dates, offset):
    T = len(dates)
    r30 = np.empty((T, n_cells), np.float32)
    r365= np.empty((T, n_cells), np.float32)
    sin_= np.empty(T, np.float32)
    cos_= np.empty(T, np.float32)
    yr_ = np.empty(T, np.float32)
    for i, d in enumerate(dates):
        pos = offset + i
        r30[i]  = rolling_sum(pos, 30)
        r365[i] = rolling_sum(pos, 365)
        doy = d.timetuple().tm_yday
        sin_[i] = np.sin(2 * np.pi * doy / 365)
        cos_[i] = np.cos(2 * np.pi * doy / 365)
        yr_[i]  = (d.year - base_year) / 10.0
    log_r365 = np.log1p(r365)
    sin_rep = np.repeat(sin_[:, None], n_cells, axis=1)
    cos_rep = np.repeat(cos_[:, None], n_cells, axis=1)
    yr_rep  = np.repeat(yr_[:, None],  n_cells, axis=1)
    return np.stack([r30, log_r365, sin_rep, cos_rep, yr_rep], axis=2).reshape(-1, 5)

feat_tr = make_block(train_dates, 0)
feat_te = make_block(test_dates,  T_tr)
cell_idx_tr = np.tile(np.arange(n_cells), T_tr).astype(np.int32)
cell_idx_te = np.tile(np.arange(n_cells), T_te).astype(np.int32)

def build_dm(feat, cidx):
    N = feat.shape[0]
    cell_oh = csr_matrix((np.ones(N, np.float32), (np.arange(N), cidx)), shape=(N, n_cells))
    return sp_hstack([cell_oh, csr_matrix(feat)], format='csr')

X_tr = build_dm(feat_tr, cell_idx_tr)
X_te = build_dm(feat_te, cell_idx_te)
y_tr = train_L.flatten().astype(np.float32)
clf  = LogisticRegression(C=GLM_C, fit_intercept=False, max_iter=2000,
                           solver='lbfgs', random_state=RAND_SEED, verbose=0)
clf.fit(X_tr, y_tr)
glm_scores = clf.predict_proba(X_te)[:, 1].reshape(T_te, n_cells).astype(np.float32)
print(f'    GLM scores shape={glm_scores.shape}, max={glm_scores.max():.4f}')

# ─── 3b. Extra Trees (補完層) ─────────────────────────────────────────────────
# 同じ特徴量で ExtraTrees を訓練 (dense 行列で fit)
print('[3b] Extra Trees 訓練中 (補完層)...', flush=True)
et_clf = ExtraTreesClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3,
                               random_state=RAND_SEED, n_jobs=-1)
et_clf.fit(X_tr.toarray() if hasattr(X_tr, 'toarray') else X_tr, y_tr)
et_scores = et_clf.predict_proba(
    X_te.toarray() if hasattr(X_te, 'toarray') else X_te)[:, 1]\
    .reshape(T_te, n_cells).astype(np.float32)
print(f'    ET  scores shape={et_scores.shape}, max={et_scores.max():.4f}')

# ─── 4. HierBayes Beta-Binomial 季節近似 ─────────────────────────────────────
print('[4] HierBayes 近似計算中 (Beta-Binomial seasonal window)...', flush=True)
train_doys  = np.array([d.timetuple().tm_yday for d in train_dates])
test_doys   = np.array([d.timetuple().tm_yday for d in test_dates])
train_years = np.array([d.year for d in train_dates])
# 直近年に高い重みを付与 (2024: weight=6, 2023: weight=5, ..., 2018: weight=1)
year_weights = np.maximum(1, 7 - (2025 - train_years)).astype(np.float32)

DOY_WINDOW   = 45
ALPHA_PRIOR  = 0.5
BETA_PRIOR   = 0.5

hb_mean = np.zeros((T_te, n_cells), dtype=np.float32)
hb_lo   = np.zeros((T_te, n_cells), dtype=np.float32)
hb_hi   = np.zeros((T_te, n_cells), dtype=np.float32)

for t_idx in range(T_te):
    target_doy = int(test_doys[t_idx])
    doy_diff = np.abs(train_doys.astype(int) - target_doy)
    doy_diff = np.minimum(doy_diff, 365 - doy_diff)   # 円環
    in_win   = (doy_diff <= DOY_WINDOW).astype(np.float32)
    w = year_weights * in_win                          # (n_train,)

    weighted_hits  = w @ train_L                       # (n_cells,)
    weighted_total = float(w.sum())

    a_post = (ALPHA_PRIOR + weighted_hits).astype(np.float64)
    b_post = (BETA_PRIOR  + weighted_total - weighted_hits).astype(np.float64)
    b_post = np.maximum(b_post, 1e-6)

    hb_mean[t_idx] = (a_post / (a_post + b_post)).astype(np.float32)
    hb_lo[t_idx]   = beta_dist.ppf(0.025, a_post, b_post).astype(np.float32)
    hb_hi[t_idx]   = beta_dist.ppf(0.975, a_post, b_post).astype(np.float32)
    if (t_idx + 1) % 60 == 0:
        print(f'    {t_idx+1}/{T_te} 完了', flush=True)

print(f'    HierBayes 完了: mean range [{hb_mean.min():.4f}, {hb_mean.max():.4f}]')

# ─── 5. グリッド座標・市町村名 ───────────────────────────────────────────────
print('[5] COORDS 構築...', flush=True)
extra_city_map = {
    '0_9':'酒田市西部','1_3':'飯豊町','1_11':'酒田市北部',
    '2_11':'鶴岡市北部','2_12':'酒田市南部','2_14':'遊佐町',
    '2_2':'長井市','2_3':'白鷹町','3_2':'川西町','5_2':'南陽市',
    '5_4':'山形市','5_6':'河北町','5_7':'大石田町','5_10':'村山市',
    '6_5':'天童市','6_7':'村山市東部','6_8':'尾花沢市',
    '7_4':'山辺町','7_7':'尾花沢市東部','7_8':'金山町南部',
    '3_12':'尾花沢市','3_14':'舟形町','5_12':'大蔵村',
    '6_11':'新庄市','8_9':'金山町',
}
# TTM HTMLからの city マッピング (フォールバック)
TTM_HTML = BASE + r'\ACM-Application Track\bear_ttm_map_2025_2.html'
try:
    with open(TTM_HTML, encoding='utf-8') as f:
        m = re.search(r'const COORDS = ({.*?});', f.read(), re.DOTALL)
    ttm_coords = json.loads(m.group(1)) if m else {}
except Exception:
    ttm_coords = {}

df_grid = pd.read_csv(GRID_CSV)
grid_info = {}
for _, row in df_grid.iterrows():
    gid  = row['Grid_ID']
    city = ttm_coords.get(gid, {}).get('city') or extra_city_map.get(gid) or gid
    grid_info[gid] = dict(
        lat=float(row['Center_Latitude']), lng=float(row['Center_Longitude']),
        lat_min=float(row['Min_Latitude']),  lat_max=float(row['Max_Latitude']),
        lng_min=float(row['Min_Longitude']), lng_max=float(row['Max_Longitude']),
        city=city,
    )

# ─── 6. 歴史的目撃統計 (cell 別) ─────────────────────────────────────────────
print('[6] 歴史的目撃統計計算...', flush=True)
# h30  : 訓練終了前 30 日 (2024-12-02 ~ 2024-12-31)
# h365 : 2024 年全体
# hall : 訓練期間全体 (2018-10-01 ~ 2024-12-31)
raw_counts_tr = df_tr[grid_cols].values.astype(np.int32)  # 実目撃数
hist_hall = raw_counts_tr.sum(axis=0)           # (n_cells,)
yr2024_mask = df_tr['Date'].dt.year == 2024
hist_h365 = raw_counts_tr[yr2024_mask.values].sum(axis=0)
hist_h30  = raw_counts_tr[-30:].sum(axis=0)

# ─── 7. DAILY データ構築 ────────────────────────────────────────────────────
print('[7] DAILY データ構築...', flush=True)
INCLUDE_THRESH = 0.008  # GLM or TTM or ET > 0.8% のセルを格納
grid_col_idx   = {gid: i for i, gid in enumerate(grid_cols)}

daily_data = {}
for t, d in enumerate(test_dates):
    ds = d.strftime('%Y-%m-%d')
    g_glm  = glm_scores[t]   # (n_cells,)
    g_hbm  = hb_mean[t]
    g_hblo = hb_lo[t]
    g_hbhi = hb_hi[t]
    g_ttm  = ttm_scores[t]
    g_et   = et_scores[t]
    g_act  = test_actual[t]  # 実目撃数

    cells = {}
    for ci, gid in enumerate(grid_cols):
        glm_v = float(g_glm[ci])
        ttm_v = float(g_ttm[ci])
        et_v  = float(g_et[ci])
        if glm_v < INCLUDE_THRESH and ttm_v < INCLUDE_THRESH * 2 and et_v < INCLUDE_THRESH * 2:
            continue
        cells[gid] = [
            round(glm_v, 4),
            round(float(g_hbm[ci]),  4),
            round(float(g_hblo[ci]), 4),
            round(float(g_hbhi[ci]), 4),
            round(ttm_v, 4),
            round(et_v,  4),
            int(g_act[ci]),
        ]
    daily_data[ds] = cells

non_empty = sum(1 for v in daily_data.values() if len(v) > 0)
print(f'    {len(daily_data)} 日  (非空: {non_empty} 日, 平均 '
      f'{sum(len(v) for v in daily_data.values())/len(daily_data):.1f} cells/日)')

# COORDS (all 144 grids + historical stats)
coords_dict = {}
for ci, gid in enumerate(grid_cols):
    if gid not in grid_info:
        continue
    info = grid_info[gid].copy()
    info['h30']  = int(hist_h30[ci])
    info['h365'] = int(hist_h365[ci])
    info['hall'] = int(hist_hall[ci])
    coords_dict[gid] = info

print(f'    COORDS: {len(coords_dict)} grids')

# ─── 8. JSON 生成 ────────────────────────────────────────────────────────────
print('[8] JSON シリアライズ...', flush=True)
DAILY_JSON  = json.dumps(daily_data,  ensure_ascii=False, separators=(',', ':'))
COORDS_JSON = json.dumps(coords_dict, ensure_ascii=False, separators=(',', ':'))
print(f'    DAILY_JSON  : {len(DAILY_JSON)/1024:.0f} KB')
print(f'    COORDS_JSON : {len(COORDS_JSON)/1024:.0f} KB')

# ─── 9. HTML 生成 ────────────────────────────────────────────────────────────
print('[9] HTML 生成中...', flush=True)

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>KumaWatch — 山形県 熊出没危険予測マップ 2025</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Meiryo','Hiragino Kaku Gothic ProN',sans-serif;background:#0d1c13;color:#eef8ec;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
/* ── header ── */
#header{{background:#162e1e;padding:8px 14px;display:flex;align-items:center;gap:10px;border-bottom:2px solid #2a5c36;flex-shrink:0;flex-wrap:wrap}}
#header h1{{font-size:17px;font-weight:700;color:#fff;white-space:nowrap}}
#layer-wrap{{display:flex;align-items:center;gap:6px;margin-left:auto}}
#layer-wrap label{{font-size:13px;color:#b0d0bc}}
#layer-sel{{background:#1e3c28;color:#eef8ec;border:1px solid #3a6a46;padding:4px 8px;font-size:13px;border-radius:5px;cursor:pointer}}
#layer-name{{font-size:14px;font-weight:700;color:#6aff90;white-space:nowrap}}
/* ── notice ── */
#notice{{background:#0a1e10;border-bottom:1px solid #1c3c22;padding:5px 14px;font-size:12px;color:#a0c0a8;flex-shrink:0}}
#notice strong{{color:#e07b2a}}
/* ── main ── */
#main{{display:flex;flex:1;overflow:hidden}}
/* ── sidebar ── */
#sidebar{{width:315px;background:#0f1e14;overflow-y:auto;flex-shrink:0;border-right:1px solid #1c3c22}}
#map{{flex:1}}
/* ── panels ── */
.panel{{padding:10px 12px;border-bottom:1px solid #1c3c22}}
.pt{{font-size:12px;color:#a0d8b0;font-weight:700;margin-bottom:6px;letter-spacing:.06em;text-transform:uppercase}}
/* ── date section ── */
#date-display{{font-size:22px;font-weight:700;color:#fff;line-height:1.1}}
#day-info{{font-size:12px;color:#a0c0a8;margin-top:2px}}
.month-row{{display:flex;gap:2px;flex-wrap:wrap;margin:6px 0 3px}}
.mbtn{{background:#1a3820;border:1px solid #2a5830;color:#a0c0a8;font-size:11px;padding:3px 3px;border-radius:3px;cursor:pointer;flex:1;min-width:24px;text-align:center;transition:background .15s}}
.mbtn:hover,.mbtn.active{{background:#2a6040;color:#fff;border-color:#4a9060}}
#slider-wrap{{margin:5px 0}}
#day-slider{{width:100%;cursor:pointer;accent-color:#3a9050}}
.play-row{{display:flex;gap:6px;align-items:center;margin-top:5px}}
#play-btn{{background:#1e5030;border:1px solid #3a8046;color:#c0ecd0;padding:5px 13px;border-radius:4px;cursor:pointer;font-size:12px}}
#play-btn:hover{{background:#2a6840}}
#speed-sel{{background:#1a2e1e;border:1px solid #2a4030;color:#c0ecd0;font-size:11px;padding:2px 4px;border-radius:3px}}
/* ── threshold ── */
#thr-val{{font-size:13px;font-weight:700;color:#6aff90}}
#thr-slider{{width:100%;cursor:pointer;accent-color:#3a9050;margin-top:4px}}
/* ── legend ── */
.leg-row{{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px}}
.leg-box{{width:16px;height:16px;border-radius:2px;flex-shrink:0;border:1px solid rgba(255,255,255,.3)}}
/* ── stats ── */
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:5px}}
.stat-item{{background:#0d1e12;border:1px solid #1c3422;border-radius:4px;padding:5px 7px}}
.stat-label{{font-size:11px;color:#8aaa94;margin-bottom:1px}}
.stat-val{{font-size:17px;font-weight:700;color:#eef8ec}}
/* ── ranking ── */
.rank-row{{display:flex;align-items:center;gap:5px;padding:4px 0;border-bottom:1px solid #1a2e1e;cursor:pointer;transition:background .12s}}
.rank-row:hover{{background:#142018}}
.rank-no{{font-size:12px;color:#8aaa94;width:18px;text-align:right;flex-shrink:0}}
.rank-city{{font-size:13px;font-weight:700;color:#eef8ec;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.rank-badge{{font-size:11px;font-weight:700;padding:2px 7px;border-radius:3px;white-space:nowrap}}
.rank-pct{{font-size:12px;color:#a8c8b4;text-align:right;width:44px;flex-shrink:0}}
/* ── detail panel ── */
#detail-panel{{display:none}}
#detail-panel .dp-city{{font-size:16px;font-weight:700;color:#fff;margin-bottom:2px}}
#detail-panel .dp-gid{{font-size:11px;color:#8aaa94}}
.dp-section{{margin:8px 0 0}}
.dp-section-title{{font-size:12px;font-weight:700;color:#a0d8b0;letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}}
.dp-row{{display:flex;align-items:center;gap:6px;margin:3px 0;font-size:13px}}
.dp-label{{color:#a8c8b4;width:82px;flex-shrink:0}}
.dp-val{{color:#eef8ec;font-weight:700}}
.dp-ci{{color:#a0c0a8;font-size:12px}}
.dp-agree{{color:#4aef7a;font-size:12px;font-weight:700}}
.dp-disagree{{color:#ef9a3a;font-size:12px;font-weight:700}}
.dp-alert-badge{{font-size:12px;font-weight:700;padding:2px 8px;border-radius:3px}}
.hist-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:4px}}
.hist-item{{background:#0d1e12;border:1px solid #1c3422;border-radius:3px;padding:4px 6px;text-align:center}}
.hist-item .hi-label{{font-size:11px;color:#8aaa94}}
.hist-item .hi-val{{font-size:15px;font-weight:700;color:#eef8ec}}
#detail-back{{background:#1e3820;border:1px solid #2a5030;color:#c0ecd0;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px;margin-bottom:6px}}
/* ── risk colors ── */
.risk-alert{{background:#e81515;color:#fff}}
.risk-high {{background:#f06520;color:#fff}}
.risk-mid  {{background:#e0b800;color:#1a1000}}
.risk-low  {{background:#28a030;color:#d8f8d0}}
</style>
</head>
<body>
<div id="header">
  <div>
    <h1>🐻 KumaWatch — 山形県 熊出没危険予測マップ 2025</h1>
  </div>
  <div id="layer-wrap">
    <label>予測レイヤー</label>
    <select id="layer-sel" onchange="switchLayer(this.value)">
      <option value="glm">GLM-Logit（主層）</option>
      <option value="hb">HierBayes（不確実性層）</option>
      <option value="ttm">TTM（補完層）</option>
      <option value="et">Extra Trees（補完層）</option>
    </select>
    <span id="layer-name">GLM-Logit 予測確率</span>
  </div>
</div>
<div id="notice">
  ⚠ <strong>注意</strong>: 本マップの予測は行政機関による公式出没情報・警告に代わるものではありません。
  訓練データ: <strong>2018年10月〜2024年12月</strong>。評価期間: 2025年1月〜12月。
</div>
<div id="main">
  <div id="sidebar">
    <!-- 日付ナビ -->
    <div class="panel">
      <div class="pt">日付</div>
      <div id="date-display">2025-10-12</div>
      <div id="day-info">読み込み中…</div>
      <div class="month-row" id="month-btns"></div>
      <div id="slider-wrap"><input type="range" id="day-slider" min="0" max="364" value="284"></div>
      <div class="play-row">
        <button id="play-btn" onclick="togglePlay()">▶ 再生</button>
        <select id="speed-sel" title="再生速度">
          <option value="600">遅い</option>
          <option value="300" selected>標準</option>
          <option value="100">速い</option>
        </select>
      </div>
    </div>
    <!-- 閾値 -->
    <div class="panel">
      <div class="pt">表示閾値 — <span id="thr-val">15%</span></div>
      <input type="range" id="thr-slider" min="1" max="50" value="15" step="1"
             oninput="setThreshold(this.value)">
    </div>
    <!-- 凡例 -->
    <div class="panel">
      <div class="pt">リスクレベル</div>
      <div class="leg-row"><div class="leg-box" style="background:#e81515"></div><span>緊急 (≥70%)</span></div>
      <div class="leg-row"><div class="leg-box" style="background:#f06520"></div><span>高危険 (45–70%)</span></div>
      <div class="leg-row"><div class="leg-box" style="background:#e0b800"></div><span>警戒 (20–45%)</span></div>
      <div class="leg-row"><div class="leg-box" style="background:#28a030"></div><span>注意 (閾値–20%)</span></div>
      <div class="leg-row"><div class="leg-box" style="background:#1a3820;border:1px dashed #2a5030"></div><span style="color:#4a7050">非表示 (閾値未満)</span></div>
    </div>
    <!-- デイリー統計 -->
    <div class="panel">
      <div class="pt">日次統計</div>
      <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">アクティブ</div><div class="stat-val" id="st-active">—</div></div>
        <div class="stat-item"><div class="stat-label">最高リスク</div><div class="stat-val" id="st-maxrisk">—</div></div>
        <div class="stat-item"><div class="stat-label">緊急セル</div><div class="stat-val" id="st-alert">—</div></div>
        <div class="stat-item"><div class="stat-label">実目撃数</div><div class="stat-val" id="st-actual">—</div></div>
      </div>
    </div>
    <!-- ランキング / 詳細 (切り替え) -->
    <div class="panel" id="rank-panel">
      <div class="pt">リスクランキング (GLM-Logit順)</div>
      <div id="rank-list"></div>
    </div>
    <div class="panel" id="detail-panel">
      <button id="detail-back" onclick="closeDetail()">← ランキングに戻る</button>
      <div class="dp-city" id="dp-city">—</div>
      <div class="dp-gid" id="dp-gid">—</div>
      <!-- 主層 -->
      <div class="dp-section">
        <div class="dp-section-title">主層 — GLM-Logit</div>
        <div class="dp-row"><span class="dp-label">予測確率</span><span class="dp-val" id="dp-glm">—</span></div>
        <div class="dp-row"><span class="dp-label">本日順位</span><span class="dp-val" id="dp-rank">—</span></div>
      </div>
      <!-- 不確実性層 -->
      <div class="dp-section">
        <div class="dp-section-title">不確実性層 — HierBayes</div>
        <div class="dp-row">
          <span class="dp-label">事後平均</span>
          <span class="dp-val" id="dp-hbm">—</span>
          <span class="dp-ci" id="dp-hbci">—</span>
        </div>
        <div class="dp-row"><span class="dp-label">グラデッド</span><span id="dp-alert-badge" class="dp-alert-badge">—</span></div>
      </div>
      <!-- 補完層 -->
      <div class="dp-section">
        <div class="dp-section-title">補完層 — TTM / Extra Trees</div>
        <div class="dp-row">
          <span class="dp-label">TTM スコア</span>
          <span class="dp-val" id="dp-ttm">—</span>
          <span id="dp-ttm-agree">—</span>
        </div>
        <div class="dp-row">
          <span class="dp-label">ET スコア</span>
          <span class="dp-val" id="dp-et">—</span>
          <span id="dp-et-agree">—</span>
        </div>
      </div>
      <!-- 目撃履歴 -->
      <div class="dp-section">
        <div class="dp-section-title">目撃履歴 (訓練データ)</div>
        <div class="hist-row">
          <div class="hist-item"><div class="hi-label">直近30日</div><div class="hi-val" id="dp-h30">—</div></div>
          <div class="hist-item"><div class="hi-label">2024年計</div><div class="hi-val" id="dp-h365">—</div></div>
          <div class="hist-item"><div class="hi-label">全期間</div><div class="hi-val" id="dp-hall">—</div></div>
        </div>
      </div>
    </div>
  </div>
  <div id="map"></div>
</div>

<script>
// ─── Data ────────────────────────────────────────────────────────────────────
const COORDS = {COORDS_JSON};
const DAILY  = {DAILY_JSON};

// ─── State ───────────────────────────────────────────────────────────────────
const DATES = Object.keys(DAILY).sort();
let currentIdx = DATES.indexOf('2025-10-12');
if (currentIdx < 0) currentIdx = 284;
let currentLayer  = 'glm';
let hideThreshold = 0.15;
let isPlaying     = false;
let playTimer     = null;

const LAYER_NAMES = {{
  glm: 'GLM-Logit 予測確率',
  hb:  'HierBayes 事後平均',
  ttm: 'TTM スコア（補完）',
  et:  'Extra Trees スコア（補完）',
}};
// Cell data indices: [glm, hb_m, hb_lo, hb_hi, ttm, et, act]
const IDX = {{glm:0, hb_m:1, hb_lo:2, hb_hi:3, ttm:4, et:5, act:6}};

// ─── Risk level ──────────────────────────────────────────────────────────────
function riskInfo(score) {{
  if (score >= 0.70) return {{level:'alert', label:'緊急',   cls:'risk-alert', color:'#e81515'}};
  if (score >= 0.45) return {{level:'high',  label:'高危険', cls:'risk-high',  color:'#f06520'}};
  if (score >= 0.20) return {{level:'mid',   label:'警戒',   cls:'risk-mid',   color:'#e0b800'}};
  return                     {{level:'low',  label:'注意',   cls:'risk-low',   color:'#28a030'}};
}}

function layerScore(cell, layer) {{
  if (!cell) return 0;
  if (layer === 'glm') return cell[IDX.glm];
  if (layer === 'hb')  return cell[IDX.hb_m];
  if (layer === 'ttm') return cell[IDX.ttm];
  if (layer === 'et')  return cell[IDX.et];
  return 0;
}}

// ─── Leaflet map ─────────────────────────────────────────────────────────────
const map = L.map('map', {{
  center: [38.7, 140.2], zoom: 8,
  zoomControl: true, scrollWheelZoom: true
}});
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors © <a href="https://carto.com/attributions">CARTO</a>',
  opacity: 0.92
}}).addTo(map);

// Rectangle layers
const rects = {{}};
Object.entries(COORDS).forEach(([gid, c]) => {{
  const rect = L.rectangle(
    [[c.lat_min, c.lng_min],[c.lat_max, c.lng_max]],
    {{weight:0.5, color:'#1a3020', fillOpacity:0.0, opacity:0}}
  ).addTo(map);
  rect.on('click', () => showCellDetail(gid));
  rects[gid] = rect;
}});

// ─── Map render ──────────────────────────────────────────────────────────────
function renderMap() {{
  const ds    = DATES[currentIdx];
  const cells = DAILY[ds] || {{}};
  Object.entries(COORDS).forEach(([gid, c]) => {{
    const cell  = cells[gid];
    const score = cell ? layerScore(cell, currentLayer) : 0;
    const rect  = rects[gid];
    if (!rect) return;
    if (score < hideThreshold) {{
      rect.setStyle({{fillOpacity:0, opacity:0}});
    }} else {{
      const ri = riskInfo(score);
      rect.setStyle({{
        fillColor: ri.color, fillOpacity: 0.78,
        color:'#ffffff', opacity:0.90, weight:1.5
      }});
    }}
  }});
}}

// ─── Sidebar update ──────────────────────────────────────────────────────────
function updateSidebar() {{
  const ds    = DATES[currentIdx];
  const cells = DAILY[ds] || {{}};
  document.getElementById('date-display').textContent = ds;

  // Day-of-week
  const dow = ['日','月','火','水','木','金','土'][new Date(ds).getDay()];
  document.getElementById('day-info').textContent =
    '(' + dow + ')  表示レイヤー: ' + LAYER_NAMES[currentLayer];

  // Stats
  let active=0, alertCnt=0, totalAct=0, maxScore=0;
  Object.values(cells).forEach(cell => {{
    const s = layerScore(cell, currentLayer);
    if (s >= hideThreshold) active++;
    if (cell[IDX.glm] >= 0.70) alertCnt++;
    totalAct += cell[IDX.act];
    if (cell[IDX.glm] > maxScore) maxScore = cell[IDX.glm];
  }});
  const ri = riskInfo(maxScore);
  document.getElementById('st-active').textContent  = active;
  document.getElementById('st-maxrisk').innerHTML   =
    '<span class="rank-badge ' + ri.cls + '">' + ri.label + '</span>';
  document.getElementById('st-alert').textContent   = alertCnt;
  document.getElementById('st-actual').textContent  = totalAct;

  // Ranking (always by GLM-Logit)
  const ranked = Object.entries(cells)
    .filter(([,c]) => c[IDX.glm] >= hideThreshold)
    .sort((a,b) => b[1][IDX.glm] - a[1][IDX.glm])
    .slice(0, 20);

  let html = '';
  ranked.forEach(([gid, cell], i) => {{
    const city = (COORDS[gid] && COORDS[gid].city) || gid;
    const s    = cell[IDX.glm];
    const ri   = riskInfo(s);
    html += `<div class="rank-row" onclick="showCellDetail('${{gid}}')">
      <span class="rank-no">${{i+1}}</span>
      <span class="rank-city" title="${{city}} [${{gid}}]">${{city}}</span>
      <span class="rank-badge ${{ri.cls}}">${{ri.label}}</span>
      <span class="rank-pct">${{(s*100).toFixed(1)}}%</span>
    </div>`;
  }});
  document.getElementById('rank-list').innerHTML = html || '<div style="color:#4a7050;font-size:11px;padding:6px 0">本日は閾値以上のセルなし</div>';

  // Slider sync
  document.getElementById('day-slider').value = currentIdx;
}};

// ─── Cell click detail ───────────────────────────────────────────────────────
function showCellDetail(gid) {{
  const ds    = DATES[currentIdx];
  const cells = DAILY[ds] || {{}};
  const cell  = cells[gid];
  const coord = COORDS[gid];
  if (!cell || !coord) return;

  document.getElementById('dp-city').textContent = coord.city || gid;
  document.getElementById('dp-gid').textContent  = '[' + gid + ']';

  // GLM rank
  const ranked = Object.entries(cells)
    .sort((a,b) => b[1][IDX.glm] - a[1][IDX.glm]);
  const glmRank = ranked.findIndex(([g]) => g === gid) + 1;
  document.getElementById('dp-glm').textContent  = (cell[IDX.glm]*100).toFixed(1) + '%';
  document.getElementById('dp-rank').textContent = '第' + glmRank + '位 / ' + ranked.length + '位中';

  // HierBayes
  const hbm  = cell[IDX.hb_m];
  const hblo = cell[IDX.hb_lo];
  const hbhi = cell[IDX.hb_hi];
  const ciW  = hbhi - hblo;
  document.getElementById('dp-hbm').textContent  = (hbm*100).toFixed(1) + '%';
  document.getElementById('dp-hbci').textContent =
    '[' + (hblo*100).toFixed(1) + '%, ' + (hbhi*100).toFixed(1) + '%]';
  // Graduated alert
  let alertTier, alertCls;
  if (hbm >= 0.25 && ciW < 0.30) {{
    alertTier = '緊急アラート'; alertCls = 'risk-alert';
  }} else if (hbm >= 0.10) {{
    alertTier = '定期警戒';    alertCls = 'risk-mid';
  }} else {{
    alertTier = '注意レベル';  alertCls = 'risk-low';
  }}
  const badge = document.getElementById('dp-alert-badge');
  badge.textContent = alertTier;
  badge.className   = 'dp-alert-badge ' + alertCls;

  // TTM / ET agrees
  const glmTop20 = ranked.slice(0, 20).map(([g]) => g);
  const ttmRanked = Object.entries(cells)
    .sort((a,b) => b[1][IDX.ttm] - a[1][IDX.ttm]);
  const etRanked  = Object.entries(cells)
    .sort((a,b) => b[1][IDX.et]  - a[1][IDX.et]);
  const ttmTop20 = ttmRanked.slice(0, 20).map(([g]) => g);
  const etTop20  = etRanked.slice(0, 20).map(([g]) => g);

  const glmIn  = glmTop20.includes(gid);
  const ttmIn  = ttmTop20.includes(gid);
  const etIn   = etTop20.includes(gid);
  const ttmAgr = (glmIn === ttmIn);
  const etAgr  = (glmIn === etIn);

  document.getElementById('dp-ttm').textContent = (cell[IDX.ttm]*100).toFixed(1) + '%';
  document.getElementById('dp-et').textContent  = (cell[IDX.et]*100).toFixed(1)  + '%';
  document.getElementById('dp-ttm-agree').innerHTML =
    ttmAgr ? '<span class="dp-agree">✓ agrees</span>' : '<span class="dp-disagree">✗ disagrees</span>';
  document.getElementById('dp-et-agree').innerHTML =
    etAgr  ? '<span class="dp-agree">✓ agrees</span>' : '<span class="dp-disagree">✗ disagrees</span>';

  // History
  document.getElementById('dp-h30').textContent  = coord.h30  ?? '—';
  document.getElementById('dp-h365').textContent = coord.h365 ?? '—';
  document.getElementById('dp-hall').textContent = coord.hall ?? '—';

  document.getElementById('rank-panel').style.display   = 'none';
  document.getElementById('detail-panel').style.display = 'block';
}}

function closeDetail() {{
  document.getElementById('detail-panel').style.display = 'none';
  document.getElementById('rank-panel').style.display   = 'block';
}}

// ─── Layer switch ─────────────────────────────────────────────────────────────
function switchLayer(layer) {{
  currentLayer = layer;
  document.getElementById('layer-name').textContent = LAYER_NAMES[layer];
  render();
}}

// ─── Threshold ───────────────────────────────────────────────────────────────
function setThreshold(val) {{
  hideThreshold = val / 100;
  document.getElementById('thr-val').textContent = val + '%';
  render();
}}

// ─── Date navigation ─────────────────────────────────────────────────────────
function goToDate(idx) {{
  currentIdx = Math.max(0, Math.min(DATES.length - 1, idx));
  render();
}}

document.getElementById('day-slider').addEventListener('input', function() {{
  goToDate(parseInt(this.value));
}});

// Month buttons
const monthNames = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
const monthBtns  = document.getElementById('month-btns');
monthNames.forEach((mn, mi) => {{
  const btn = document.createElement('button');
  btn.className = 'mbtn';
  btn.textContent = mn;
  btn.onclick = () => {{
    const target = '2025-' + String(mi+1).padStart(2,'0') + '-01';
    const idx = DATES.findIndex(d => d >= target);
    if (idx >= 0) goToDate(idx);
    document.querySelectorAll('.mbtn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }};
  monthBtns.appendChild(btn);
}});

// ─── Playback ────────────────────────────────────────────────────────────────
function togglePlay() {{
  isPlaying = !isPlaying;
  document.getElementById('play-btn').textContent = isPlaying ? '⏸ 停止' : '▶ 再生';
  if (isPlaying) advancePlay();
}}

function advancePlay() {{
  if (!isPlaying) return;
  const speed = parseInt(document.getElementById('speed-sel').value);
  goToDate(currentIdx < DATES.length - 1 ? currentIdx + 1 : 0);
  playTimer = setTimeout(advancePlay, speed);
}}

// ─── Master render ────────────────────────────────────────────────────────────
function render() {{
  renderMap();
  updateSidebar();
}}

// ─── Init ─────────────────────────────────────────────────────────────────────
render();
// Highlight October button
document.querySelectorAll('.mbtn')[9].classList.add('active');
</script>
</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

size_kb = len(html.encode('utf-8')) / 1024
print(f'[完了] 出力: {OUT_HTML}')
print(f'       ファイルサイズ: {size_kb:.0f} KB')
