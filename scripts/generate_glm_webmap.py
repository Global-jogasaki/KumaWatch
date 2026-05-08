"""
GLM-Logit 2025 熊出没危険予測マップ生成スクリプト
=================================================
山形県 10km グリッドに対して GLM-Logit モデルで 2025 年の日次予測を行い、
bear_ttm_map_2025_2.html と同じフォーマットの HTML マップを生成する。
"""

import sys
import re
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ─── パス設定 ───────────────────────────────────────────────────────────────
SIGHTINGS_CSV = r'F:\SSD-PGU3\電動モビリティ専門職大学\山形大学申請\山形県データ\Yamagata_10km_AllGrid_144cells_Daily_TimeSeries.csv'
GRID_CSV      = r'F:\SSD-PGU3\電動モビリティ専門職大学\山形大学申請\山形県データ\Yamagata_10km_Grid_0.csv'
TTM_HTML      = r'F:\SSD-PGU3\電動モビリティ専門職大学\山形大学申請\ACM-Application Track\bear_ttm_map_2025_2.html'
OUT_HTML      = r'F:\SSD-PGU3\電動モビリティ専門職大学\山形大学申請\ACM-Application Track\bear_glm_map_2025.html'

TRAIN_START = '2018-10-01'
TRAIN_END   = '2024-12-31'
TEST_START  = '2025-01-01'
TEST_END    = '2025-12-31'
GLM_C       = 1.0
RAND_SEED   = 42

# ─── 1. データ読み込み ───────────────────────────────────────────────────────
print('[1] データ読み込み...', flush=True)
df_raw = pd.read_csv(SIGHTINGS_CSV)
df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='mixed')

grid_cols = [c for c in df_raw.columns if re.match(r'^\d+_\d+$', c)]
print(f'    グリッド数: {len(grid_cols)}')

# train / test split
mask_tr = (df_raw['Date'] >= TRAIN_START) & (df_raw['Date'] <= TRAIN_END)
mask_te = (df_raw['Date'] >= TEST_START)  & (df_raw['Date'] <= TEST_END)

df_tr = df_raw[mask_tr].copy()
df_te = df_raw[mask_te].copy()

train_L = (df_tr[grid_cols].values > 0).astype(np.float32)   # (T_tr, n_cells)
test_L  = (df_te[grid_cols].values > 0).astype(np.float32)   # (T_te, n_cells)
test_actual = df_te[grid_cols].values.astype(np.int32)        # 実出没件数

train_dates = pd.to_datetime(df_tr['Date'].values)
test_dates  = pd.to_datetime(df_te['Date'].values)

n_cells = len(grid_cols)
T_tr    = len(train_dates)
T_te    = len(test_dates)
print(f'    訓練期間: {train_dates[0].date()} - {train_dates[-1].date()} ({T_tr} 日)')
print(f'    評価期間: {test_dates[0].date()} - {test_dates[-1].date()} ({T_te} 日)')

# ─── 2. 特徴量生成 ───────────────────────────────────────────────────────────
print('[2] 特徴量生成...', flush=True)

all_L = np.concatenate([train_L, test_L], axis=0).astype(np.float64)
cs    = np.zeros((len(all_L) + 1, n_cells), dtype=np.float64)
np.cumsum(all_L, axis=0, out=cs[1:])

def rolling_sum(pos, window):
    start = max(0, pos - window)
    return (cs[pos] - cs[start]).astype(np.float32)

base_year = train_dates[0].year

def make_block(dates, offset):
    T = len(dates)
    r30  = np.empty((T, n_cells), np.float32)
    r365 = np.empty((T, n_cells), np.float32)
    sin_ = np.empty(T, np.float32)
    cos_ = np.empty(T, np.float32)
    yr_  = np.empty(T, np.float32)
    for i, d in enumerate(dates):
        pos = offset + i
        r30[i]  = rolling_sum(pos, 30)
        r365[i] = rolling_sum(pos, 365)
        doy      = d.timetuple().tm_yday
        sin_[i]  = np.sin(2 * np.pi * doy / 365)
        cos_[i]  = np.cos(2 * np.pi * doy / 365)
        yr_[i]   = (d.year - base_year) / 10.0
    log_r365 = np.log1p(r365)
    # (T, n_cells, 5) -> (T*n_cells, 5)
    sin_rep = np.repeat(sin_[:, None], n_cells, axis=1)
    cos_rep = np.repeat(cos_[:, None], n_cells, axis=1)
    yr_rep  = np.repeat(yr_[:, None],  n_cells, axis=1)
    feat = np.stack([r30, log_r365, sin_rep, cos_rep, yr_rep], axis=2)
    return feat.reshape(-1, 5)

feat_tr = make_block(train_dates, offset=0)
feat_te = make_block(test_dates,  offset=T_tr)

cell_idx_tr = np.tile(np.arange(n_cells), T_tr).astype(np.int32)
cell_idx_te = np.tile(np.arange(n_cells), T_te).astype(np.int32)

print(f'    feat_tr: {feat_tr.shape}  feat_te: {feat_te.shape}')

# ─── 3. GLM-Logit 訓練 ──────────────────────────────────────────────────────
print('[3] GLM-Logit 訓練中...', flush=True)

def build_design_matrix(feat, cell_idx, n_cells):
    """shared feature cols + cell one-hot"""
    N = feat.shape[0]
    # cell one-hot (sparse)
    rows = np.arange(N)
    data = np.ones(N, dtype=np.float32)
    cell_oh = csr_matrix((data, (rows, cell_idx)), shape=(N, n_cells))
    # shared features (dense -> sparse)
    from scipy.sparse import csr_matrix as csrm
    feat_sp = csrm(feat)
    return sp_hstack([cell_oh, feat_sp], format='csr')

X_tr = build_design_matrix(feat_tr, cell_idx_tr, n_cells)
X_te = build_design_matrix(feat_te, cell_idx_te, n_cells)
y_tr = train_L.flatten().astype(np.float32)

clf = LogisticRegression(C=GLM_C, fit_intercept=False, max_iter=2000,
                          solver='lbfgs', random_state=RAND_SEED, verbose=1)
clf.fit(X_tr, y_tr)
print('    訓練完了')

proba = clf.predict_proba(X_te)[:, 1]
glm_scores = proba.reshape(T_te, n_cells).astype(np.float32)   # (T_te, n_cells)
print(f'    scores shape={glm_scores.shape}  mean={glm_scores.mean():.4f}  max={glm_scores.max():.4f}')

# ─── 4. COORDS 構築 ──────────────────────────────────────────────────────────
print('[4] COORDS 構築...', flush=True)

# TTM HTML からの既存 city 名マッピング
with open(TTM_HTML, encoding='utf-8') as f:
    ttm_content = f.read()
m = re.search(r'const COORDS = ({.*?});', ttm_content, re.DOTALL)
ttm_coords = json.loads(m.group(1))

# TTM にない追加グリッドの市町村名マッピング（座標から推定）
extra_city_map = {
    # 庄内地方 (西部・沿岸)
    '0_9':  '酒田市西部',
    '1_3':  '飯豊町',
    '1_11': '酒田市北部',
    '2_11': '鶴岡市北部',
    '2_12': '酒田市南部',
    '2_14': '遊佐町',
    # 置賜地方 (南部)
    '2_2':  '長井市',
    '2_3':  '白鷹町',
    '3_2':  '川西町',
    '5_2':  '南陽市',
    # 村山地方 (中央)
    '5_4':  '山形市',
    '5_6':  '河北町',
    '5_7':  '大石田町',
    '5_10': '村山市',
    '6_5':  '天童市',
    '6_7':  '村山市東部',
    '6_8':  '尾花沢市',
    '7_4':  '山辺町・中山町',
    '7_7':  '尾花沢市東部',
    '7_8':  '金山町南部',
    # 最上地方 (北部)
    '3_14': '舟形町',
    '2_14': '遊佐町北部',
    '5_12': '大蔵村',
    '6_11': '新庄市',
    '8_9':  '金山町',
}

# 全 144 グリッドの座標を Yamagata_10km_Grid_0.csv から取得
df_grid = pd.read_csv(GRID_CSV)
grid_info = {}
for _, row in df_grid.iterrows():
    gid = row['Grid_ID']
    # city名: TTM COORDS → extra_city_map → grid ID
    city = ttm_coords.get(gid, {}).get('city') or extra_city_map.get(gid) or gid
    grid_info[gid] = {
        'lat': float(row['Center_Latitude']),
        'lng': float(row['Center_Longitude']),
        'lat_min': float(row['Min_Latitude']),
        'lat_max': float(row['Max_Latitude']),
        'lng_min': float(row['Min_Longitude']),
        'lng_max': float(row['Max_Longitude']),
        'city': city,
    }

known = sum(1 for gid, v in grid_info.items() if v['city'] != gid)
print(f'    グリッド情報: {len(grid_info)} 件  (city 名あり: {known} 件)')

# grid_cols の順に並べた COORDS だけを含める
# (実際にスコアが出るグリッドのみ)
# 全 144 を COORDS に含めると JS が重くなるので上位候補グリッドに絞る
# 2025 年を通じて少なくとも 1 日でも Top-20 に入ったグリッドを対象とする
TOP_K = 20  # 1 日の表示上限

active_grids = set()
for t in range(T_te):
    day_scores = glm_scores[t]
    top_idx = np.argsort(day_scores)[::-1][:TOP_K]
    for idx in top_idx:
        if day_scores[idx] > 0:
            active_grids.add(grid_cols[idx])

print(f'    年間アクティブグリッド数: {len(active_grids)}')

# COORDS 辞書を構築（アクティブグリッドのみ）
coords_dict = {}
for gid in active_grids:
    if gid in grid_info:
        coords_dict[gid] = grid_info[gid]
    else:
        print(f'    WARNING: {gid} not in grid_info')

# ─── 5. DAILY データ構築 ────────────────────────────────────────────────────
print('[5] DAILY データ構築...', flush=True)

# scores を正規化: TTM マップは raw score をそのまま使い、max を "m" として格納
# GLM も同様に probability (0-1) をスコアとして使う

daily_data = {}
for t, d in enumerate(test_dates):
    ds = d.strftime('%Y-%m-%d')
    day_scores = glm_scores[t]       # (n_cells,)
    day_actual = test_actual[t]      # (n_cells,)

    # 上位 TOP_K グリッドのみ格納
    top_idx = np.argsort(day_scores)[::-1][:TOP_K]
    grids = {}
    for idx in top_idx:
        gid = grid_cols[idx]
        score = float(day_scores[idx])
        if score <= 0:
            continue
        actual_cnt = int(day_actual[idx])
        # [score (0-1), actual_count, raw (= score for GLM)]
        grids[gid] = [round(score, 5), actual_cnt, round(score, 5)]

    mx = float(day_scores.max()) if len(grids) > 0 else 0.0
    daily_data[ds] = {'m': round(mx, 5), 'g': grids}

print(f'    日数: {len(daily_data)}')
non_empty = sum(1 for v in daily_data.values() if len(v['g']) > 0)
print(f'    出没予測ありの日: {non_empty}')

# ─── 6. HTML 生成 ────────────────────────────────────────────────────────────
print('[6] HTML 生成中...', flush=True)

DAILY_JSON  = json.dumps(daily_data,  ensure_ascii=False, separators=(',', ':'))
COORDS_JSON = json.dumps(coords_dict, ensure_ascii=False, separators=(',', ':'))

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>山形県 熊出没危険予測マップ 2025 — GLM-Logit</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Meiryo',sans-serif;background:#0f1a14;color:#e0eed5;height:100vh;display:flex;flex-direction:column}}
#header{{background:#1a3a2a;padding:10px 16px;display:flex;align-items:center;gap:12px;border-bottom:2px solid #2c5f3a;flex-shrink:0}}
#header h1{{font-size:17px;font-weight:700;color:#fff}}
#header p{{font-size:10px;color:#88aa94;margin-top:2px}}
.badge{{font-size:11px;font-weight:700;padding:3px 10px;border-radius:12px;white-space:nowrap;background:#2c5f3a;color:#fff}}
#notice{{background:#0d2a1a;border-bottom:1px solid #1e4028;padding:5px 16px;font-size:11px;color:#88aa94;flex-shrink:0}}
#notice strong{{color:#e07b2a}}
#main{{display:flex;flex:1;overflow:hidden}}
#sidebar{{width:290px;background:#111f17;overflow-y:auto;flex-shrink:0}}
#map{{flex:1}}
.panel{{padding:11px 13px;border-bottom:1px solid #1e3828}}
.pt{{font-size:11px;color:#88aa94;font-weight:700;margin-bottom:7px;letter-spacing:.04em;text-transform:uppercase}}
#date-display{{font-size:22px;font-weight:700;color:#fff;line-height:1.1}}
#day-info{{font-size:11px;color:#88aa94;margin-top:2px}}
.month-row{{display:flex;gap:2px;flex-wrap:wrap;margin:6px 0 2px}}
.mbtn{{background:#1e3828;border:1px solid #2c5f3a;color:#88aa94;font-size:10px;padding:2px 4px;border-radius:3px;cursor:pointer;flex:1;min-width:26px;text-align:center}}
.mbtn:hover,.mbtn.act{{background:#e07b2a;color:#fff;border-color:#e07b2a}}
#date-slider{{width:100%;accent-color:#e07b2a;cursor:pointer;margin:6px 0 2px}}
.ctrl-row{{display:flex;gap:5px;margin-top:5px}}
.btn{{background:#1e3828;border:1px solid #2c5f3a;color:#cce8d4;font-size:12px;padding:5px 8px;border-radius:5px;cursor:pointer;flex:1;text-align:center}}
.btn:hover{{background:#2c5f3a}}
.btn.play{{background:#e07b2a;border-color:#e07b2a;color:#fff}}
#speed-sel{{background:#1e3828;border:1px solid #2c5f3a;color:#cce8d4;font-size:12px;padding:4px 5px;border-radius:5px;cursor:pointer}}
.thresh-row{{display:flex;align-items:center;gap:6px;margin-top:5px}}
.thresh-row label{{font-size:11px;color:#88aa94;white-space:nowrap}}
#thresh-slider{{flex:1;accent-color:#1D9E75;cursor:pointer}}
#thresh-val{{font-size:11px;color:#1D9E75;font-weight:700;min-width:30px;text-align:right}}
.leg-row{{display:flex;gap:6px;flex-wrap:wrap}}
.leg-item{{display:flex;align-items:center;gap:4px;font-size:11px;color:#88aa94;flex:1;min-width:80px}}
.leg-dot{{width:13px;height:13px;border-radius:50%;flex-shrink:0}}
.stat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}
.sc{{background:#1a2e20;border-radius:5px;padding:7px 9px}}
.sc-l{{font-size:10px;color:#88aa94}}
.sc-v{{font-size:19px;font-weight:700;color:#fff;line-height:1.2}}
#grid-panel{{background:#0d1e14;display:none}}
#grid-title{{font-size:12px;font-weight:700;color:#e07b2a;margin-bottom:3px}}
#grid-city{{font-size:15px;font-weight:700;color:#fff;margin-bottom:1px}}
#grid-rank{{font-size:11px;color:#88aa94;margin-bottom:7px}}
.lv-badge{{display:inline-block;font-size:13px;font-weight:700;padding:3px 13px;border-radius:4px;margin-bottom:5px}}
.lv0{{background:#2ECC71;color:#0a3a1a}}.lv1{{background:#F39C12;color:#3a1a00}}
.lv2{{background:#E74C3C;color:#fff}}.lv3{{background:#8E0000;color:#fff}}
.bar-wrap{{background:#1e3828;border-radius:3px;height:7px;overflow:hidden;margin:3px 0}}
.bar-fill{{height:100%;border-radius:3px;transition:width .3s}}
.grid-sub{{font-size:11px;color:#88aa94;margin-top:3px}}
.pred-note{{font-size:10px;color:#4a7a5a;margin-top:4px;font-style:italic}}
#rl{{margin-top:2px}}
.rr{{display:flex;align-items:center;gap:5px;padding:4px 0;border-bottom:1px solid #1a2e20;cursor:pointer}}
.rr:hover{{background:#1a2e20}}
.rn{{font-size:11px;color:#88aa94;width:15px;text-align:right;flex-shrink:0}}
.rc{{font-size:12px;color:#cce8d4;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.rb-w{{width:65px;background:#1e3828;border-radius:2px;height:5px;flex-shrink:0}}
.rb{{height:100%;border-radius:2px}}
.rl{{font-size:10px;font-weight:700;width:36px;text-align:right;flex-shrink:0}}
#glm-badge{{display:inline-flex;align-items:center;gap:5px;background:#2a1a3a;border:1px solid #7a3fa5;border-radius:8px;padding:3px 9px;font-size:10px;color:#c9a0e8;font-weight:700;white-space:nowrap}}
</style>
</head>
<body>
<div id="header">
  <span style="font-size:22px">🐻</span>
  <div>
    <h1>山形県　熊出没危険予測マップ</h1>
    <p>YAMAGATA BEAR ENCOUNTER RISK FORECAST — GLM-Logit (L2-Regularized Logistic Regression) · DAILY GRID ANALYSIS 2025</p>
  </div>
  <div style="margin-left:auto;display:flex;align-items:center;gap:8px;flex-shrink:0">
    <div id="glm-badge">📊 GLM-Logit 予測確率</div>
    <div style="display:flex;gap:4px;align-items:center">
      <div style="width:70px;height:8px;border-radius:4px;background:linear-gradient(to right,#2ECC71,#F39C12,#E74C3C,#8E0000)"></div>
      <span style="font-size:10px;color:#88aa94">低→警戒</span>
    </div>
    <span id="top-badge" class="badge">● 算出中</span>
  </div>
</div>
<div id="notice">
  ⚠ <strong>本マップはGLM-Logit（L2正則化ロジスティック回帰）による予測確率を表示しています。</strong>　訓練期間: 2018年10月〜2024年12月。グリッド位置は山形県10kmグリッド定義に基づく正確な座標。対象は上位20地区のみ。表示のない地区が安全であることを意味しません。　公式警戒情報は山形県・各市町村の発表に従ってください。
</div>
<div id="main">
<div id="sidebar">
  <div class="panel">
    <div class="pt">日付選択</div>
    <div id="date-display">2025-01-01</div>
    <div id="day-info">第1日目</div>
    <div class="month-row" id="mrow"></div>
    <input type="range" id="date-slider" min="0" max="364" value="0">
    <div class="ctrl-row">
      <button class="btn" id="prev-btn">◀</button>
      <button class="btn play" id="play-btn">▶ 再生</button>
      <button class="btn" id="next-btn">▶</button>
      <select id="speed-sel">
        <option value="700">普通</option>
        <option value="250">速い</option>
        <option value="100">最速</option>
      </select>
    </div>
  </div>
  <div class="panel">
    <div class="pt">表示閾値（GLM予測確率）</div>
    <div class="thresh-row">
      <label>閾値:</label>
      <input type="range" id="thresh-slider" min="5" max="80" value="15">
      <span id="thresh-val">15%</span>
    </div>
  </div>
  <div class="panel">
    <div class="pt">リスクレベル凡例</div>
    <div class="leg-row">
      <div class="leg-item"><div class="leg-dot" style="background:#2ECC71"></div>低危険 (&lt;20%)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#F39C12"></div>中危険 (20-45%)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#E74C3C"></div>高危険 (45-70%)</div>
      <div class="leg-item"><div class="leg-dot" style="background:#8E0000"></div>警戒 (≥70%)</div>
    </div>
  </div>
  <div class="panel">
    <div class="pt">本日の状況</div>
    <div class="stat-grid">
      <div class="sc"><div class="sc-l">アクティブグリッド</div><div class="sc-v" id="s-active">-</div></div>
      <div class="sc"><div class="sc-l">最高警戒</div><div class="sc-v" id="s-top">-</div></div>
      <div class="sc"><div class="sc-l">警戒グリッド数</div><div class="sc-v" id="s-warn">-</div></div>
      <div class="sc"><div class="sc-l">実出没件数</div><div class="sc-v" id="s-actual">-</div></div>
    </div>
    <div class="pred-note">GLM-Logit による予測確率 (訓練: 2018-10〜2024-12)</div>
  </div>
  <div class="panel" id="grid-panel">
    <div id="grid-title">グリッド詳細</div>
    <div id="grid-city"></div>
    <div id="grid-rank"></div>
    <span class="lv-badge" id="grid-lv-badge"></span>
    <div class="bar-wrap"><div class="bar-fill" id="grid-bar" style="width:0%"></div></div>
    <div id="grid-sub"></div>
  </div>
  <div class="panel">
    <div class="pt">本日リスクランキング <span style="color:#4a7a5a;font-size:9px">（GLM予測確率順）</span></div>
    <div id="rl"></div>
  </div>
</div>
<div id="map"></div>
</div>
<script>
const DAILY  = {DAILY_JSON};
const COORDS = {COORDS_JSON};
const DATES  = Object.keys(DAILY).sort();

const LV_COLOR = ['#2ECC71','#F39C12','#E74C3C','#8E0000'];
const LV_LABEL = ['低危険','中危険','高危険','警戒'];
const LV_CLASS = ['lv0','lv1','lv2','lv3'];

function getLevel(s){{ if(s>=0.7)return 3; if(s>=0.45)return 2; if(s>=0.2)return 1; return 0; }}

const map = L.map('map',{{zoomControl:true}}).setView([38.4,140.0],8);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
  attribution:'© OpenStreetMap contributors', maxZoom:18
}}).addTo(map);

const gridLayers = {{}};
Object.entries(COORDS).forEach(([gid,info])=>{{
  const b = [[info.lat_min, info.lng_min],[info.lat_max, info.lng_max]];
  const r = L.rectangle(b,{{
    color:'#444', weight:0.8,
    fillColor:'#2ECC71', fillOpacity:0, opacity:0
  }}).addTo(map);
  r.on('click',()=>showDetail(gid));
  r.bindTooltip(info.city, {{permanent:false, direction:'center', className:'grid-tip'}});
  gridLayers[gid] = r;
}});

let curIdx=0, playing=false, timer=null, thresh=0.15;

function dayNum(d){{ return Math.round((new Date(d)-new Date('2025-01-01'))/86400000)+1; }}
function season(d){{ const m=parseInt(d.slice(5,7)); return m>=3&&m<=5?'🌸春':m>=6&&m<=8?'☀️夏':m>=9&&m<=11?'🍂秋':'❄️冬'; }}

function updateMap(idx){{
  const ds=DATES[idx], day=DAILY[ds], gs=day.g||{{}}, mx=day.m||0, thr=mx*thresh;
  document.getElementById('date-display').textContent = ds;
  document.getElementById('day-info').textContent = `第${{dayNum(ds)}}日目　${{season(ds)}}`;
  document.getElementById('date-slider').value = idx;

  const cm = parseInt(ds.slice(5,7));
  document.querySelectorAll('.mbtn').forEach((b,i)=>b.classList.toggle('act', i+1===cm));

  let active=0, warn=0, actualSum=0;
  const ranked=[];

  Object.entries(COORDS).forEach(([gid,info])=>{{
    const g=gs[gid], score=g?g[0]:0, actual=g?g[1]:0, raw=g?g[2]:0;
    actualSum += actual;
    const vis = score>=thr && score>0;
    const lv  = getLevel(score);
    if(vis){{
      active++;
      if(lv===3) warn++;
      ranked.push({{gid,score,actual,raw,lv,city:info.city}});
      gridLayers[gid].setStyle({{
        fillColor:LV_COLOR[lv], fillOpacity:0.6,
        color:LV_COLOR[lv], weight:1.5, opacity:0.9
      }});
    }} else {{
      gridLayers[gid].setStyle({{fillOpacity:0, opacity:0.15, color:'#444', weight:0.5}});
    }}
  }});

  ranked.sort((a,b)=>b.score-a.score);

  document.getElementById('s-active').textContent  = active;
  document.getElementById('s-warn').textContent    = warn;
  document.getElementById('s-actual').textContent  = actualSum;

  const topLv = ranked.length>0 ? ranked[0].lv : -1;
  const badge = document.getElementById('top-badge');
  if(topLv>=0){{
    badge.textContent = '● '+LV_LABEL[topLv];
    badge.style.background = LV_COLOR[topLv];
    badge.style.color = topLv<=1?'#1a1a1a':'#fff';
  }} else {{
    badge.textContent='● 出没予測なし';
    badge.style.background='#2c5f3a';
    badge.style.color='#fff';
  }}
  document.getElementById('s-top').textContent = topLv>=0?LV_LABEL[topLv]:'-';

  const rl = document.getElementById('rl');
  rl.innerHTML = '';
  ranked.slice(0,15).forEach((r,i)=>{{
    const pct   = Math.round(r.score*100);
    const medal = i===0?'🥇':i===1?'🥈':i===2?'🥉':String(i+1);
    rl.innerHTML += `<div class="rr" onclick="showDetail('${{r.gid}}')">
      <span class="rn">${{medal}}</span>
      <span class="rc">${{r.city}}<span style="color:#3a5a3a;font-size:10px"> [${{r.gid}}]</span></span>
      <div class="rb-w"><div class="rb" style="width:${{pct}}%;background:${{LV_COLOR[r.lv]}}"></div></div>
      <span class="rl" style="color:${{LV_COLOR[r.lv]}}">${{LV_LABEL[r.lv]}}</span>
    </div>`;
  }});
  if(!ranked.length) rl.innerHTML='<div style="font-size:11px;color:#3a5a3a;padding:6px 0">本日の高リスクグリッドなし</div>';
}}

function showDetail(gid){{
  const ds=DATES[curIdx], day=DAILY[ds], gs=day.g||{{}}, mx=day.m||0, thr=mx*thresh;
  const g=gs[gid]; if(!g) return;
  const score=g[0], actual=g[1], raw=g[2], lv=getLevel(score), info=COORDS[gid];
  const ranked = Object.entries(gs).filter(([,v])=>v[0]>=thr).sort((a,b)=>b[1][0]-a[1][0]);
  const rank   = ranked.findIndex(([k])=>k===gid)+1;
  document.getElementById('grid-panel').style.display = 'block';
  document.getElementById('grid-title').textContent   = `グリッド ${{gid}}`;
  document.getElementById('grid-city').textContent    = info.city;
  document.getElementById('grid-rank').textContent    =
    rank>0?`本日ランキング #${{rank}} / ${{ranked.length}}地区中`:'閾値以下（参考）';
  const badge=document.getElementById('grid-lv-badge');
  badge.textContent=LV_LABEL[lv]; badge.className=`lv-badge ${{LV_CLASS[lv]}}`;
  const bar=document.getElementById('grid-bar');
  bar.style.width=`${{Math.round(score*100)}}%`; bar.style.background=LV_COLOR[lv];
  document.getElementById('grid-sub').textContent =
    `GLM予測確率: ${{(score*100).toFixed(1)}}%　実出没: ${{actual}}件`;
}}

const months=['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
const mrow=document.getElementById('mrow');
months.forEach((m,i)=>{{
  const b=document.createElement('button');
  b.className='mbtn'; b.textContent=m;
  b.onclick=()=>{{
    const t=`2025-${{String(i+1).padStart(2,'0')}}-01`;
    const idx=DATES.findIndex(d=>d>=t);
    if(idx>=0){{curIdx=idx; updateMap(idx);}}
  }};
  mrow.appendChild(b);
}});

document.getElementById('date-slider').addEventListener('input',e=>{{curIdx=parseInt(e.target.value);updateMap(curIdx);}});
document.getElementById('prev-btn').addEventListener('click',()=>{{if(curIdx>0){{curIdx--;updateMap(curIdx);}}}});
document.getElementById('next-btn').addEventListener('click',()=>{{if(curIdx<DATES.length-1){{curIdx++;updateMap(curIdx);}}}});
document.getElementById('thresh-slider').addEventListener('input',e=>{{
  thresh=parseInt(e.target.value)/100;
  document.getElementById('thresh-val').textContent=e.target.value+'%';
  updateMap(curIdx);
}});

let playSpeed=700;
function doPlay(){{
  timer=setInterval(()=>{{
    if(curIdx>=DATES.length-1){{
      playing=false; clearInterval(timer);
      document.getElementById('play-btn').textContent='▶ 再生';
      document.getElementById('play-btn').className='btn play';
      return;
    }}
    curIdx++; updateMap(curIdx);
  }}, playSpeed);
}}
document.getElementById('play-btn').addEventListener('click',()=>{{
  playing=!playing;
  document.getElementById('play-btn').textContent = playing?'⏸ 停止':'▶ 再生';
  document.getElementById('play-btn').className   = playing?'btn':'btn play';
  playing ? doPlay() : clearInterval(timer);
}});
document.getElementById('speed-sel').addEventListener('change',e=>{{
  playSpeed=parseInt(e.target.value);
  if(playing){{clearInterval(timer); doPlay();}}
}});
document.addEventListener('keydown',e=>{{
  if(e.key==='ArrowLeft'&&curIdx>0){{curIdx--;updateMap(curIdx);}}
  if(e.key==='ArrowRight'&&curIdx<DATES.length-1){{curIdx++;updateMap(curIdx);}}
}});

updateMap(0);
</script>
</body>
</html>"""

with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'[完了] 出力: {OUT_HTML}')
sz = len(html) / 1024
print(f'       ファイルサイズ: {sz:.1f} KB')
