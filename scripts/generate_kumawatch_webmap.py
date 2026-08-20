"""
KumaWatch Web Decision Support Map Generator
=============================================
公開済みスコアファイルから意思決定支援マップを生成する。

表示する 4 手法は、いずれも論文 Table 1 を生成したスコアそのものを
`data/scores/` から読み込む。**マップ生成時にモデルを学習し直さない。**

  GLM-Logit  yamagata_glm_logit_scores_2025.npy   (Recall@20 = 0.5470)
  HierBayes  yamagata_hier_mean_scores_2025.npy   (Recall@20 = 0.5425)
  TTM        yamagata_ttm_scores_2025.csv         (Recall@20 = 0.4917)
  ET         yamagata_et_scores_2025.csv          (Recall@20 = 0.4739)

以前の版はマップ生成時に GLM-Logit と Extra Trees をその場で学習し、
HierBayes を Beta-Binomial 季節近似で代用していた。ET は環境・人口・
土地被覆の共変量を持たず、HierBayes は階層ベイズですらなかったため、
デモに表示される「Extra Trees」「HierBayes」は論文で評価した手法とは
別物だった。本版はその再学習経路を削除している。

埋め込み直前に verify_scores() が Recall@20・日数・セル数・列整列を
検査し、1 つでも論文値から外れれば HTML を出力せず異常終了する。

Paper: KumaWatch: Benchmarking Wildlife Encounter Prediction for
       Municipal Decision Support in Northern Japan
       ACM SIGSPATIAL 2026
"""

import sys, re, json, csv
from pathlib import Path

import numpy as np
import pandas as pd

# ─── パス設定 ───────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parent.parent
SCORES = REPO / 'data' / 'scores'

SIGHTINGS_CSV = REPO / 'data' / 'yamagata_10km_daily_timeseries.csv'
GRID_CSV      = REPO / 'data' / 'yamagata_10km_grid_coords.csv'
GLM_SCORES_NPY = SCORES / 'yamagata_glm_logit_scores_2025.npy'
HB_SCORES_NPY  = SCORES / 'yamagata_hier_mean_scores_2025.npy'
TTM_SCORES_CSV = SCORES / 'yamagata_ttm_scores_2025.csv'
ET_SCORES_CSV  = SCORES / 'yamagata_et_scores_2025.csv'
OUT_HTML       = REPO / 'maps' / 'kumawatch_primary_layer.html'

TRAIN_START = '2018-10-01'; TRAIN_END = '2024-12-31'
TEST_START  = '2025-01-01'; TEST_END  = '2025-12-31'
RAND_SEED   = 42

# 論文 Table 1 の Recall@20。埋め込み前の検査に使う。
PAPER_R20 = {'GLM-Logit': 0.5470, 'HierBayes': 0.5425,
             'TTM': 0.4917, 'ET': 0.4739}
R20_TOL   = 0.001
N_DAYS_EXPECTED  = 365
N_CELLS_EXPECTED = 144

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

# ─── 2. 公開スコアの読み込みと検査 ───────────────────────────────────────────
print('[2] 公開スコア読み込み (再学習なし)...', flush=True)


def load_score_csv(path, grid_cols, test_dates):
    """CSV スコアを (T_te, n_cells) で読む。列は**名称で整列**する。

    ET の CSV は sightings CSV と列順が異なるため、位置で揃えるとグリッドが
    黙って入れ替わる。欠損列があれば例外にする。
    """
    df = pd.read_csv(path)
    df['_dt'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.set_index('_dt').sort_index()
    missing = [c for c in grid_cols if c not in df.columns]
    if missing:
        raise SystemExit(f'{path.name}: セル列が {len(missing)} 個不足 '
                         f'(先頭: {missing[:3]})')
    out = df.reindex(test_dates)[grid_cols]
    if out.isna().any().any():
        raise SystemExit(f'{path.name}: 評価期間の日付が欠落している')
    return out.values.astype(np.float32)


def load_score_npy(path, n_days, n_cells):
    arr = np.load(path).astype(np.float32)
    if arr.shape != (n_days, n_cells):
        raise SystemExit(f'{path.name}: 形状 {arr.shape} は '
                         f'{(n_days, n_cells)} と一致しない')
    return arr


raw = {
    'GLM-Logit': load_score_npy(GLM_SCORES_NPY, T_te, n_cells),
    'HierBayes': load_score_npy(HB_SCORES_NPY,  T_te, n_cells),
    'TTM':       load_score_csv(TTM_SCORES_CSV, grid_cols, test_dates),
    'ET':        load_score_csv(ET_SCORES_CSV,  grid_cols, test_dates),
}
for name, arr in raw.items():
    print(f'    {name:10s} {arr.shape}  範囲 [{arr.min():.4f}, {arr.max():.4f}]')

# ─── 3. 埋め込み前の検査 ─────────────────────────────────────────────────────
print('[3] 検査 (Recall@20 / 日数 / セル数 / 列整列)...', flush=True)


def recall_at_k(scores, labels, k=20):
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = np.take_along_axis(labels.astype(np.int32), topk, axis=1).sum(axis=1)
    npos = labels.sum(axis=1)
    valid = npos > 0
    return float((hits[valid] / npos[valid]).mean())


def verify_scores(raw, labels):
    ok = True
    if T_te != N_DAYS_EXPECTED:
        print(f'    [NG] 日数 {T_te} != {N_DAYS_EXPECTED}'); ok = False
    else:
        print(f'    [OK] 日数 {T_te}')
    if n_cells != N_CELLS_EXPECTED:
        print(f'    [NG] セル数 {n_cells} != {N_CELLS_EXPECTED}'); ok = False
    else:
        print(f'    [OK] セル数 {n_cells}')
    for name, arr in raw.items():
        got = recall_at_k(arr, labels)
        ref = PAPER_R20[name]
        hit = abs(got - ref) <= R20_TOL
        ok &= hit
        print(f'    [{"OK" if hit else "NG"}] {name:10s} Recall@20 = {got:.4f} '
              f'(論文 {ref:.4f})')
    return ok


if not verify_scores(raw, test_L):
    raise SystemExit('検査に失敗した。論文と異なるスコアを埋め込まないため中止する。')
print('    検査通過 — 論文 Table 1 と同一のスコアを埋め込む')

# ─── 4. 表示用スケーリング ───────────────────────────────────────────────────
# 生の確率は分布が 0 付近に密集するため、そのままではリスク帯 (20/45/70%) に
# ほとんど乗らない。手法ごとに正の値の 95 パーセンタイルで割って表示範囲を
# 揃える。**正のスカラー除算のみで、クリップはしない**ので、各手法内の順位は
# 生スコアと完全に一致する。表示値は確率ではなく正規化リスク指標である。
print('[4] 表示用スケーリング (順位を保つ正規化)...', flush=True)


def normalize_preserving_rank(scores, pct=95):
    pos = scores[scores > 0]
    if len(pos) == 0:
        return scores
    p = float(np.percentile(pos, pct))
    return (scores / p).astype(np.float32) if p > 0 else scores


P95 = {}
for name, arr in raw.items():
    scaled = normalize_preserving_rank(arr)
    assert abs(recall_at_k(arr, test_L) - recall_at_k(scaled, test_L)) < 1e-9, \
        f'{name}: 表示用スケーリングで順位が変化した'
    pos = arr[arr > 0]
    P95[name] = float(np.percentile(pos, 95)) if len(pos) else 1.0
    print(f'    {name:10s} p95 = {P95[name]:.5f}  '
          f'表示レンジ [0, {arr.max() / P95[name]:.2f}]  (順位不変)')

# HTML には**生スコアをそのまま**埋め込む。詳細パネルに出る数値は論文の
# スコアそのもので、順位もそれに一致する。色分けと閾値スライダーだけが
# p95 で割った値を使う (正のスカラー除算なので順位は変わらない)。
glm_scores = raw['GLM-Logit']
hb_mean    = raw['HierBayes']
ttm_scores = raw['TTM']
et_scores  = raw['ET']


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
# 既刊マップからの city マッピング (フォールバック、無ければ空)
try:
    with open(OUT_HTML, encoding='utf-8') as f:
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
INCLUDE_TOP_K = 40  # 各手法の上位 K セルを格納する (top-20 表示を厳密に保つ)
grid_col_idx   = {gid: i for i, gid in enumerate(grid_cols)}

daily_data = {}
for t, d in enumerate(test_dates):
    ds = d.strftime('%Y-%m-%d')
    g_glm  = glm_scores[t]   # (n_cells,)
    g_hbm  = hb_mean[t]
    g_ttm  = ttm_scores[t]
    g_et   = et_scores[t]
    g_act  = test_actual[t]  # 実目撃数

    # 格納するセルは**順位**で決める。各手法の上位 INCLUDE_TOP_K と、実際に
    # 目撃があったセルを必ず含める。スコアの絶対値で足切りすると、その日の
    # 上位セルでもスコアが小さいと落ち、マップ上の順位が論文のモデルと
    # ずれてしまう (旧版はこれで top-20 一致率が 0.88 まで落ちていた)。
    keep = set(np.flatnonzero(g_act > 0).tolist())
    for arr in (g_glm, g_hbm, g_ttm, g_et):
        keep.update(np.argpartition(-arr, INCLUDE_TOP_K)[:INCLUDE_TOP_K].tolist())

    cells = {}
    for ci, gid in enumerate(grid_cols):
        if ci not in keep:
            continue
        glm_v = float(g_glm[ci])
        hb_v  = float(g_hbm[ci])
        ttm_v = float(g_ttm[ci])
        et_v  = float(g_et[ci])
        # 6 桁で保持する。4 桁だと丸めで同点が生まれ、その日の順位が
        # 論文のモデルとわずかにずれる。
        cells[gid] = [
            round(glm_v, 6),
            round(hb_v,  6),
            round(ttm_v, 6),
            round(et_v,  6),
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
p95_json = json.dumps({'glm': round(P95['GLM-Logit'], 6),
                       'hb':  round(P95['HierBayes'], 6),
                       'ttm': round(P95['TTM'], 6),
                       'et':  round(P95['ET'], 6)}, separators=(',', ':'))
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
/* ── on-map cell labels ── */
.cell-label{{background:transparent!important;border:none!important;box-shadow:none!important;color:#0a200f;font-size:10px;font-weight:700;font-family:'Meiryo','Hiragino Kaku Gothic ProN',sans-serif;text-align:center;white-space:nowrap;pointer-events:none!important;text-shadow:0 0 4px #fff,0 0 4px #fff,1px 1px 0 rgba(255,255,255,.9),-1px -1px 0 rgba(255,255,255,.9)}}
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
    <span id="layer-name">GLM-Logit スコア</span>
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
        <div class="dp-row"><span class="dp-label">スコア</span><span class="dp-val" id="dp-glm">—</span></div>
        <div class="dp-row"><span class="dp-label">本日順位</span><span class="dp-val" id="dp-rank">—</span></div>
      </div>
      <!-- 不確実性層 -->
      <div class="dp-section">
        <div class="dp-section-title">不確実性層 — HierBayes</div>
        <div class="dp-row">
          <span class="dp-label">事後平均</span>
          <span class="dp-val" id="dp-hbm">—</span>
        </div>
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
  glm: 'GLM-Logit スコア',
  hb:  'HierBayes 事後平均',
  ttm: 'TTM スコア',
  et:  'Extra Trees スコア',
}};
// Cell data indices: [glm, hb_m, ttm, et, act]
// All four are rank-preserving normalisations of the released score files.
const IDX = {{glm:0, hb_m:1, ttm:2, et:3, act:4}};

// Stored values are the released raw scores, so the detail panel and the daily
// ranking are exactly the paper's. Colour tiers and the threshold slider work on
// a per-method 95th-percentile rescaling — a positive scalar divide, so it never
// reorders anything.
const P95 = {p95_json};
function disp(v, layer) {{ return v / (P95[layer] || 1); }}

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
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  opacity: 1.0,
  maxZoom: 18
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

// City-name label markers (divIcon, centered on each cell)
const labelMarkers = {{}};
Object.entries(COORDS).forEach(([gid, c]) => {{
  const city = c.city;
  if (!city || city === gid) return;
  const lm = L.marker([c.lat, c.lng], {{
    icon: L.divIcon({{
      className: 'cell-label',
      html: city,
      iconSize: [96, 16],
      iconAnchor: [48, 8]
    }}),
    interactive: false,
    zIndexOffset: 500
  }}).addTo(map);
  const el = lm.getElement();
  if (el) el.style.visibility = 'hidden';
  labelMarkers[gid] = lm;
}});

// ─── Map render ──────────────────────────────────────────────────────────────
function renderMap() {{
  const ds    = DATES[currentIdx];
  const cells = DAILY[ds] || {{}};
  Object.entries(COORDS).forEach(([gid, c]) => {{
    const cell  = cells[gid];
    const score = cell ? disp(layerScore(cell, currentLayer), currentLayer) : 0;
    const rect  = rects[gid];
    if (!rect) return;
    if (score < hideThreshold) {{
      rect.setStyle({{fillOpacity:0, opacity:0}});
      const lm = labelMarkers[gid];
      if (lm) {{ const el = lm.getElement(); if (el) el.style.visibility = 'hidden'; }}
    }} else {{
      const ri = riskInfo(score);
      rect.setStyle({{
        fillColor: ri.color, fillOpacity: 0.45,
        color:'#ffffff', opacity:0.80, weight:1.5
      }});
      const lm = labelMarkers[gid];
      if (lm) {{ const el = lm.getElement(); if (el) el.style.visibility = 'visible'; }}
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
    const s = disp(layerScore(cell, currentLayer), currentLayer);
    if (s >= hideThreshold) active++;
    if (disp(cell[IDX.glm], 'glm') >= 0.70) alertCnt++;
    totalAct += cell[IDX.act];
    if (cell[IDX.glm] > maxScore) maxScore = cell[IDX.glm];
  }});
  const ri = riskInfo(disp(maxScore, 'glm'));
  document.getElementById('st-active').textContent  = active;
  document.getElementById('st-maxrisk').innerHTML   =
    '<span class="rank-badge ' + ri.cls + '">' + ri.label + '</span>';
  document.getElementById('st-alert').textContent   = alertCnt;
  document.getElementById('st-actual').textContent  = totalAct;

  // Ranking (always by GLM-Logit)
  const ranked = Object.entries(cells)
    .filter(([,c]) => disp(c[IDX.glm], 'glm') >= hideThreshold)
    .sort((a,b) => b[1][IDX.glm] - a[1][IDX.glm])
    .slice(0, 20);

  let html = '';
  ranked.forEach(([gid, cell], i) => {{
    const city = (COORDS[gid] && COORDS[gid].city) || gid;
    const s    = cell[IDX.glm];
    const ri   = riskInfo(disp(s, 'glm'));
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
  document.getElementById('dp-glm').textContent  = cell[IDX.glm].toFixed(3);
  document.getElementById('dp-rank').textContent = '第' + glmRank + '位 / ' + ranked.length + '位中';

  // HierBayes
  const hbm = cell[IDX.hb_m];
  document.getElementById('dp-hbm').textContent = hbm.toFixed(3);

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
