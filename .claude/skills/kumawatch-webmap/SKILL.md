---
name: kumawatch-webmap
description: KumaWatchのインタラクティブWebマップ(三層マップ kumawatch_primary_layer.html / GLM単層マップ)を scripts/generate_kumawatch_webmap.py 等で生成・更新し、GitHub Pages に公開する手順。マップの再生成、maps/ 配下のHTML更新、公開デモの更新時に使う。
---

# KumaWatch Webマップ生成

Leaflet.js ベースの自己完結型 HTML マップ(10kmグリッド、365日スライダー、レイヤー切替)を生成する。

- 公開デモ: https://todalaba.github.io/KumaWatch/maps/kumawatch_primary_layer.html
- リポジトリ内成果物: `maps/kumawatch_primary_layer.html`(三層)/ `maps/kumawatch_complementary_layer.html`(補完層ビュー)

**注意**: `kumawatch_complementary_layer.html` を生成するスクリプトは**リポジトリに含まれていない**(作者環境で生成された成果物のみコミットされている)。再生成できるのは三層マップ(下記1)だけ。補完層ビューを更新する必要が出たら、三層マップ生成スクリプトを流用して補完層のみのバリアントを作るのが現実的。

データ形式・パス対応は **kumawatch-data-reference** Skill を参照。

## 1. 三層マップ生成(`scripts/generate_kumawatch_webmap.py`)

GLM-Logit(主層)+ HierBayes Beta-Binomial 近似(不確実性層)+ TTM・Extra Trees(補完層)を1枚のHTMLに焼き込む。ET はスクリプト内で時間特徴量から訓練するため ET スコアファイルは不要。TTM スコア CSV のみ事前に必要。

### 手順

1. **依存**: `pip install numpy pandas scipy scikit-learn`

2. **パス設定を書き換える**。スクリプト冒頭(`generate_kumawatch_webmap.py:24-28`)は作者の Windows パス(`F:\SSD-PGU3\...`)なので、リポジトリ内ファイルに変更する:

   ```python
   SIGHTINGS_CSV  = 'data/yamagata_10km_daily_timeseries.csv'
   GRID_CSV       = 'data/yamagata_10km_grid_coords.csv'
   TTM_SCORES_CSV = 'data/scores/yamagata_ttm_scores_2025.csv'
   OUT_HTML       = 'kumawatch_map_2025.html'
   ```

   (`BASE` 変数はこれらの組み立てにしか使われないので、絶対/相対パスを直接書けば削除可)

3. **実行**:

   ```bash
   python scripts/generate_kumawatch_webmap.py
   ```

   学習期間 2018-10-01〜2024-12-31、評価 2025年、`GLM_C=1.0`、`RAND_SEED=42`(論文設定)。

4. **出力の配置**(READMEに欠けていた手順): 出力 `kumawatch_map_2025.html`(自己完結、約1.5MB)をブラウザで動作確認後、公開用に配置する:

   ```bash
   mv kumawatch_map_2025.html maps/kumawatch_primary_layer.html
   ```

## 2. GLM単層マップ(`scripts/generate_glm_webmap.py`)— 参考

GLM-Logit のみの単層マップを生成する(READMEには未記載のスクリプト)。

**注意**: このスクリプトは `TTM_HTML`(`bear_ttm_map_2025_2.html`)という**リポジトリに含まれない既存マップHTMLをテンプレートとして読み込む**(`generate_glm_webmap.py:134`)。テンプレートが手元に無い場合はそのままでは動かない。三層マップ(上記1)が上位互換なので、通常はそちらを使う。

動かす場合はヘッダー(`generate_glm_webmap.py:18-21`)の `SIGHTINGS_CSV` / `GRID_CSV` をリポジトリ内ファイルに、`TTM_HTML` を手元のテンプレートHTMLに、`OUT_HTML` を出力先に書き換えて `python scripts/generate_glm_webmap.py` を実行。

## 3. 動作確認

生成 HTML をブラウザで開き(サーバ不要・外部依存なし)、以下を確認:

- [ ] 日付スライダーで 2025-01-01〜2025-12-31 を移動できる
- [ ] レイヤー切替(GLM-Logit / HierBayes / TTM / Extra Trees)が動く
- [ ] セルをクリックすると統計パネル(予測確率・95%CI・実出没数)が出る
- [ ] グリッドが山形県(144セル)の位置に正しく重なっている

## 4. GitHub Pages への公開

`main` ブランチにコミット&プッシュすれば GitHub Pages に反映される(`.nojekyll` 設置済み、`index.html` がランディングページ):

```bash
git add maps/kumawatch_primary_layer.html
git commit -m "Update primary layer web map"
git push origin main
```

反映後 https://todalaba.github.io/KumaWatch/ で確認。
