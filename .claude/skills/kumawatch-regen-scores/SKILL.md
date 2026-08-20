---
name: kumawatch-regen-scores
description: KumaWatchの事前計算スコア(Extra Trees / TTM / GLM-Logit / HierBayes の2025年日次スコア)を再生成する手順。data/scores/ のファイルを作り直すとき、新しいデータ期間でスコアを計算し直すときに使う。
---

# KumaWatch スコア再生成

`data/scores/` の事前計算スコアを再生成する手順。モデルごとに生成経路が異なる。

ファイル形式・パス対応表は **kumawatch-data-reference** Skill を参照。

## どのモデルをどこで再生成するか(全体像)

| スコア | 生成場所 | 追加要件 |
|--------|---------|---------|
| Extra Trees (`*_et_scores_2025.csv`) | `scripts/et_benchmark_{yamagata,akita}.py` | **リポジトリ外の生データ+外部共変量データ**(下記) |
| TTM (`*_ttm_scores_2025.csv` / `.npy`) | `notebooks/ttm_{yamagata,akita}.ipynb`(Colab) | IBM watsonx.ai APIキー(無料枠あり) |
| GLM-Logit / HierBayes (`yamagata_glm_logit_*.npy`, `yamagata_hier_*.npy`) | `notebooks/kumawatch_benchmark.ipynb` 内で計算 | PyMC + NumPyro(kumawatch-benchmark Skill 参照) |

## 1. Extra Trees スコア

```bash
python scripts/et_benchmark_yamagata.py   # → yamagata_et_scores_2025.csv
python scripts/et_benchmark_akita.py      # → akita_et_scores_2025.csv
```

### 前提(READMEに書かれていない重要な点)

これらのスクリプトは **リポジトリ同梱の日次時系列CSVではなく、生の出没ポイントデータ**を読む:

- 山形 (`et_benchmark_yamagata.py:49`): `SIGHTINGS_DIR = ../bear-sighting-data/data/yamagata/`
  配下の `sightings_{年}.csv`(`event_date` 列と緯度経度を持つ年別CSV)
- 秋田 (`et_benchmark_akita.py`): `DATA_CSV = ../bear-sighting-data/data/akita/050008_kumadas.csv`

これらの生データは**このリポジトリに含まれない**。県の公開出没データベースから取得して同じレイアウトで配置するか、スクリプト冒頭のパス変数を自分のデータ位置に書き換える。

### パスの書き換え

各スクリプト冒頭の「── paths ──」ブロックを編集する:

| 変数 | デフォルト | 書き換え先 |
|------|-----------|-----------|
| `SIGHTINGS_DIR` / `DATA_CSV` | `../bear-sighting-data/...` | 生データの実際の場所 |
| `GRID_CSV`(山形) | `F:\...\Yamagata_10km_Grid_0.csv`(作者ローカル) | `data/yamagata_10km_grid_coords.csv`(同形式・差し替え可) |

出力は `scripts/results_10km/`(山形)/ `scripts/results_akita_10km/`(秋田)に生成される。生成された wide形式 CSV を `data/scores/{yamagata,akita}_et_scores_2025.csv` にコピーする。

### 外部共変量データ(論文値の再現に必要)

| データ | 入手先 | 使う特徴量 |
|--------|--------|-----------|
| 土地被覆 | [JAXA ALOS Land Cover](https://www.eorc.jaxa.jp/ALOS/en/dataset/lc_e.htm) | セル別の森林/農地/住宅地比率 |
| 国勢調査 | [総務省統計局](https://www.stat.go.jp/english/data/kokusei/) | 人口密度・高齢化率 |
| 気象 | [気象庁](https://www.data.jma.go.jp/gmd/risk/obsdl/) | 日別気温・降水量・積雪深 |
| 堅果類豊凶指数 | [林野庁](https://www.rinya.maff.go.jp/j/hogo/higai/dounami.html) | ブナ・ミズナラの年間豊凶 |

これらが無い場合は時間特徴量(過去の出没履歴)のみで学習され、**結果は論文値と異なる**。依存: `pip install scikit-learn imbalanced-learn matplotlib pandas numpy requests`

## 2. TTM スコア

| 県 | モデル | コンテキスト長 | 予測ホライズン |
|----|--------|--------------|--------------|
| 山形 | IBM Granite **TTM 1536-96-R2** (`ibm/granite-ttm-1536-96-r2`) | 1,536日(学習期間末尾) | 96日 |
| 秋田 | IBM Granite **TTM 512-96-R2** | 512日(学習データが1,536日未満のため) | 96日 |

365日の評価期間は **96日窓×4回のオーバーラップ推論**でカバーする(ゼロショット・in-context learning、追加学習なし)。モデル重みは [Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)。

### 手順

1. `notebooks/ttm_yamagata.ipynb` または `notebooks/ttm_akita.ipynb` を **Google Colab** で開く
2. 依存インストールセル(冒頭の `!pip install` 群: `ibm-watsonx-ai`, `ibm-granite-community/utils` ほかバージョン固定)を実行
3. **watsonx.ai 認証セル**に自分の認証情報を設定(無料枠あり):

   ```python
   os.environ['WX_APIKEY']     = '<APIキー>'
   os.environ['WX_URL']        = 'https://jp-tok.ml.cloud.ibm.com'  # 契約リージョンに合わせる
   os.environ['WX_PROJECT_ID'] = '<プロジェクトID>'
   ```

   ※ APIキーをノートブックに残したままコミットしないこと

4. データ読み込みセルは **Colab の `files.upload()`** でCSVを受け取る。リポジトリの `data/{yamagata,akita}_10km_daily_timeseries.csv` をアップロードする
5. 全セル実行。カレントディレクトリに3つのCSVが出る:
   - `{pref}_{n}cells_predictions_2025.csv` — 全予測(**long形式**: `GridID, timestamp, pred, actual, ...`)
   - `{pref}_{n}cells_percell_evaluation_2025.csv` — セル別評価
   - `{pref}_{n}cells_global_pkrk_2025.csv` — グローバル P@K / R@K

### wide形式スコアCSVへの変換(未記載だった手順)

`data/scores/*_ttm_scores_2025.csv` は wide形式(`Date` + セル列)なので、predictions CSV を pivot する:

```python
import pandas as pd
df = pd.read_csv('yamagata_144cells_predictions_2025.csv', parse_dates=['timestamp'])
wide = df.pivot(index='timestamp', columns='GridID', values='pred')
wide.index.name = 'Date'
wide.reset_index().to_csv('data/scores/yamagata_ttm_scores_2025.csv', index=False)
```

列順・列名が `data/yamagata_10km_daily_timeseries.csv` のセル列(`0_0`〜`8_15`)と一致していることを確認する。`.npy` 版が必要なら `np.save(..., wide.values.astype(np.float32))`。

## 3. GLM-Logit / HierBayes スコア

ベンチマークノートブック(`notebooks/kumawatch_benchmark.ipynb`、Table 3 用は `notebooks/kumawatch_benchmark_table3_colab.ipynb`)の実行中に計算され、`OUTPUT_DIR/scores/` 配下に npy 保存される。実行手順・Cell 2 設定・Windows での `PYTENSOR_FLAGS` 対策は **kumawatch-benchmark** Skill を参照。

生成物を `data/scores/` の対応ファイル名(`yamagata_glm_logit_scores_2025.npy` / `yamagata_hier_mean_scores_2025.npy` / `yamagata_hier_std_scores_2025.npy`)で配置する。

⚠️ 事後平均と事後標準偏差は**必ず同一 trace から同時に**出力すること。現在公開中の std は
過去の別実行のもので、公開中の事後平均とは対にならない。片方だけ差し替えると、
両者を組み合わせた信頼区間・確信度がすべて無効になる。

## 再生成後の検証

1. 形状確認: 山形 = (365, 144)、秋田 = (365, 260)。CSV は `Date` 列+セル列(`0_0` 形式)で365行
2. **kumawatch-benchmark** Skill の手順でベンチマークを回し、Recall@20 が同Skill記載の論文参照値から大きく外れないことを確認
