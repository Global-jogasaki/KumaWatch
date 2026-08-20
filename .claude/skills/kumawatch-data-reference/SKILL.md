---
name: kumawatch-data-reference
description: KumaWatchのデータファイル(時系列CSV・グリッド座標CSV・スコアファイル)の正確な形式・グリッド定義・学習/評価期間のリファレンス。データの列構成、ファイル形式、期間設定を確認するとき、またはスクリプトにデータパスを設定するときに使う。
---

# KumaWatch データリファレンス

KumaWatch の全データファイルの正確な仕様。他の Skill(benchmark / regen-scores / webmap / calibration)から参照される唯一の情報源。

## グリッド定義

| 県 | 列×行 | セル数 | 原点 (lat_min, lon_min) | ステップ |
|----|------|-------|------------------------|---------|
| 山形 | 9×16 | **144** | 37.758430, 139.549091 | lat 0.090090° / lon 0.114326° (≈10km) |
| 秋田 | 13×20 | **260** | 38.839510, 139.663417 | 同上 |

- セルIDは **`{col}_{row}`** 形式(列が先)。例: `0_0`, `4_9`, `8_15`
- 解像度10kmの根拠: データ密度(1kmセルの99%超が年間出没ゼロ)・パトロール可動域・行政界との整合

## 学習/評価期間

| 県 | 学習期間 | 評価期間 |
|----|---------|---------|
| 山形 | 2018-10-01 〜 2024-12-31 (~2,284日) | 2025-01-01 〜 2025-12-31 (365日) |
| 秋田 | 2022-04-01 〜 2024-12-31 | 2025-01-01 〜 2025-12-31 (365日) |

2025年の365日全てがホールドアウトテスト。学習・特徴量計算にテスト期間のデータは一切使わない(厳密な時間分離)。

※ CSVファイル自体は山形 2018-04以降、秋田 2020-04以降のデータを含むが、学習開始日は上表の通り。

## データファイルと形式

### 日次時系列 (`data/`)

| ファイル | 県 | 形状 |
|---------|----|------|
| `data/yamagata_10km_daily_timeseries.csv` | 山形 | 日×144セル |
| `data/akita_10km_daily_timeseries.csv` | 秋田 | 日×260セル |

**wide形式**。列構成:

```
Date, Year, Month, Week, Weekday, 0_0, 1_0, ..., 8_15, Sum
```

- `Date` は `2018/4/11` のようなスラッシュ区切り(パース時は `pd.to_datetime(..., format='mixed')` が安全)
- メタ列は `{Date, Year, Month, Week, Weekday, Sum}`、残りが `{col}_{row}` セル列(値=その日のそのセルの出没件数)
- **注意**: long形式(date, grid_id, sightings)ではない。スクレイピングや変換は不要で、scripts/ と notebooks/ はこのwide形式をそのまま読む

### グリッド座標 (`data/`)

| ファイル | 県 |
|---------|----|
| `data/yamagata_10km_grid_coords.csv` | 山形 |
| `data/akita_10km_grid_coords.csv` | 秋田 |

列構成:

```
Grid_ID, Grid_Row, Grid_Col, Center_Latitude, Center_Longitude,
Min_Latitude, Max_Latitude, Min_Longitude, Max_Longitude
```

`Grid_ID` は時系列CSVのセル列名と一致する。スクリプト内で参照される `Yamagata_10km_Grid_0.csv`(作者ローカルファイル)はこのファイルと同形式なので、`GRID_CSV` 変数はこのリポジトリ内ファイルに差し替えられる。

### 事前計算スコア (`data/scores/`)

CSV は wide形式(`Date` + セル列、365行 = 2025-01-01〜2025-12-31)。NPY は `(日数, セル数)` の配列。

| ファイル | モデル | 県 | 形状 |
|---------|-------|----|------|
| `data/scores/yamagata_glm_logit_scores_2025.npy` | GLM-Logit | 山形 | (365,144) float32 |
| `data/scores/yamagata_hier_mean_scores_2025.npy` | HierBayes 事後平均 | 山形 | (365,144) float64 |
| `data/scores/yamagata_hier_std_scores_2025.npy` | HierBayes 事後標準偏差 ⚠️ **事後平均とは別実行。組み合わせて使わないこと** | 山形 | (365,144) float64 |
| `data/scores/yamagata_et_scores_2025.csv` | Extra Trees | 山形 | 365×144 |
| `data/scores/yamagata_ttm_scores_2025.csv` | IBM Granite TTM | 山形 | 365×144 |
| `data/scores/yamagata_ttm_scores.npy` | IBM Granite TTM | 山形 | (365,144) float32 |
| `data/scores/akita_et_scores_2025.csv` | Extra Trees | 秋田 | 365×260 |
| `data/scores/akita_ttm_scores_2025.csv` | IBM Granite TTM | 秋田 | 365×260 |
| `data/scores/akita_ttm_scores.npy` | IBM Granite TTM | 秋田 | (365,260) float32 |

## スクリプト内の作者ローカルパスとの対応

scripts/ と notebooks/ には作者環境のパスがハードコードされている。以下のように読み替える:

| スクリプト内のパス/ファイル名 | リポジトリ内の対応ファイル |
|------------------------------|--------------------------|
| `Yamagata_10km_AllGrid_144cells_Daily_TimeSeries.csv` | `data/yamagata_10km_daily_timeseries.csv` |
| `Akita_10km_AllGrid_260cells_Daily_TimeSeries.csv` | `data/akita_10km_daily_timeseries.csv` |
| `Yamagata_10km_Grid_0.csv` | `data/yamagata_10km_grid_coords.csv` |
| `yamagata_ttm_scores_2025.csv` (Drive) | `data/scores/yamagata_ttm_scores_2025.csv` |
| `*_et_scores_2025.csv` (Drive) | `data/scores/*_et_scores_2025.csv` |
| `bear-sighting-data/data/{yamagata,akita}/`(生データ) | **リポジトリに含まれない**(kumawatch-regen-scores Skill 参照) |

## ライセンス

- コード: Apache 2.0 / データ: CC BY 4.0
- 出典: 山形県・秋田県の公開野生動物出没データベース
