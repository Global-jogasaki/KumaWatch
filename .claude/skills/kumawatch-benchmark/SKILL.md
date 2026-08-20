---
name: kumawatch-benchmark
description: KumaWatchの11手法ベンチマーク(GLM-Logit / HierBayes / TTM / Extra Trees / ベースラインB0-B5)を notebooks/kumawatch_benchmark.ipynb で実行する手順。ベンチマークの再現、Recall@K・較正指標・並べ替え検定の再計算、Table 3(信頼度フィルタ分析)の実行時に使う。
---

# KumaWatch ベンチマーク実行

`notebooks/kumawatch_benchmark.ipynb` で論文の11手法(ベースライン B0〜B5、Poisson-GLM、GLM-Logit、HierBayes、Extra Trees、TTM)を同一条件で評価する。ノートブックには参考実装として Poisson-GLM-cs(セル別季節性付き)も含まれる。

データ形式・期間の詳細は **kumawatch-data-reference** Skill を参照。

## 前提条件

1. **依存ライブラリ**(Cell 1 が未導入分を自動 pip install するが、手動なら):

   ```bash
   pip install scikit-learn pandas numpy scipy pymc numpyro statsmodels arviz "jax[cpu]"
   ```

2. **事前計算スコアファイル**(リポジトリに同梱済み):
   - `data/scores/yamagata_et_scores_2025.csv` / `data/scores/yamagata_ttm_scores_2025.csv`
   - `data/scores/akita_et_scores_2025.csv` / `data/scores/akita_ttm_scores_2025.csv`

   再生成したい場合は **kumawatch-regen-scores** Skill を参照。

## 手順

### 1. Cell 2(★ USER EDIT HERE)のパス設定

デフォルトは作者の Google Drive パス(`/content/drive/MyDrive/bear/...`)なので書き換える。リポジトリルートから実行する場合:

```python
SCORE_FORMAT = 'CSV'

YAMA_SIGHTINGS_CSV  = 'data/yamagata_10km_daily_timeseries.csv'
YAMA_TTM_SCORES_CSV = 'data/scores/yamagata_ttm_scores_2025.csv'
YAMA_ET_SCORES_CSV  = 'data/scores/yamagata_et_scores_2025.csv'

AKITA_SIGHTINGS_CSV  = 'data/akita_10km_daily_timeseries.csv'
AKITA_TTM_SCORES_CSV = 'data/scores/akita_ttm_scores_2025.csv'
AKITA_ET_SCORES_CSV  = 'data/scores/akita_et_scores_2025.csv'

OUTPUT_DIR = 'benchmark_output'          # デフォルトは Drive パスなので変更必須
CACHE_PATH = 'benchmark_output/glm_hier_cache.pkl'
```

- `*_NPY` 変数は `SCORE_FORMAT='NPY'` のときのみ使用(`data/scores/*_ttm_scores.npy` が使える)
- 学習/評価期間(`YAMA_TRAIN_START='2018-10-01'`、`AKITA_TRAIN_START='2022-04-01'` など)はデフォルトのままでよい
- Colab で実行する場合はスコアファイルを Drive にアップロードし、Drive パスを指定(Cell 3 が自動でマウント。ローカルではマウント失敗を自動スキップするので無視してよい)

### 2. 主要な設定値(必要なら調整)

| 変数 | デフォルト | 意味 |
|------|-----------|------|
| `GLM_C` | 1.0 | GLM-Logit の L2 正則化 (C=1/λ) |
| `HIER_TRAIN_YEARS` | 3 | HierBayes に使う直近学習年数(0=全期間、数時間かかる) |
| `MCMC_DRAWS / MCMC_TUNE / MCMC_CHAINS` | 1000 / 500 / 2 | NUTS 設定(論文設定) |
| `USE_MCMC` | True | False で ADVI 近似(高速・精度低下) |
| `B_BOOT / B_PERM` | 5000 / 5000 | ブートストラップ/並べ替え検定の反復数 |
| `RAND_SEED` | 42 | 再現用シード |

### 3. 全セル実行

- 実行時間: **Colab CPU で約60〜120分**(MCMC がボトルネック)
- **Windows で HierBayes を動かす場合**は C++ コンパイルを無効化する:

  ```
  PYTENSOR_FLAGS=device=cpu,floatX=float64,optimizer=fast_compile,cxx=
  ```

  (環境変数として設定してから Jupyter を起動する)

### 4. 結果のサニティチェック

以下の論文値と照合する(ノートブック内にも `PAPER_*_R20` 参照値によるチェックがあり、許容誤差は TTM が `SANITY_TOL=0.005`、ET は緩め `ET_SANITY_TOL=0.15`):

| 指標 | 山形 | 秋田 |
|------|------|------|
| GLM-Logit Recall@20 | **0.547** | **0.454** |
| HierBayes Recall@20(全日) | 0.542 | 0.431 |
| TTM Recall@20 | 0.492 | 0.395 |
| Extra Trees Recall@20 | 0.474 | 0.326 |

HierBayes の収束確認: R̂ < 1.01、divergent transitions = 0。

## Table 3(信頼度フィルタ分析)

論文 Table 3(top-25%/50% 信頼度サブセットの Recall@20 と並べ替え検定)は別ノートブック **`notebooks/kumawatch_benchmark_table3_colab.ipynb`** を使う。事前計算済みの
`data/scores/yamagata_glm_logit_scores_2025.npy` / `yamagata_hier_mean_scores_2025.npy` / `yamagata_hier_std_scores_2025.npy` を入力にするため MCMC の再実行は不要。

期待値(山形): HierBayes top-50% Recall@20 = **0.639**(全日 0.542 から向上)、GLM-Logit top-25% = 0.889。
