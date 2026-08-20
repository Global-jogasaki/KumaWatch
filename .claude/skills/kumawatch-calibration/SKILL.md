---
name: kumawatch-calibration
description: KumaWatchの較正検証(Extra Trees スコアへの Platt / Isotonic 事後較正と GLM-Logit との比較)を scripts/calibration_validation.py で実行する手順。Brier・ECE・MAE・Recall@K の較正評価を行うときに使う。
---

# KumaWatch 較正検証

`scripts/calibration_validation.py` で Extra Trees スコアに事後較正(Platt scaling / Isotonic regression)を適用し、較正指標と運用指標を GLM-Logit 主層と比較する。GLM-Logit スコアはスクリプト内で出没データから計算されるため、入力は出没CSVとETスコアCSVの2つだけ。

## 依存

```bash
pip install numpy pandas scikit-learn
```

## 実行方法

デフォルトパスは Google Colab / Drive 前提(`/content/drive/MyDrive/bear/...`)だが、**ファイルを編集しなくても CLI 引数でパスを渡せる**(READMEには `--prefecture` しか書かれていないが、パス引数がある)。リポジトリルートから:

```bash
# 山形のみ
python scripts/calibration_validation.py --prefecture yamagata \
  --sightings_yamagata data/yamagata_10km_daily_timeseries.csv \
  --et_scores_yamagata data/scores/yamagata_et_scores_2025.csv

# 秋田のみ
python scripts/calibration_validation.py --prefecture akita \
  --sightings_akita data/akita_10km_daily_timeseries.csv \
  --et_scores_akita data/scores/akita_et_scores_2025.csv

# 両県まとめて(デフォルト。--prefecture all と同じ)
python scripts/calibration_validation.py \
  --sightings_yamagata data/yamagata_10km_daily_timeseries.csv \
  --et_scores_yamagata data/scores/yamagata_et_scores_2025.csv \
  --sightings_akita data/akita_10km_daily_timeseries.csv \
  --et_scores_akita data/scores/akita_et_scores_2025.csv
```

| 引数 | 既定値 | 意味 |
|------|--------|------|
| `--prefecture` | `all` | `yamagata` / `akita` / `all` |
| `--sightings_yamagata` / `--sightings_akita` | Drive パス | 出没時系列CSV(wide形式) |
| `--et_scores_yamagata` / `--et_scores_akita` | Drive パス | ET スコアCSV(wide形式) |

学習/評価期間はスクリプト内 `DEFAULT_PATHS` に固定(山形 2018-10-01〜 / 秋田 2022-04-01〜、テスト2025年)。期間を変える場合のみファイル編集が必要。

## 出力の見方

標準出力に県ごとの指標テーブルが出る:

- **較正指標**(出没≥1のセルで平均): Brier(小さいほど良い)、ECE(10ビン)、MAE
- **運用指標**: Recall@K / Precision@K(K = 10, 20, 30)

判断基準:

- 事後較正(Platt / Isotonic)で ET の Brier が改善しても、**Recall@K のランキング性能は較正では変わらない**(単調変換のため)。ET の較正が GLM-Logit に届かないことが、GLM-Logit を主層とする論文の根拠のひとつ
- 参考値(論文 Calibration Metrics): GLM-Logit Brier = 0.034(山形)/ 0.041(秋田)、無較正 ET = 0.097 / 0.126

## 関連

- スコアファイルの再生成 → **kumawatch-regen-scores** Skill
- ベンチマーク全体(BSS・並べ替え検定を含む)→ **kumawatch-benchmark** Skill
