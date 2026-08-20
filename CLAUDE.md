# KumaWatch 開発ガイド(Claude Code 用)

クマ出没リスク予測システム(ACM SIGSPATIAL 2026)。GLM-Logit(主層)+ HierBayes(不確実性層)+ TTM・Extra Trees(補完層)の三層構成。詳細は `README.md`。

## Skill 一覧(作業の入口)

作業手順は `.claude/skills/` の Skill に集約されている。READMEより先にこちらを参照すること:

| Skill | 使うとき |
|-------|---------|
| `kumawatch-data-reference` | データ形式・グリッド定義・期間を確認する / スクリプトにパスを設定する |
| `kumawatch-benchmark` | 11手法ベンチマーク・Table 3 を実行/再現する |
| `kumawatch-regen-scores` | `data/scores/` のスコアファイルを再生成する |
| `kumawatch-webmap` | Webマップを生成し GitHub Pages に公開する |
| `kumawatch-calibration` | 較正検証(Platt/Isotonic vs GLM-Logit)を実行する |

## 全体で共通の注意

- `scripts/` と `notebooks/` には**作者ローカルのパスがハードコード**されている(`F:\SSD-PGU3\...`、`/content/drive/MyDrive/...`)。実行前に必ず kumawatch-data-reference Skill のパス対応表に従って書き換える
- ET スコアの再生成には**リポジトリ外の生データと外部共変量**が必要(kumawatch-regen-scores Skill 参照)。同梱の `data/scores/` を使えば再生成せずにベンチマーク・マップ生成が可能
- データCSVは **wide形式**(`Date` + `{col}_{row}` セル列)。long形式ではない
- 2025年365日は厳密なホールドアウト。学習・特徴量にテスト期間のデータを混ぜないこと
- 論文値の再現確認には kumawatch-benchmark Skill のサニティチェック表を使う
