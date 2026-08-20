"""
table2_significance.py — KumaWatch Pairwise Significance (Table 2)

Reproduces Table 2 of the paper — the effect size, 95% confidence interval and
permutation p-value for each pairwise comparison at K = 20 — directly from the
score files in `data/scores/`. No model is retrained and no MCMC is rerun, so
the statistical claims can be checked from the released artifact alone.

Test procedure (identical to `notebooks/kumawatch_benchmark.ipynb` Cell 17):
  * per-day Recall@K is computed on days with at least one sighting
  * effect size Δ = mean over those days of (Recall_A − Recall_B)
  * 95% CI  — day-level paired bootstrap, B = 5,000, percentile interval
  * p-value — day-level paired permutation (sign-flip), P = 5,000, two-sided,
              reported as (#{|perm| ≥ |obs|} + 1) / (P + 1)
  * significance at the Bonferroni-corrected α = 0.05 / 13 = 0.0038

Both resamplers are seeded with RAND_SEED = 42, so results are deterministic.

Usage:
    python scripts/table2_significance.py              # the six published rows
    python scripts/table2_significance.py --all        # every available pair
    python scripts/table2_significance.py --markdown
    python scripts/table2_significance.py --k 10 30    # other budgets

Dependencies:
    pip install numpy pandas
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "data" / "scores"

META_COLS = {"Date", "Year", "Month", "Week", "Weekday", "Sum"}
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

RAND_SEED = 42
B_BOOT = P_PERM = 5000
ALPHA_BONFERRONI = 0.05 / 13

PREFECTURES = {
    "yamagata": {
        "sightings": REPO / "data" / "yamagata_10km_daily_timeseries.csv",
        "train_start": "2018-10-01", "train_end": "2024-12-31",
        "scores": {
            "GLM-Logit": SCORES / "yamagata_glm_logit_scores_2025.npy",
            "HierBayes": SCORES / "yamagata_hier_mean_scores_2025.npy",
            "TTM":       SCORES / "yamagata_ttm_scores_2025.csv",
            "ET":        SCORES / "yamagata_et_scores_2025.csv",
        },
    },
    "akita": {
        "sightings": REPO / "data" / "akita_10km_daily_timeseries.csv",
        "train_start": "2022-04-01", "train_end": "2024-12-31",
        "scores": {
            "GLM-Logit": SCORES / "akita_glm_logit_scores_2025.npy",
            "HierBayes": SCORES / "akita_hier_mean_scores_2025.npy",
            "TTM":       SCORES / "akita_ttm_scores_2025.csv",
            "ET":        SCORES / "akita_et_scores_2025.csv",
        },
    },
}

# The six rows printed in Table 2, with the published values for comparison.
PUBLISHED = [
    ("akita",    "GLM-Logit", "ET",        +0.128, (+0.101, +0.157), 0.0002),
    ("yamagata", "TTM",       "B1",        -0.041, (-0.066, -0.019), 0.0004),
    ("yamagata", "GLM-Logit", "B1",        +0.014, (-0.006, +0.032), 0.155),
    ("akita",    "GLM-Logit", "B1",        +0.050, (+0.028, +0.072), 0.0002),
    ("yamagata", "HierBayes", "GLM-Logit", -0.005, (-0.023, +0.013), 0.624),
    ("akita",    "HierBayes", "GLM-Logit", -0.023, (-0.039, -0.007), 0.003),
]


def load_prefecture(pref):
    cfg = PREFECTURES[pref]
    df = pd.read_csv(cfg["sightings"])
    df["_dt"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("_dt").sort_index()
    cells = [c for c in df.columns if c not in META_COLS]
    tr = (df.index >= cfg["train_start"]) & (df.index <= cfg["train_end"])
    te = (df.index >= TEST_START) & (df.index <= TEST_END)
    train_L = df.loc[tr, cells].values.astype(np.float32)
    test_L = df.loc[te, cells].values.astype(np.float32)
    test_dates = df.index[te]

    scores = {"B1": np.tile(train_L.mean(axis=0),
                            (len(test_dates), 1)).astype(np.float32)}
    for name, path in cfg["scores"].items():
        path = Path(path)
        if not path.exists():
            continue
        if path.suffix == ".npy":
            scores[name] = np.load(path).astype(np.float32)
        else:
            # CSV score files are aligned by column name: the ET files order
            # their cell columns differently from the sightings CSV.
            d = pd.read_csv(path)
            d["_dt"] = pd.to_datetime(d["Date"], format="mixed")
            d = d.set_index("_dt").sort_index()
            scores[name] = d.loc[test_dates, cells].values.astype(np.float32)
    return scores, test_L


def per_day_recall(scores, labels, k):
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = np.take_along_axis(labels.astype(np.int32), topk, axis=1).sum(axis=1)
    npos = labels.sum(axis=1).astype(np.float64)
    valid = npos > 0
    recall = np.where(valid, hits / np.where(valid, npos, 1.0), 0.0)
    return recall.astype(np.float64), valid


def compare(scores_a, scores_b, labels, k, seed=RAND_SEED):
    ra, valid = per_day_recall(scores_a, labels, k)
    rb, _ = per_day_recall(scores_b, labels, k)
    diff = (ra - rb)[valid]
    n_valid = int(valid.sum())
    obs = float(diff.mean())

    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, n_valid, size=(B_BOOT, n_valid))].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(P_PERM, n_valid))
    perm = (signs * diff).mean(axis=1)
    p_perm = float((np.abs(perm) >= abs(obs)).sum() + 1) / (P_PERM + 1)

    return obs, (float(ci_lo), float(ci_hi)), p_perm, n_valid


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce Table 2 from the released score files.")
    ap.add_argument("--k", type=int, nargs="+", default=[20])
    ap.add_argument("--all", action="store_true",
                    help="report every available method pair, not just Table 2")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    cache = {}

    def get(pref):
        if pref not in cache:
            cache[pref] = load_prefecture(pref)
        return cache[pref]

    for k in args.k:
        print(f"Pairwise comparisons at K = {k} — bootstrap B = {B_BOOT}, "
              f"permutation P = {P_PERM}, seed = {RAND_SEED}, "
              f"Bonferroni alpha = {ALPHA_BONFERRONI:.4f}\n")

        if args.all:
            rows = []
            for pref in PREFECTURES:
                scores, _ = get(pref)
                for a, b in itertools.combinations(sorted(scores), 2):
                    rows.append((pref, a, b, None, None, None))
        else:
            rows = PUBLISHED

        if args.markdown:
            print("| Comparison | Δ | 95% CI | p | Significant |")
            print("|------------|--:|:------:|--:|:-----------:|")

        for pref, a, b, pub_d, pub_ci, pub_p in rows:
            scores, labels = get(pref)
            if a not in scores or b not in scores:
                print(f"  {a} vs {b} ({pref}): score file unavailable — skipped")
                continue
            obs, ci, p_perm, n = compare(scores[a], scores[b], labels, k)
            sig = "Yes" if p_perm < ALPHA_BONFERRONI else "No"
            tag = "AKT" if pref == "akita" else "YGT"
            label = f"{a} vs {b} ({tag})"

            if args.markdown:
                print(f"| {label} | {obs:+.3f} | "
                      f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {p_perm:.4f} | {sig} |")
                continue

            line = (f"  {label:32s} Δ = {obs:+.4f}  "
                    f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                    f"p = {p_perm:.4f}  {sig:>3s}  (n = {n})")
            if pub_d is not None:
                agree = (abs(obs - pub_d) <= 0.0015
                         and abs(ci[0] - pub_ci[0]) <= 0.0015
                         and abs(ci[1] - pub_ci[1]) <= 0.0015)
                line += f"   published Δ = {pub_d:+.3f} p = {pub_p}"
                line += "  [OK]" if agree else "  [DIFFERS]"
            print(line)
        print()


if __name__ == "__main__":
    main()
