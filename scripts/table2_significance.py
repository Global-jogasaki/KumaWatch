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
RESULTS = REPO / "results"

META_COLS = {"Date", "Year", "Month", "Week", "Weekday", "Sum"}
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

RAND_SEED = 42
B_BOOT = P_PERM = 5000
INCLUDE_ALL_METHODS = False   # set by --all
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
    if INCLUDE_ALL_METHODS:
        # B0-B5 and Poisson-GLM are regenerated deterministically, so --all can
        # cover all eleven benchmarked methods rather than only the released four
        # plus the static prior. Shared with all_vs_static_prior.py so the two
        # scripts cannot drift apart.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import all_vs_static_prior as avs
        scores, test_L = avs.load(pref)
        return scores, test_L
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
    ap.add_argument("--no-save", action="store_true",
                    help="with --all, print only; do not write results/")
    args = ap.parse_args()

    global INCLUDE_ALL_METHODS
    INCLUDE_ALL_METHODS = args.all

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
            # This family is every unordered pair per prefecture, not the paper's
            # thirteen, so it gets its own Bonferroni threshold.
            n_pairs = len(rows) // len(PREFECTURES)
            alpha = 0.05 / n_pairs
            print(f"  --all: {n_pairs} pairs per prefecture over "
                  f"{len(get('yamagata')[0])} methods; "
                  f"Bonferroni alpha = {alpha:.5f}\n")
        else:
            rows = PUBLISHED
            alpha = ALPHA_BONFERRONI

        collected = []
        if args.markdown:
            print("| Comparison | Δ | 95% CI | p | Significant |")
            print("|------------|--:|:------:|--:|:-----------:|")

        for pref, a, b, pub_d, pub_ci, pub_p in rows:
            scores, labels = get(pref)
            if a not in scores or b not in scores:
                print(f"  {a} vs {b} ({pref}): score file unavailable — skipped")
                continue
            obs, ci, p_perm, n = compare(scores[a], scores[b], labels, k)
            sig = "Yes" if p_perm < alpha else "No"
            collected.append(dict(prefecture=pref, method_a=a, method_b=b,
                                  delta=obs, ci_low=ci[0], ci_high=ci[1],
                                  p_perm=p_perm,
                                  significant_family=sig.lower(),
                                  significant_paper_alpha=(
                                      "yes" if p_perm < ALPHA_BONFERRONI else "no"),
                                  n_days=n, k=k))
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
                # Report the largest deviation rather than only a verdict: a
                # tolerance wide enough to pass also hides a 0.001 disagreement,
                # which is exactly the kind of thing worth seeing.
                dev = max(abs(obs - pub_d),
                          abs(ci[0] - pub_ci[0]), abs(ci[1] - pub_ci[1]))
                line += f"   published Δ = {pub_d:+.3f} p = {pub_p}"
                line += f"  max|dev| = {dev:.4f}"
                line += "  [OK]" if dev <= 0.0015 else "  [DIFFERS]"
            print(line)

        if args.all and not args.no_save and collected:
            save_all(collected, k, alpha)
        print()


def save_all(rows, k, alpha):
    """Write every pairwise test to results/, as CSV and as Markdown."""
    RESULTS.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    for pref, sub in df.groupby("prefecture"):
        stem = RESULTS / f"all_pairwise_tests_{pref}_2025"
        sub = sub.sort_values("p_perm")
        sub.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.6f")

        tag = "Yamagata" if pref == "yamagata" else "Akita"
        n_days = int(sub.n_days.max())
        md = [f"# Every pairwise comparison — {tag}, 2025", "",
              f"All {len(sub)} unordered pairs among the {len(PREFECTURES) and ''}"
              f"eleven benchmarked methods, at K = {k}, on the {n_days} days with "
              "at least one reported sighting.", "",
              f"Δ is the mean per-day difference in Recall@{k} (first method minus "
              f"second); the interval is a day-level paired bootstrap "
              f"(B = {B_BOOT:,}) and *p* a day-level sign-flip permutation test "
              f"(P = {P_PERM:,}), both seeded with {RAND_SEED} — the same procedure "
              "as Table 2 of the paper.", "",
              f"Two Bonferroni thresholds are reported, because the answer "
              f"depends on which family the comparison is read as part of. "
              f"**This family** is the {len(sub)} pairs in this table, "
              f"α = {alpha:.5f}. **Table 2** of the paper corrects over its own "
              "family of thirteen, α = 0.0038, and that is the column to use "
              "when checking a claim made in the paper. Rows are sorted by *p*.",
              "",
              f"The two disagree on {int((sub.significant_family != sub.significant_paper_alpha).sum())} "
              f"of the {len(sub)} pairs — those with *p* between {alpha:.5f} and "
              "0.0038, which the stricter family-wide threshold rejects. "
              "Among them on Akita is HierBayes against GLM-Logit "
              "(*p* = 0.0026), the comparison Section 6 describes as "
              "significantly worse; it is significant at the paper's threshold "
              "and not at this table's.", "",
              "| Comparison | Δ | 95% CI | *p* | Sig. (this family, "
              f"α = {alpha:.5f}) | Sig. (paper, α = 0.0038) |",
              "|------------|--:|:------:|----:|:------------------:|"
              ":-----------------------:|"]
        for _, r in sub.iterrows():
            md.append(f"| {r.method_a} vs {r.method_b} | {r.delta:+.4f} | "
                      f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] | {r.p_perm:.4f} | "
                      f"{r.significant_family} | {r.significant_paper_alpha} |")
        md += ["", "Generated by `python scripts/table2_significance.py --all`."]
        stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
        print(f"  wrote {stem.with_suffix('.csv').relative_to(REPO)} and .md "
              f"({len(sub)} pairs)")


if __name__ == "__main__":
    main()
