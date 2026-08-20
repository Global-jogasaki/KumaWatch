"""
crosslayer_jaccard.py — KumaWatch Cross-Method Top-K Agreement (Jaccard@K)

Recomputes the pairwise Jaccard overlap of top-K cell sets between every pair of
released methods, directly from the score files in `data/scores/`. Every number
this script prints is reproducible from files committed to this repository, so
the cross-method agreement figures quoted in the paper and README can be checked
without rerunning any model.

Definition (identical to `jaccard_at_k_daily` used in the robustness analysis):
for each evaluation day t, take the top-K cells by score under each method and
report |A(t) ∩ B(t)| / |A(t) ∪ B(t)|, then average over days. Ties are broken by
`np.argpartition`, matching the Recall@K implementation in the benchmark
notebook.

Two day subsets are reported because they are not interchangeable:
  * all days      — all 365 evaluation days
  * sighting days — days with at least one reported sighting (the subset the
                    paper's cross-layer table was computed on)

Usage:
    python scripts/crosslayer_jaccard.py
    python scripts/crosslayer_jaccard.py --prefecture yamagata --k 20
    python scripts/crosslayer_jaccard.py --markdown
    python scripts/crosslayer_jaccard.py --check          # verify Recall@K first

Dependencies:
    pip install numpy pandas
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

META_COLS = {"Date", "Year", "Month", "Week", "Weekday", "Sum"}

TEST_START, TEST_END = "2025-01-01", "2025-12-31"

PREFECTURES = {
    "yamagata": {
        "sightings": REPO / "data" / "yamagata_10km_daily_timeseries.csv",
        "scores": {
            "GLM-Logit": REPO / "data" / "scores" / "yamagata_glm_logit_scores_2025.npy",
            "HierBayes": REPO / "data" / "scores" / "yamagata_hier_mean_scores_2025.npy",
            "TTM":       REPO / "data" / "scores" / "yamagata_ttm_scores_2025.csv",
            "ET":        REPO / "data" / "scores" / "yamagata_et_scores_2025.csv",
        },
        # Recall@20 recomputed from these files; Table 1 reports the same
        # values to three decimals. Used by --check.
        "expected_r20": {"GLM-Logit": 0.5470, "HierBayes": 0.5425,
                         "TTM": 0.4917, "ET": 0.4739},
    },
    "akita": {
        "sightings": REPO / "data" / "akita_10km_daily_timeseries.csv",
        "scores": {
            "GLM-Logit": REPO / "data" / "scores" / "akita_glm_logit_scores_2025.npy",
            "HierBayes": REPO / "data" / "scores" / "akita_hier_mean_scores_2025.npy",
            "TTM":       REPO / "data" / "scores" / "akita_ttm_scores_2025.csv",
            "ET":        REPO / "data" / "scores" / "akita_et_scores_2025.csv",
        },
        "expected_r20": {"GLM-Logit": 0.4543, "HierBayes": 0.4316,
                         "TTM": 0.3950, "ET": 0.3258},
    },
}


def load_labels(sightings_csv):
    """Return (labels, cell_cols, test_dates) for the 2025 evaluation window."""
    df = pd.read_csv(sightings_csv)
    df["_dt"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("_dt").sort_index()
    cell_cols = [c for c in df.columns if c not in META_COLS]
    test_dates = df.index[(df.index >= TEST_START) & (df.index <= TEST_END)]
    labels = df.loc[test_dates, cell_cols].values.astype(np.float32)
    return labels, cell_cols, test_dates


def load_scores(path, cell_cols, test_dates):
    """Load a (n_days, n_cells) score matrix from .npy or wide-format .csv.

    CSV score files are aligned to the sightings grid **by column name**: the ET
    files order their cell columns differently from the sightings CSV, so
    positional alignment silently scrambles the grid.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.shape != (len(test_dates), len(cell_cols)):
            raise ValueError(f"{path.name}: expected "
                             f"{(len(test_dates), len(cell_cols))}, got {arr.shape}")
        return arr
    df = pd.read_csv(path)
    df["_dt"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("_dt").sort_index()
    missing = [c for c in cell_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing {len(missing)} cell columns "
                         f"(first: {missing[:3]})")
    return df.loc[test_dates, cell_cols].values.astype(np.float32)


def recall_at_k(scores, labels, k):
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = np.take_along_axis(labels.astype(np.int32), topk, axis=1).sum(axis=1)
    npos = labels.sum(axis=1)
    valid = npos > 0
    return float((hits[valid] / npos[valid]).mean())


def jaccard_at_k_daily(scores_a, scores_b, k):
    """Per-day Jaccard of the two top-K cell sets. Returns one value per day."""
    n_days, n_cells = scores_a.shape
    k_eff = min(k, n_cells)
    top_a = np.argpartition(-scores_a, k_eff, axis=1)[:, :k_eff]
    top_b = np.argpartition(-scores_b, k_eff, axis=1)[:, :k_eff]
    out = np.empty(n_days, dtype=np.float64)
    for t in range(n_days):
        sa, sb = set(top_a[t]), set(top_b[t])
        union = len(sa | sb)
        out[t] = len(sa & sb) / union if union > 0 else 0.0
    return out


def run(prefecture, k_values, as_markdown, check):
    cfg = PREFECTURES[prefecture]
    labels, cell_cols, test_dates = load_labels(cfg["sightings"])
    sighting_days = labels.sum(axis=1) > 0

    scores = {}
    for name, path in cfg["scores"].items():
        scores[name] = load_scores(path, cell_cols, test_dates)

    header = (f"{prefecture.capitalize()} — {len(cell_cols)} cells, "
              f"{len(test_dates)} evaluation days, "
              f"{int(sighting_days.sum())} with >=1 sighting")
    print(header)
    print("=" * len(header))

    if check:
        print("\nRecall@20 sanity check (paper Table 1):")
        ok = True
        for name, expected in cfg["expected_r20"].items():
            got = recall_at_k(scores[name], labels, 20)
            # 0.001 would let the superseded GLM-Logit file (0.5460 against
            # 0.5470) pass, so the check is tighter than the reporting precision.
            hit = abs(got - expected) <= 0.0005
            ok &= hit
            print(f"  [{'OK' if hit else 'NG'}] {name:10s} "
                  f"computed={got:.4f}  expected={expected:.4f}")
        print("  all methods match" if ok else
              "  WARNING: score files do not match the reported Recall@20")

    pairs = list(itertools.combinations(scores.keys(), 2))

    for k in k_values:
        print(f"\nJaccard@{k}")
        if as_markdown:
            print(f"\n| Method pair | Jaccard@{k} (all days) | "
                  f"Jaccard@{k} (sighting days) |")
            print("|-------------|:---------------------:|"
                  ":---------------------------:|")
        for a, b in pairs:
            v = jaccard_at_k_daily(scores[a], scores[b], k)
            all_days, sight = v.mean(), v[sighting_days].mean()
            if as_markdown:
                print(f"| {a} vs {b} | {all_days:.3f} | {sight:.3f} |")
            else:
                print(f"  {a + ' vs ' + b:24s} all days = {all_days:.3f}   "
                      f"sighting days = {sight:.3f}   (sd = {v.std():.3f})")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Recompute cross-method Jaccard@K from released score files.")
    ap.add_argument("--prefecture", choices=["yamagata", "akita", "all"],
                    default="all")
    ap.add_argument("--k", type=int, nargs="+", default=[10, 20, 30],
                    help="K values to report (default: 10 20 30)")
    ap.add_argument("--markdown", action="store_true",
                    help="emit Markdown tables instead of aligned text")
    ap.add_argument("--check", action="store_true",
                    help="verify Recall@20 against the paper before reporting")
    args = ap.parse_args()

    targets = (["yamagata", "akita"] if args.prefecture == "all"
               else [args.prefecture])
    for pref in targets:
        run(pref, args.k, args.markdown, args.check)


if __name__ == "__main__":
    main()
