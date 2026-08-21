"""
daily_diagnostics.py — KumaWatch per-day diagnostics and supplementary metrics

Emits, from the released score files in `data/scores/` and nothing else:

  1. `results/daily_diagnostics_<pref>_2025.csv` — one row per evaluation day per
     method, with the positive-cell count, Recall@20, Precision@20 and the
     true-positive / false-positive / false-negative cell ids for that day, plus
     the pairwise Jaccard@20 against every other method.
  2. Precision@K and Recall@K tables, reported separately for the two day sets
     that are **not** interchangeable (see below).
  3. Three ROC-AUC variants, which answer three different questions and must not
     be collapsed into one number.
  4. Case-study days chosen automatically by the deterministic criteria below,
     rather than picked by hand. The set includes both low-performing and
     relatively favourable cases.

Coverage: ten of the paper's eleven methods. The four learned methods come from
released score matrices; B0-B5 are regenerated deterministically from Cell 5 of
the benchmark notebook with RAND_SEED = 42. Poisson-GLM is the exception - no
score matrix for it is released.

Metric conventions, stated because they change the numbers
----------------------------------------------------------
* Global top-K: for each day the K cells with the highest score across the whole
  prefectural grid are selected. Grids are 144 cells (Yamagata) and 260 (Akita);
  K ∈ {10, 20, 30}. The evaluation window is the held-out year 2025, 365 days.
* Ties are resolved by `numpy.argpartition`, matching the benchmark notebook.
  Tied cells have no semantic ordering at the K-th boundary; the exact selection
  depends on the pinned versions in `requirements-diagnostics.txt`.
* **Sighting days** — the mean is taken over days with at least one reported
  sighting (Yamagata 213 of 365, Akita 323 of 365). Recall@K is only defined on
  these days, so this is the set the paper reports.
* **All days** — the mean is taken over all 365 days; a day with no sighting
  contributes a precision of 0. This is the number a patrol that goes out every
  day experiences, and it is always lower. The two are reported side by side and
  never mixed.

ROC-AUC is a supplementary threshold-free diagnostic. It does not replace
Recall@K or Precision@K, which correspond directly to the fixed daily patrol
budget. A method can lead on one and not the other; the paper's point is that the
metric has to be chosen from the decision.

Usage:
    python scripts/daily_diagnostics.py
    python scripts/daily_diagnostics.py --prefecture yamagata --no-csv

Dependencies:
    pip install -r requirements-diagnostics.txt
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "data" / "scores"
RESULTS = REPO / "results"
RAND_SEED = 42

META_COLS = {"Date", "Year", "Month", "Week", "Weekday", "Sum"}
TEST_START, TEST_END = "2025-01-01", "2025-12-31"
K_VALUES = (10, 20, 30)

PREFECTURES = {
    "yamagata": {
        "sightings": REPO / "data" / "yamagata_10km_daily_timeseries.csv",
        "train": ("2018-10-01", "2024-12-31"),
        "scores": {
            "GLM-Logit": SCORES / "yamagata_glm_logit_scores_2025.npy",
            "HierBayes": SCORES / "yamagata_hier_mean_scores_2025.npy",
            "TTM":       SCORES / "yamagata_ttm_scores_2025.csv",
            "ET":        SCORES / "yamagata_et_scores_2025.csv",
        },
    },
    "akita": {
        "sightings": REPO / "data" / "akita_10km_daily_timeseries.csv",
        "train": ("2022-04-01", "2024-12-31"),
        "scores": {
            "GLM-Logit": SCORES / "akita_glm_logit_scores_2025.npy",
            "HierBayes": SCORES / "akita_hier_mean_scores_2025.npy",
            "TTM":       SCORES / "akita_ttm_scores_2025.csv",
            "ET":        SCORES / "akita_et_scores_2025.csv",
        },
    },
}


def build_baselines(train_L, test_L, train_dates, test_dates):
    """B0-B5 exactly as notebooks/kumawatch_benchmark.ipynb Cell 5 (RAND_SEED=42).

    These are regenerated deterministically rather than read from a file, so the
    diagnostics can cover ten of the paper's eleven methods. Poisson-GLM is the
    exception: no score matrix for it is released.
    """
    n_test, n_cells = test_L.shape

    def daynorm(s):
        mu = s.mean(axis=1, keepdims=True)
        sd = s.std(axis=1, keepdims=True)
        return (s - mu) / np.where(sd == 0, 1.0, sd)

    b1 = np.tile(train_L.mean(axis=0), (n_test, 1)).astype(np.float32)

    all_L = np.concatenate([train_L, test_L], axis=0).astype(np.float64)
    cs = np.zeros((len(all_L) + 1, n_cells)); np.cumsum(all_L, axis=0, out=cs[1:])
    b2 = np.empty((n_test, n_cells), np.float32)
    for d in range(n_test):
        pos = len(train_L) + d
        start = max(0, pos - 30)
        b2[d] = ((cs[pos] - cs[start]) / (pos - start)).astype(np.float32)

    tdoy = np.array([d.timetuple().tm_yday for d in train_dates], np.int32)
    gm = train_L.mean(axis=0).astype(np.float32)
    b3 = np.empty((n_test, n_cells), np.float32)
    for i, d in enumerate(test_dates):
        doy = d.timetuple().tm_yday
        diff = np.minimum(np.abs(tdoy - doy), 365 - np.abs(tdoy - doy))
        m = diff <= 7
        b3[i] = train_L[m].mean(axis=0).astype(np.float32) if m.sum() else gm

    rng = np.random.default_rng(RAND_SEED)
    return {
        "GLM-Logit": None, "HierBayes": None, "TTM": None, "ET": None,
        "B0: Random":       rng.random((n_test, n_cells)).astype(np.float32),
        "B1: Static prior": b1,
        "B2: Recent MA":    b2,
        "B3: DoY season":   b3,
        "B4: B1+B3":        daynorm(b1 * 0.5 + b3 * 0.5).astype(np.float32),
        "B5: B2+B3":        daynorm(b2 * 0.5 + b3 * 0.5).astype(np.float32),
    }


def load(pref):
    cfg = PREFECTURES[pref]
    df = pd.read_csv(cfg["sightings"])
    df["_dt"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.set_index("_dt").sort_index()
    cells = [c for c in df.columns if c not in META_COLS]
    tr = (df.index >= cfg["train"][0]) & (df.index <= cfg["train"][1])
    te = (df.index >= TEST_START) & (df.index <= TEST_END)
    train_L = df.loc[tr, cells].values.astype(np.float32)
    labels = df.loc[te, cells].values.astype(np.float32)
    dates = df.index[te]

    scores = build_baselines(train_L, labels, df.index[tr], dates)
    for name, path in cfg["scores"].items():
        path = Path(path)
        if path.suffix == ".npy":
            scores[name] = np.load(path).astype(np.float32)
        else:
            d = pd.read_csv(path)
            d["_dt"] = pd.to_datetime(d["Date"], format="mixed")
            d = d.set_index("_dt").sort_index()
            # aligned by column name: the ET files order their cells differently
            scores[name] = d.loc[dates, cells].values.astype(np.float32)
    order = ["GLM-Logit", "HierBayes", "B1: Static prior", "TTM", "ET",
             "B5: B2+B3", "B4: B1+B3", "B2: Recent MA", "B3: DoY season",
             "B0: Random"]
    return {k: scores[k] for k in order}, labels, cells, dates


def topk_idx(scores, k):
    return np.argpartition(-scores, k, axis=1)[:, :k]


def per_day(scores, labels, k):
    """Per-day hits, positives, recall and precision at K."""
    top = topk_idx(scores, k)
    hits = np.take_along_axis(labels.astype(np.int32), top, axis=1).sum(axis=1)
    npos = labels.sum(axis=1)
    recall = np.where(npos > 0, hits / np.where(npos > 0, npos, 1), np.nan)
    precision = hits / k
    return hits, npos, recall, precision


def jaccard_day(a, b, k):
    ta, tb = topk_idx(a, k), topk_idx(b, k)
    out = np.empty(a.shape[0])
    for t in range(a.shape[0]):
        sa, sb = set(ta[t]), set(tb[t])
        out[t] = len(sa & sb) / len(sa | sb)
    return out


def roc_variants(scores, labels):
    """Three ROC-AUC definitions that answer three different questions."""
    y = (labels > 0).astype(int)

    pooled = roc_auc_score(y.ravel(), scores.ravel())

    daily = []
    for t in range(y.shape[0]):
        if 0 < y[t].sum() < y.shape[1]:
            daily.append(roc_auc_score(y[t], scores[t]))

    per_cell = []
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < y.shape[0]:
            per_cell.append(roc_auc_score(y[:, c], scores[:, c]))

    return {
        "pooled_cell_day": float(pooled),
        "mean_daily_cross_sectional": float(np.mean(daily)),
        "n_days_scored": len(daily),
        "mean_per_cell_temporal": float(np.mean(per_cell)),
        "n_cells_scored": len(per_cell),
    }


def write_daily_csv(pref, scores, labels, cells, dates, k=20):
    RESULTS.mkdir(exist_ok=True)
    cell_arr = np.array(cells)
    names = list(scores)
    rows = []
    tops = {n: topk_idx(scores[n], k) for n in names}
    for t, d in enumerate(dates):
        actual = set(np.flatnonzero(labels[t] > 0).tolist())
        for n in names:
            pred = set(tops[n][t].tolist())
            tp, fp, fn = pred & actual, pred - actual, actual - pred
            row = {
                "date": d.strftime("%Y-%m-%d"),
                "method": n,
                "n_positive_cells": len(actual),
                f"recall_at_{k}": (len(tp) / len(actual)) if actual else np.nan,
                f"precision_at_{k}": len(tp) / k,
                "true_positive_cells": " ".join(sorted(cell_arr[list(tp)])),
                "false_positive_cells": " ".join(sorted(cell_arr[list(fp)])),
                "false_negative_cells": " ".join(sorted(cell_arr[list(fn)])),
            }
            for other in names:
                if other == n:
                    continue
                pj = set(tops[other][t].tolist())
                row[f"jaccard_{k}_vs_{other}"] = len(pred & pj) / len(pred | pj)
            rows.append(row)
    out = RESULTS / f"daily_diagnostics_{pref}_2025.csv"
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.6f")
    print(f"  wrote {out.relative_to(REPO)}  ({len(rows):,} rows)")
    return out


def case_days(scores, labels, dates, k=20):
    """Case studies chosen by deterministic criteria rather than by hand.

    The lowest-scoring day and the largest cross-method disagreement are
    included by construction, so the selection cannot be read as a highlight reel.
    """
    _, npos, rec_glm, _ = per_day(scores["GLM-Logit"], labels, k)
    _, _, rec_b1, _ = per_day(scores["B1: Static prior"], labels, k)
    jac_et = jaccard_day(scores["GLM-Logit"], scores["ET"], k)
    valid = npos > 0

    def pick(mask_values, how):
        v = np.where(valid, mask_values, np.nan)
        i = int(np.nanargmax(v) if how == "max" else np.nanargmin(v))
        return i

    med = np.nanmedian(rec_glm[valid])
    picks = [
        ("most sightings in 2025", pick(npos.astype(float), "max")),
        ("median GLM-Logit Recall@20", pick(-np.abs(rec_glm - med), "max")),
        ("lowest GLM-Logit Recall@20", pick(rec_glm, "min")),
        ("largest GLM-Logit − static-prior gap", pick(rec_glm - rec_b1, "max")),
        ("lowest GLM-Logit vs ET agreement", pick(jac_et, "min")),
    ]
    out = []
    for label, i in picks:
        out.append({
            "criterion": label,
            "date": dates[i].strftime("%Y-%m-%d"),
            "n_positive": int(npos[i]),
            "recall_glm": float(rec_glm[i]),
            "recall_b1": float(rec_b1[i]),
            "jaccard_glm_et": float(jac_et[i]),
        })
    return out


def report(pref, write_csv=True):
    scores, labels, cells, dates = load(pref)
    sighting = labels.sum(axis=1) > 0
    names = list(scores)

    head = (f"{pref.capitalize()} — {len(cells)} cells, {len(dates)} days, "
            f"{int(sighting.sum())} with ≥1 sighting")
    print("\n" + head); print("=" * len(head))

    print("\nRecall@K and Precision@K — sighting days only "
          f"(n = {int(sighting.sum())})")
    print(f"  {'Method':18s}" + "".join(f"  R@{k:<5d} P@{k:<5d}" for k in K_VALUES))
    for n in names:
        line = f"  {n:18s}"
        for k in K_VALUES:
            _, _, r, p = per_day(scores[n], labels, k)
            line += f"  {np.nanmean(r[sighting]):.4f} {p[sighting].mean():.4f}"
        print(line)

    print(f"\nPrecision@K — all days (n = {len(dates)}; a day with no sighting "
          "contributes 0)")
    print(f"  {'Method':18s}" + "".join(f"  P@{k:<8d}" for k in K_VALUES))
    for n in names:
        line = f"  {n:18s}"
        for k in K_VALUES:
            _, _, _, p = per_day(scores[n], labels, k)
            line += f"  {p.mean():.4f}    "
        print(line)

    print("\nROC-AUC — three definitions, three questions "
          "(supplementary; does not replace Recall@K / Precision@K)")
    print(f"  {'Method':18s}  {'pooled':>8s}  {'daily':>8s}  {'per-cell':>9s}")
    for n in names:
        v = roc_variants(scores[n], labels)
        print(f"  {n:18s}  {v['pooled_cell_day']:8.4f}  "
              f"{v['mean_daily_cross_sectional']:8.4f}  "
              f"{v['mean_per_cell_temporal']:9.4f}")
    v0 = roc_variants(scores[names[0]], labels)
    print(f"  (daily AUC averaged over {v0['n_days_scored']} days with a mixed "
          f"outcome; per-cell over {v0['n_cells_scored']} cells)")

    print("\nCase-study days — deterministic criteria, including "
          "low-performing and relatively favourable cases")
    for c in case_days(scores, labels, dates):
        print(f"  {c['criterion']:38s} {c['date']}  "
              f"positives={c['n_positive']:3d}  "
              f"GLM R@20={c['recall_glm']:.3f}  B1={c['recall_b1']:.3f}  "
              f"J(GLM,ET)={c['jaccard_glm_et']:.3f}")

    if write_csv:
        print()
        write_daily_csv(pref, scores, labels, cells, dates)


def main():
    ap = argparse.ArgumentParser(
        description="Per-day diagnostics and supplementary metrics.")
    ap.add_argument("--prefecture", choices=["yamagata", "akita", "all"],
                    default="all")
    ap.add_argument("--no-csv", action="store_true")
    a = ap.parse_args()
    for pref in (["yamagata", "akita"] if a.prefecture == "all" else [a.prefecture]):
        report(pref, write_csv=not a.no_csv)
    print()


if __name__ == "__main__":
    main()
