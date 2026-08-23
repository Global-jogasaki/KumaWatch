"""
all_vs_static_prior.py — every benchmarked method against the static prior B1

Table 2 of the paper reports six selected pairwise comparisons. The question it
does not answer directly is the obvious one to ask at a poster: *for every other
method, how does it compare with the static prior?* This script answers it for
all eleven methods on both prefectures, under exactly the test the paper uses,
and writes the result to `results/` so the answer is fixed on disk rather than
recomputed from memory.

Output per prefecture:
  results/all_vs_static_prior_<pref>_2025.csv
  results/all_vs_static_prior_<pref>_2025.md

Columns: method, recall_at_20, delta_vs_b1, ci_low, ci_high, p_perm,
         bonferroni_significant, n_days.

Test procedure — identical to `notebooks/kumawatch_benchmark.ipynb` Cell 17 and
to `scripts/table2_significance.py`:
  * per-day Recall@20 on days with at least one sighting
  * Δ = mean over those days of (Recall_method − Recall_B1)
  * 95% CI — day-level paired bootstrap, B = 5,000, percentile interval
  * p — day-level paired permutation (sign-flip), P = 5,000, two-sided
  * both resamplers seeded with RAND_SEED = 42, so the numbers are fixed

Multiple comparisons: this is a family of ten comparisons per prefecture (every
method except B1 itself), so the Bonferroni threshold is α = 0.05 / 10 = 0.005.
The paper's Table 2 uses α = 0.05 / 13 = 0.0038 for its own family of thirteen.
Both thresholds are printed; no comparison in this table falls between them, so
the two agree on every row.

Method coverage: all eleven. The four learned methods come from released score
matrices; B0–B5 and Poisson-GLM are regenerated deterministically from the
benchmark notebook (Cell 5 and Cell 10, RAND_SEED = 42, POISSON_ALPHA = 0.5).

Usage:
    python scripts/all_vs_static_prior.py
    python scripts/all_vs_static_prior.py --prefecture yamagata

Dependencies:
    pip install -r requirements-diagnostics.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.linear_model import PoissonRegressor

REPO = Path(__file__).resolve().parent.parent
SCORES = REPO / "data" / "scores"
RESULTS = REPO / "results"

META_COLS = {"Date", "Year", "Month", "Week", "Weekday", "Sum"}
TEST_START, TEST_END = "2025-01-01", "2025-12-31"

K = 20
RAND_SEED = 42
B_BOOT = P_PERM = 5000
POISSON_ALPHA = 0.5

BASELINE = "B1: Static prior"

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


# ── features and Poisson-GLM, mirroring the benchmark notebook ──────────────
def rolling_features(train_L, test_L, train_dates, test_dates):
    T_tr, n_cells = train_L.shape
    all_L = np.concatenate([train_L, test_L], axis=0).astype(np.float64)
    cs = np.zeros((len(all_L) + 1, n_cells)); np.cumsum(all_L, axis=0, out=cs[1:])

    def rsum(pos, w):
        return (cs[pos] - cs[max(0, pos - w)]).astype(np.float32)

    base_year = train_dates[0].year

    def block(dates, off):
        T = len(dates)
        r30 = np.empty((T, n_cells), np.float32); r365 = np.empty((T, n_cells), np.float32)
        s_ = np.empty(T, np.float32); c_ = np.empty(T, np.float32); y_ = np.empty(T, np.float32)
        for i, d in enumerate(dates):
            p = off + i
            r30[i] = rsum(p, 30); r365[i] = rsum(p, 365)
            doy = d.timetuple().tm_yday
            s_[i] = np.sin(2 * np.pi * doy / 365); c_[i] = np.cos(2 * np.pi * doy / 365)
            y_[i] = float(d.year - base_year)
        feat = np.stack([r30, np.log1p(r365),
                         np.repeat(s_, n_cells).reshape(T, n_cells),
                         np.repeat(c_, n_cells).reshape(T, n_cells),
                         np.repeat(y_, n_cells).reshape(T, n_cells)], axis=2)
        return (feat.reshape(T * n_cells, 5).astype(np.float32),
                np.tile(np.arange(n_cells, dtype=np.int32), T))

    ftr, ctr = block(train_dates, 0)
    fte, cte = block(test_dates, T_tr)
    return ftr, fte, ctr, cte


def design(feat, cidx, n_cells):
    n = feat.shape[0]
    oh = csr_matrix((np.ones(n, np.float32), (np.arange(n), cidx)), shape=(n, n_cells))
    return sp_hstack([oh, csr_matrix(feat)], format="csr")


def train_poisson_glm(train_L, ftr, ctr, fte, cte, n_cells):
    reg = PoissonRegressor(alpha=POISSON_ALPHA, fit_intercept=False,
                           max_iter=2000, tol=1e-4, warm_start=False)
    reg.fit(design(ftr, ctr, n_cells), train_L.flatten().astype(np.float64))
    lam = reg.predict(design(fte, cte, n_cells))
    return (1.0 - np.exp(-lam)).astype(np.float32).reshape(
        fte.shape[0] // n_cells, n_cells)


def build_baselines(train_L, test_L, train_dates, test_dates):
    n_test, n_cells = test_L.shape

    def daynorm(s):
        mu = s.mean(axis=1, keepdims=True); sd = s.std(axis=1, keepdims=True)
        return (s - mu) / np.where(sd == 0, 1.0, sd)

    b1 = np.tile(train_L.mean(axis=0), (n_test, 1)).astype(np.float32)

    all_L = np.concatenate([train_L, test_L], axis=0).astype(np.float64)
    cs = np.zeros((len(all_L) + 1, n_cells)); np.cumsum(all_L, axis=0, out=cs[1:])
    b2 = np.empty((n_test, n_cells), np.float32)
    for d in range(n_test):
        pos = len(train_L) + d; start = max(0, pos - 30)
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
        "B0: Random":       rng.random((n_test, n_cells)).astype(np.float32),
        BASELINE:           b1,
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
    dates, train_dates = df.index[te], df.index[tr]

    scores = build_baselines(train_L, labels, train_dates, dates)
    for name, path in cfg["scores"].items():
        path = Path(path)
        if path.suffix == ".npy":
            scores[name] = np.load(path).astype(np.float32)
        else:
            d = pd.read_csv(path)
            d["_dt"] = pd.to_datetime(d["Date"], format="mixed")
            d = d.set_index("_dt").sort_index()
            scores[name] = d.loc[dates, cells].values.astype(np.float32)

    ftr, fte, ctr, cte = rolling_features(train_L, labels, train_dates, dates)
    scores["Poisson-GLM"] = train_poisson_glm(train_L, ftr, ctr, fte, cte, len(cells))
    return scores, labels


def per_day_recall(scores, labels, k=K):
    topk = np.argpartition(-scores, k, axis=1)[:, :k]
    hits = np.take_along_axis(labels.astype(np.int32), topk, axis=1).sum(axis=1)
    npos = labels.sum(axis=1).astype(np.float64)
    valid = npos > 0
    r = np.where(valid, hits / np.where(valid, npos, 1.0), 0.0)
    return r.astype(np.float64), valid


def compare(sa, sb, labels, seed=RAND_SEED):
    ra, valid = per_day_recall(sa, labels)
    rb, _ = per_day_recall(sb, labels)
    diff = (ra - rb)[valid]
    n = int(valid.sum())
    obs = float(diff.mean())

    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, n, size=(B_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    rng = np.random.default_rng(seed)
    perm = (rng.choice([-1.0, 1.0], size=(P_PERM, n)) * diff).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs)).sum() + 1) / (P_PERM + 1)
    return obs, float(lo), float(hi), p, n


ORDER = ["GLM-Logit", "HierBayes", "B5: B2+B3", BASELINE, "B4: B1+B3", "TTM",
         "B2: Recent MA", "B3: DoY season", "ET", "B0: Random", "Poisson-GLM"]


def run(pref):
    scores, labels = load(pref)
    base = scores[BASELINE]
    n_comparisons = len(ORDER) - 1
    alpha = 0.05 / n_comparisons

    rows = []
    for name in ORDER:
        recall, valid = per_day_recall(scores[name], labels)
        r20 = float(recall[valid].mean())
        if name == BASELINE:
            rows.append(dict(method=name, recall_at_20=r20, delta_vs_b1=0.0,
                             ci_low=np.nan, ci_high=np.nan, p_perm=np.nan,
                             bonferroni_significant="—", n_days=int(valid.sum())))
            continue
        d, lo, hi, p, n = compare(scores[name], base, labels)
        rows.append(dict(method=name, recall_at_20=r20, delta_vs_b1=d,
                         ci_low=lo, ci_high=hi, p_perm=p,
                         bonferroni_significant="yes" if p < alpha else "no",
                         n_days=n))
    df = pd.DataFrame(rows)

    RESULTS.mkdir(exist_ok=True)
    stem = RESULTS / f"all_vs_static_prior_{pref}_2025"
    df.to_csv(stem.with_suffix(".csv"), index=False, float_format="%.6f")

    title = pref.capitalize()
    md = [f"# Every method against the static prior — {title}, 2025", "",
          f"Recall@20 on the {int(df.n_days.max())} days with at least one reported "
          f"sighting. Δ is the mean per-day difference against B1; the interval is a "
          f"day-level paired bootstrap (B = {B_BOOT:,}) and *p* a day-level "
          f"sign-flip permutation test (P = {P_PERM:,}), both seeded with 42 — the "
          "same procedure as Table 2 of the paper.", "",
          f"Bonferroni over the {n_comparisons} comparisons in this table: "
          f"α = {alpha:.4f}. (Table 2 of the paper uses α = 0.0038 over its own "
          "family of thirteen; no row here falls between the two thresholds.)", "",
          "| Method | Recall@20 | Δ vs B1 | 95% CI | *p* | Bonferroni sig. | n days |",
          "|--------|:---------:|:-------:|:------:|:---:|:---------------:|:------:|"]
    for r in rows:
        if r["method"] == BASELINE:
            md.append(f"| **{r['method']}** | {r['recall_at_20']:.4f} | — | — | — "
                      f"| — | {r['n_days']} |")
        else:
            md.append(f"| {r['method']} | {r['recall_at_20']:.4f} | "
                      f"{r['delta_vs_b1']:+.4f} | "
                      f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                      f"{r['p_perm']:.4f} | {r['bonferroni_significant']} | "
                      f"{r['n_days']} |")
    md.append("")
    md.append("Generated by `scripts/all_vs_static_prior.py`.")
    stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n{title} — {int(df.n_days.max())} sighting days, "
          f"Bonferroni α = {alpha:.4f}")
    print(f"  {'Method':18s} {'R@20':>7s} {'Δ vs B1':>9s} "
          f"{'95% CI':>20s} {'p':>8s}  sig")
    for r in rows:
        if r["method"] == BASELINE:
            print(f"  {r['method']:18s} {r['recall_at_20']:7.4f} "
                  f"{'—':>9s} {'—':>20s} {'—':>8s}   —")
        else:
            print(f"  {r['method']:18s} {r['recall_at_20']:7.4f} "
                  f"{r['delta_vs_b1']:+9.4f} "
                  f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] "
                  f"{r['p_perm']:8.4f}  {r['bonferroni_significant']}")
    print(f"  wrote {stem.with_suffix('.csv').relative_to(REPO)} and .md")


def main():
    ap = argparse.ArgumentParser(
        description="Compare every benchmarked method with the static prior B1.")
    ap.add_argument("--prefecture", choices=["yamagata", "akita", "all"],
                    default="all")
    a = ap.parse_args()
    for pref in (["yamagata", "akita"] if a.prefecture == "all" else [a.prefecture]):
        run(pref)
    print()


if __name__ == "__main__":
    main()
