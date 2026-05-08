"""
calibration_validation.py — KumaWatch Calibration Validation Script

Applies post-hoc Platt scaling and Isotonic regression to Extra Trees (ET) scores
and evaluates calibration metrics (Brier, ECE, MAE) and operational metrics
(Recall@K, Precision@K) against the GLM-Logit primary layer.

Paper: KumaWatch: A Multi-Method Wildlife Encounter Alert System for
       Operational Municipal Deployment in Northern Japan [Applications]
       ACM SIGSPATIAL 2026

Usage:
    python calibration_validation.py --prefecture yamagata
    python calibration_validation.py --prefecture akita
    python calibration_validation.py --prefecture all

Dependencies:
    pip install numpy pandas scikit-learn
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Default file paths (Google Colab / Drive layout — adjust for local runs)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PATHS = {
    "yamagata": {
        "sightings":  "/content/drive/MyDrive/bear/Yamagata_10km_AllGrid_144cells_Daily_TimeSeries.csv",
        "et_scores":  "/content/drive/MyDrive/bear/yamagata_et_scores_2025.csv",
        "glm_scores": None,   # computed inline from sightings
        "train_start": "2018-10-01",
        "train_end":   "2024-12-31",
        "test_start":  "2025-01-01",
        "test_end":    "2025-12-31",
    },
    "akita": {
        "sightings":  "/content/drive/MyDrive/bear/Akita_10km_AllGrid_260cells_Daily_TimeSeries.csv",
        "et_scores":  "/content/drive/MyDrive/bear/akita_et_scores_2025.csv",
        "glm_scores": None,
        "train_start": "2022-04-01",
        "train_end":   "2024-12-31",
        "test_start":  "2025-01-01",
        "test_end":    "2025-12-31",
    },
}

K_VALUES = [10, 20, 30]
N_BINS_ECE = 10
GLM_C = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sightings(csv_path, train_start, train_end, test_start, test_end):
    """Load bear sightings CSV → (train_labels, test_labels, cell_cols, test_dates)."""
    df = pd.read_csv(csv_path)
    df["_dt"] = pd.to_datetime(df["Date"])
    df = df.set_index("_dt").sort_index()
    meta = {"Date", "Year", "Month", "Week", "Weekday", "Sum"}
    cell_cols = [c for c in df.columns if c not in meta]

    tr_mask = (df.index >= train_start) & (df.index <= train_end)
    te_mask = (df.index >= test_start)  & (df.index <= test_end)
    train_L = df.loc[tr_mask, cell_cols].values.astype(np.float32)
    test_L  = df.loc[te_mask, cell_cols].values.astype(np.float32)
    test_dates = df.index[te_mask]
    return train_L, test_L, cell_cols, test_dates


def load_scores_csv(csv_path, cell_cols, test_dates):
    """Load wide-format score CSV (Date, cell1, cell2, ...) → (n_days, n_cells) array."""
    df = pd.read_csv(csv_path)
    df["_dt"] = pd.to_datetime(df["Date"])
    df = df.set_index("_dt").sort_index()
    return df.loc[test_dates, cell_cols].values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GLM-Logit training (primary layer — reproduced for calibration baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_design_matrix(labels, n_cells, start_day=0):
    """Build GLM-Logit design matrix: cell one-hot + temporal features."""
    n_days = labels.shape[0]
    rows_X, rows_y = [], []

    for t in range(start_day, n_days):
        # Temporal features (shared across cells)
        doy = t % 365
        sin_doy = np.sin(2 * np.pi * doy / 365)
        cos_doy = np.cos(2 * np.pi * doy / 365)
        year_idx = t / 365.0

        for c in range(n_cells):
            # Rolling features (no data leakage: only days before t)
            recent30  = float(labels[max(0, t - 30):t, c].sum())
            recent365 = float(labels[max(0, t - 365):t, c].sum())
            log_r365  = np.log1p(recent365)

            # Cell one-hot + shared features
            feat = np.zeros(n_cells + 5, dtype=np.float32)
            feat[c] = 1.0                     # cell fixed effect
            feat[n_cells + 0] = recent30
            feat[n_cells + 1] = log_r365
            feat[n_cells + 2] = sin_doy
            feat[n_cells + 3] = cos_doy
            feat[n_cells + 4] = year_idx

            rows_X.append(feat)
            rows_y.append(float(labels[t, c] > 0))

    return np.array(rows_X, dtype=np.float32), np.array(rows_y, dtype=np.float32)


def train_glm_logit(train_L):
    """Train GLM-Logit and return a predict function."""
    print("  Training GLM-Logit (this may take a few minutes)...")
    n_days, n_cells = train_L.shape
    start = 30  # skip first 30 days (warm-up for rolling features)
    X_tr, y_tr = build_design_matrix(train_L, n_cells, start_day=start)

    model = LogisticRegression(
        C=GLM_C, fit_intercept=False, max_iter=2000, solver="lbfgs",
        multi_class="ovr", n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    return model, n_cells


def predict_glm(model, n_cells, train_L, test_L):
    """Generate GLM-Logit test-period probability scores (n_test_days, n_cells)."""
    concat_L = np.vstack([train_L, test_L])
    n_train  = train_L.shape[0]
    n_test   = test_L.shape[0]
    scores   = np.zeros((n_test, n_cells), dtype=np.float32)

    for t in range(n_test):
        abs_t = n_train + t
        doy   = abs_t % 365
        sin_doy = np.sin(2 * np.pi * doy / 365)
        cos_doy = np.cos(2 * np.pi * doy / 365)
        year_idx = abs_t / 365.0

        feat = np.zeros((n_cells, n_cells + 5), dtype=np.float32)
        for c in range(n_cells):
            recent30  = float(concat_L[max(0, abs_t - 30):abs_t, c].sum())
            recent365 = float(concat_L[max(0, abs_t - 365):abs_t, c].sum())
            feat[c, c] = 1.0
            feat[c, n_cells + 0] = recent30
            feat[c, n_cells + 1] = np.log1p(recent365)
            feat[c, n_cells + 2] = sin_doy
            feat[c, n_cells + 3] = cos_doy
            feat[c, n_cells + 4] = year_idx

        probs = model.predict_proba(feat)[:, 1]
        scores[t] = probs

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_scores(train_scores_flat, train_labels_flat, method="sigmoid"):
    """
    Fit a post-hoc calibrator on flattened training scores.

    Parameters
    ----------
    method : "sigmoid"  → Platt scaling (logistic regression on scores)
             "isotonic" → Isotonic regression
    """
    s = train_scores_flat.clip(0, 1).reshape(-1, 1)
    y = (train_labels_flat > 0).astype(int)

    if method == "sigmoid":
        # Platt scaling: logistic regression on raw probability scores
        cal = LogisticRegression(C=1e6, fit_intercept=True, max_iter=1000)
        cal.fit(s, y)
        def predict(scores):
            return cal.predict_proba(scores.clip(0, 1).reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(s.ravel(), y)
        def predict(scores):
            return cal.predict(scores.clip(0, 1).ravel())
    else:
        raise ValueError(f"Unknown method: {method}")

    return predict


def apply_calibration(raw_scores, train_scores, train_labels, method):
    """
    Apply post-hoc calibration to test-period scores.
    Fits calibrator on training-period scores, applies to test period.

    Returns: calibrated test scores (n_days, n_cells)
    """
    cal_fn = calibrate_scores(
        train_scores.ravel(), train_labels.ravel(), method=method
    )
    n_days, n_cells = raw_scores.shape
    cal_scores = cal_fn(raw_scores.ravel()).reshape(n_days, n_cells)
    return cal_scores.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(scores, labels, K):
    K = min(K, scores.shape[1] - 1)
    topk = np.argpartition(-scores, K, axis=1)[:, :K]
    hits = np.take_along_axis(labels.astype(np.int32), topk, axis=1).sum(axis=1)
    npos = labels.sum(axis=1)
    valid = npos > 0
    return float((hits[valid] / npos[valid]).mean()) if valid.any() else 0.0


def precision_at_k(scores, labels, K):
    K = min(K, scores.shape[1] - 1)
    topk = np.argpartition(-scores, K, axis=1)[:, :K]
    hits = np.take_along_axis(labels.astype(np.int32), topk, axis=1).sum(axis=1)
    valid = labels.sum(axis=1) > 0
    return float((hits[valid] / K).mean()) if valid.any() else 0.0


def compute_calibration_metrics(scores, labels, n_bins=N_BINS_ECE):
    """Compute Brier, ECE, MAE, BSS."""
    s = np.clip(scores.ravel().astype(np.float64), 0.0, 1.0)
    y = labels.ravel().astype(np.float64)
    brier = float(np.mean((s - y) ** 2))
    mae   = float(np.mean(np.abs(s - y)))
    base  = float(np.mean((y.mean() - y) ** 2))
    bss   = float(1.0 - brier / base) if base > 0 else 0.0

    # Expected Calibration Error
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == n_bins - 1 else (s >= lo) & (s < hi)
        if mask.sum() > 0:
            ece += (mask.sum() / len(s)) * abs(s[mask].mean() - y[mask].mean())

    return {"Brier": brier, "ECE": ece, "MAE": mae, "BSS": bss}


def evaluate(name, scores, labels):
    """Full evaluation dict for one method."""
    result = {"Method": name}
    for K in K_VALUES:
        result[f"Recall@{K}"]  = recall_at_k(scores, labels, K)
        result[f"Prec@{K}"]    = precision_at_k(scores, labels, K)
    result.update(compute_calibration_metrics(scores, labels))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration_validation(prefecture, paths):
    print(f"\n{'='*60}")
    print(f"  KumaWatch Calibration Validation — {prefecture.upper()}")
    print(f"{'='*60}")

    cfg = paths[prefecture]
    print("Loading data...")
    train_L, test_L, cell_cols, test_dates = load_sightings(
        cfg["sightings"],
        cfg["train_start"], cfg["train_end"],
        cfg["test_start"],  cfg["test_end"],
    )
    print(f"  Train: {train_L.shape}  Test: {test_L.shape}  Cells: {len(cell_cols)}")

    print("Loading ET raw scores...")
    et_raw = load_scores_csv(cfg["et_scores"], cell_cols, test_dates)

    # For ET calibration, we need training-period ET scores
    # Use a simple heuristic: compute rolling-window probability as proxy for
    # training ET scores (full ET training predictions are needed for proper
    # calibration; if available, replace `train_et_proxy` with actual train scores)
    n_train, n_cells = train_L.shape
    print("  Building ET training score proxy for calibrator fitting...")
    # Proxy: recent-30-day moving average (similar information to ET temporal features)
    train_et_proxy = np.zeros_like(train_L, dtype=np.float32)
    for t in range(n_train):
        recent = train_L[max(0, t - 30):t].sum(axis=0)
        train_et_proxy[t] = recent / 30.0

    print("Applying Platt scaling to ET scores...")
    et_platt = apply_calibration(et_raw, train_et_proxy, train_L, method="sigmoid")

    print("Applying Isotonic regression to ET scores...")
    et_iso = apply_calibration(et_raw, train_et_proxy, train_L, method="isotonic")

    print("Training GLM-Logit primary layer...")
    glm_model, _ = train_glm_logit(train_L)
    glm_scores = predict_glm(glm_model, n_cells, train_L, test_L)
    print("  GLM-Logit training complete.")

    # ── Evaluate all variants
    results = []
    for name, scores in [
        ("GLM-Logit",     glm_scores),
        ("ET-Raw",        et_raw),
        ("ET-Platt",      et_platt),
        ("ET-Isotonic",   et_iso),
    ]:
        r = evaluate(name, scores, test_L)
        results.append(r)
        print(f"  [{name:14s}]  Recall@20={r['Recall@20']:.4f}  "
              f"Brier={r['Brier']:.4f}  ECE={r['ECE']:.4f}")

    # ── Print table
    df = pd.DataFrame(results).set_index("Method")
    print(f"\n{'─'*70}")
    print(f"  {prefecture.upper()} — Full Results")
    print(f"{'─'*70}")
    cols = [f"Recall@{k}" for k in K_VALUES] + \
           [f"Prec@{k}"   for k in K_VALUES] + \
           ["Brier", "ECE", "MAE", "BSS"]
    print(df[[c for c in cols if c in df.columns]].to_string(float_format="{:.4f}".format))

    # ── Calibration advantage summary
    r20_glm = df.loc["GLM-Logit", "Recall@20"]
    brier_glm = df.loc["GLM-Logit", "Brier"]
    print(f"\n  Calibration advantage (GLM-Logit vs ET variants):")
    for variant in ["ET-Raw", "ET-Platt", "ET-Isotonic"]:
        if variant in df.index:
            brier_adv = df.loc[variant, "Brier"] / brier_glm
            recall_adv = r20_glm / df.loc[variant, "Recall@20"]
            print(f"    vs {variant:14s}: Brier ratio = {brier_adv:.2f}×  "
                  f"Recall@20 ratio = {recall_adv:.2f}×")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KumaWatch calibration validation: Platt/Isotonic vs GLM-Logit"
    )
    parser.add_argument(
        "--prefecture", choices=["yamagata", "akita", "all"], default="all",
        help="Prefecture to evaluate (default: all)"
    )
    parser.add_argument(
        "--sightings_yamagata", default=DEFAULT_PATHS["yamagata"]["sightings"],
        help="Path to Yamagata sightings CSV"
    )
    parser.add_argument(
        "--et_scores_yamagata", default=DEFAULT_PATHS["yamagata"]["et_scores"],
        help="Path to Yamagata ET scores CSV (wide format)"
    )
    parser.add_argument(
        "--sightings_akita", default=DEFAULT_PATHS["akita"]["sightings"],
        help="Path to Akita sightings CSV"
    )
    parser.add_argument(
        "--et_scores_akita", default=DEFAULT_PATHS["akita"]["et_scores"],
        help="Path to Akita ET scores CSV (wide format)"
    )
    args = parser.parse_args()

    paths = {
        "yamagata": {**DEFAULT_PATHS["yamagata"],
                     "sightings": args.sightings_yamagata,
                     "et_scores": args.et_scores_yamagata},
        "akita":    {**DEFAULT_PATHS["akita"],
                     "sightings": args.sightings_akita,
                     "et_scores": args.et_scores_akita},
    }

    prefectures = ["yamagata", "akita"] if args.prefecture == "all" else [args.prefecture]
    for pref in prefectures:
        run_calibration_validation(pref, paths)

    print("\nDone. Calibration validation complete.")
