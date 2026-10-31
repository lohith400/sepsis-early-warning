"""End-to-end training pipeline (GA -> ANFIS -> PSO) producing a JSON model artifact.

Usage (from the repository root)::

    python -m sepsis.train --download            # all 20,336 patients (about 10-15 minutes)
    python -m sepsis.train --download --quick    # 3,000 patients (a few minutes)

Run with ``PYTHONPATH=backend`` or via ``make train``.
"""
import argparse
import dataclasses
import datetime as dt
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import __version__
from .anfis import class_weights, decode, lse, phi, score
from .config import CLIP, TrainConfig
from .features import FEATURE_NAMES, build_features, prepare
from .ga import ga_select
from .metrics import auc_fast, curves, evaluate, pick_threshold
from .pso import pso

warnings.filterwarnings("ignore")


def _subset(rng, pid, y, patients, n_pos, n_neg):
    m = np.isin(pid, patients)
    pos = np.where(m & (y == 1))[0]
    neg = np.where(m & (y == 0))[0]
    pos = rng.choice(pos, min(n_pos, len(pos)), replace=False)
    neg = rng.choice(neg, min(n_neg, len(neg)), replace=False)
    idx = np.concatenate([pos, neg])
    rng.shuffle(idx)
    return idx


def train(df: pd.DataFrame, cfg: TrainConfig | None = None, source: str = "physionet-2019-setA",
          name: str = "GA + ANFIS + PSO sepsis early-warning model", log=print) -> dict:
    """Train on a DataFrame (columns: pid, raw measurements, Age, ICULOS, SepsisLabel). Returns the artifact dict."""
    cfg = cfg or TrainConfig()
    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)
    pid = df["pid"].values
    y = df["SepsisLabel"].values.astype(int)

    # ---- 1. patient-level split: train / validation / test, and fit / select inside train ----
    pid_all = df.groupby("pid")["SepsisLabel"].max()
    tr_p, tmp_p = train_test_split(pid_all.index.values, test_size=0.4, random_state=cfg.seed, stratify=pid_all.values)
    va_p, te_p = train_test_split(tmp_p, test_size=0.5, random_state=cfg.seed, stratify=pid_all.loc[tmp_p].values)
    fit_p, sel_p = train_test_split(tr_p, test_size=0.3, random_state=cfg.seed, stratify=pid_all.loc[tr_p].values)
    log(f"Patients: {len(pid_all)} | sepsis patients: {int(pid_all.sum())} | records: {len(df)}")

    # ---- 2. features, imputation and scaling (statistics from training patients only) ----
    F = prepare(build_features(df))
    tr_mask = np.isin(pid, tr_p)
    medians = F[tr_mask].median().fillna(0.0)
    F = F.fillna(medians)
    mu = F[tr_mask].mean()
    sd = F[tr_mask].std().replace(0, 1).fillna(1.0)
    X = ((F - mu) / sd).clip(-CLIP, CLIP).values.astype(np.float32)
    nF = X.shape[1]
    va_idx = np.where(np.isin(pid, va_p))[0]
    te_idx = np.where(np.isin(pid, te_p))[0]
    fit_idx = _subset(rng, pid, y, fit_p, cfg.fit_pos, cfg.fit_neg)
    sel_idx = _subset(rng, pid, y, sel_p, cfg.sel_pos, cfg.sel_neg)
    log(f"Rows | fit {len(fit_idx)} | select {len(sel_idx)} | validation {len(va_idx)} | test {len(te_idx)} "
        f"(test sepsis rate {100 * y[te_idx].mean():.2f}%)")

    # ---- 3. Genetic Algorithm: feature selection ----
    log("Stage 1/3: Genetic Algorithm feature selection")
    t0 = time.time()
    mask, ga_best, ga_mean, ga_quick_auc = ga_select(X[fit_idx], y[fit_idx], X[sel_idx], y[sel_idx], cfg, rng, log)
    ga_time = time.time() - t0
    sel_cols = np.flatnonzero(mask)
    sel_names = [FEATURE_NAMES[i] for i in sel_cols]
    lr_all = LogisticRegression(max_iter=200, class_weight="balanced").fit(X[fit_idx], y[fit_idx])
    all_auc = auc_fast(y[sel_idx], lr_all.decision_function(X[sel_idx]))
    log(f"  selected {len(sel_names)} features: {sel_names} (quick AUROC {ga_quick_auc:.4f}, all features {all_auc:.4f})")

    # ---- 4. ANFIS (k-means rules) + PSO ----
    log("Stage 2/3: ANFIS rules (k-means) and PSO training")
    R, K = cfg.n_rules, len(sel_cols)
    Xs = X[:, sel_cols].astype(np.float64)
    Xf, yf = Xs[fit_idx], y[fit_idx].astype(float)
    Xv, yv = Xs[sel_idx], y[sel_idx].astype(float)
    swf, swv = class_weights(yf), class_weights(yv)
    km = KMeans(R, n_init=5, random_state=cfg.seed).fit(Xf)
    C0 = km.cluster_centers_
    S0 = np.stack([Xf[km.labels_ == r].std(0) if (km.labels_ == r).sum() > 5 else np.ones(K) for r in range(R)])
    S0 = S0.clip(0.3, 2.5)
    v0 = np.concatenate([C0.ravel(), np.log(S0).ravel()])
    lo = np.concatenate([np.full(R * K, -5.0), np.full(R * K, np.log(0.15))])
    hi = np.concatenate([np.full(R * K, 5.0), np.full(R * K, np.log(4.0))])

    def fitness(v):
        C, S = decode(v, R, K)
        _, Pf = phi(Xf, C, S)
        th = lse(Pf, yf, swf)
        _, Pv = phi(Xv, C, S)
        pv = Pv @ th
        return (1 - auc_fast(yv, pv)) + 0.1 * np.average((pv - yv) ** 2, weights=swv)

    noise = np.concatenate([np.full(R * K, 0.3), np.full(R * K, 0.2)])
    t0 = time.time()
    gbest, gfit, pso_hist, init_fit = pso(fitness, v0, lo, hi, cfg.pso_particles, cfg.pso_iters, rng, noise, log)
    pso_time = time.time() - t0

    u_idx = np.concatenate([fit_idx, sel_idx])           # final consequents on fit + select rows
    Xu, yu = Xs[u_idx], y[u_idx].astype(float)

    def train_theta(v):
        C, S = decode(v, R, K)
        _, P = phi(Xu, C, S)
        return lse(P, yu, class_weights(yu))

    th0, th1 = train_theta(v0), train_theta(gbest)
    C0d, S0d = decode(v0, R, K)
    C1, S1 = decode(gbest, R, K)
    val_auc_kmeans = auc_fast(y[va_idx], score(Xs[va_idx], C0d, S0d, th0))
    s_va = score(Xs[va_idx], C1, S1, th1)
    val_auc_pso = auc_fast(y[va_idx], s_va)
    log(f"  validation AUROC: k-means rules only {val_auc_kmeans:.4f} -> after PSO {val_auc_pso:.4f}")

    # ---- 5. calibration + alert threshold (validation), test (once) ----
    log("Stage 3/3: calibration, threshold and test evaluation")
    s_te = score(Xs[te_idx], C1, S1, th1)
    platt = LogisticRegression(C=1e6).fit(s_va.reshape(-1, 1), y[va_idx])
    p_va = platt.predict_proba(s_va.reshape(-1, 1))[:, 1]
    p_te = platt.predict_proba(s_te.reshape(-1, 1))[:, 1]
    thr = pick_threshold(y[va_idx], p_va, cfg.target_sens)
    res, cm = evaluate("GA + ANFIS + PSO (ours)", y[te_idx], p_te, thr, pid[te_idx])
    t = time.time()
    score(Xs[te_idx[:20000]], C1, S1, th1)
    per_record_us = (time.time() - t) / min(20000, len(te_idx)) * 1e6
    res["train_time_s"] = ga_time + pso_time
    models = [res]
    roc_all, pr_all = {}, {}
    roc_all[res["model"]], pr_all[res["model"]] = curves(y[te_idx], p_te)

    def baseline(label, model, cols=None):
        Xb = X if cols is None else X[:, cols]
        t = time.time()
        model.fit(Xb[u_idx], y[u_idx])
        tt = time.time() - t
        sv = model.predict_proba(Xb[va_idx])[:, 1]
        st = model.predict_proba(Xb[te_idx])[:, 1]
        r, _ = evaluate(label, y[te_idx], st, pick_threshold(y[va_idx], sv, cfg.target_sens), pid[te_idx])
        r["train_time_s"] = tt
        models.append(r)
        roc_all[label], pr_all[label] = curves(y[te_idx], st)

    def rf():
        return RandomForestClassifier(cfg.rf_trees, max_depth=12, min_samples_leaf=20,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=cfg.seed)

    baseline("Logistic Regression (GA features)", LogisticRegression(max_iter=300, class_weight="balanced"), sel_cols)
    baseline("Random Forest (GA features)", rf(), sel_cols)
    baseline("Logistic Regression (all features)", LogisticRegression(max_iter=300, class_weight="balanced"))
    baseline("Random Forest (all features)", rf())
    log(f"  test AUROC {res['auroc']:.4f} | sensitivity {res['sensitivity']:.3f} | precision {res['precision']:.3f}")

    # ---- 6. artifact ----
    f = lambda v: float(v)  # noqa: E731
    art = {
        "format_version": 1,
        "name": name,
        "version": __version__,
        "source": source,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": dataclasses.asdict(cfg),
        "dataset": {
            "patients": int(len(pid_all)), "records": int(len(df)),
            "sepsis_patients": int(pid_all.sum()), "sepsis_records": int(y.sum()),
            "sepsis_rate": f(y.mean()), "train_patients": int(len(tr_p)),
            "validation_patients": int(len(va_p)), "test_patients": int(len(te_p)),
            "validation_rows": int(len(va_idx)), "test_rows": int(len(te_idx)),
            "test_sepsis_rate": f(y[te_idx].mean()), "fit_rows": int(len(fit_idx)), "select_rows": int(len(sel_idx)),
        },
        "features": {"all": FEATURE_NAMES, "selected": sel_names, "selected_idx": [int(i) for i in sel_cols]},
        "preprocessing": {
            "clip": CLIP,
            "medians": {k: f(v) for k, v in medians.items()},
            "mean": {k: f(v) for k, v in mu.items()},
            "std": {k: f(v) for k, v in sd.items()},
            "range_min": {n: f(F[tr_mask][n].min()) for n in sel_names},
            "range_max": {n: f(F[tr_mask][n].max()) for n in sel_names},
        },
        "anfis": {"n_rules": R, "centers": C1.tolist(), "sigmas": S1.tolist(), "theta": th1.tolist()},
        "calibration": {"coef": f(platt.coef_[0][0]), "intercept": f(platt.intercept_[0])},
        "alert": {"threshold": thr, "target_sensitivity": cfg.target_sens},
        "metrics": {"anfis": res, "confusion": cm, "models": models,
                    "ga_quick_auc_selected": ga_quick_auc, "ga_quick_auc_all": all_auc,
                    "val_auc_kmeans_only": f(val_auc_kmeans), "val_auc_after_pso": f(val_auc_pso)},
        "curves": {"roc": roc_all, "pr": pr_all, "test_prevalence": f(y[te_idx].mean())},
        "history": {"ga_best": ga_best, "ga_mean": ga_mean, "pso": [f(v) for v in pso_hist], "pso_start": f(init_fit)},
        "timing": {"ga_s": ga_time, "pso_s": pso_time, "predict_us_per_record": per_record_us},
    }
    return art


def save_artifact(art: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(art, fh)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the GA + ANFIS + PSO sepsis model.")
    ap.add_argument("--data-dir", default="data/setA", help="folder with PhysioNet .psv files")
    ap.add_argument("--download", action="store_true", help="download Training Set A first (skips existing files)")
    ap.add_argument("--quick", action="store_true", help="use only 3,000 patients")
    ap.add_argument("--patients", type=int, default=None, help="limit the number of patients")
    ap.add_argument("--out", default="artifacts/model.json", help="where to save the trained model")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    cfg = TrainConfig(seed=a.seed, n_patients=3000 if a.quick else a.patients)
    if a.download:
        from .data import download_set_a
        download_set_a(a.data_dir, limit=cfg.n_patients)
    from .data import load_patients
    df = load_patients(a.data_dir, cfg.n_patients)
    art = train(df, cfg, source="physionet-2019-setA")
    save_artifact(art, a.out)
    print(f"\nSaved model to {a.out}")


if __name__ == "__main__":
    main()
