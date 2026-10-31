"""Evaluation helpers."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve


def auc_fast(y, s) -> float:
    """Rank-based AUROC (fast, no sorting of thresholds)."""
    y = np.asarray(y)
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pick_threshold(y, s, target_sens: float) -> float:
    """Highest threshold whose sensitivity is at least ``target_sens`` (chosen on validation data)."""
    _, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr >= target_sens)])


def evaluate(name, y, s, thr, pids):
    pred = (np.asarray(s) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    pa = pd.Series(pred).groupby(pids).max()
    pl = pd.Series(y).groupby(pids).max()
    prec = tp / max(tp + fp, 1)
    sens = tp / max(tp + fn, 1)
    res = {
        "model": name,
        "auroc": auc_fast(y, s),
        "auprc": float(average_precision_score(y, s)),
        "sensitivity": float(sens),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(prec),
        "f1": float(2 * prec * sens / max(prec + sens, 1e-9)),
        "accuracy": float((tp + tn) / len(y)),
        "false_alarms_per_true_alert": float(fp / max(tp, 1)),
        "sepsis_patients_alerted": float(pa[pl == 1].mean()) if (pl == 1).any() else 0.0,
        "non_sepsis_patients_alerted": float(pa[pl == 0].mean()) if (pl == 0).any() else 0.0,
    }
    return res, {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def curves(y, s, points: int = 200):
    """Down-sampled ROC and precision-recall curves for the UI."""
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(points, len(fpr))).astype(int))
    roc = {"fpr": [round(float(v), 5) for v in fpr[idx]], "tpr": [round(float(v), 5) for v in tpr[idx]]}
    p, r, _ = precision_recall_curve(y, s)
    idx = np.unique(np.linspace(0, len(p) - 1, min(points, len(p))).astype(int))
    pr = {"recall": [round(float(v), 5) for v in r[idx]], "precision": [round(float(v), 5) for v in p[idx]]}
    return roc, pr
