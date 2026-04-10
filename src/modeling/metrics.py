"""Classification metrics aligned with PRD Phase E (E4)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def apnea_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score_positive: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Apnea / binary detection: accuracy, sensitivity (recall on positive), specificity, AUC-ROC if scores given.

    Expects binary labels (typically 0/1) consistent with `y_score_positive` if provided.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(yt, yp)),
        "n_samples": int(len(yt)),
    }
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    out["sensitivity"] = float(sens)
    out["specificity"] = float(spec)
    out["confusion_matrix_tn_fp_fn_tp"] = [int(tn), int(fp), int(fn), int(tp)]

    if y_score_positive is not None:
        scores = np.asarray(y_score_positive, dtype=float)
        if len(scores) == len(yt) and len(np.unique(yt)) > 1:
            try:
                # sklearn>=1.4: binary AUC without pos_label kwarg on roc_auc_score
                out["auc_roc"] = float(roc_auc_score(yt, scores))
            except ValueError:
                out["auc_roc"] = None
        else:
            out["auc_roc"] = None
    else:
        out["auc_roc"] = None

    return out


def multiclass_sleep_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Sleep staging: accuracy, macro-F1, Cohen's kappa, per-class F1 (e.g. N1 analysis).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    lab = np.asarray(labels) if labels is not None else np.unique(np.concatenate([yt, yp]))
    per_f1 = f1_score(yt, yp, labels=lab, average=None, zero_division=0)
    per_class_f1: Dict[str, float] = {}
    for i, c in enumerate(lab):
        per_class_f1[str(c)] = float(per_f1[i])
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(yt, yp)),
        "per_class_f1": per_class_f1,
        "labels": [str(x) for x in lab],
    }


def cohen_kappa(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    return float(cohen_kappa_score(np.asarray(y_true), np.asarray(y_pred)))


def macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), average="macro", zero_division=0))


def mcnemar_exact(
    y_true: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
) -> Tuple[float, float]:
    """
    McNemar test for two classifiers (pred_a vs pred_b) on the same labels. Returns (statistic, p-value).

    Discordant counts follow the usual McNemar setup; p-value uses a two-sided exact binomial test on
    ``b + c`` (compatible across SciPy versions; avoids deprecated ``scipy.stats.mcnemar`` import paths).
    """
    from scipy.stats import binomtest

    yt = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    a_ok = a == yt
    b_ok = b == yt
    # A correct / wrong vs B correct / wrong (discordant pairs drive McNemar)
    b_disc = int(np.sum(a_ok & ~b_ok))
    c_disc = int(np.sum(~a_ok & b_ok))
    n = b_disc + c_disc
    if n == 0:
        return 0.0, 1.0
    # Continuity-corrected chi-square statistic (Fagerland et al.)
    stat = (abs(b_disc - c_disc) - 1) ** 2 / n if n else 0.0
    k = min(b_disc, c_disc)
    p_value = float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    return float(stat), p_value


def fold_metrics_summary(rows: Sequence[Mapping[str, float]], metric_keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Mean ± std across folds for CSV reporting."""
    out: Dict[str, Dict[str, float]] = {}
    for key in metric_keys:
        vals = []
        for r in rows:
            if key not in r or r[key] is None:
                continue
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[key] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}
    return out
