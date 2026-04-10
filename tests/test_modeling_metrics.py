"""Tests for modeling.metrics (Phase E4 helpers)."""

import numpy as np
import pytest

from modeling.metrics import apnea_binary_metrics, mcnemar_exact, multiclass_sleep_metrics


def test_apnea_binary_metrics_basic() -> None:
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    m = apnea_binary_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.6)
    assert "sensitivity" in m and "specificity" in m
    assert m["auc_roc"] is None


def test_apnea_binary_metrics_with_auc() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    m = apnea_binary_metrics(y_true, y_pred, y_score_positive=scores)
    assert m["auc_roc"] is not None
    assert 0.0 <= m["auc_roc"] <= 1.0


def test_multiclass_sleep_metrics() -> None:
    y_true = ["W", "N1", "N2", "N2", "REM"]
    y_pred = ["W", "N2", "N2", "N2", "REM"]
    m = multiclass_sleep_metrics(y_true, y_pred)
    assert m["macro_f1"] > 0
    assert "N1" in m["per_class_f1"] or any("N1" in k for k in m["per_class_f1"])


def test_mcnemar_exact_runs() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    a = np.array([0, 1, 0, 0, 0, 1])
    b = np.array([0, 1, 1, 1, 0, 1])
    stat, p = mcnemar_exact(y_true, a, b)
    assert np.isfinite(stat)
    assert 0.0 <= p <= 1.0
