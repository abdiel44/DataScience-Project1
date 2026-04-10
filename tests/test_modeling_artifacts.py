"""Tests for saving predictions and confusion matrix figures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modeling.artifacts import save_confusion_matrix_figure, save_predictions_dataframe


def test_save_predictions_dataframe(tmp_path: Path) -> None:
    p = save_predictions_dataframe(
        tmp_path / "pred.csv",
        y_true=[0, 1],
        y_pred=[0, 0],
        subject_id=["s1", "s2"],
        fold_id=2,
    )
    df = pd.read_csv(p)
    assert list(df.columns) == ["y_true", "y_pred", "subject_id", "fold_id"]
    assert df["fold_id"].iloc[0] == 2


def test_save_confusion_matrix_figure(tmp_path: Path) -> None:
    out = tmp_path / "cm.png"
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    path = save_confusion_matrix_figure(y_true, y_pred, out, labels=[0, 1], title="t")
    assert path.exists()
    assert path.stat().st_size > 0
