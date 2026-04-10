"""Save predictions and figures for Phase E3 (CSV + PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_predictions_dataframe(
    path: Union[str, Path],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    y_score: Optional[Sequence[float]] = None,
    subject_id: Optional[Sequence[Any]] = None,
    fold_id: Optional[int] = None,
    extra_columns: Optional[dict[str, Sequence[Any]]] = None,
) -> Path:
    """
    Write per-sample predictions for the informe / estadística (PRD: CSV).

    Columns: y_true, y_pred, optional y_score, subject_id, fold_id, plus any extra_columns.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }
    if y_score is not None:
        data["y_score"] = np.asarray(y_score)
    if subject_id is not None:
        data["subject_id"] = np.asarray(subject_id)
    if fold_id is not None:
        data["fold_id"] = fold_id
    if extra_columns:
        for k, v in extra_columns.items():
            data[k] = np.asarray(v)
    pd.DataFrame(data).to_csv(out, index=False)
    return out


def save_confusion_matrix_figure(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    path: Union[str, Path],
    *,
    labels: Optional[Sequence[Any]] = None,
    title: Optional[str] = None,
) -> Path:
    """Save confusion matrix PNG (normalized or counts — sklearn default counts)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    lab = np.asarray(labels) if labels is not None else np.unique(np.concatenate([yt, yp]))
    cm = confusion_matrix(yt, yp, labels=lab)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lab)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax)
    if title:
        ax.set_title(title)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out
