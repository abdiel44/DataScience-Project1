"""Subject-wise cross-validation splits (PRD Phase E3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass(frozen=True)
class SubjectFoldConfig:
    n_splits: int = 5
    random_state: int = 42
    stratify: bool = True
    shuffle: bool = True


def _groups_from_frame(df: pd.DataFrame, subject_col: str) -> np.ndarray:
    if subject_col not in df.columns:
        raise KeyError(f"Column '{subject_col}' not in dataframe.")
    return pd.factorize(df[subject_col].astype(str))[0].astype(np.int64)


def subject_wise_fold_indices(
    df: pd.DataFrame,
    *,
    subject_col: str,
    y: Union[pd.Series, np.ndarray, Sequence[Any]],
    config: Optional[SubjectFoldConfig] = None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield train/test row indices for each fold so that **no subject appears in both** train and test.

    Uses StratifiedGroupKFold when `config.stratify` is True (multiclass/binary labels),
    otherwise GroupKFold.

    Parameters
    ----------
    df : DataFrame whose rows align with y (same length / order).
    subject_col : Column identifying subject / recording (any dtype; factorized internally).
        Use ``subject_unit_id`` after :func:`modeling.subject_id.ensure_subject_unit_column`, or
        ``record_id`` / ``recording_id`` when present.
    y : Target labels aligned with df rows.
    config : Fold settings (n_splits, random_state, stratify, shuffle).
    """
    cfg = config or SubjectFoldConfig()
    if len(df) != len(np.asarray(y)):
        raise ValueError("df and y must have the same number of rows.")

    groups = _groups_from_frame(df, subject_col)
    y_arr = np.asarray(y)

    if cfg.stratify:
        cv = StratifiedGroupKFold(
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            random_state=cfg.random_state,
        )
        split_iter = cv.split(np.zeros(len(df)), y_arr, groups)
    else:
        cv = GroupKFold(n_splits=cfg.n_splits)
        split_iter = cv.split(np.zeros(len(df)), y_arr, groups)

    for train_idx, test_idx in split_iter:
        yield np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def list_subject_ids(df: pd.DataFrame, subject_col: str) -> List[Any]:
    """Unique subject values in column order of appearance."""
    if subject_col not in df.columns:
        raise KeyError(f"Column '{subject_col}' not in dataframe.")
    return df[subject_col].drop_duplicates().tolist()
