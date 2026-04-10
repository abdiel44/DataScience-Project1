"""Tests for subject-wise CV (Phase E3)."""

import numpy as np
import pandas as pd
import pytest

from modeling.cv_split import SubjectFoldConfig, list_subject_ids, subject_wise_fold_indices


def test_subject_wise_no_subject_leakage() -> None:
    df = pd.DataFrame(
        {
            "sub": ["a", "a", "b", "b", "c", "c"],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [0, 0, 1, 1, 0, 1],
        }
    )
    y = df["y"]
    cfg = SubjectFoldConfig(n_splits=3, random_state=0, stratify=True)
    folds = list(subject_wise_fold_indices(df, subject_col="sub", y=y, config=cfg))
    assert len(folds) == 3
    all_train_subs = set()
    for tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        train_subs = set(df.iloc[tr]["sub"].unique())
        test_subs = set(df.iloc[te]["sub"].unique())
        assert train_subs.isdisjoint(test_subs)
        all_train_subs |= train_subs
    assert all_train_subs == {"a", "b", "c"}


def test_list_subject_ids() -> None:
    df = pd.DataFrame({"sub": ["x", "x", "y"], "v": [1, 2, 3]})
    assert list_subject_ids(df, "sub") == ["x", "y"]


def test_missing_subject_col_raises() -> None:
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        next(subject_wise_fold_indices(df, subject_col="missing", y=[0], config=SubjectFoldConfig(n_splits=2)))
