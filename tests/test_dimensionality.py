import numpy as np
import pandas as pd
import pytest

from src.dimensionality import (
    DimensionalityOptions,
    apply_dimensionality,
    dimensionality_options_from_dict,
    write_dimensionality_report,
)


def test_pca_column_count_and_variance() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    out, rep = apply_dimensionality(
        df,
        DimensionalityOptions(method="pca", pca_n_components=2, random_state=0),
    )
    assert [c for c in out.columns if c.startswith("pca_")] == ["pca_1", "pca_2"]
    assert len(rep.explained_variance_ratio) == 2
    assert 0.99 >= sum(rep.explained_variance_ratio) > 0


def test_lda_reduces_features_multiclass() -> None:
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
            "y": np.repeat([0, 1, 2], n // 3),
        }
    )
    out, rep = apply_dimensionality(
        df,
        DimensionalityOptions(method="lda", target_column="y", lda_n_components=None),
    )
    lda_cols = [c for c in out.columns if c.startswith("lda_")]
    assert len(lda_cols) <= 2
    assert len(lda_cols) < 3
    assert "y" in out.columns
    assert rep.method == "lda"


def test_variance_threshold_drops_constant() -> None:
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0]})
    out, rep = apply_dimensionality(
        df,
        DimensionalityOptions(method="variance_threshold", variance_threshold_value=0.0),
    )
    assert "a" not in out.columns
    assert "b" in out.columns
    assert "a" in rep.dropped_constant_features


def test_select_k_best_k_less_than_n_features() -> None:
    rng = np.random.default_rng(2)
    n = 40
    df = pd.DataFrame(
        {
            "f0": rng.standard_normal(n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
            "y": np.repeat([0, 1], n // 2),
        }
    )
    out, rep = apply_dimensionality(
        df,
        DimensionalityOptions(method="select_k_best", target_column="y", select_k=2, select_score_func="f_classif"),
    )
    feat_cols = [c for c in out.columns if c.startswith("f")]
    assert len(feat_cols) == 2
    assert len(rep.selected_feature_scores) == 2


def test_lda_without_target_raises() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
    with pytest.raises(ValueError, match="target_column"):
        apply_dimensionality(df, DimensionalityOptions(method="lda", target_column=None))


def test_lda_continuous_many_uniques_raises() -> None:
    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n), "y": rng.standard_normal(n)})
    with pytest.raises(ValueError, match="classification"):
        apply_dimensionality(df, DimensionalityOptions(method="lda", target_column="y"))


def test_dimensionality_options_from_dict() -> None:
    opt = dimensionality_options_from_dict(
        {
            "dimensionality_method": "pca",
            "pca_n_components": 3,
        }
    )
    assert opt is not None
    assert opt.method == "pca"
    assert opt.pca_n_components == 3


def test_write_dimensionality_report(tmp_path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out, rep = apply_dimensionality(df, DimensionalityOptions(method="pca", pca_n_components=1))
    p = tmp_path / "d.md"
    write_dimensionality_report(p, report=rep, options=DimensionalityOptions(method="pca", pca_n_components=1))
    text = p.read_text(encoding="utf-8")
    assert "Topic 6" in text
    assert "PCA" in text
