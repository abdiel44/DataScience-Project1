"""Smoke tests for modeling.train_runner (Phase E)."""

from pathlib import Path

import pandas as pd
import pytest
from modeling.train_runner import (
    resolve_csv_path,
    resolve_feature_columns,
    resolve_feature_columns_cross,
    run_cross_dataset,
    run_cv,
    run_experiment,
)


@pytest.fixture()
def tiny_cv_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["A", "B", "A", "B"],
            "f1": [1.0, 2.0, 1.5, 2.5],
            "f2": [0.5, 1.0, 0.7, 1.2],
        }
    )
    p = tmp_path / "train.csv"
    df.to_csv(p, index=False)
    return p


def test_resolve_feature_columns_include_only_listed() -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": [1, 1],
            "y": [0, 1],
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "f3": [5.0, 6.0],
        }
    )
    ex = {"subject_unit_id", "y"}
    cols = resolve_feature_columns(df, ex, ["f2"])
    assert cols == ["f2"]


def test_resolve_csv_path_next_to_config_file(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    csv_path = cfg_dir / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)
    found = resolve_csv_path("data.csv", cfg_dir / "exp.yaml")
    assert found == csv_path.resolve()


def test_run_cv_rejects_multiple_epoch_signal_stems(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["A", "B", "A", "B"],
            "eeg_a_mean": [1.0, 2.0, 1.5, 2.5],
            "eeg_a_std": [0.1, 0.2, 0.15, 0.25],
            "eeg_b_mean": [3.0, 4.0, 3.5, 4.5],
            "eeg_b_std": [0.3, 0.4, 0.35, 0.45],
        }
    )
    cfg = {
        "experiment_name": "mc",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_exclude": [],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    with pytest.raises(ValueError, match="Several epoch feature"):
        run_cv(cfg, df, tmp_path / "exp" / "mc")


def test_run_cv_allows_multi_channel_when_flag_set(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["A", "B", "A", "B"],
            "eeg_a_mean": [1.0, 2.0, 1.5, 2.5],
            "eeg_a_std": [0.1, 0.2, 0.15, 0.25],
            "eeg_b_mean": [3.0, 4.0, 3.5, 4.5],
            "eeg_b_std": [0.3, 0.4, 0.35, 0.45],
        }
    )
    cfg = {
        "experiment_name": "mc2",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_exclude": [],
        "allow_multi_channel_features": True,
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "mc2"
    run_cv(cfg, df, out)
    assert (out / "metrics_per_fold.csv").is_file()


def test_run_cv_binary_rejects_non_zero_one_labels(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["low", "high", "low", "high"],
            "f1": [1.0, 2.0, 1.5, 2.5],
        }
    )
    cfg = {
        "experiment_name": "bin_bad",
        "random_seed": 0,
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "binary",
        "binary_require_zero_one_labels": True,
        "feature_exclude": [],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    with pytest.raises(ValueError, match="task=binary requires"):
        run_cv(cfg, df, tmp_path / "exp" / "bin_bad")


def test_run_cv_binary_zero_one_numeric_ok(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": [0, 1, 0, 1],
            "f1": [1.0, 2.0, 1.5, 2.5],
        }
    )
    cfg = {
        "experiment_name": "bin01",
        "random_seed": 0,
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "binary",
        "feature_exclude": [],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "bin01"
    run_cv(cfg, df, out)
    assert (out / "metrics_per_fold.csv").is_file()


def test_run_cv_binary_relaxed_two_label_strings(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["low", "high", "low", "high"],
            "f1": [1.0, 2.0, 1.5, 2.5],
        }
    )
    cfg = {
        "experiment_name": "bin_relax",
        "random_seed": 0,
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "binary",
        "binary_require_zero_one_labels": False,
        "feature_exclude": [],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "bin_relax"
    run_cv(cfg, df, out)
    assert (out / "metrics_per_fold.csv").is_file()


def test_resolve_feature_columns_cross_include_both_frames() -> None:
    train_df = pd.DataFrame({"s": [1], "y": [0], "a": [1.0], "b": [2.0]})
    eval_df = pd.DataFrame({"s": [2], "y": [1], "a": [3.0], "b": [4.0]})
    ex = ["s", "y"]
    cols = resolve_feature_columns_cross(train_df, eval_df, ex, ["a"])
    assert cols == ["a"]


def test_run_cv_smoke(tiny_cv_csv: Path, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "smoke",
        "random_seed": 0,
        "train_csv": str(tiny_cv_csv),
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_exclude": [],
        "feature_include": ["f1"],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "smoke"
    run_cv(cfg, pd.read_csv(tiny_cv_csv), out)
    assert (out / "metrics_per_fold.csv").is_file()
    assert (out / "summary.json").is_file()
    mdf = pd.read_csv(out / "metrics_per_fold.csv")
    assert len(mdf) == 2
    assert (out / "predictions" / "random_forest_fold0.csv").is_file()


def test_run_cross_dataset_smoke(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "subject_unit_id": ["a", "a", "b"],
            "y": [0, 1, 0],
            "f1": [1.0, 2.0, 1.5],
            "f2": [0.1, 0.2, 0.15],
        }
    )
    ev = pd.DataFrame(
        {
            "subject_unit_id": ["c", "c"],
            "y": [1, 0],
            "f1": [2.0, 1.0],
            "f2": [0.2, 0.1],
        }
    )
    cfg = {
        "random_seed": 0,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "binary",
        "feature_exclude": [],
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "cross_out"
    run_cross_dataset(cfg, train, ev, out)
    assert (out / "metrics_cross_eval.csv").is_file()
    assert (out / "predictions" / "random_forest_cross_eval.csv").is_file()


@pytest.mark.slow
def test_run_cv_all_three_models_smoke(tiny_cv_csv: Path, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "smoke_all_models",
        "random_seed": 0,
        "train_csv": str(tiny_cv_csv),
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_exclude": [],
        "feature_include": ["f1"],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": True, "random_forest": True, "xgboost": True},
        "hyperparams": {
            "random_forest": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
            "xgboost": {"n_estimators": 5, "max_depth": 2, "n_jobs": 1},
        },
    }
    out = tmp_path / "exp" / "smoke_all_models"
    run_cv(cfg, pd.read_csv(tiny_cv_csv), out)
    mdf = pd.read_csv(out / "metrics_per_fold.csv")
    assert set(mdf["model"].unique()) >= {"random_forest"}


def test_run_experiment_from_yaml_file(tiny_cv_csv: Path, tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "smoke_phase_e.yaml"
    text = fixture.read_text(encoding="utf-8")
    # Replace longest token first — PLACEHOLDER is a prefix of PLACEHOLDER_OUT
    text = text.replace("PLACEHOLDER_OUT", str(tmp_path / "exp_root").replace("\\", "/"))
    text = text.replace("PLACEHOLDER", str(tiny_cv_csv).replace("\\", "/"))
    ypath = tmp_path / "run.yaml"
    ypath.write_text(text, encoding="utf-8")
    run_experiment(ypath)
    out = tmp_path / "exp_root" / "_smoke_phase_e"
    assert (out / "metrics_per_fold.csv").exists()
