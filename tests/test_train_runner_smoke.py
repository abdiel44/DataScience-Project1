"""Smoke tests for modeling.train_runner (Phase E)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from modeling.train_runner import (
    _apply_train_resampling,
    _tuning_train_subject_subsample,
    cast_feature_frame_float32,
    make_model,
    read_table_file,
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


@pytest.fixture()
def tiny_cv_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["A", "B", "A", "B"],
            "f1": [1.0, 2.0, 1.5, 2.5],
            "f2": [0.5, 1.0, 0.7, 1.2],
        }
    )
    p = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
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


def test_read_table_file_parquet_matches_csv(tiny_cv_csv: Path, tiny_cv_parquet: Path) -> None:
    csv_df = read_table_file(tiny_cv_csv)
    parquet_df = read_table_file(tiny_cv_parquet)
    pd.testing.assert_frame_equal(csv_df, parquet_df)


def test_cast_feature_frame_float32_casts_all_columns() -> None:
    df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.5, 4.5]}, dtype="float64")
    out = cast_feature_frame_float32(df)
    assert list(out.dtypes.astype(str)) == ["float32", "float32"]


def test_make_model_svm_exact_hyperparams_are_forwarded() -> None:
    model = make_model(
        "svm_rbf",
        random_state=7,
        hyperparams={
            "svm_rbf": {
                "C": 2.0,
                "gamma": 0.01,
                "class_weight": "balanced",
                "cache_size": 2048,
                "tol": 0.01,
                "shrinking": True,
                "max_iter": 1234,
            }
        },
        task="multiclass",
    )
    assert model.cache_size == 2048
    assert model.tol == 0.01
    assert model.shrinking is True
    assert model.max_iter == 1234
    assert model.probability is False


def test_apply_train_resampling_smote_to_reference_minus() -> None:
    X = np.array(
        [
            [0.0],
            [0.1],
            [1.0],
            [1.1],
            [1.2],
            [1.3],
            [1.4],
            [1.5],
        ],
        dtype=np.float32,
    )
    y = np.array(["W", "W", "N2", "N2", "N2", "N2", "N2", "N2"])
    cfg = {
        "random_seed": 0,
        "train_resampling": {
            "enabled": True,
            "method": "smote_to_reference_minus",
            "reference_class": "N2",
            "reference_offset": 2,
            "target_labels": ["W"],
            "k_neighbors": 1,
            "random_state": 0,
        },
    }
    X_res, y_res = _apply_train_resampling(cfg=cfg, X_train=X, y_train_raw=y)
    counts = pd.Series(y_res).value_counts().to_dict()
    assert counts["W"] == 4
    assert counts["N2"] == 6
    assert X_res.shape[0] == len(y_res)


def test_tuning_train_subject_subsample_preserves_class_coverage() -> None:
    rows = []
    labels = ["N1", "N2", "N3"]
    for idx in range(40):
        label = labels[idx % len(labels)]
        rows.append({"subject_unit_id": f"s{idx:02d}", "sleep_stage": label, "eeg_mean": float(idx)})
        rows.append({"subject_unit_id": f"s{idx:02d}", "sleep_stage": label, "eeg_mean": float(idx) + 0.1})
    df = pd.DataFrame(rows)
    cfg = {
        "tuning": {
            "enabled": True,
            "mode": "nested_cv",
            "inner_cv_splits": 3,
            "train_subject_subsample": {"enabled": True, "fraction": 0.25, "min_subjects": 32},
        }
    }
    sampled = _tuning_train_subject_subsample(
        cfg=cfg,
        model_name="svm_rbf",
        seed=13,
        train_df=df,
        subject_col="subject_unit_id",
        target_col="sleep_stage",
        inner_splits=3,
    )
    assert sampled["subject_unit_id"].nunique() == 32
    assert set(sampled["sleep_stage"].astype(str)) == {"N1", "N2", "N3"}


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


def test_run_cv_saves_fold_and_final_models(tiny_cv_csv: Path, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "smoke_saved_models",
        "random_seed": 0,
        "train_csv": str(tiny_cv_csv),
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_exclude": [],
        "feature_include": ["f1"],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {
            "root": str(tmp_path / "exp"),
            "save_models": True,
            "save_fold_models": True,
            "save_final_model": True,
        },
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "smoke_saved_models"
    train_df = pd.read_csv(tiny_cv_csv)
    run_cv(cfg, train_df, out, train_csv_path=tiny_cv_csv)

    fold_bundle_path = out / "models" / "folds" / "random_forest_fold0.joblib"
    final_bundle_path = out / "models" / "random_forest_final.joblib"
    registry_path = out / "models" / "model_registry.json"
    assert fold_bundle_path.is_file()
    assert final_bundle_path.is_file()
    assert registry_path.is_file()

    bundle = joblib.load(final_bundle_path)
    X = bundle["imputer"].transform(train_df[bundle["feature_columns"]])
    if bundle.get("scaler") is not None:
        X = bundle["scaler"].transform(X)
    pred_enc = bundle["model"].predict(X)
    pred = bundle["label_encoder"].inverse_transform(pred_enc.astype(int))
    assert len(pred) == len(train_df)
    assert bundle["artifact_kind"] == "final"
    assert bundle["class_labels"] == ["A", "B"]
    assert bundle["scaler"] is None


def test_run_cv_svm_bundle_persists_scaler(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "y": ["A", "B", "A", "B"],
            "f1": [0.0, 1.0, 0.1, 1.1],
        }
    )
    cfg = {
        "experiment_name": "svm_saved_models",
        "random_seed": 0,
        "train_csv": "train.csv",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_include": ["f1"],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": True, "random_forest": False, "xgboost": False},
        "hyperparams": {"svm_rbf": {"C": 1.0, "gamma": "scale"}},
    }
    out = tmp_path / "exp" / "svm_saved_models"
    run_cv(cfg, df, out)
    bundle = joblib.load(out / "models" / "svm_rbf_final.joblib")
    assert bundle["scaler"] is not None


def test_run_cv_accepts_one_hot_target_and_reports_n1(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1"],
            "eeg_c4_a1_mean": [1.0, 2.0, 1.5, 2.5],
            "eeg_c4_a1_std": [0.1, 0.2, 0.15, 0.25],
            "sleep_stage_w": [1.0, 0.0, 1.0, 0.0],
            "sleep_stage_1": [0.0, 1.0, 0.0, 1.0],
        }
    )
    cfg = {
        "experiment_name": "one_hot_staging",
        "random_seed": 0,
        "subject_column": "subject_unit_id",
        "target_column": "sleep_stage",
        "task": "multiclass",
        "feature_exclude": [],
        "feature_include": ["eeg_c4_a1_mean", "eeg_c4_a1_std"],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "one_hot_staging"
    run_cv(cfg, df, out)
    mdf = pd.read_csv(out / "metrics_per_fold.csv")
    assert "per_class_f1_n1" in mdf.columns
    pred = pd.read_csv(out / "predictions" / "random_forest_fold0.csv")
    assert set(pred["y_true"].astype(str)).issubset({"W", "N1"})


def test_run_cv_rejects_when_not_enough_subjects(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s0"],
            "y": ["A", "B", "A"],
            "f1": [1.0, 2.0, 1.5],
        }
    )
    cfg = {
        "experiment_name": "too_few_subjects",
        "random_seed": 0,
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
    with pytest.raises(ValueError, match="requires at least 2 unique subjects"):
        run_cv(cfg, df, tmp_path / "exp" / "too_few_subjects")


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


def test_run_cross_dataset_saves_final_model_only(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "subject_unit_id": ["a", "a", "b"],
            "y": [0, 1, 0],
            "f1": [1.0, 2.0, 1.5],
        }
    )
    ev = pd.DataFrame(
        {
            "subject_unit_id": ["c", "c"],
            "y": [1, 0],
            "f1": [2.0, 1.0],
        }
    )
    cfg = {
        "experiment_name": "cross_saved_models",
        "random_seed": 0,
        "train_csv": "train.csv",
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "binary",
        "feature_exclude": [],
        "output": {
            "root": str(tmp_path / "exp"),
            "save_models": True,
            "save_fold_models": True,
            "save_final_model": True,
        },
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "cross_saved_models"
    run_cross_dataset(cfg, train, ev, out)
    assert (out / "models" / "random_forest_final.joblib").is_file()
    assert not (out / "models" / "folds").exists()


def test_run_cv_tuning_writes_best_params_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "subject_unit_id": ["s0", "s0", "s1", "s1", "s2", "s2"],
            "y": ["A", "B", "A", "B", "A", "B"],
            "f1": [1.0, 2.0, 1.1, 2.1, 0.9, 1.9],
        }
    )
    cfg = {
        "experiment_name": "tuned_cv",
        "random_seed": 0,
        "train_csv": "train.csv",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_include": ["f1"],
        "cv": {"n_splits": 3, "stratify": True, "shuffle": True},
        "tuning": {
            "enabled": True,
            "mode": "nested_cv",
            "inner_cv_splits": 2,
            "scoring": "macro_f1",
            "search_method": "grid",
            "search_space": {
                "random_forest": {
                    "n_estimators": [5, 10],
                    "max_depth": [2, 3],
                }
            },
        },
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_jobs": 1}},
    }
    out = tmp_path / "exp" / "tuned_cv"
    run_cv(cfg, df, out)
    assert (out / "best_params_per_fold.csv").is_file()
    assert (out / "best_params_summary.json").is_file()
    reg = pd.read_csv(out / "best_params_per_fold.csv")
    assert set(reg["fold"].astype(str)) >= {"0", "1", "2", "final"}


def test_run_cv_resume_completed_skips_duplicate_rows(tiny_cv_csv: Path, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "resume_cv",
        "random_seed": 0,
        "train_csv": str(tiny_cv_csv),
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "target_column": "y",
        "task": "multiclass",
        "feature_include": ["f1"],
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "output": {
            "root": str(tmp_path / "exp"),
            "save_models": True,
            "save_fold_models": True,
            "save_final_model": True,
            "resume_completed": True,
        },
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "exp" / "resume_cv"
    df = pd.read_csv(tiny_cv_csv)
    run_cv(cfg, df, out, train_csv_path=tiny_cv_csv)
    run_cv(cfg, df, out, train_csv_path=tiny_cv_csv)

    metrics_df = pd.read_csv(out / "metrics_per_fold.csv")
    assert len(metrics_df) == 2
    with (out / "models" / "model_registry.json").open(encoding="utf-8") as f:
        registry = pd.DataFrame.from_records(json.load(f))
    assert len(registry) == 3
    best_df = pd.read_csv(out / "best_params_per_fold.csv")
    assert set(best_df["fold"].astype(str)) == {"0", "1", "final"}


def test_run_cross_dataset_label_subset_filters_eval_labels(tmp_path: Path) -> None:
    train = pd.DataFrame(
        {
            "subject_unit_id": ["a", "a", "b", "b", "c", "c"],
            "sleep_stage": ["N1", "N2", "N1", "N2", "N3", "N3"],
            "eeg_mean": [1.0, 2.0, 1.1, 2.1, 3.0, 3.1],
        }
    )
    ev = pd.DataFrame(
        {
            "subject_unit_id": ["x", "x", "y", "y"],
            "sleep_stage": ["N1", "W", "N2", "W"],
            "eeg_mean": [1.5, 0.5, 2.5, 0.4],
        }
    )
    cfg = {
        "experiment_name": "cross_subset",
        "random_seed": 0,
        "train_csv": "train.csv",
        "subject_column": "subject_unit_id",
        "target_column": "sleep_stage",
        "task": "multiclass",
        "feature_include": ["eeg_mean"],
        "label_subset": ["N1", "N2", "N3"],
        "output": {"root": str(tmp_path / "exp")},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }
    out = tmp_path / "cross_subset"
    run_cross_dataset(cfg, train, ev, out)
    pred = pd.read_csv(out / "predictions" / "random_forest_cross_eval.csv")
    assert set(pred["y_true"].astype(str)) <= {"N1", "N2", "N3"}


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


def test_run_experiment_from_parquet_file(tiny_cv_parquet: Path, tmp_path: Path) -> None:
    ypath = tmp_path / "run_parquet.yaml"
    ypath.write_text(
        "\n".join(
            [
                "experiment_name: parquet_smoke",
                "random_seed: 0",
                f"train_csv: {str(tiny_cv_parquet).replace(chr(92), '/')}",
                "cross_dataset: false",
                "subject_column: subject_unit_id",
                "target_column: y",
                "task: multiclass",
                "feature_include:",
                "  - f1",
                "cv:",
                "  n_splits: 2",
                "  stratify: true",
                "  shuffle: true",
                "output:",
                f"  root: {str(tmp_path / 'exp_root').replace(chr(92), '/')}",
                "models:",
                "  svm_rbf: false",
                "  random_forest: true",
                "  xgboost: false",
                "hyperparams:",
                "  random_forest:",
                "    n_estimators: 10",
                "    max_depth: 3",
                "    n_jobs: 1",
            ]
        ),
        encoding="utf-8",
    )
    run_experiment(ypath)
    out = tmp_path / "exp_root" / "parquet_smoke"
    assert (out / "metrics_per_fold.csv").exists()
