from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modeling.classic_multitarget_runner import run_cross_dataset, run_cv
from modeling.epoch_store import export_epoch_store_features, materialize_epoch_store


def _tiny_multitarget_df() -> pd.DataFrame:
    rows = []
    stage_cycle = ["W", "N1", "N2", "N3", "REM"]
    for subj_idx in range(6):
        for epoch_idx in range(4):
            rows.append(
                {
                    "dataset_id": "mit_bih_psg",
                    "subject_unit_id": f"s{subj_idx}",
                    "recording_id": f"rec{subj_idx}",
                    "epoch_index": epoch_idx,
                    "epoch_start_sec": float(epoch_idx * 3),
                    "epoch_end_sec": float((epoch_idx + 1) * 3),
                    "source_file": f"rec{subj_idx}.edf",
                    "eeg_channel_standardized": "EEG",
                    "apnea_binary": int((subj_idx + epoch_idx) % 2),
                    "sleep_stage": stage_cycle[(subj_idx + epoch_idx) % len(stage_cycle)],
                }
            )
    return pd.DataFrame(rows)


def _signal_loader(row, _raw_root: Path, _cfg):
    rec_idx = int(str(row["recording_id"]).replace("rec", ""))
    t = np.linspace(0, 1, 120, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * (rec_idx + 1) * t).astype(np.float32), 10.0


def _classic_cfg(tmp_path: Path) -> dict:
    return {
        "experiment_name": "classic_multitarget_smoke",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "recording_column": "recording_id",
        "targets": {"stage_column": "sleep_stage", "apnea_column": "apnea_binary"},
        "feature_include": ["eeg_mean", "eeg_std", "eeg_var", "eeg_rms"],
        "feature_exclude": [],
        "cv": {"n_splits": 3, "shuffle": True},
        "output": {"root": str(tmp_path / "exp"), "save_models": True, "save_fold_models": True, "save_final_model": True},
        "models": {"svm_rbf": False, "random_forest": True, "xgboost": False},
        "hyperparams": {"random_forest": {"n_estimators": 10, "max_depth": 3, "n_jobs": 1}},
    }


def test_classic_multitarget_run_cv_smoke(tmp_path: Path) -> None:
    base_df = _tiny_multitarget_df()
    manifest = materialize_epoch_store(
        base_df,
        store_root=tmp_path / "store",
        manifest_path=tmp_path / "manifest.parquet",
        raw_root=tmp_path,
        dataset_cfg={"sample_hz": 10, "epoch_seconds": 3},
        signal_loader=_signal_loader,
    )
    feat_df = export_epoch_store_features(manifest, store_root=tmp_path / "store", output_path=tmp_path / "features.parquet")
    cfg = _classic_cfg(tmp_path)
    out = tmp_path / "exp" / "classic_multitarget_smoke"
    run_cv(cfg, feat_df, out)

    metrics = pd.read_csv(out / "metrics_per_fold.csv")
    assert "apnea_auc_roc" in metrics.columns
    assert "stage_cohen_kappa" in metrics.columns
    assert (out / "fold_assignments.csv").is_file()
    assert (out / "predictions_apnea" / "random_forest_fold0.csv").is_file()
    assert (out / "predictions_stage" / "random_forest_fold0.csv").is_file()
    assert (out / "models" / "apnea" / "random_forest_final.joblib").is_file()
    assert (out / "models" / "stage" / "random_forest_final.joblib").is_file()


def test_classic_multitarget_run_cross_dataset_smoke(tmp_path: Path) -> None:
    base_df = _tiny_multitarget_df()
    manifest = materialize_epoch_store(
        base_df,
        store_root=tmp_path / "store",
        manifest_path=tmp_path / "manifest.parquet",
        raw_root=tmp_path,
        dataset_cfg={"sample_hz": 10, "epoch_seconds": 3},
        signal_loader=_signal_loader,
    )
    feat_df = export_epoch_store_features(manifest, store_root=tmp_path / "store", output_path=tmp_path / "features.parquet")
    train_df = feat_df[feat_df["subject_unit_id"].isin(["s0", "s1", "s2", "s3"])].copy()
    train_df["dataset_id"] = "mit_bih_psg"
    eval_df = feat_df[feat_df["subject_unit_id"].isin(["s4", "s5"])].copy()
    eval_df["dataset_id"] = "st_vincent_apnea"

    cfg = _classic_cfg(tmp_path)
    cfg["cross_dataset"] = True
    out = tmp_path / "exp" / "classic_multitarget_cross"
    run_cross_dataset(cfg, train_df, eval_df, out)

    metrics = pd.read_csv(out / "metrics_cross_eval.csv")
    assert "apnea_sensitivity" in metrics.columns
    assert "stage_macro_f1" in metrics.columns
    assert (out / "predictions_apnea" / "random_forest_cross_eval.csv").is_file()
    assert (out / "predictions_stage" / "random_forest_cross_eval.csv").is_file()
