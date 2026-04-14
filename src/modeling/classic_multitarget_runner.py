from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from modeling.artifacts import (
    save_confusion_matrix_figure,
    save_model_bundle,
    save_predictions_dataframe,
    save_roc_curve_figure,
    write_model_registry,
)
from modeling.metrics import apnea_binary_metrics, fold_metrics_summary, multiclass_sleep_metrics
from modeling.subject_id import ensure_subject_unit_column
from modeling.target_utils import normalize_sleep_stage_series
from modeling.train_runner import (
    _fit_model_bundle_only,
    _fit_predict_bundle,
    _nested_best_hyperparams,
    _output_settings,
    _registry_row,
    _train_csv_string,
    _tuning_settings,
    _write_tuning_artifacts,
    cast_feature_frame_float32,
    enforce_single_channel_epoch_features,
    load_config,
    make_model,
    read_table_file,
    resolve_csv_path,
    resolve_feature_columns,
    resolve_feature_columns_cross,
)

_VALID_STAGE_LABELS = {"W", "N1", "N2", "N3", "REM"}
_SUMMARY_METRIC_KEYS = [
    "apnea_accuracy",
    "apnea_sensitivity",
    "apnea_specificity",
    "apnea_auc_roc",
    "stage_accuracy",
    "stage_macro_f1",
    "stage_cohen_kappa",
]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _resume_completed_enabled(cfg: Mapping[str, Any]) -> bool:
    out = cfg.get("output", {}) or {}
    return bool(out.get("resume_completed", False))


def _existing_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _existing_registry_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(row) for row in data]
    return []


def _coerce_fold_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value) and float(value).is_integer():
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    try:
        as_float = float(text)
    except ValueError:
        return None
    if np.isfinite(as_float) and float(as_float).is_integer():
        return int(as_float)
    return None


def _targets_cfg(cfg: Mapping[str, Any]) -> Tuple[str, str]:
    targets = cfg.get("targets", {}) or {}
    if not isinstance(targets, Mapping):
        raise ValueError("targets must be a mapping.")
    stage_col = str(targets.get("stage_column", "sleep_stage"))
    apnea_col = str(targets.get("apnea_column", "apnea_binary"))
    return stage_col, apnea_col


def normalize_classic_multitarget_dataframe(
    df: pd.DataFrame,
    *,
    subject_col: str,
    recording_col: str,
    stage_col: str,
    apnea_col: str,
) -> pd.DataFrame:
    out = ensure_subject_unit_column(df.copy(), output_col=subject_col, overwrite=False)
    if subject_col not in out.columns:
        raise KeyError(f"Column {subject_col!r} not in dataframe.")
    if recording_col not in out.columns:
        raise KeyError(f"Column {recording_col!r} not in dataframe.")
    if stage_col not in out.columns:
        raise KeyError(f"Column {stage_col!r} not in dataframe.")
    if apnea_col not in out.columns:
        raise KeyError(f"Column {apnea_col!r} not in dataframe.")

    out[subject_col] = out[subject_col].astype(str)
    out[recording_col] = out[recording_col].astype(str)
    if "dataset_id" not in out.columns:
        out["dataset_id"] = "unknown"
    out["dataset_id"] = out["dataset_id"].astype(str)
    out[apnea_col] = pd.to_numeric(out[apnea_col], errors="coerce")
    out[stage_col] = normalize_sleep_stage_series(out[stage_col])
    out["label_mask_apnea"] = out[apnea_col].notna().astype(int)
    out["label_mask_stage"] = out[stage_col].astype(str).isin(_VALID_STAGE_LABELS).astype(int)
    out.loc[out["label_mask_stage"] == 0, stage_col] = pd.NA
    if int(out["label_mask_apnea"].sum()) <= 0:
        raise ValueError("No apnea labels available after normalization.")
    return out.reset_index(drop=True)


def _subject_profiles(
    df: pd.DataFrame,
    *,
    subject_col: str,
    stage_col: str,
    apnea_col: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subject_id, group in df.groupby(subject_col, sort=False):
        apnea_vals = pd.to_numeric(group[apnea_col], errors="coerce").dropna()
        apnea_rate = float(apnea_vals.mean()) if not apnea_vals.empty else 0.0
        stage_vals = group[stage_col].dropna().astype(str)
        dominant_stage = str(stage_vals.value_counts().index[0]) if not stage_vals.empty else "UNK"
        rows.append(
            {
                "subject_unit_id": str(subject_id),
                "apnea_rate": apnea_rate,
                "dominant_stage": dominant_stage,
                "n_rows": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No subject profiles available.")
    n_unique = int(out["apnea_rate"].nunique())
    q = min(3, n_unique) if n_unique > 1 else 1
    out["apnea_bin"] = pd.qcut(out["apnea_rate"], q=q, duplicates="drop").astype(str) if q > 1 else "all"
    out["composite_label"] = out["apnea_bin"].astype(str) + "__" + out["dominant_stage"].astype(str)
    return out


def _best_stratify_label(subject_profiles: pd.DataFrame, n_splits: int) -> Tuple[Optional[pd.Series], str]:
    candidates = [
        ("composite_subject_stratified", subject_profiles["composite_label"].astype(str)),
        ("apnea_bin_only", subject_profiles["apnea_bin"].astype(str)),
        ("dominant_stage_only", subject_profiles["dominant_stage"].astype(str)),
    ]
    for strategy, series in candidates:
        counts = series.value_counts()
        if not counts.empty and int(counts.min()) >= int(n_splits):
            return series, strategy
    return None, "group_only"


def build_shared_fold_assignments(
    df: pd.DataFrame,
    *,
    subject_col: str,
    stage_col: str,
    apnea_col: str,
    n_splits: int,
    random_state: int,
    shuffle: bool,
) -> pd.DataFrame:
    subject_profiles = _subject_profiles(df, subject_col=subject_col, stage_col=stage_col, apnea_col=apnea_col)
    y_subject, strategy_used = _best_stratify_label(subject_profiles, n_splits)
    groups = subject_profiles["subject_unit_id"].astype(str).to_numpy()
    X = np.zeros((len(subject_profiles), 1), dtype=np.float32)
    if y_subject is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_iter = splitter.split(X, y_subject.to_numpy(), groups)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, np.zeros(len(subject_profiles), dtype=int), groups)
    subject_profiles = subject_profiles.copy()
    subject_profiles["fold"] = -1
    for fold_id, (_train_idx, test_idx) in enumerate(split_iter):
        subject_profiles.loc[test_idx, "fold"] = int(fold_id)
    if (subject_profiles["fold"] < 0).any():
        raise RuntimeError("Shared fold assignment left subjects without a fold.")
    fold_map = dict(zip(subject_profiles["subject_unit_id"].astype(str), subject_profiles["fold"].astype(int)))
    out = df.copy()
    out["fold"] = out[subject_col].astype(str).map(fold_map).astype(int)
    out["fold_strategy_used"] = strategy_used
    return out


def _feature_columns(df: pd.DataFrame, cfg: Mapping[str, Any], *, stage_col: str, apnea_col: str, subject_col: str) -> List[str]:
    exclude = {
        subject_col,
        str(cfg.get("recording_column", "recording_id")),
        stage_col,
        apnea_col,
        "fold",
        "label_mask_apnea",
        "label_mask_stage",
        *cfg.get("feature_exclude", []),
    }
    feat_cols = resolve_feature_columns(df, exclude, cfg.get("feature_include"))
    if not feat_cols:
        raise ValueError("No numeric feature columns after exclusions / feature_include.")
    if not any(str(col).startswith("eeg_") for col in feat_cols):
        raise ValueError("Classic multitarget requires shared EEG feature columns with prefix 'eeg_'.")
    enforce_single_channel_epoch_features(feat_cols, dict(cfg))
    return feat_cols


def _feature_columns_cross(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    stage_col: str,
    apnea_col: str,
    subject_col: str,
) -> List[str]:
    exclude = [
        subject_col,
        str(cfg.get("recording_column", "recording_id")),
        stage_col,
        apnea_col,
        "fold",
        "label_mask_apnea",
        "label_mask_stage",
        *cfg.get("feature_exclude", []),
    ]
    feat_cols = resolve_feature_columns_cross(train_df, eval_df, exclude, cfg.get("feature_include"))
    if not feat_cols:
        raise ValueError("No overlapping numeric feature columns between train/eval after exclusions.")
    if not any(str(col).startswith("eeg_") for col in feat_cols):
        raise ValueError("Cross-dataset classic multitarget requires shared EEG feature columns with prefix 'eeg_'.")
    enforce_single_channel_epoch_features(feat_cols, dict(cfg))
    return feat_cols


def _enabled_models(cfg: Mapping[str, Any]) -> List[str]:
    models_on = cfg.get("models", {}) or {}
    names: List[str] = []
    if models_on.get("random_forest", True):
        names.append("random_forest")
    if models_on.get("xgboost", True):
        names.append("xgboost")
    if models_on.get("svm_rbf", True):
        names.append("svm_rbf")
    if not names:
        raise ValueError("No models enabled under config.models")
    return names


def _safe_stage_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    mm = multiclass_sleep_metrics(y_true, y_pred)
    out = {
        "stage_accuracy": mm["accuracy"],
        "stage_macro_f1": mm["macro_f1"],
        "stage_cohen_kappa": mm["cohen_kappa"],
    }
    for label, score in mm["per_class_f1"].items():
        out[f"stage_per_class_f1_{str(label).lower()}"] = score
    return out


def _safe_apnea_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Optional[Sequence[float]]) -> Dict[str, Any]:
    mm = apnea_binary_metrics(y_true, y_pred, y_score_positive=y_score)
    return {
        "apnea_accuracy": mm["accuracy"],
        "apnea_sensitivity": mm["sensitivity"],
        "apnea_specificity": mm["specificity"],
        "apnea_auc_roc": mm["auc_roc"],
        "apnea_n_samples": mm["n_samples"],
    }


def _write_summary(out_dir: Path, rows: Sequence[Mapping[str, Any]], names: Sequence[str], *, cross_dataset: bool) -> None:
    summary = {"models": list(names)}
    if cross_dataset:
        summary["aggregate"] = fold_metrics_summary(rows, _SUMMARY_METRIC_KEYS)
    else:
        aggregate: Dict[str, Any] = {}
        for model_name in names:
            model_rows = [dict(r) for r in rows if str(r.get("model")) == model_name]
            if model_rows:
                aggregate[model_name] = fold_metrics_summary(model_rows, _SUMMARY_METRIC_KEYS)
        summary["aggregate"] = aggregate
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _task_df(df: pd.DataFrame, *, target_col: str, mask_col: str) -> pd.DataFrame:
    out = df[df[mask_col].astype(int) > 0].copy()
    if out.empty:
        raise ValueError(f"No rows available for task target {target_col!r}.")
    return out


def _fit_eval_task(
    *,
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    task: str,
    target_col: str,
    subject_col: str,
    feat_cols: Sequence[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[Any, Any, Any, Any, np.ndarray, Optional[np.ndarray], Dict[str, Any], Optional[float]]:
    tuned_cfg = _task_tuned_cfg(cfg, task=task)
    selected_hp, inner_best_score = _nested_best_hyperparams(
        cfg=tuned_cfg,
        model_name=model_name,
        seed=seed,
        task=task,
        subject_col=subject_col,
        feat_cols=feat_cols,
        train_df=train_df.copy(),
        target_col=target_col,
    )
    model, imputer, scaler, le, pred, y_score = _fit_predict_bundle(
        cfg=tuned_cfg,
        model_name=model_name,
        seed=seed,
        hyperparams={model_name: selected_hp},
        X_train_df=cast_feature_frame_float32(train_df[list(feat_cols)]),
        X_test_df=cast_feature_frame_float32(test_df[list(feat_cols)]),
        y_train_raw=train_df[target_col].values,
        y_test_raw=test_df[target_col].values,
        task=task,
    )
    return model, imputer, scaler, le, pred, y_score, selected_hp, inner_best_score


def _task_tuned_cfg(cfg: Mapping[str, Any], *, task: str) -> Dict[str, Any]:
    tuned_cfg = dict(cfg)
    tuning_cfg = dict(cfg.get("tuning", {}) or {})
    if tuning_cfg:
        scoring = str(tuning_cfg.get("scoring", "macro_f1")).strip().lower()
        if task == "binary" and scoring not in {"accuracy", "sensitivity", "specificity", "auc_roc"}:
            tuning_cfg["scoring"] = "auc_roc"
        elif task == "multiclass" and scoring not in {"accuracy", "macro_f1", "cohen_kappa"}:
            tuning_cfg["scoring"] = "macro_f1"
        tuned_cfg["tuning"] = tuning_cfg
    return tuned_cfg


def _completed_fold_ids_from_metrics(metrics_rows: Sequence[Mapping[str, Any]], *, model_name: str) -> set[int]:
    completed: set[int] = set()
    for row in metrics_rows:
        if str(row.get("model")) != model_name:
            continue
        fold_id = _coerce_fold_id(row.get("fold"))
        if fold_id is not None:
            completed.add(fold_id)
    return completed


def _fold_artifacts_complete(out_dir: Path, *, model_name: str, fold_id: int) -> bool:
    required = [
        out_dir / "predictions_stage" / f"{model_name}_fold{fold_id}.csv",
        out_dir / "predictions_apnea" / f"{model_name}_fold{fold_id}.csv",
        out_dir / "figures" / f"cm_stage_{model_name}_fold{fold_id}.png",
        out_dir / "figures" / f"cm_apnea_{model_name}_fold{fold_id}.png",
    ]
    return all(path.is_file() for path in required)


def _final_artifacts_complete(out_dir: Path, *, model_name: str) -> bool:
    required = [
        out_dir / "models" / "stage" / f"{model_name}_final.joblib",
        out_dir / "models" / "apnea" / f"{model_name}_final.joblib",
    ]
    return all(path.is_file() for path in required)


def _recover_metrics_rows_from_predictions(out_dir: Path, metrics_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    existing_keys = {
        (str(row.get("model")), _coerce_fold_id(row.get("fold")))
        for row in metrics_rows
        if _coerce_fold_id(row.get("fold")) is not None
    }
    recovered: List[Dict[str, Any]] = [dict(row) for row in metrics_rows]
    stage_dir = out_dir / "predictions_stage"
    apnea_dir = out_dir / "predictions_apnea"
    if not stage_dir.is_dir() or not apnea_dir.is_dir():
        return recovered

    for stage_path in sorted(stage_dir.glob("*_fold*.csv")):
        stem = stage_path.stem
        if "_fold" not in stem:
            continue
        model_name, fold_text = stem.rsplit("_fold", 1)
        fold_id = _coerce_fold_id(fold_text)
        if fold_id is None or (model_name, fold_id) in existing_keys:
            continue
        apnea_path = apnea_dir / f"{model_name}_fold{fold_id}.csv"
        if not apnea_path.is_file():
            continue
        stage_df = pd.read_csv(stage_path)
        apnea_df = pd.read_csv(apnea_path)
        if stage_df.empty or apnea_df.empty:
            continue
        stage_score = None
        apnea_score = pd.to_numeric(apnea_df["y_score"], errors="coerce").to_numpy() if "y_score" in apnea_df.columns else None
        row = {
            "model": model_name,
            "fold": int(fold_id),
            **_safe_stage_metrics(stage_df["y_true"].astype(str).to_numpy(), stage_df["y_pred"].astype(str).to_numpy()),
            **_safe_apnea_metrics(
                pd.to_numeric(apnea_df["y_true"], errors="coerce").fillna(0).astype(int).to_numpy(),
                pd.to_numeric(apnea_df["y_pred"], errors="coerce").fillna(0).astype(int).to_numpy(),
                apnea_score,
            ),
        }
        recovered.append(row)
        existing_keys.add((model_name, fold_id))
    return recovered


def _save_task_predictions(
    *,
    path: Path,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_score: Optional[Sequence[float]],
    subject_id: Sequence[Any],
    fold_id: Any,
    model_name: str,
    task_name: str,
    extra_columns: Optional[Dict[str, Sequence[Any]]] = None,
) -> None:
    extras = {"model": [model_name] * len(y_true), "task": [task_name] * len(y_true)}
    if extra_columns:
        extras.update(extra_columns)
    save_predictions_dataframe(
        path,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        subject_id=subject_id,
        fold_id=fold_id,
        extra_columns=extras,
    )


def _model_bundle(
    *,
    model: Any,
    imputer: Any,
    scaler: Any,
    label_encoder: Any,
    feature_columns: Sequence[str],
    target_column: str,
    subject_column: str,
    task: str,
    random_seed: int,
    train_csv: str,
    model_name: str,
    artifact_kind: str,
    training_mode: str,
    fold: Optional[int],
    task_name: str,
) -> Dict[str, Any]:
    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_columns": list(feature_columns),
        "target_column": target_column,
        "subject_column": subject_column,
        "task": task,
        "task_name": task_name,
        "random_seed": random_seed,
        "train_csv": train_csv,
        "class_labels": list(label_encoder.classes_),
        "target_dummy_columns": [],
        "model_name": model_name,
        "artifact_kind": artifact_kind,
        "training_mode": training_mode,
        "fold": fold,
    }


def run_cv(cfg: Dict[str, Any], df: pd.DataFrame, out_dir: Path, *, train_csv_path: Optional[Path] = None) -> None:
    seed = int(cfg.get("random_seed", 42))
    subject_col = str(cfg.get("subject_column", "subject_unit_id"))
    recording_col = str(cfg.get("recording_column", "recording_id"))
    stage_col, apnea_col = _targets_cfg(cfg)
    df = normalize_classic_multitarget_dataframe(
        df,
        subject_col=subject_col,
        recording_col=recording_col,
        stage_col=stage_col,
        apnea_col=apnea_col,
    )
    n_splits = int((cfg.get("cv", {}) or {}).get("n_splits", 5))
    if df[subject_col].astype(str).nunique() < n_splits:
        raise ValueError(f"Need at least {n_splits} unique subjects for subject-wise CV.")
    df = build_shared_fold_assignments(
        df,
        subject_col=subject_col,
        stage_col=stage_col,
        apnea_col=apnea_col,
        n_splits=n_splits,
        random_state=seed,
        shuffle=bool((cfg.get("cv", {}) or {}).get("shuffle", True)),
    )
    feat_cols = _feature_columns(df, cfg, stage_col=stage_col, apnea_col=apnea_col, subject_col=subject_col)
    model_names = _enabled_models(cfg)
    training_mode = "tuned" if bool(_tuning_settings(cfg).get("enabled", False)) else "fixed"
    train_csv = _train_csv_string(train_csv_path, cfg)
    save_models, save_fold_models, save_final_model = _output_settings(cfg)
    resume_completed = _resume_completed_enabled(cfg)
    metrics_path = out_dir / "metrics_per_fold.csv"
    registry_path = out_dir / "models" / "model_registry.json"
    best_params_path = out_dir / "best_params_per_fold.csv"
    metrics_rows: List[Dict[str, Any]] = _existing_csv_rows(metrics_path) if resume_completed else []
    registry_rows: List[Dict[str, Any]] = _existing_registry_rows(registry_path) if resume_completed else []
    best_param_rows: List[Dict[str, Any]] = _existing_csv_rows(best_params_path) if resume_completed else []

    out_dir.mkdir(parents=True, exist_ok=True)
    df[[subject_col, "fold", "fold_strategy_used"]].drop_duplicates().to_csv(out_dir / "fold_assignments.csv", index=False)
    if resume_completed:
        metrics_rows = _recover_metrics_rows_from_predictions(out_dir, metrics_rows)
    print(
        f"[classic_multitarget_cv] rows={len(df)} subjects={df[subject_col].nunique()} "
        f"recordings={df[recording_col].nunique()} features={len(feat_cols)} "
        f"fold_strategy={df['fold_strategy_used'].iloc[0]}",
        flush=True,
    )

    for fold_id in range(n_splits):
        train_df = df[df["fold"] != fold_id].copy()
        test_df = df[df["fold"] == fold_id].copy()
        print(
            f"[classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
            f"train_rows={len(train_df)} test_rows={len(test_df)}",
            flush=True,
        )
        stage_train = _task_df(train_df, target_col=stage_col, mask_col="label_mask_stage")
        stage_test = _task_df(test_df, target_col=stage_col, mask_col="label_mask_stage")
        apnea_train = _task_df(train_df, target_col=apnea_col, mask_col="label_mask_apnea")
        apnea_test = _task_df(test_df, target_col=apnea_col, mask_col="label_mask_apnea")

        for model_name in model_names:
            if resume_completed:
                completed_folds = _completed_fold_ids_from_metrics(metrics_rows, model_name=model_name)
                if fold_id in completed_folds or _fold_artifacts_complete(out_dir, model_name=model_name, fold_id=fold_id):
                    print(
                        f"[{_ts()}] [classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
                        f"model={model_name} step=skip_completed",
                        flush=True,
                    )
                    continue
            try:
                make_model(model_name, seed, cfg.get("hyperparams", {}), task="multiclass")
            except RuntimeError as e:
                print(f"  Skip {model_name}: {e}", flush=True)
                continue
            model_t0 = time.perf_counter()
            print(
                f"[{_ts()}] [classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
                f"model={model_name} step=stage_tuning_fit start",
                flush=True,
            )
            stage_model, stage_imp, stage_scaler, stage_le, stage_pred, _stage_score, stage_hp, stage_inner = _fit_eval_task(
                cfg=cfg,
                model_name=model_name,
                seed=seed,
                task="multiclass",
                target_col=stage_col,
                subject_col=subject_col,
                feat_cols=feat_cols,
                train_df=stage_train,
                test_df=stage_test,
            )
            print(
                f"[{_ts()}] [classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
                f"model={model_name} step=stage_tuning_fit done "
                f"elapsed_sec={time.perf_counter() - model_t0:.1f}",
                flush=True,
            )
            apnea_t0 = time.perf_counter()
            print(
                f"[{_ts()}] [classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
                f"model={model_name} step=apnea_tuning_fit start",
                flush=True,
            )
            apnea_model, apnea_imp, apnea_scaler, apnea_le, apnea_pred, apnea_score, apnea_hp, apnea_inner = _fit_eval_task(
                cfg=cfg,
                model_name=model_name,
                seed=seed,
                task="binary",
                target_col=apnea_col,
                subject_col=subject_col,
                feat_cols=feat_cols,
                train_df=apnea_train,
                test_df=apnea_test,
            )
            print(
                f"[{_ts()}] [classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
                f"model={model_name} step=apnea_tuning_fit done "
                f"elapsed_sec={time.perf_counter() - apnea_t0:.1f}",
                flush=True,
            )

            row = {
                "model": model_name,
                "fold": fold_id,
                **_safe_stage_metrics(stage_test[stage_col].astype(str).values, stage_pred),
                **_safe_apnea_metrics(apnea_test[apnea_col].astype(int).values, apnea_pred.astype(int), apnea_score),
            }
            metrics_rows.append(row)
            print(
                f"[{_ts()}] [classic_multitarget_cv] fold={fold_id + 1}/{n_splits} "
                f"model={model_name} metrics "
                f"stage_acc={row['stage_accuracy']:.4f} apnea_auc={row['apnea_auc_roc']}",
                flush=True,
            )

            _save_task_predictions(
                path=out_dir / "predictions_stage" / f"{model_name}_fold{fold_id}.csv",
                y_true=stage_test[stage_col].astype(str).values,
                y_pred=stage_pred,
                y_score=None,
                subject_id=stage_test[subject_col].astype(str).values,
                fold_id=fold_id,
                model_name=model_name,
                task_name="stage",
                extra_columns={"recording_id": stage_test[recording_col].astype(str).values},
            )
            _save_task_predictions(
                path=out_dir / "predictions_apnea" / f"{model_name}_fold{fold_id}.csv",
                y_true=apnea_test[apnea_col].astype(int).values,
                y_pred=apnea_pred.astype(int),
                y_score=apnea_score,
                subject_id=apnea_test[subject_col].astype(str).values,
                fold_id=fold_id,
                model_name=model_name,
                task_name="apnea",
                extra_columns={"recording_id": apnea_test[recording_col].astype(str).values},
            )
            save_confusion_matrix_figure(
                stage_test[stage_col].astype(str).values,
                stage_pred,
                out_dir / "figures" / f"cm_stage_{model_name}_fold{fold_id}.png",
                title=f"stage {model_name} fold {fold_id}",
            )
            save_confusion_matrix_figure(
                apnea_test[apnea_col].astype(int).values,
                apnea_pred.astype(int),
                out_dir / "figures" / f"cm_apnea_{model_name}_fold{fold_id}.png",
                labels=[0, 1],
                title=f"apnea {model_name} fold {fold_id}",
            )
            if apnea_score is not None:
                save_roc_curve_figure(
                    apnea_test[apnea_col].astype(int).values,
                    apnea_score,
                    out_dir / "figures" / f"roc_apnea_{model_name}_fold{fold_id}.png",
                    title=f"ROC apnea {model_name} fold {fold_id}",
                )

            best_param_rows.extend(
                [
                    {
                        "model": model_name,
                        "task_name": "stage",
                        "fold": fold_id,
                        "training_mode": training_mode,
                        "inner_best_score": stage_inner,
                        "selected_params_json": json.dumps(stage_hp, sort_keys=True, default=str),
                    },
                    {
                        "model": model_name,
                        "task_name": "apnea",
                        "fold": fold_id,
                        "training_mode": training_mode,
                        "inner_best_score": apnea_inner,
                        "selected_params_json": json.dumps(apnea_hp, sort_keys=True, default=str),
                    },
                ]
            )
            if save_models and save_fold_models:
                for task_name, task_type, model_obj, imp, scaler, le, target_column, bundle_path in (
                    ("stage", "multiclass", stage_model, stage_imp, stage_scaler, stage_le, stage_col, out_dir / "models" / "stage" / "folds" / f"{model_name}_fold{fold_id}.joblib"),
                    ("apnea", "binary", apnea_model, apnea_imp, apnea_scaler, apnea_le, apnea_col, out_dir / "models" / "apnea" / "folds" / f"{model_name}_fold{fold_id}.joblib"),
                ):
                    saved_path = save_model_bundle(
                        bundle_path,
                        _model_bundle(
                            model=model_obj,
                            imputer=imp,
                            scaler=scaler,
                            label_encoder=le,
                            feature_columns=feat_cols,
                            target_column=target_column,
                            subject_column=subject_col,
                            task=task_type,
                            random_seed=seed,
                            train_csv=train_csv,
                            model_name=model_name,
                            artifact_kind="fold",
                            training_mode=training_mode,
                            fold=fold_id,
                            task_name=task_name,
                        ),
                    )
                    registry_rows.append(
                        _registry_row(
                            out_dir=out_dir,
                            artifact_path=saved_path,
                            experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                            dataset_origin=train_csv,
                            algorithm=f"{model_name}_{task_name}",
                            artifact_type="fold",
                            training_mode=training_mode,
                            feature_columns=feat_cols,
                            class_labels=le.classes_,
                            fold=fold_id,
                        )
                    )

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_per_fold.csv", index=False)
    _write_summary(out_dir, metrics_rows, model_names, cross_dataset=False)
    if registry_rows:
        write_model_registry(out_dir / "models" / "model_registry.json", registry_rows)
    _write_tuning_artifacts(out_dir, best_param_rows)

    if save_models and save_final_model:
        full_stage = _task_df(df, target_col=stage_col, mask_col="label_mask_stage")
        full_apnea = _task_df(df, target_col=apnea_col, mask_col="label_mask_apnea")
        for model_name in model_names:
            if resume_completed and _final_artifacts_complete(out_dir, model_name=model_name):
                print(f"[{_ts()}] [classic_multitarget_cv] final model={model_name} step=skip_completed", flush=True)
                continue
            try:
                make_model(model_name, seed, cfg.get("hyperparams", {}), task="multiclass")
            except RuntimeError:
                continue
            print(
                f"[{_ts()}] [classic_multitarget_cv] final model={model_name} step=stage_tuning_fit start",
                flush=True,
            )
            stage_cfg = _task_tuned_cfg(cfg, task="multiclass")
            final_stage_hp, stage_inner = _nested_best_hyperparams(
                cfg=stage_cfg,
                model_name=model_name,
                seed=seed,
                task="multiclass",
                subject_col=subject_col,
                feat_cols=feat_cols,
                train_df=full_stage.copy(),
                target_col=stage_col,
            )
            print(
                f"[{_ts()}] [classic_multitarget_cv] final model={model_name} step=apnea_tuning_fit start",
                flush=True,
            )
            apnea_cfg = _task_tuned_cfg(cfg, task="binary")
            final_apnea_hp, apnea_inner = _nested_best_hyperparams(
                cfg=apnea_cfg,
                model_name=model_name,
                seed=seed,
                task="binary",
                subject_col=subject_col,
                feat_cols=feat_cols,
                train_df=full_apnea.copy(),
                target_col=apnea_col,
            )
            print(
                f"[{_ts()}] [classic_multitarget_cv] final model={model_name} step=fit_all start",
                flush=True,
            )
            stage_model, stage_imp, stage_scaler, stage_le = _fit_model_bundle_only(
                cfg=stage_cfg,
                model_name=model_name,
                seed=seed,
                hyperparams={model_name: final_stage_hp},
                X_train_df=cast_feature_frame_float32(full_stage[list(feat_cols)]),
                y_train_raw=full_stage[stage_col].astype(str).values,
                task="multiclass",
            )
            apnea_model, apnea_imp, apnea_scaler, apnea_le = _fit_model_bundle_only(
                cfg=apnea_cfg,
                model_name=model_name,
                seed=seed,
                hyperparams={model_name: final_apnea_hp},
                X_train_df=cast_feature_frame_float32(full_apnea[list(feat_cols)]),
                y_train_raw=full_apnea[apnea_col].astype(int).values,
                task="binary",
            )
            print(
                f"[{_ts()}] [classic_multitarget_cv] final model={model_name} step=fit_all done",
                flush=True,
            )
            best_param_rows.extend(
                [
                    {
                        "model": model_name,
                        "task_name": "stage",
                        "fold": "final",
                        "training_mode": training_mode,
                        "inner_best_score": stage_inner,
                        "selected_params_json": json.dumps(final_stage_hp, sort_keys=True, default=str),
                    },
                    {
                        "model": model_name,
                        "task_name": "apnea",
                        "fold": "final",
                        "training_mode": training_mode,
                        "inner_best_score": apnea_inner,
                        "selected_params_json": json.dumps(final_apnea_hp, sort_keys=True, default=str),
                    },
                ]
            )
            for task_name, task_type, model_obj, imp, scaler, le, target_column, bundle_path in (
                ("stage", "multiclass", stage_model, stage_imp, stage_scaler, stage_le, stage_col, out_dir / "models" / "stage" / f"{model_name}_final.joblib"),
                ("apnea", "binary", apnea_model, apnea_imp, apnea_scaler, apnea_le, apnea_col, out_dir / "models" / "apnea" / f"{model_name}_final.joblib"),
            ):
                saved_path = save_model_bundle(
                    bundle_path,
                    _model_bundle(
                        model=model_obj,
                        imputer=imp,
                        scaler=scaler,
                        label_encoder=le,
                        feature_columns=feat_cols,
                        target_column=target_column,
                        subject_column=subject_col,
                        task=task_type,
                        random_seed=seed,
                        train_csv=train_csv,
                        model_name=model_name,
                        artifact_kind="final",
                        training_mode=training_mode,
                        fold=None,
                        task_name=task_name,
                    ),
                )
                registry_rows.append(
                    _registry_row(
                        out_dir=out_dir,
                        artifact_path=saved_path,
                        experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                        dataset_origin=train_csv,
                        algorithm=f"{model_name}_{task_name}",
                        artifact_type="final",
                        training_mode=training_mode,
                        feature_columns=feat_cols,
                        class_labels=le.classes_,
                    )
                )

        if registry_rows:
            write_model_registry(out_dir / "models" / "model_registry.json", registry_rows)
        _write_tuning_artifacts(out_dir, best_param_rows)
    print(f"[{_ts()}] [classic_multitarget_cv] wrote metrics and summary under {out_dir}", flush=True)


def run_cross_dataset(
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    train_csv_path: Optional[Path] = None,
) -> None:
    seed = int(cfg.get("random_seed", 42))
    subject_col = str(cfg.get("subject_column", "subject_unit_id"))
    recording_col = str(cfg.get("recording_column", "recording_id"))
    stage_col, apnea_col = _targets_cfg(cfg)
    train_df = normalize_classic_multitarget_dataframe(
        train_df,
        subject_col=subject_col,
        recording_col=recording_col,
        stage_col=stage_col,
        apnea_col=apnea_col,
    )
    eval_df = normalize_classic_multitarget_dataframe(
        eval_df,
        subject_col=subject_col,
        recording_col=recording_col,
        stage_col=stage_col,
        apnea_col=apnea_col,
    )
    feat_cols = _feature_columns_cross(
        train_df,
        eval_df,
        cfg,
        stage_col=stage_col,
        apnea_col=apnea_col,
        subject_col=subject_col,
    )
    model_names = _enabled_models(cfg)
    training_mode = "tuned" if bool(_tuning_settings(cfg).get("enabled", False)) else "fixed"
    train_csv = _train_csv_string(train_csv_path, cfg)
    save_models, _save_fold_models, save_final_model = _output_settings(cfg)
    metrics_rows: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []
    best_param_rows: List[Dict[str, Any]] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[classic_multitarget_cross] train_rows={len(train_df)} eval_rows={len(eval_df)} "
        f"train_subjects={train_df[subject_col].nunique()} eval_subjects={eval_df[subject_col].nunique()} "
        f"features={len(feat_cols)}",
        flush=True,
    )

    stage_train = _task_df(train_df, target_col=stage_col, mask_col="label_mask_stage")
    stage_eval = _task_df(eval_df, target_col=stage_col, mask_col="label_mask_stage")
    apnea_train = _task_df(train_df, target_col=apnea_col, mask_col="label_mask_apnea")
    apnea_eval = _task_df(eval_df, target_col=apnea_col, mask_col="label_mask_apnea")

    for model_name in model_names:
        try:
            make_model(model_name, seed, cfg.get("hyperparams", {}), task="multiclass")
        except RuntimeError as e:
            print(f"  Skip {model_name}: {e}", flush=True)
            continue
        print(f"[{_ts()}] [classic_multitarget_cross] model={model_name} step=stage_tuning_fit start", flush=True)

        stage_model, stage_imp, stage_scaler, stage_le, stage_pred, _stage_score, stage_hp, stage_inner = _fit_eval_task(
            cfg=cfg,
            model_name=model_name,
            seed=seed,
            task="multiclass",
            target_col=stage_col,
            subject_col=subject_col,
            feat_cols=feat_cols,
            train_df=stage_train,
            test_df=stage_eval,
        )
        print(f"[{_ts()}] [classic_multitarget_cross] model={model_name} step=apnea_tuning_fit start", flush=True)
        apnea_model, apnea_imp, apnea_scaler, apnea_le, apnea_pred, apnea_score, apnea_hp, apnea_inner = _fit_eval_task(
            cfg=cfg,
            model_name=model_name,
            seed=seed,
            task="binary",
            target_col=apnea_col,
            subject_col=subject_col,
            feat_cols=feat_cols,
            train_df=apnea_train,
            test_df=apnea_eval,
        )

        row = {
            "model": model_name,
            "fold": "cross_eval",
            "train_dataset": ",".join(sorted(train_df["dataset_id"].astype(str).unique())),
            "eval_dataset": ",".join(sorted(eval_df["dataset_id"].astype(str).unique())),
            **_safe_stage_metrics(stage_eval[stage_col].astype(str).values, stage_pred),
            **_safe_apnea_metrics(apnea_eval[apnea_col].astype(int).values, apnea_pred.astype(int), apnea_score),
        }
        metrics_rows.append(row)
        print(
            f"[{_ts()}] [classic_multitarget_cross] model={model_name} metrics "
            f"stage_acc={row['stage_accuracy']:.4f} apnea_auc={row['apnea_auc_roc']}",
            flush=True,
        )

        _save_task_predictions(
            path=out_dir / "predictions_stage" / f"{model_name}_cross_eval.csv",
            y_true=stage_eval[stage_col].astype(str).values,
            y_pred=stage_pred,
            y_score=None,
            subject_id=stage_eval[subject_col].astype(str).values,
            fold_id="cross_eval",
            model_name=model_name,
            task_name="stage",
            extra_columns={"recording_id": stage_eval[recording_col].astype(str).values},
        )
        _save_task_predictions(
            path=out_dir / "predictions_apnea" / f"{model_name}_cross_eval.csv",
            y_true=apnea_eval[apnea_col].astype(int).values,
            y_pred=apnea_pred.astype(int),
            y_score=apnea_score,
            subject_id=apnea_eval[subject_col].astype(str).values,
            fold_id="cross_eval",
            model_name=model_name,
            task_name="apnea",
            extra_columns={"recording_id": apnea_eval[recording_col].astype(str).values},
        )
        save_confusion_matrix_figure(
            stage_eval[stage_col].astype(str).values,
            stage_pred,
            out_dir / "figures" / f"cm_stage_{model_name}_cross_eval.png",
            title=f"stage {model_name} cross_eval",
        )
        save_confusion_matrix_figure(
            apnea_eval[apnea_col].astype(int).values,
            apnea_pred.astype(int),
            out_dir / "figures" / f"cm_apnea_{model_name}_cross_eval.png",
            labels=[0, 1],
            title=f"apnea {model_name} cross_eval",
        )
        if apnea_score is not None:
            save_roc_curve_figure(
                apnea_eval[apnea_col].astype(int).values,
                apnea_score,
                out_dir / "figures" / f"roc_apnea_{model_name}_cross_eval.png",
                title=f"ROC apnea {model_name} cross_eval",
            )

        best_param_rows.extend(
            [
                {
                    "model": model_name,
                    "task_name": "stage",
                    "fold": "cross_eval",
                    "training_mode": training_mode,
                    "inner_best_score": stage_inner,
                    "selected_params_json": json.dumps(stage_hp, sort_keys=True, default=str),
                },
                {
                    "model": model_name,
                    "task_name": "apnea",
                    "fold": "cross_eval",
                    "training_mode": training_mode,
                    "inner_best_score": apnea_inner,
                    "selected_params_json": json.dumps(apnea_hp, sort_keys=True, default=str),
                },
            ]
        )

        if save_models and save_final_model:
            for task_name, task_type, model_obj, imp, scaler, le, target_column, bundle_path in (
                ("stage", "multiclass", stage_model, stage_imp, stage_scaler, stage_le, stage_col, out_dir / "models" / "stage" / f"{model_name}_final.joblib"),
                ("apnea", "binary", apnea_model, apnea_imp, apnea_scaler, apnea_le, apnea_col, out_dir / "models" / "apnea" / f"{model_name}_final.joblib"),
            ):
                saved_path = save_model_bundle(
                    bundle_path,
                    _model_bundle(
                        model=model_obj,
                        imputer=imp,
                        scaler=scaler,
                        label_encoder=le,
                        feature_columns=feat_cols,
                        target_column=target_column,
                        subject_column=subject_col,
                        task=task_type,
                        random_seed=seed,
                        train_csv=train_csv,
                        model_name=model_name,
                        artifact_kind="final",
                        training_mode=training_mode,
                        fold=None,
                        task_name=task_name,
                    ),
                )
                registry_rows.append(
                    _registry_row(
                        out_dir=out_dir,
                        artifact_path=saved_path,
                        experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                        dataset_origin=train_csv,
                        algorithm=f"{model_name}_{task_name}",
                        artifact_type="final",
                        training_mode=training_mode,
                        feature_columns=feat_cols,
                        class_labels=le.classes_,
                    )
                )

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_cross_eval.csv", index=False)
    _write_summary(out_dir, metrics_rows, model_names, cross_dataset=True)
    if registry_rows:
        write_model_registry(out_dir / "models" / "model_registry.json", registry_rows)
    _write_tuning_artifacts(out_dir, best_param_rows)
    print(f"[{_ts()}] [classic_multitarget_cross] wrote cross metrics under {out_dir}", flush=True)


def run_experiment(config_path: Path) -> None:
    cfg = load_config(config_path)
    out_root = Path((cfg.get("output", {}) or {}).get("root", "reports/experiments"))
    out_dir = out_root / str(cfg.get("experiment_name", "classic_multitarget_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    train_path = resolve_csv_path(str(cfg["train_csv"]), config_path)
    train_df = read_table_file(train_path)
    train_df = ensure_subject_unit_column(train_df, output_col=str(cfg.get("subject_column", "subject_unit_id")), overwrite=False)
    if bool(cfg.get("cross_dataset", False)):
        eval_raw = cfg.get("eval_csv")
        if not eval_raw:
            raise ValueError("cross_dataset true requires eval_csv")
        eval_path = resolve_csv_path(str(eval_raw), config_path)
        eval_df = read_table_file(eval_path)
        eval_df = ensure_subject_unit_column(eval_df, output_col=str(cfg.get("subject_column", "subject_unit_id")), overwrite=False)
        run_cross_dataset(cfg, train_df, eval_df, out_dir, train_csv_path=train_path)
    else:
        run_cv(cfg, train_df, out_dir, train_csv_path=train_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run classic multitarget EEG experiments for sleep_stage + apnea_binary.")
    parser.add_argument("--config", required=True, type=str, help="Path to classic multitarget experiment YAML.")
    args = parser.parse_args(argv)
    run_experiment(Path(args.config))


if __name__ == "__main__":
    main()
