"""
Phase E experiment runner: subject-wise CV and optional cross-dataset evaluation.

CLI: python -m modeling.train_runner --config path/to.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from modeling.artifacts import (
    save_confusion_matrix_figure,
    save_model_bundle,
    save_predictions_dataframe,
    write_model_registry,
)
from modeling.cv_split import SubjectFoldConfig, subject_wise_fold_indices
from modeling.metrics import apnea_binary_metrics, fold_metrics_summary, multiclass_sleep_metrics
from modeling.subject_id import ensure_subject_unit_column
from modeling.target_utils import ensure_target_column, normalize_sleep_stage_series

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[misc, assignment]


def load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping at the root.")
    return data


def resolve_csv_path(path_str: str, config_path: Path) -> Path:
    """
    Resolve ``train_csv`` / ``eval_csv``: absolute as-is, then cwd, then directory of the config file.
    """
    raw = Path(path_str)
    if raw.is_file():
        return raw.resolve()
    if raw.is_absolute():
        raise FileNotFoundError(
            f"CSV not found (absolute path): {raw}\n"
            "Export or copy your processed table, or fix train_csv / eval_csv in the YAML."
        )
    cand_cwd = (Path.cwd() / raw).resolve()
    if cand_cwd.is_file():
        return cand_cwd
    cand_cfg = (config_path.resolve().parent / raw).resolve()
    if cand_cfg.is_file():
        return cand_cfg
    raise FileNotFoundError(
        f"CSV not found: {path_str!r}\n"
        f"  Tried: {cand_cwd}\n"
        f"  Tried: {cand_cfg}\n"
        "Generate it (e.g. WFDB export via main.py) or point train_csv to the real file."
    )


def read_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format {suffix!r} for {path}. Use .csv or .parquet.")


def _scalar_is_strict_binary_zero_one(v: Any) -> None:
    """Raise if ``v`` is not a representable 0/1 label (apnea / binary branch, raw CSV)."""
    if isinstance(v, (bool, np.bool_)):
        return
    if isinstance(v, str):
        t = v.strip()
        if t in ("0", "1"):
            return
        raise ValueError(
            f"task=binary requires target values '0' or '1' as strings; got {v!r}. "
            "Map labels in preprocessing or set binary_require_zero_one_labels: false (two arbitrary classes)."
        )
    if isinstance(v, (int, np.integer)):
        if int(v) in (0, 1):
            return
    elif isinstance(v, (float, np.floating)):
        if np.isfinite(v) and float(v) in (0.0, 1.0):
            return
    raise ValueError(
        f"task=binary requires numeric target 0 or 1; got {v!r}. "
        "Add a derived 0/1 column or set binary_require_zero_one_labels: false."
    )


def validate_binary_target_training(y: pd.Series, col: str, *, require_zero_one: bool) -> None:
    s = y.dropna()
    if s.empty:
        raise ValueError(f"Target column {col!r} is all NaN.")
    if s.nunique() < 2:
        raise ValueError(
            f"task=binary needs at least two classes in {col!r} for training (found {s.nunique()})."
        )
    if require_zero_one:
        for v in s.unique():
            _scalar_is_strict_binary_zero_one(v)


def validate_binary_target_eval(y: pd.Series, col: str, *, require_zero_one: bool) -> None:
    """Cross-dataset eval: labels must be valid; eval may be single-class."""
    s = y.dropna()
    if s.empty:
        raise ValueError(f"Eval target column {col!r} is all NaN.")
    if require_zero_one:
        for v in s.unique():
            _scalar_is_strict_binary_zero_one(v)
    elif s.nunique() > 2:
        raise ValueError(
            f"task=binary with binary_require_zero_one_labels: false still allows at most 2 classes in {col!r}; "
            f"found {s.nunique()}."
        )


def epoch_feature_signal_bases(feat_cols: Sequence[str]) -> List[str]:
    bases = set()
    for c in feat_cols:
        if c.endswith("_mean"):
            bases.add(c[:-5])
        elif c.endswith("_std"):
            bases.add(c[:-4])
    return sorted(bases)


def enforce_single_channel_epoch_features(feat_cols: Sequence[str], cfg: Dict[str, Any]) -> None:
    """PRD §4.2: a single signal prefix among *_mean / *_std unless explicitly overridden."""
    if cfg.get("allow_multi_channel_features", False):
        return
    inc = cfg.get("feature_include")
    if inc is not None and not (isinstance(inc, list) and len(inc) == 0):
        return
    bases = epoch_feature_signal_bases(feat_cols)
    if len(bases) <= 1:
        return
    raise ValueError(
        "Several epoch feature stems (*_mean / *_std) found without feature_include: "
        f"{bases!r}. Use feature_include with exactly one signal (e.g. eeg_c3_a2_mean, eeg_c3_a2_std). "
        "Names match wfdb export after snake_case. See config/experiment_scope.yaml → eeg_single_channel. "
        "Override only if needed: allow_multi_channel_features: true."
    )


def _default_hyperparams() -> Dict[str, Any]:
    return {
        "svm_rbf": {
            "C": 1.0,
            "gamma": "scale",
            "cache_size": 200,
            "tol": 1e-3,
            "shrinking": True,
            "max_iter": -1,
        },
        "random_forest": {"n_estimators": 200, "max_depth": None, "n_jobs": -1},
        "xgboost": {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.1,
            "n_jobs": -1,
            "tree_method": "hist",
        },
    }


def numeric_feature_columns(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    ex = {str(x) for x in exclude}
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num if c not in ex]


def resolve_feature_columns(
    df: pd.DataFrame,
    exclude: Sequence[str],
    include: Optional[Sequence[str]],
) -> List[str]:
    """
    If ``feature_include`` is set in YAML, use only those columns (must exist, numeric, not excluded).
    Otherwise all numeric columns except ``exclude`` (PRD: EEG monocanal → listar solo columnas *_mean/*_std del canal elegido).
    """
    ex = {str(x) for x in exclude}
    if include is None or (isinstance(include, list) and len(include) == 0):
        return numeric_feature_columns(df, ex)
    out: List[str] = []
    for c in include:
        key = str(c)
        if key in ex:
            raise ValueError(f"feature_include column {key!r} is excluded (target/subject/feature_exclude).")
        if key not in df.columns:
            raise ValueError(f"feature_include column {key!r} not in dataframe.")
        if not pd.api.types.is_numeric_dtype(df[key]):
            raise ValueError(
                f"feature_include column {key!r} is not numeric; convert in preprocessing or omit from include."
            )
        out.append(key)
    return out


def resolve_feature_columns_cross(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    exclude_cols: Sequence[str],
    include: Optional[Sequence[str]],
) -> List[str]:
    """Same columns in train and eval; with ``feature_include``, both must list the same names."""
    ex = {str(x) for x in exclude_cols}
    if include is None or (isinstance(include, list) and len(include) == 0):
        tr = set(numeric_feature_columns(train_df, ex))
        ev = set(numeric_feature_columns(eval_df, ex))
        inter = sorted(tr & ev)
        if not inter:
            raise ValueError("No overlapping numeric feature columns between train and eval CSV.")
        return inter
    out: List[str] = []
    for c in include:
        key = str(c)
        if key in ex:
            raise ValueError(f"feature_include column {key!r} is excluded.")
        for df, label in ((train_df, "train_csv"), (eval_df, "eval_csv")):
            if key not in df.columns:
                raise ValueError(f"feature_include {key!r} missing in {label}.")
            if not pd.api.types.is_numeric_dtype(df[key]):
                raise ValueError(f"feature_include {key!r} is not numeric in {label}.")
        out.append(key)
    return out


def cast_feature_frame_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(np.float32, copy=False)


def impute_fit(X_train: pd.DataFrame) -> Tuple[np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy="median")
    X_train_f = cast_feature_frame_float32(X_train)
    Xt = imp.fit_transform(X_train_f).astype(np.float32, copy=False)
    return Xt, imp


def impute_apply(imputer: SimpleImputer, X: pd.DataFrame) -> np.ndarray:
    X_f = cast_feature_frame_float32(X)
    return imputer.transform(X_f).astype(np.float32, copy=False)


def impute_fit_transform(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer]:
    Xt, imp = impute_fit(X_train)
    Xv = impute_apply(imp, X_test)
    return Xt, Xv, imp


def _needs_standard_scaler(model_name: str) -> bool:
    return model_name == "svm_rbf"


def scale_fit_transform(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    if not enabled:
        return X_train, X_test, None
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)
    return X_train_scaled, X_test_scaled, scaler


def scale_apply(
    scaler: Optional[StandardScaler],
    X: np.ndarray,
) -> np.ndarray:
    if scaler is None:
        return X.astype(np.float32, copy=False)
    return scaler.transform(X).astype(np.float32, copy=False)


def make_model(
    name: str,
    random_state: int,
    hyperparams: Dict[str, Any],
    *,
    task: str = "multiclass",
) -> Any:
    hp = hyperparams.get(name, {})
    if name == "svm_rbf":
        base = _default_hyperparams()["svm_rbf"]
        base.update(hp)
        gamma = base["gamma"]
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            gamma = float(gamma)
        class_weight = base.get("class_weight")
        return SVC(
            kernel="rbf",
            C=float(base["C"]),
            gamma=gamma,
            probability=(task == "binary"),
            class_weight=class_weight,
            cache_size=float(base.get("cache_size", 200)),
            tol=float(base.get("tol", 1e-3)),
            shrinking=bool(base.get("shrinking", True)),
            max_iter=int(base.get("max_iter", -1)),
            random_state=random_state,
        )
    if name == "random_forest":
        base = _default_hyperparams()["random_forest"]
        base.update(hp)
        md = base.get("max_depth")
        return RandomForestClassifier(
            n_estimators=int(base["n_estimators"]),
            max_depth=None if md is None else int(md),
            min_samples_leaf=int(base.get("min_samples_leaf", 1)),
            class_weight=base.get("class_weight"),
            random_state=random_state,
            n_jobs=int(base.get("n_jobs", -1)),
        )
    if name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed; pip install xgboost or disable models.xgboost")
        base = _default_hyperparams()["xgboost"]
        base.update(hp)
        return XGBClassifier(
            n_estimators=int(base["n_estimators"]),
            max_depth=int(base["max_depth"]),
            learning_rate=float(base["learning_rate"]),
            subsample=float(base.get("subsample", 1.0)),
            colsample_bytree=float(base.get("colsample_bytree", 1.0)),
            random_state=random_state,
            n_jobs=int(base.get("n_jobs", -1)),
            tree_method=str(base.get("tree_method", "hist")),
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model: {name}")


def encode_y_safe(y_train: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_tr = le.fit_transform(np.asarray(y_train).astype(str))
    y_test_str = np.asarray(y_test).astype(str)
    missing = set(y_test_str) - set(le.classes_)
    if missing:
        raise ValueError(f"eval/test contains labels not in train: {missing}")
    y_te = le.transform(y_test_str)
    return y_tr, y_te, le


def _metrics_row(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    label_encoder: Optional[LabelEncoder] = None,
) -> Dict[str, Any]:
    if task == "binary":
        if label_encoder is None:
            raise ValueError("label_encoder required for binary metrics")
        yt = label_encoder.transform(np.asarray(y_true).astype(str))
        yp = label_encoder.transform(np.asarray(y_pred).astype(str))
        m = apnea_binary_metrics(yt, yp, y_score_positive=y_score)
        flat = {k: v for k, v in m.items() if k != "per_class_f1"}
        return flat
    m = multiclass_sleep_metrics(y_true, y_pred)
    flat = {
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "cohen_kappa": m["cohen_kappa"],
    }
    for label, score in m["per_class_f1"].items():
        flat[f"per_class_f1_{str(label).lower()}"] = score
    return flat


def _output_settings(cfg: Dict[str, Any]) -> Tuple[bool, bool, bool]:
    out = cfg.get("output", {}) or {}
    save_models = bool(out.get("save_models", True))
    save_fold_models = bool(out.get("save_fold_models", True))
    save_final_model = bool(out.get("save_final_model", True))
    return save_models, save_fold_models, save_final_model


def _resume_completed_enabled(cfg: Dict[str, Any]) -> bool:
    out = cfg.get("output", {}) or {}
    return bool(out.get("resume_completed", False))


def _train_csv_string(train_csv_path: Optional[Path], cfg: Dict[str, Any]) -> str:
    if train_csv_path is not None:
        return str(train_csv_path)
    value = cfg.get("train_csv")
    return "" if value is None else str(value)


def _model_bundle(
    *,
    model: Any,
    imputer: SimpleImputer,
    scaler: Optional[StandardScaler],
    label_encoder: LabelEncoder,
    feature_columns: Sequence[str],
    target_column: str,
    subject_column: str,
    task: str,
    random_seed: int,
    train_csv: str,
    target_dummy_columns: Sequence[str],
    model_name: str,
    artifact_kind: str,
    training_mode: str,
    fold: Optional[int],
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
        "random_seed": random_seed,
        "train_csv": train_csv,
        "class_labels": list(label_encoder.classes_),
        "target_dummy_columns": list(target_dummy_columns),
        "model_name": model_name,
        "artifact_kind": artifact_kind,
        "training_mode": training_mode,
        "fold": fold,
    }


def _registry_row(
    *,
    out_dir: Path,
    artifact_path: Path,
    experiment_name: str,
    dataset_origin: str,
    algorithm: str,
    artifact_type: str,
    training_mode: str,
    feature_columns: Sequence[str],
    class_labels: Sequence[Any],
    fold: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "dataset_origin": dataset_origin,
        "algorithm": algorithm,
        "artifact_type": artifact_type,
        "training_mode": training_mode,
        "fold": fold,
        "path": str(artifact_path.relative_to(out_dir)),
        "feature_columns": list(feature_columns),
        "class_labels": [str(x) for x in class_labels],
    }


def validate_subject_wise_cv_ready(df: pd.DataFrame, subject_col: str, n_splits: int) -> None:
    if subject_col not in df.columns:
        raise KeyError(f"Column '{subject_col}' not in dataframe.")
    n_subjects = int(df[subject_col].astype(str).nunique())
    if n_subjects < n_splits:
        raise ValueError(
            f"Subject-wise CV requires at least {n_splits} unique subjects in '{subject_col}', "
            f"but only {n_subjects} were found."
        )


def _apply_label_subset(df: pd.DataFrame, target_col: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    subset_raw = cfg.get("label_subset")
    if not subset_raw:
        return df
    allowed = {str(x) for x in subset_raw}
    out = df[df[target_col].astype(str).isin(allowed)].copy()
    if out.empty:
        raise ValueError(f"label_subset={sorted(allowed)!r} removed every row from target {target_col!r}.")
    return out


def _tuning_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tuning = cfg.get("tuning", {}) or {}
    if not isinstance(tuning, dict):
        raise ValueError("tuning must be a mapping when present.")
    return tuning


def _existing_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _existing_registry_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


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


def _completed_fold_ids(
    *,
    out_dir: Path,
    model_name: str,
    metrics_rows: Sequence[Dict[str, Any]],
) -> set[int]:
    completed: set[int] = set()
    for row in metrics_rows:
        if str(row.get("model")) != model_name:
            continue
        fold_id = _coerce_fold_id(row.get("fold"))
        if fold_id is None:
            continue
        pred_path = out_dir / "predictions" / f"{model_name}_fold{fold_id}.csv"
        fig_path = out_dir / "figures" / f"cm_{model_name}_fold{fold_id}.png"
        if pred_path.is_file() and fig_path.is_file():
            completed.add(fold_id)
    return completed


def _final_model_complete(
    *,
    out_dir: Path,
    model_name: str,
    best_param_rows: Sequence[Dict[str, Any]],
    registry_rows: Sequence[Dict[str, Any]],
) -> bool:
    model_path = out_dir / "models" / f"{model_name}_final.joblib"
    if not model_path.is_file():
        return False
    has_tuning_row = any(
        str(row.get("model")) == model_name and str(row.get("fold")) == "final" for row in best_param_rows
    )
    has_registry_row = any(
        str(row.get("algorithm")) == model_name and str(row.get("artifact_type")) == "final" for row in registry_rows
    )
    return has_tuning_row and has_registry_row


def _score_key_for_tuning(task: str, scoring: str) -> str:
    key = str(scoring).strip().lower()
    if task == "multiclass":
        allowed = {"accuracy", "macro_f1", "cohen_kappa"}
    else:
        allowed = {"accuracy", "sensitivity", "specificity", "auc_roc"}
    if key not in allowed:
        raise ValueError(f"Unsupported tuning scoring={scoring!r} for task={task!r}.")
    return key


def _parameter_candidates(tuning: Dict[str, Any], model_name: str) -> List[Dict[str, Any]]:
    search_space = tuning.get("search_space", {}) or {}
    model_space = search_space.get(model_name)
    if not model_space:
        return []
    if not isinstance(model_space, dict):
        raise ValueError(f"tuning.search_space.{model_name} must be a mapping of param -> list.")
    return [dict(x) for x in ParameterGrid(model_space)]


def _extract_metric(metrics_row: Dict[str, Any], key: str) -> float:
    if key not in metrics_row:
        raise KeyError(f"Metric {key!r} missing from computed metrics row.")
    return float(metrics_row[key])


def _train_resampling_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    settings = cfg.get("train_resampling", {}) or {}
    if not isinstance(settings, dict):
        raise ValueError("train_resampling must be a mapping when present.")
    return settings


def _apply_train_resampling(
    *,
    cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    resampling = _train_resampling_settings(cfg)
    if not bool(resampling.get("enabled", False)):
        return X_train, np.asarray(y_train_raw)

    method = str(resampling.get("method", "none"))
    if method != "smote_to_reference_minus":
        raise ValueError(
            f"Unsupported train_resampling.method={method!r}; only 'smote_to_reference_minus' is implemented."
        )

    y_str = np.asarray(y_train_raw).astype(str)
    counts = Counter(y_str)
    reference_class = str(resampling.get("reference_class", "N2"))
    reference_count = int(counts.get(reference_class, 0))
    if reference_count <= 1:
        return X_train, y_str

    reference_offset = int(resampling.get("reference_offset", 7000))
    target_count = max(1, reference_count - reference_offset)
    target_labels = [str(x) for x in resampling.get("target_labels", ["W", "N1", "N3", "REM"])]
    sampling_strategy: Dict[str, int] = {}
    for label in target_labels:
        count = int(counts.get(label, 0))
        if count > 0 and count < target_count:
            sampling_strategy[label] = target_count
    if not sampling_strategy:
        return X_train, y_str

    minority_counts = [int(counts[label]) for label in sampling_strategy]
    if not minority_counts:
        return X_train, y_str
    min_class_count = min(minority_counts)
    if min_class_count <= 1:
        return X_train, y_str

    requested_k = int(resampling.get("k_neighbors", 5))
    effective_k = min(requested_k, min_class_count - 1)
    if effective_k < 1:
        return X_train, y_str

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=int(resampling.get("random_state", cfg.get("random_seed", 42))),
        k_neighbors=effective_k,
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_str)
    return (
        np.asarray(X_resampled, dtype=np.float32),
        np.asarray(y_resampled),
    )


def _xgboost_sample_weight(
    *,
    model_name: str,
    hyperparams: Dict[str, Any],
    y_train_raw: np.ndarray,
) -> Optional[np.ndarray]:
    if model_name != "xgboost":
        return None
    model_hp = hyperparams.get(model_name, {}) or {}
    class_weight = model_hp.get("class_weight")
    if class_weight not in ("balanced", "balanced_subsample"):
        return None
    return np.asarray(compute_sample_weight(class_weight="balanced", y=np.asarray(y_train_raw).astype(str)))


def _fit_model_bundle_only(
    *,
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    hyperparams: Dict[str, Any],
    X_train_df: pd.DataFrame,
    y_train_raw: np.ndarray,
    task: str,
) -> Tuple[Any, SimpleImputer, Optional[StandardScaler], LabelEncoder]:
    X_tr_imputed, imputer = impute_fit(X_train_df)
    X_tr_ready, _unused, scaler = scale_fit_transform(
        X_tr_imputed,
        X_tr_imputed,
        enabled=_needs_standard_scaler(model_name),
    )
    X_fit, y_fit_raw = _apply_train_resampling(cfg=cfg, X_train=X_tr_ready, y_train_raw=y_train_raw)
    y_fit_enc, _y_ignore, le = encode_y_safe(y_fit_raw, y_fit_raw)
    model = make_model(model_name, seed, hyperparams, task=task)
    fit_kwargs: Dict[str, Any] = {}
    sample_weight = _xgboost_sample_weight(model_name=model_name, hyperparams=hyperparams, y_train_raw=y_fit_raw)
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_fit, y_fit_enc, **fit_kwargs)
    return model, imputer, scaler, le


def _fit_predict_bundle(
    *,
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    hyperparams: Dict[str, Any],
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    task: str,
) -> Tuple[Any, SimpleImputer, Optional[StandardScaler], LabelEncoder, np.ndarray, Optional[np.ndarray]]:
    model, imputer, scaler, le = _fit_model_bundle_only(
        cfg=cfg,
        model_name=model_name,
        seed=seed,
        hyperparams=hyperparams,
        X_train_df=X_train_df,
        y_train_raw=y_train_raw,
        task=task,
    )
    X_te_imputed = impute_apply(imputer, X_test_df)
    X_te = scale_apply(scaler, X_te_imputed)
    y_test_str = np.asarray(y_test_raw).astype(str)
    missing = set(y_test_str) - set(le.classes_)
    if missing:
        raise ValueError(f"eval/test contains labels not in train: {missing}")
    pred_enc = model.predict(X_te)
    pred = le.inverse_transform(pred_enc.astype(int))
    y_score = None
    if task == "binary" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_te)
        if proba.shape[1] == 2:
            y_score = proba[:, 1]
    return model, imputer, scaler, le, pred, y_score


def _tuning_train_subject_subsample(
    *,
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    train_df: pd.DataFrame,
    subject_col: str,
    target_col: str,
    inner_splits: int,
) -> pd.DataFrame:
    if model_name != "svm_rbf":
        return train_df
    tuning = _tuning_settings(cfg)
    subsample = tuning.get("train_subject_subsample", {}) or {}
    if not isinstance(subsample, dict) or not bool(subsample.get("enabled", False)):
        return train_df

    subjects = train_df[subject_col].astype(str)
    unique_subjects = pd.Index(subjects.dropna().unique())
    n_subjects = len(unique_subjects)
    if n_subjects == 0:
        return train_df

    fraction = float(subsample.get("fraction", 0.25))
    min_subjects = int(subsample.get("min_subjects", 32))
    target_n = max(inner_splits, min_subjects, int(math.ceil(n_subjects * fraction)))
    target_n = min(target_n, n_subjects)
    if target_n >= n_subjects:
        return train_df

    rng = np.random.RandomState(seed)
    order = list(unique_subjects[rng.permutation(n_subjects)])
    selected = list(order[:target_n])
    selected_set = set(selected)
    full_labels = set(train_df[target_col].dropna().astype(str))

    def current_labels() -> set[str]:
        return set(train_df.loc[subjects.isin(selected_set), target_col].dropna().astype(str))

    label_coverage = current_labels()
    if label_coverage != full_labels:
        for subject in order[target_n:]:
            if label_coverage == full_labels:
                break
            selected.append(subject)
            selected_set.add(subject)
            label_coverage = current_labels()

    sampled = train_df.loc[subjects.isin(selected_set)].copy()
    try:
        validate_subject_wise_cv_ready(sampled, subject_col, inner_splits)
    except ValueError:
        return train_df
    return sampled


def _nested_best_hyperparams(
    *,
    cfg: Dict[str, Any],
    model_name: str,
    seed: int,
    task: str,
    subject_col: str,
    feat_cols: Sequence[str],
    train_df: pd.DataFrame,
    target_col: str,
) -> Tuple[Dict[str, Any], Optional[float]]:
    tuning = _tuning_settings(cfg)
    if not bool(tuning.get("enabled", False)):
        return dict(cfg.get("hyperparams", {}).get(model_name, {})), None
    mode = str(tuning.get("mode", "nested_cv"))
    if mode != "nested_cv":
        raise ValueError(f"Unsupported tuning.mode={mode!r}; only 'nested_cv' is implemented.")
    if str(tuning.get("search_method", "grid")) != "grid":
        raise ValueError("Only tuning.search_method='grid' is implemented.")

    scoring_key = _score_key_for_tuning(task, str(tuning.get("scoring", "macro_f1")))
    inner_splits = int(tuning.get("inner_cv_splits", 3))
    tuning_train_df = _tuning_train_subject_subsample(
        cfg=cfg,
        model_name=model_name,
        seed=seed,
        train_df=train_df,
        subject_col=subject_col,
        target_col=target_col,
        inner_splits=inner_splits,
    )
    validate_subject_wise_cv_ready(tuning_train_df, subject_col, inner_splits)

    base_hp = dict(cfg.get("hyperparams", {}).get(model_name, {}))
    candidates = _parameter_candidates(tuning, model_name)
    if not candidates:
        return base_hp, None

    y_raw = tuning_train_df[target_col].values
    inner_cfg = SubjectFoldConfig(
        n_splits=inner_splits,
        random_state=seed,
        stratify=bool(tuning_train_df.shape[0] > 0 and cfg.get("cv", {}).get("stratify", True)),
        shuffle=bool(cfg.get("cv", {}).get("shuffle", True)),
    )

    best_params = dict(base_hp)
    best_score = -np.inf
    for params in candidates:
        hp = dict(base_hp)
        hp.update(params)
        inner_scores: List[float] = []
        failed = False
        for inner_train_idx, inner_val_idx in subject_wise_fold_indices(
            tuning_train_df,
            subject_col=subject_col,
            y=y_raw,
            config=inner_cfg,
        ):
            X_inner_train_df = tuning_train_df.iloc[inner_train_idx][list(feat_cols)]
            X_inner_val_df = tuning_train_df.iloc[inner_val_idx][list(feat_cols)]
            y_inner_train = y_raw[inner_train_idx]
            y_inner_val = y_raw[inner_val_idx]
            try:
                _model, _imp, _scaler, le, pred, y_score = _fit_predict_bundle(
                    cfg=cfg,
                    model_name=model_name,
                    seed=seed,
                    hyperparams={model_name: hp},
                    X_train_df=X_inner_train_df,
                    X_test_df=X_inner_val_df,
                    y_train_raw=y_inner_train,
                    y_test_raw=y_inner_val,
                    task=task,
                )
            except ValueError:
                failed = True
                break
            row = _metrics_row(task, y_inner_val, pred, y_score, label_encoder=le)
            inner_scores.append(_extract_metric(row, scoring_key))
        if failed or not inner_scores:
            continue
        mean_score = float(np.mean(inner_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = hp

    if not np.isfinite(best_score):
        return base_hp, None
    return best_params, best_score


def _write_tuning_artifacts(out_dir: Path, best_param_rows: List[Dict[str, Any]]) -> None:
    if not best_param_rows:
        return
    csv_path = out_dir / "best_params_per_fold.csv"
    pd.DataFrame(best_param_rows).to_csv(csv_path, index=False)

    summary: Dict[str, Any] = {}
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in best_param_rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    for model_name, rows in by_model.items():
        params_counter = Counter(str(r["selected_params_json"]) for r in rows)
        most_common_params_json, most_common_count = params_counter.most_common(1)[0]
        scores = [float(r["inner_best_score"]) for r in rows if pd.notna(r["inner_best_score"])]
        summary[model_name] = {
            "folds": len(rows),
            "most_common_params_json": most_common_params_json,
            "most_common_count": most_common_count,
            "inner_best_score_mean": float(np.mean(scores)) if scores else None,
            "inner_best_score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0 if scores else None,
        }
    with (out_dir / "best_params_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def _write_metrics_summary(
    *,
    out_dir: Path,
    metrics_rows: List[Dict[str, Any]],
    task: str,
    names: Sequence[str],
) -> None:
    met_path = out_dir / "metrics_per_fold.csv"
    pd.DataFrame(metrics_rows).to_csv(met_path, index=False)

    summary: Dict[str, Any] = {}
    if task == "binary":
        keys = ["accuracy", "sensitivity", "specificity"]
        if any("auc_roc" in r for r in metrics_rows):
            keys.append("auc_roc")
    else:
        keys = ["accuracy", "macro_f1", "cohen_kappa"]
    for mn in names:
        sub = [r for r in metrics_rows if r["model"] == mn]
        if not sub:
            continue
        extra = sorted(k for k in sub[0].keys() if k.startswith("per_class_f1_"))
        summary[mn] = fold_metrics_summary(sub, [k for k in keys if k in sub[0]] + extra)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def run_cv(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    *,
    model_filter: Optional[Sequence[str]] = None,
    train_csv_path: Optional[Path] = None,
) -> None:
    seed = int(cfg.get("random_seed", 42))
    subject_col = str(cfg["subject_column"])
    target_col = str(cfg["target_column"])
    task = str(cfg.get("task", "multiclass"))
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'")

    df, target_col, target_dummy_cols = ensure_target_column(df, target_col_raw=target_col)
    if target_col == "sleep_stage":
        df[target_col] = normalize_sleep_stage_series(df[target_col])
    df = _apply_label_subset(df, target_col, cfg)

    exclude = {subject_col, target_col, *cfg.get("feature_exclude", []), *target_dummy_cols}
    feat_cols = resolve_feature_columns(df, exclude, cfg.get("feature_include"))
    if not feat_cols:
        raise ValueError("No numeric feature columns after exclusions / feature_include.")
    enforce_single_channel_epoch_features(feat_cols, cfg)

    req_bin01 = bool(cfg.get("binary_require_zero_one_labels", True))
    if task == "binary":
        validate_binary_target_training(df[target_col], target_col, require_zero_one=req_bin01)

    y_raw = df[target_col].values
    subj = df[subject_col].values
    X_df = df[feat_cols]

    cv_cfg = cfg.get("cv", {})
    fold_conf = SubjectFoldConfig(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        random_state=seed,
        stratify=bool(cv_cfg.get("stratify", True)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
    )
    validate_subject_wise_cv_ready(df, subject_col, fold_conf.n_splits)

    models_on = cfg.get("models", {})
    names = []
    if models_on.get("random_forest", True):
        names.append("random_forest")
    if models_on.get("xgboost", True):
        names.append("xgboost")
    if models_on.get("svm_rbf", True):
        names.append("svm_rbf")
    if model_filter is not None:
        names = [n for n in names if n in model_filter]
    if not names:
        raise ValueError("No models enabled under config.models")

    hp = cfg.get("hyperparams", {})
    save_models, save_fold_models, save_final_model = _output_settings(cfg)
    resume_completed = _resume_completed_enabled(cfg)
    train_csv = _train_csv_string(train_csv_path, cfg)
    models_dir = out_dir / "models"
    fold_models_dir = models_dir / "folds"
    registry_path = models_dir / "model_registry.json"
    metrics_path = out_dir / "metrics_per_fold.csv"
    best_params_path = out_dir / "best_params_per_fold.csv"
    registry_rows: List[Dict[str, Any]] = _existing_registry_rows(registry_path) if resume_completed else []
    best_param_rows: List[Dict[str, Any]] = _existing_csv_rows(best_params_path) if resume_completed else []
    metrics_rows: List[Dict[str, Any]] = _existing_csv_rows(metrics_path) if resume_completed else []

    for model_name in names:
        try:
            make_model(model_name, seed, hp)
        except RuntimeError as e:
            print(f"  Skip {model_name}: {e}")
            continue

        completed_folds = _completed_fold_ids(out_dir=out_dir, model_name=model_name, metrics_rows=metrics_rows)
        if resume_completed and len(completed_folds) >= fold_conf.n_splits and _final_model_complete(
            out_dir=out_dir,
            model_name=model_name,
            best_param_rows=best_param_rows,
            registry_rows=registry_rows,
        ):
            continue

        fold_id = 0
        for train_idx, test_idx in subject_wise_fold_indices(
            df,
            subject_col=subject_col,
            y=y_raw,
            config=fold_conf,
        ):
            X_tr_df = X_df.iloc[train_idx]
            X_te_df = X_df.iloc[test_idx]
            y_tr_raw = y_raw[train_idx]
            y_te_raw = y_raw[test_idx]
            sub_te = subj[test_idx]

            if resume_completed and fold_id in completed_folds:
                fold_id += 1
                continue

            try:
                selected_hp, inner_best_score = _nested_best_hyperparams(
                    cfg=cfg,
                    model_name=model_name,
                    seed=seed,
                    task=task,
                    subject_col=subject_col,
                    feat_cols=feat_cols,
                    train_df=df.iloc[train_idx].copy(),
                    target_col=target_col,
                )
                model, imputer, scaler, le, pred, y_score = _fit_predict_bundle(
                    cfg=cfg,
                    model_name=model_name,
                    seed=seed,
                    hyperparams={model_name: selected_hp},
                    X_train_df=X_tr_df,
                    X_test_df=X_te_df,
                    y_train_raw=y_tr_raw,
                    y_test_raw=y_te_raw,
                    task=task,
                )
            except ValueError as e:
                raise RuntimeError(
                    f"Fold {fold_id} label encoding failed (try cv.stratify: false or more data): {e}"
                ) from e

            pred_dir = out_dir / "predictions"
            fig_dir = out_dir / "figures"
            save_predictions_dataframe(
                pred_dir / f"{model_name}_fold{fold_id}.csv",
                y_true=y_te_raw,
                y_pred=pred,
                y_score=y_score,
                subject_id=sub_te,
                fold_id=fold_id,
                extra_columns={"model": [model_name] * len(pred)},
            )
            save_confusion_matrix_figure(
                y_te_raw,
                pred,
                fig_dir / f"cm_{model_name}_fold{fold_id}.png",
                title=f"{model_name} fold {fold_id}",
            )

            row = {
                "model": model_name,
                "fold": fold_id,
                **_metrics_row(task, y_te_raw, pred, y_score, label_encoder=le),
            }
            metrics_rows.append(row)
            training_mode = "tuned" if bool(_tuning_settings(cfg).get("enabled", False)) else "fixed"
            best_param_rows.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "training_mode": training_mode,
                    "inner_best_score": inner_best_score,
                    "selected_params_json": json.dumps(selected_hp, sort_keys=True, default=str),
                }
                )
            if save_models and save_fold_models:
                fold_model_path = save_model_bundle(
                    fold_models_dir / f"{model_name}_fold{fold_id}.joblib",
                    _model_bundle(
                        model=model,
                        imputer=imputer,
                        label_encoder=le,
                        feature_columns=feat_cols,
                        target_column=target_col,
                        subject_column=subject_col,
                        task=task,
                        random_seed=seed,
                        train_csv=train_csv,
                        target_dummy_columns=target_dummy_cols,
                        model_name=model_name,
                        artifact_kind="fold",
                        training_mode=training_mode,
                        fold=fold_id,
                        scaler=scaler,
                    ),
                )
                registry_rows.append(
                    _registry_row(
                        out_dir=out_dir,
                        artifact_path=fold_model_path,
                        experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                        dataset_origin=train_csv,
                        algorithm=model_name,
                        artifact_type="fold",
                        training_mode=training_mode,
                        feature_columns=feat_cols,
                        class_labels=le.classes_,
                        fold=fold_id,
                    )
                )
            fold_id += 1

        completed_folds = _completed_fold_ids(out_dir=out_dir, model_name=model_name, metrics_rows=metrics_rows)
        if save_models and save_final_model and len(completed_folds) >= fold_conf.n_splits:
            if resume_completed and _final_model_complete(
                out_dir=out_dir,
                model_name=model_name,
                best_param_rows=best_param_rows,
                registry_rows=registry_rows,
            ):
                _write_metrics_summary(out_dir=out_dir, metrics_rows=metrics_rows, task=task, names=names)
                if save_models and registry_rows:
                    write_model_registry(models_dir / "model_registry.json", registry_rows)
                _write_tuning_artifacts(out_dir, best_param_rows)
                continue
            final_hp, final_inner_score = _nested_best_hyperparams(
                cfg=cfg,
                model_name=model_name,
                seed=seed,
                task=task,
                subject_col=subject_col,
                feat_cols=feat_cols,
                train_df=df.copy(),
                target_col=target_col,
            )
            final_model, full_imputer, full_scaler, full_le = _fit_model_bundle_only(
                cfg=cfg,
                model_name=model_name,
                seed=seed,
                hyperparams={model_name: final_hp},
                X_train_df=X_df,
                y_train_raw=y_raw,
                task=task,
            )
            training_mode = "tuned" if bool(_tuning_settings(cfg).get("enabled", False)) else "fixed"
            final_model_path = save_model_bundle(
                models_dir / f"{model_name}_final.joblib",
                _model_bundle(
                    model=final_model,
                    imputer=full_imputer,
                    scaler=full_scaler,
                    label_encoder=full_le,
                    feature_columns=feat_cols,
                    target_column=target_col,
                    subject_column=subject_col,
                    task=task,
                    random_seed=seed,
                    train_csv=train_csv,
                    target_dummy_columns=target_dummy_cols,
                    model_name=model_name,
                    artifact_kind="final",
                    training_mode=training_mode,
                    fold=None,
                ),
            )
            best_param_rows.append(
                {
                    "model": model_name,
                    "fold": "final",
                    "training_mode": training_mode,
                    "inner_best_score": final_inner_score,
                    "selected_params_json": json.dumps(final_hp, sort_keys=True, default=str),
                }
            )
            registry_rows.append(
                _registry_row(
                    out_dir=out_dir,
                    artifact_path=final_model_path,
                    experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                    dataset_origin=train_csv,
                    algorithm=model_name,
                    artifact_type="final",
                    training_mode=training_mode,
                    feature_columns=feat_cols,
                    class_labels=full_le.classes_,
                )
            )

        _write_metrics_summary(out_dir=out_dir, metrics_rows=metrics_rows, task=task, names=names)
        if save_models and registry_rows:
            write_model_registry(models_dir / "model_registry.json", registry_rows)
        _write_tuning_artifacts(out_dir, best_param_rows)

    met_path = out_dir / "metrics_per_fold.csv"
    print(f"Wrote {met_path} and summary.json under {out_dir}")


def run_cross_dataset(
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    train_csv_path: Optional[Path] = None,
) -> None:
    seed = int(cfg.get("random_seed", 42))
    subject_col = str(cfg["subject_column"])
    target_col = str(cfg["target_column"])
    task = str(cfg.get("task", "multiclass"))
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'")

    train_df, target_col, train_target_dummy_cols = ensure_target_column(train_df, target_col_raw=target_col)
    eval_df, target_col, eval_target_dummy_cols = ensure_target_column(eval_df, target_col_raw=target_col)
    if target_col == "sleep_stage":
        train_df[target_col] = normalize_sleep_stage_series(train_df[target_col])
        eval_df[target_col] = normalize_sleep_stage_series(eval_df[target_col])
    train_df = _apply_label_subset(train_df, target_col, cfg)
    eval_df = _apply_label_subset(eval_df, target_col, cfg)

    exclude_cols = [subject_col, target_col, *cfg.get("feature_exclude", []), *train_target_dummy_cols, *eval_target_dummy_cols]
    feat_cols = resolve_feature_columns_cross(
        train_df,
        eval_df,
        exclude_cols,
        cfg.get("feature_include"),
    )
    enforce_single_channel_epoch_features(feat_cols, cfg)

    req_bin01 = bool(cfg.get("binary_require_zero_one_labels", True))
    if task == "binary":
        validate_binary_target_training(train_df[target_col], target_col, require_zero_one=req_bin01)
        validate_binary_target_eval(eval_df[target_col], target_col, require_zero_one=req_bin01)

    X_train_df = train_df[feat_cols]
    X_eval_df = eval_df[feat_cols]
    y_train_raw = train_df[target_col].values
    y_eval_raw = eval_df[target_col].values
    sub_eval = eval_df[subject_col].values

    models_on = cfg.get("models", {})
    names = []
    if models_on.get("random_forest", True):
        names.append("random_forest")
    if models_on.get("xgboost", True):
        names.append("xgboost")
    if models_on.get("svm_rbf", True):
        names.append("svm_rbf")
    if not names:
        raise ValueError("No models enabled under config.models")
    hp = cfg.get("hyperparams", {})
    save_models, _save_fold_models, save_final_model = _output_settings(cfg)
    train_csv = _train_csv_string(train_csv_path, cfg)
    models_dir = out_dir / "models"
    registry_rows: List[Dict[str, Any]] = []
    best_param_rows: List[Dict[str, Any]] = []

    metrics_rows: List[Dict[str, Any]] = []
    pred_dir = out_dir / "predictions"
    fig_dir = out_dir / "figures"

    for model_name in names:
        try:
            make_model(model_name, seed, hp, task=task)
        except RuntimeError as e:
            print(f"  Skip {model_name}: {e}")
            continue

        selected_hp, inner_best_score = _nested_best_hyperparams(
            cfg=cfg,
            model_name=model_name,
            seed=seed,
            task=task,
            subject_col=subject_col,
            feat_cols=feat_cols,
            train_df=train_df.copy(),
            target_col=target_col,
        )
        model, imputer, scaler, le, pred, y_score = _fit_predict_bundle(
            cfg=cfg,
            model_name=model_name,
            seed=seed,
            hyperparams={model_name: selected_hp},
            X_train_df=X_train_df,
            X_test_df=X_eval_df,
            y_train_raw=y_train_raw,
            y_test_raw=y_eval_raw,
            task=task,
        )
        training_mode = "tuned" if bool(_tuning_settings(cfg).get("enabled", False)) else "fixed"

        save_predictions_dataframe(
            pred_dir / f"{model_name}_cross_eval.csv",
            y_true=y_eval_raw,
            y_pred=pred,
            y_score=y_score,
            subject_id=sub_eval,
            fold_id=-1,
            extra_columns={"model": [model_name] * len(pred)},
        )
        save_confusion_matrix_figure(
            y_eval_raw,
            pred,
            fig_dir / f"cm_{model_name}_cross_eval.png",
            title=f"{model_name} cross-dataset",
        )
        row = {
            "model": model_name,
            "fold": "cross_eval",
            **_metrics_row(task, y_eval_raw, pred, y_score, label_encoder=le),
        }
        metrics_rows.append(row)
        best_param_rows.append(
            {
                "model": model_name,
                "fold": "cross_eval",
                "training_mode": training_mode,
                "inner_best_score": inner_best_score,
                "selected_params_json": json.dumps(selected_hp, sort_keys=True, default=str),
            }
        )
        if save_models and save_final_model:
            model_path = save_model_bundle(
                models_dir / f"{model_name}_final.joblib",
                _model_bundle(
                    model=model,
                    imputer=imputer,
                    scaler=scaler,
                    label_encoder=le,
                    feature_columns=feat_cols,
                    target_column=target_col,
                    subject_column=subject_col,
                    task=task,
                    random_seed=seed,
                    train_csv=train_csv,
                    target_dummy_columns=train_target_dummy_cols,
                    model_name=model_name,
                    artifact_kind="final",
                    training_mode=training_mode,
                    fold=None,
                ),
            )
            registry_rows.append(
                _registry_row(
                    out_dir=out_dir,
                    artifact_path=model_path,
                    experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                    dataset_origin=train_csv,
                    algorithm=model_name,
                    artifact_type="final",
                    training_mode=training_mode,
                    feature_columns=feat_cols,
                    class_labels=le.classes_,
                )
            )

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_cross_eval.csv", index=False)
    if save_models and registry_rows:
        write_model_registry(models_dir / "model_registry.json", registry_rows)
    _write_tuning_artifacts(out_dir, best_param_rows)
    print(f"Cross-dataset metrics written under {out_dir}")


def run_experiment(config_path: Path) -> None:
    cfg = load_config(config_path)
    root = Path(cfg.get("output", {}).get("root", "reports/experiments"))
    name = str(cfg.get("experiment_name", "run"))
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    train_path = resolve_csv_path(str(cfg["train_csv"]), config_path)
    train_df = read_table_file(train_path)
    train_df = ensure_subject_unit_column(train_df)

    cross = bool(cfg.get("cross_dataset", False))
    eval_path_raw = cfg.get("eval_csv")
    if cross:
        if not eval_path_raw:
            raise ValueError("cross_dataset true requires eval_csv")
        eval_path = resolve_csv_path(str(eval_path_raw), config_path)
        eval_df = read_table_file(eval_path)
        eval_df = ensure_subject_unit_column(eval_df)
        run_cross_dataset(cfg, train_df, eval_df, out_dir, train_csv_path=train_path)
    else:
        run_cv(cfg, train_df, out_dir, train_csv_path=train_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Phase E: subject-wise CV or cross-dataset eval from YAML config.")
    p.add_argument("--config", type=str, required=True, help="Path to experiment YAML.")
    args = p.parse_args(argv)
    run_experiment(Path(args.config))


if __name__ == "__main__":
    main()
