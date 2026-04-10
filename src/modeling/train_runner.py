"""
Phase E experiment runner: subject-wise CV and optional cross-dataset evaluation.

CLI: python -m modeling.train_runner --config path/to.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from modeling.artifacts import save_confusion_matrix_figure, save_predictions_dataframe
from modeling.cv_split import SubjectFoldConfig, subject_wise_fold_indices
from modeling.metrics import apnea_binary_metrics, fold_metrics_summary, multiclass_sleep_metrics
from modeling.subject_id import ensure_subject_unit_column

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
        "svm_rbf": {"C": 1.0, "gamma": "scale"},
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


def impute_fit_transform(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imp = SimpleImputer(strategy="median")
    Xt = imp.fit_transform(X_train)
    Xv = imp.transform(X_test)
    return Xt, Xv, imp


def make_model(
    name: str,
    random_state: int,
    hyperparams: Dict[str, Any],
) -> Any:
    hp = hyperparams.get(name, {})
    if name == "svm_rbf":
        base = _default_hyperparams()["svm_rbf"]
        base.update(hp)
        gamma = base["gamma"]
        if isinstance(gamma, str) and gamma not in ("scale", "auto"):
            gamma = float(gamma)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=float(base["C"]),
                        gamma=gamma,
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        base = _default_hyperparams()["random_forest"]
        base.update(hp)
        md = base.get("max_depth")
        return RandomForestClassifier(
            n_estimators=int(base["n_estimators"]),
            max_depth=None if md is None else int(md),
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
    return flat


def run_cv(
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    *,
    model_filter: Optional[Sequence[str]] = None,
) -> None:
    seed = int(cfg.get("random_seed", 42))
    subject_col = str(cfg["subject_column"])
    target_col = str(cfg["target_column"])
    task = str(cfg.get("task", "multiclass"))
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'")

    exclude = {subject_col, target_col, *cfg.get("feature_exclude", [])}
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

    models_on = cfg.get("models", {})
    names = []
    if models_on.get("svm_rbf", True):
        names.append("svm_rbf")
    if models_on.get("random_forest", True):
        names.append("random_forest")
    if models_on.get("xgboost", True):
        names.append("xgboost")
    if model_filter is not None:
        names = [n for n in names if n in model_filter]
    if not names:
        raise ValueError("No models enabled under config.models")

    hp = cfg.get("hyperparams", {})

    metrics_rows: List[Dict[str, Any]] = []

    for model_name in names:
        try:
            make_model(model_name, seed, hp)
        except RuntimeError as e:
            print(f"  Skip {model_name}: {e}")
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

            X_tr, X_te, _ = impute_fit_transform(X_tr_df, X_te_df)
            try:
                y_tr_enc, y_te_enc, le = encode_y_safe(y_tr_raw, y_te_raw)
            except ValueError as e:
                raise RuntimeError(
                    f"Fold {fold_id} label encoding failed (try cv.stratify: false or more data): {e}"
                ) from e

            model = make_model(model_name, seed, hp)
            model.fit(X_tr, y_tr_enc)
            pred_enc = model.predict(X_te)
            pred = le.inverse_transform(pred_enc.astype(int))

            y_score = None
            if task == "binary" and hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)
                if proba.shape[1] == 2:
                    y_score = proba[:, 1]

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
            fold_id += 1

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
        summary[mn] = fold_metrics_summary(sub, [k for k in keys if k in sub[0]])

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {met_path} and summary.json under {out_dir}")


def run_cross_dataset(cfg: Dict[str, Any], train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> None:
    seed = int(cfg.get("random_seed", 42))
    subject_col = str(cfg["subject_column"])
    target_col = str(cfg["target_column"])
    task = str(cfg.get("task", "multiclass"))
    if task not in ("binary", "multiclass"):
        raise ValueError("task must be 'binary' or 'multiclass'")

    exclude_cols = [subject_col, target_col, *cfg.get("feature_exclude", [])]
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

    X_tr, X_ev, _ = impute_fit_transform(X_train_df, X_eval_df)
    y_tr_enc, y_ev_enc, le = encode_y_safe(y_train_raw, y_eval_raw)

    models_on = cfg.get("models", {})
    names = []
    if models_on.get("svm_rbf", True):
        names.append("svm_rbf")
    if models_on.get("random_forest", True):
        names.append("random_forest")
    if models_on.get("xgboost", True):
        names.append("xgboost")
    if not names:
        raise ValueError("No models enabled under config.models")
    hp = cfg.get("hyperparams", {})

    metrics_rows: List[Dict[str, Any]] = []
    pred_dir = out_dir / "predictions"
    fig_dir = out_dir / "figures"

    for model_name in names:
        try:
            model = make_model(model_name, seed, hp)
        except RuntimeError as e:
            print(f"  Skip {model_name}: {e}")
            continue

        model.fit(X_tr, y_tr_enc)
        pred_enc = model.predict(X_ev)
        pred = le.inverse_transform(pred_enc.astype(int))

        y_score = None
        if task == "binary" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_ev)
            if proba.shape[1] == 2:
                y_score = proba[:, 1]

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

    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_cross_eval.csv", index=False)
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
    train_df = pd.read_csv(train_path)
    train_df = ensure_subject_unit_column(train_df)

    cross = bool(cfg.get("cross_dataset", False))
    eval_path_raw = cfg.get("eval_csv")
    if cross:
        if not eval_path_raw:
            raise ValueError("cross_dataset true requires eval_csv")
        eval_path = resolve_csv_path(str(eval_path_raw), config_path)
        eval_df = pd.read_csv(eval_path)
        eval_df = ensure_subject_unit_column(eval_df)
        run_cross_dataset(cfg, train_df, eval_df, out_dir)
    else:
        run_cv(cfg, train_df, out_dir)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Phase E: subject-wise CV or cross-dataset eval from YAML config.")
    p.add_argument("--config", type=str, required=True, help="Path to experiment YAML.")
    args = p.parse_args(argv)
    run_experiment(Path(args.config))


if __name__ == "__main__":
    main()
