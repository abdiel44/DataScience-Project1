"""Build LaTeX-ready tables, figures, and statistical summaries for the final report."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.metrics import mcnemar_exact


REPORT_ROOT = ROOT / "report" / "latex"
GENERATED_ROOT = REPORT_ROOT / "generated"
TABLES_ROOT = GENERATED_ROOT / "tables"
FIGURES_ROOT = GENERATED_ROOT / "figures"
DATA_ROOT = GENERATED_ROOT / "data"

MODEL_ORDER = ["svm_rbf", "random_forest", "xgboost"]
MODEL_LABELS = {
    "svm_rbf": "SVM-RBF",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
STAGE_LABEL_ORDER = ["W", "N1", "N2", "N3", "REM"]


def _ensure_dirs() -> None:
    for path in (GENERATED_ROOT, TABLES_ROOT, FIGURES_ROOT, DATA_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _escape_tex(value: Any) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    return text


def _metric_stats(df: pd.DataFrame, group_col: str, metric_cols: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    if df.empty:
        return out
    for model_name, group in df.groupby(group_col):
        model_stats: Dict[str, Dict[str, float]] = {}
        for metric in metric_cols:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue
            model_stats[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        out[str(model_name)] = model_stats
    return out


def _fmt_mean_std(stats: Optional[Mapping[str, float]], *, nd: str = "N/D") -> str:
    if not stats:
        return nd
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None or not np.isfinite(float(mean)):
        return nd
    return f"{float(mean):.3f} $\\pm$ {float(std):.3f}"


def _fmt_float(value: Any, *, nd: str = "N/D") -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return nd
    if not np.isfinite(value_f):
        return nd
    return f"{value_f:.3f}"


def _bootstrap_mean_ci(values: Sequence[float], *, seed: int = 42, n_boot: int = 5000) -> Tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _extract_fold_id(path: Path) -> int:
    match = re.search(r"_fold(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Could not parse fold id from {path.name!r}")
    return int(match.group(1))


def _load_prediction_bundle(exp_dir: Path, subdir: str, model_name: str) -> pd.DataFrame:
    pred_dir = exp_dir / subdir
    parts: List[pd.DataFrame] = []
    for path in sorted(pred_dir.glob(f"{model_name}_fold*.csv")):
        frame = pd.read_csv(path)
        frame["fold_id"] = int(frame["fold_id"].iloc[0]) if "fold_id" in frame.columns else _extract_fold_id(path)
        frame["_row_id"] = np.arange(len(frame), dtype=int)
        parts.append(frame)
    if not parts:
        raise FileNotFoundError(f"No predictions for {model_name} under {pred_dir}")
    return pd.concat(parts, ignore_index=True)


def _align_predictions(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = df_a.merge(df_b, on=["fold_id", "_row_id"], suffixes=("_a", "_b"))
    if not np.array_equal(merged["y_true_a"].astype(str).to_numpy(), merged["y_true_b"].astype(str).to_numpy()):
        raise ValueError("Prediction alignment failed: y_true mismatch between models.")
    return merged


def _bootstrap_auc_diff(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    *,
    seed: int = 42,
    n_boot: int = 5000,
) -> Tuple[float, float, float]:
    yt = np.asarray(y_true, dtype=int)
    sa = np.asarray(score_a, dtype=float)
    sb = np.asarray(score_b, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(sa) & np.isfinite(sb)
    yt = yt[valid]
    sa = sa[valid]
    sb = sb[valid]
    base = float(roc_auc_score(yt, sa) - roc_auc_score(yt, sb))
    rng = np.random.default_rng(seed)
    diffs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt), len(yt))
        yt_b = yt[idx]
        if np.unique(yt_b).size < 2:
            continue
        diffs.append(float(roc_auc_score(yt_b, sa[idx]) - roc_auc_score(yt_b, sb[idx])))
    if not diffs:
        return base, math.nan, math.nan
    lo, hi = np.percentile(np.asarray(diffs, dtype=float), [2.5, 97.5])
    return base, float(lo), float(hi)


def _write_table_tex(
    *,
    path: Path,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    alignment: str,
    wide: bool = False,
) -> None:
    env = "table*" if wide else "table"
    width_cmd = r"\textwidth" if wide else r"\columnwidth"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\resizebox{{{width_cmd}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\end{{{env}}}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_confusion_figure(path: Path, y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any], title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_roc_figure(path: Path, y_true: Sequence[int], y_score: Sequence[float], title: str) -> None:
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    fpr, tpr, _ = roc_curve(yt, ys)
    auc = roc_auc_score(yt, ys)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance(path_png: Path, path_csv: Path, bundle_path: Path, title: str, *, top_k: int = 10) -> List[Dict[str, Any]]:
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_names = list(bundle.get("feature_columns", []))
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model at {bundle_path} does not expose feature_importances_.")
    importances = np.asarray(model.feature_importances_, dtype=float)
    frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .head(top_k)
    )
    frame.to_csv(path_csv, index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(frame["feature"][::-1], frame["importance"][::-1], color="#1f77b4")
    ax.set_xlabel("Feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return frame.to_dict(orient="records")


def _copy_best_existing_figure(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def build_assets() -> Dict[str, Any]:
    _ensure_dirs()

    sleep_dir = ROOT / "reports" / "experiments" / "sleep_edf_expanded_tuned"
    mit_dir = ROOT / "reports" / "experiments" / "mitbih_apnea_stage_classic"
    cross_sleep_isruc_dir = ROOT / "reports" / "experiments" / "cross_sleep_edf_to_isruc_n123"
    cross_isruc_sleep_dir = ROOT / "reports" / "experiments" / "cross_isruc_to_sleep_edf_n123"
    cross_mit_stv_dir = ROOT / "reports" / "experiments" / "cross_dataset_mitbih_to_st_vincent_classic"
    cross_stv_mit_dir = ROOT / "reports" / "experiments" / "cross_dataset_st_vincent_to_mitbih_classic"

    sleep_metrics = _read_csv(sleep_dir / "metrics_per_fold.csv")
    mit_metrics = _read_csv(mit_dir / "metrics_per_fold.csv")
    cross_sleep_isruc = _read_csv(cross_sleep_isruc_dir / "metrics_cross_eval.csv")
    cross_isruc_sleep = _read_csv(cross_isruc_sleep_dir / "metrics_cross_eval.csv")
    cross_mit_stv = _read_csv(cross_mit_stv_dir / "metrics_cross_eval.csv")
    cross_stv_mit = _read_csv(cross_stv_mit_dir / "metrics_cross_eval.csv")

    sleep_stats = _metric_stats(
        sleep_metrics,
        "model",
        ["accuracy", "macro_f1", "cohen_kappa", "per_class_f1_n1", "per_class_f1_n2", "per_class_f1_n3", "per_class_f1_rem", "per_class_f1_w"],
    )
    mit_stage_stats = _metric_stats(mit_metrics, "model", ["stage_accuracy", "stage_macro_f1", "stage_cohen_kappa"])
    mit_apnea_stats = _metric_stats(mit_metrics, "model", ["apnea_accuracy", "apnea_sensitivity", "apnea_specificity", "apnea_auc_roc"])

    table1_rows = []
    for metric, label in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("cohen_kappa", "Kappa"),
    ]:
        row = [_escape_tex(label)]
        for model_name in MODEL_ORDER:
            row.append(_fmt_mean_std(sleep_stats.get(model_name, {}).get(metric)))
        table1_rows.append(row)
    _write_table_tex(
        path=TABLES_ROOT / "table_stage_intra.tex",
        caption="Sleep-EDF Expanded tuned: metricas promedio (media$\\pm$desviacion estandar) por modelo para staging intra-dataset.",
        label="tab:stage_intra",
        headers=["Metrica", "SVM-RBF", "Random Forest", "XGBoost"],
        rows=table1_rows,
        alignment="lccc",
        wide=False,
    )

    table2_rows = []
    for class_name, metric in [("N1", "per_class_f1_n1"), ("N2", "per_class_f1_n2"), ("N3", "per_class_f1_n3"), ("REM", "per_class_f1_rem"), ("W", "per_class_f1_w")]:
        row = [_escape_tex(class_name)]
        for model_name in MODEL_ORDER:
            row.append(_fmt_mean_std(sleep_stats.get(model_name, {}).get(metric)))
        table2_rows.append(row)
    _write_table_tex(
        path=TABLES_ROOT / "table_stage_f1_by_class.tex",
        caption="Sleep-EDF Expanded tuned: F1 por clase en staging (media$\\pm$desviacion estandar sobre folds).",
        label="tab:stage_f1_class",
        headers=["Clase", "SVM-RBF", "Random Forest", "XGBoost"],
        rows=table2_rows,
        alignment="lccc",
        wide=False,
    )

    table3_rows = []
    for model_name in MODEL_ORDER:
        row = [
            _escape_tex(MODEL_LABELS[model_name]),
            _fmt_mean_std(mit_apnea_stats.get(model_name, {}).get("apnea_accuracy")),
            _fmt_mean_std(mit_apnea_stats.get(model_name, {}).get("apnea_sensitivity")),
            _fmt_mean_std(mit_apnea_stats.get(model_name, {}).get("apnea_specificity")),
            _fmt_mean_std(mit_apnea_stats.get(model_name, {}).get("apnea_auc_roc")),
            _fmt_mean_std(mit_stage_stats.get(model_name, {}).get("stage_accuracy")),
            _fmt_mean_std(mit_stage_stats.get(model_name, {}).get("stage_macro_f1")),
            _fmt_mean_std(mit_stage_stats.get(model_name, {}).get("stage_cohen_kappa")),
        ]
        table3_rows.append(row)
    _write_table_tex(
        path=TABLES_ROOT / "table_mitbih_multitarget.tex",
        caption="MIT-BIH PSG multitarget: media$\\pm$desviacion estandar por modelo para apnea y staging.",
        label="tab:mitbih_multitarget",
        headers=[
            "Modelo",
            "Apnea Acc.",
            "Sens.",
            "Spec.",
            "AUC",
            "Stage Acc.",
            "Stage Macro-F1",
            "Stage Kappa",
        ],
        rows=table3_rows,
        alignment="lccccccc",
        wide=True,
    )

    cross_stage_rows = []
    for direction, frame in [
        ("Sleep-EDF -> ISRUC", cross_sleep_isruc),
        ("ISRUC -> Sleep-EDF", cross_isruc_sleep),
        ("MIT-BIH -> St. Vincent", cross_mit_stv),
        ("St. Vincent -> MIT-BIH", cross_stv_mit),
    ]:
        for _, row in frame.iterrows():
            cross_stage_rows.append(
                [
                    _escape_tex(direction),
                    _escape_tex(MODEL_LABELS.get(str(row["model"]), str(row["model"]))),
                    _fmt_float(row["stage_accuracy"] if "stage_accuracy" in row else row.get("accuracy")),
                    _fmt_float(row["stage_macro_f1"] if "stage_macro_f1" in row else row.get("macro_f1")),
                    _fmt_float(row["stage_cohen_kappa"] if "stage_cohen_kappa" in row else row.get("cohen_kappa")),
                ]
            )
    _write_table_tex(
        path=TABLES_ROOT / "table_cross_stage.tex",
        caption="Resultados cross-dataset de staging para las direcciones evaluadas.",
        label="tab:cross_stage",
        headers=["Direccion", "Modelo", "Accuracy", "Macro-F1", "Kappa"],
        rows=cross_stage_rows,
        alignment="llccc",
        wide=True,
    )

    cross_apnea_rows = []
    for direction, frame in [
        ("MIT-BIH -> St. Vincent", cross_mit_stv),
        ("St. Vincent -> MIT-BIH", cross_stv_mit),
    ]:
        for _, row in frame.iterrows():
            cross_apnea_rows.append(
                [
                    _escape_tex(direction),
                    _escape_tex(MODEL_LABELS.get(str(row["model"]), str(row["model"]))),
                    _fmt_float(row["apnea_accuracy"]),
                    _fmt_float(row["apnea_sensitivity"]),
                    _fmt_float(row["apnea_specificity"]),
                    _fmt_float(row["apnea_auc_roc"]),
                ]
            )
    _write_table_tex(
        path=TABLES_ROOT / "table_cross_apnea.tex",
        caption="Resultados cross-dataset de apnea para las direcciones evaluadas.",
        label="tab:cross_apnea",
        headers=["Direccion", "Modelo", "Accuracy", "Sensitivity", "Specificity", "AUC-ROC"],
        rows=cross_apnea_rows,
        alignment="llcccc",
        wide=True,
    )

    sleep_boot_rows = []
    for model_name in MODEL_ORDER:
        model_df = sleep_metrics[sleep_metrics["model"] == model_name]
        if model_df.empty:
            continue
        acc = _bootstrap_mean_ci(pd.to_numeric(model_df["accuracy"], errors="coerce"))
        macro_f1 = _bootstrap_mean_ci(pd.to_numeric(model_df["macro_f1"], errors="coerce"))
        kappa = _bootstrap_mean_ci(pd.to_numeric(model_df["cohen_kappa"], errors="coerce"))
        sleep_boot_rows.append(
            [
                _escape_tex(MODEL_LABELS[model_name]),
                f"{acc[0]:.3f} [{acc[1]:.3f}, {acc[2]:.3f}]",
                f"{macro_f1[0]:.3f} [{macro_f1[1]:.3f}, {macro_f1[2]:.3f}]",
                f"{kappa[0]:.3f} [{kappa[1]:.3f}, {kappa[2]:.3f}]",
            ]
        )
    _write_table_tex(
        path=TABLES_ROOT / "table_bootstrap_sleep_stage.tex",
        caption="Intervalos de confianza bootstrap para staging intra-dataset en Sleep-EDF Expanded tuned.",
        label="tab:bootstrap_sleep_stage",
        headers=["Modelo", "Accuracy", "Macro-F1", "Kappa"],
        rows=sleep_boot_rows,
        alignment="lccc",
        wide=False,
    )

    mit_boot_rows = []
    for model_name in MODEL_ORDER:
        model_df = mit_metrics[mit_metrics["model"] == model_name]
        if model_df.empty:
            continue
        auc = _bootstrap_mean_ci(pd.to_numeric(model_df["apnea_auc_roc"], errors="coerce"))
        stage_acc = _bootstrap_mean_ci(pd.to_numeric(model_df["stage_accuracy"], errors="coerce"))
        mit_boot_rows.append(
            [
                _escape_tex(MODEL_LABELS[model_name]),
                f"{auc[0]:.3f} [{auc[1]:.3f}, {auc[2]:.3f}]",
                f"{stage_acc[0]:.3f} [{stage_acc[1]:.3f}, {stage_acc[2]:.3f}]",
            ]
        )
    _write_table_tex(
        path=TABLES_ROOT / "table_bootstrap_mitbih.tex",
        caption="Intervalos de confianza bootstrap para MIT-BIH en apnea (AUC) y staging (accuracy).",
        label="tab:bootstrap_mitbih",
        headers=["Modelo", "Apnea AUC-ROC", "Stage Accuracy"],
        rows=mit_boot_rows,
        alignment="lcc",
        wide=False,
    )

    sleep_best = "xgboost"
    sleep_second = "random_forest"
    mit_stage_best = "xgboost"
    mit_stage_second = "random_forest"
    mit_apnea_best = "xgboost"
    mit_apnea_second = "random_forest"

    sleep_best_pred = _load_prediction_bundle(sleep_dir, "predictions", sleep_best)
    sleep_second_pred = _load_prediction_bundle(sleep_dir, "predictions", sleep_second)
    sleep_merged = _align_predictions(sleep_best_pred, sleep_second_pred)
    sleep_mcnemar = mcnemar_exact(
        sleep_merged["y_true_a"].astype(str).to_numpy(),
        sleep_merged["y_pred_a"].astype(str).to_numpy(),
        sleep_merged["y_pred_b"].astype(str).to_numpy(),
    )

    mit_stage_best_pred = _load_prediction_bundle(mit_dir, "predictions_stage", mit_stage_best)
    mit_stage_second_pred = _load_prediction_bundle(mit_dir, "predictions_stage", mit_stage_second)
    mit_stage_merged = _align_predictions(mit_stage_best_pred, mit_stage_second_pred)
    mit_stage_mcnemar = mcnemar_exact(
        mit_stage_merged["y_true_a"].astype(str).to_numpy(),
        mit_stage_merged["y_pred_a"].astype(str).to_numpy(),
        mit_stage_merged["y_pred_b"].astype(str).to_numpy(),
    )

    mit_apnea_best_pred = _load_prediction_bundle(mit_dir, "predictions_apnea", mit_apnea_best)
    mit_apnea_second_pred = _load_prediction_bundle(mit_dir, "predictions_apnea", mit_apnea_second)
    mit_apnea_merged = _align_predictions(mit_apnea_best_pred, mit_apnea_second_pred)
    mit_apnea_mcnemar = mcnemar_exact(
        pd.to_numeric(mit_apnea_merged["y_true_a"], errors="coerce").fillna(0).astype(int).to_numpy(),
        pd.to_numeric(mit_apnea_merged["y_pred_a"], errors="coerce").fillna(0).astype(int).to_numpy(),
        pd.to_numeric(mit_apnea_merged["y_pred_b"], errors="coerce").fillna(0).astype(int).to_numpy(),
    )
    auc_diff = _bootstrap_auc_diff(
        pd.to_numeric(mit_apnea_merged["y_true_a"], errors="coerce").fillna(0).astype(int).to_numpy(),
        pd.to_numeric(mit_apnea_merged["y_score_a"], errors="coerce").to_numpy(),
        pd.to_numeric(mit_apnea_merged["y_score_b"], errors="coerce").to_numpy(),
    )

    stats_test_rows = [
        [
            "Sleep-EDF staging",
            f"{MODEL_LABELS[sleep_best]} vs {MODEL_LABELS[sleep_second]}",
            f"{sleep_mcnemar[0]:.3f}",
            f"{sleep_mcnemar[1]:.4f}",
            "N/A",
        ],
        [
            "MIT-BIH staging",
            f"{MODEL_LABELS[mit_stage_best]} vs {MODEL_LABELS[mit_stage_second]}",
            f"{mit_stage_mcnemar[0]:.3f}",
            f"{mit_stage_mcnemar[1]:.4f}",
            "N/A",
        ],
        [
            "MIT-BIH apnea",
            f"{MODEL_LABELS[mit_apnea_best]} vs {MODEL_LABELS[mit_apnea_second]}",
            f"{mit_apnea_mcnemar[0]:.3f}",
            f"{mit_apnea_mcnemar[1]:.4f}",
            f"{auc_diff[0]:.3f} [{auc_diff[1]:.3f}, {auc_diff[2]:.3f}]",
        ],
    ]
    _write_table_tex(
        path=TABLES_ROOT / "table_stat_tests.tex",
        caption="Pruebas pareadas: McNemar exacto y diferencia bootstrap de AUC.",
        label="tab:stat_tests",
        headers=["Escenario", "Comparacion", "McNemar", "p-value", "Delta AUC"],
        rows=stats_test_rows,
        alignment="llccc",
        wide=True,
    )

    literature_rows = [
        [r"\cite{mousavi2019sleepeegnet}", "Sleep-EDF", "EEG monocanal", "Sleep staging", "Seq2Seq deep", "No", "Usa deep learning; comparable solo por dataset y tarea"],
        [r"\cite{barnes2022apneaeegcnn}", "Apnea cohort", "EEG monocanal", "Apnea", "CNN explicable", "Parcial", "Comparable por tarea y canal, no por familia de modelo"],
        [r"\cite{gao2023psdrf}", "Sleep-EDF", "Fpz-Cz / Pz-Oz", "Sleep staging", "PSD + Random Forest", "Si", "Muy cercano a nuestro baseline de staging"],
        [r"\cite{nakamura2017complexity}", "Sleep EEG", "EEG", "Sleep staging", "Complejidad/HMM-style", "Parcial", "Comparable por monocanal y features, no por pipeline final"],
        [r"\cite{ghimatgar2019hmm}", "Sleep EEG", "EEG monocanal", "Sleep staging", "HMM", "Parcial", "Sirve como contraste secuencial, no como linea principal"],
        [r"\cite{li2022cascadedsvm}", "Sleep EEG", "EEG monocanal", "Sleep staging", "SVM en cascada", "Si", "Muy comparable con la familia SVM del proyecto"],
        [r"\cite{wang2024svmxgboost}", "Sleep EEG", "EEG", "Sleep staging", "SVM + XGBoost", "Si", "Comparable por familias clasicas y objetivo"],
        [r"\cite{satapathy2024multimodality}", "Multimodal sleep data", "Multimodal", "Sleep staging", "ML multimodal", "No", "Sirve como referencia externa, pero no es justo por usar mas senales"],
    ]
    _write_table_tex(
        path=TABLES_ROOT / "table_literature_baselines.tex",
        caption="Baselines publicados y grado de comparabilidad con el protocolo del proyecto.",
        label="tab:literature_baselines",
        headers=["Referencia", "Dataset", "Canal", "Tarea", "Modelo", "Comparable", "Nota"],
        rows=literature_rows,
        alignment="lllllll",
        wide=True,
    )

    _save_confusion_figure(
        FIGURES_ROOT / "sleep_stage_best_confusion.png",
        sleep_best_pred["y_true"].astype(str).to_numpy(),
        sleep_best_pred["y_pred"].astype(str).to_numpy(),
        STAGE_LABEL_ORDER,
        "Sleep-EDF Expanded tuned: confusion matrix (best staging model)",
    )
    _save_confusion_figure(
        FIGURES_ROOT / "mitbih_apnea_best_confusion.png",
        pd.to_numeric(mit_apnea_best_pred["y_true"], errors="coerce").fillna(0).astype(int).to_numpy(),
        pd.to_numeric(mit_apnea_best_pred["y_pred"], errors="coerce").fillna(0).astype(int).to_numpy(),
        [0, 1],
        "MIT-BIH PSG: confusion matrix (best apnea model)",
    )
    _save_roc_figure(
        FIGURES_ROOT / "mitbih_apnea_best_roc.png",
        pd.to_numeric(mit_apnea_best_pred["y_true"], errors="coerce").fillna(0).astype(int).to_numpy(),
        pd.to_numeric(mit_apnea_best_pred["y_score"], errors="coerce").to_numpy(),
        "MIT-BIH PSG: ROC curve (best apnea model)",
    )

    importance_sleep_xgb = _save_feature_importance(
        FIGURES_ROOT / "sleep_stage_xgboost_importance.png",
        DATA_ROOT / "sleep_stage_xgboost_importance.csv",
        sleep_dir / "models" / "xgboost_final.joblib",
        "Sleep-EDF staging: XGBoost feature importance",
    )
    importance_sleep_rf = _save_feature_importance(
        FIGURES_ROOT / "sleep_stage_random_forest_importance.png",
        DATA_ROOT / "sleep_stage_random_forest_importance.csv",
        sleep_dir / "models" / "random_forest_final.joblib",
        "Sleep-EDF staging: Random Forest feature importance",
    )
    importance_mit_apnea_xgb = _save_feature_importance(
        FIGURES_ROOT / "mitbih_apnea_xgboost_importance.png",
        DATA_ROOT / "mitbih_apnea_xgboost_importance.csv",
        mit_dir / "models" / "apnea" / "xgboost_final.joblib",
        "MIT-BIH apnea: XGBoost feature importance",
    )
    importance_mit_stage_rf = _save_feature_importance(
        FIGURES_ROOT / "mitbih_stage_random_forest_importance.png",
        DATA_ROOT / "mitbih_stage_random_forest_importance.csv",
        mit_dir / "models" / "stage" / "random_forest_final.joblib",
        "MIT-BIH staging: Random Forest feature importance",
    )

    report_summary = {
        "sleep_stage_intra_best_accuracy": "xgboost",
        "sleep_stage_intra_best_macro_f1": "random_forest",
        "mitbih_apnea_best_auc": "xgboost",
        "mitbih_stage_best_accuracy": "xgboost",
        "sleep_stage_mcnemar": {"statistic": sleep_mcnemar[0], "p_value": sleep_mcnemar[1]},
        "mitbih_stage_mcnemar": {"statistic": mit_stage_mcnemar[0], "p_value": mit_stage_mcnemar[1]},
        "mitbih_apnea_mcnemar": {"statistic": mit_apnea_mcnemar[0], "p_value": mit_apnea_mcnemar[1]},
        "mitbih_apnea_auc_diff": {"delta": auc_diff[0], "ci_low": auc_diff[1], "ci_high": auc_diff[2]},
        "importance_files": {
            "sleep_stage_xgboost": importance_sleep_xgb,
            "sleep_stage_random_forest": importance_sleep_rf,
            "mitbih_apnea_xgboost": importance_mit_apnea_xgb,
            "mitbih_stage_random_forest": importance_mit_stage_rf,
        },
    }
    (DATA_ROOT / "report_summary.json").write_text(json.dumps(report_summary, indent=2), encoding="utf-8")
    return report_summary


def main() -> None:
    summary = build_assets()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
