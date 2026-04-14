from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.metrics import fold_metrics_summary, multiclass_sleep_metrics  # noqa: E402


EXPERIMENTS = [
    ("sleep_edf_expanded", "within_dataset", "Sleep-EDF baseline"),
    ("sleep_edf_expanded_tuned", "within_dataset", "Sleep-EDF tuned"),
    ("sleep_edf_2013_fpzcz", "within_dataset", "Sleep-EDF 2013 comparable baseline"),
    ("sleep_edf_2013_fpzcz_tuned", "within_dataset", "Sleep-EDF 2013 comparable tuned"),
    ("isruc_sleep_stage", "within_dataset", "ISRUC baseline"),
    ("isruc_sleep_stage_tuned", "within_dataset", "ISRUC tuned"),
    ("cross_sleep_edf_to_isruc_n123", "cross_dataset", "Sleep-EDF -> ISRUC (N1/N2/N3)"),
    ("cross_isruc_to_sleep_edf_n123", "cross_dataset", "ISRUC -> Sleep-EDF (N1/N2/N3)"),
]

SLEEP_EEGNET_LABELS = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _metric(summary: Dict[str, Any], model: str, metric: str) -> tuple[Any, Any]:
    model_block = summary.get(model, {})
    metric_block = model_block.get(metric, {})
    return metric_block.get("mean"), metric_block.get("std")


def _sleep_eegnet_summary(root: Path) -> List[Dict[str, Any]]:
    baseline_dir = root / "Baselines" / "SleepEEGNet-master" / "SleepEEGNet-master" / "outputs_2013" / "outputs_eeg_fpz_cz"
    if not baseline_dir.is_dir():
        return []

    rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
    for npz_path in sorted(baseline_dir.glob("output_fold*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        y_true = np.vectorize(SLEEP_EEGNET_LABELS.get)(data["y_true"])
        y_pred = np.vectorize(SLEEP_EEGNET_LABELS.get)(data["y_pred"])
        metrics = multiclass_sleep_metrics(y_true, y_pred, labels=["W", "N1", "N2", "N3", "REM"])
        row = {
            "model": "sleep_eegnet_fpz_cz",
            "fold": npz_path.stem.replace("output_fold", ""),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "cohen_kappa": metrics["cohen_kappa"],
        }
        for label, score in metrics["per_class_f1"].items():
            row[f"per_class_f1_{str(label).lower()}"] = score
        metric_rows.append(row)

    if not metric_rows:
        return []

    summary = fold_metrics_summary(
        metric_rows,
        [
            "accuracy",
            "macro_f1",
            "cohen_kappa",
            "per_class_f1_w",
            "per_class_f1_n1",
            "per_class_f1_n2",
            "per_class_f1_n3",
            "per_class_f1_rem",
        ],
    )
    rows.append(
        {
            "experiment_name": "sleep_eegnet_outputs_2013_fpz_cz",
            "experiment_type": "external_baseline",
            "label": "SleepEEGNet 2013 Fpz-Cz",
            "model": "sleep_eegnet_fpz_cz",
            "accuracy_mean": summary.get("accuracy", {}).get("mean"),
            "accuracy_std": summary.get("accuracy", {}).get("std"),
            "macro_f1_mean": summary.get("macro_f1", {}).get("mean"),
            "macro_f1_std": summary.get("macro_f1", {}).get("std"),
            "cohen_kappa_mean": summary.get("cohen_kappa", {}).get("mean"),
            "cohen_kappa_std": summary.get("cohen_kappa", {}).get("std"),
            "per_class_f1_n1_mean": summary.get("per_class_f1_n1", {}).get("mean"),
            "per_class_f1_n1_std": summary.get("per_class_f1_n1", {}).get("std"),
            "summary_path": str(baseline_dir),
        }
    )
    return rows


def main() -> None:
    root = Path("reports/experiments")
    rows: List[Dict[str, Any]] = []
    md_lines: List[str] = ["# Staging Phase E Summary", ""]

    for exp_name, exp_type, label in EXPERIMENTS:
        summary_path = root / exp_name / "summary.json"
        if not summary_path.is_file():
            summary_path = root / exp_name / "metrics_cross_eval.csv"
        if not summary_path.exists():
            continue

        if summary_path.name == "summary.json":
            summary = _load_summary(summary_path)
            for model_name in sorted(summary.keys()):
                acc_mean, acc_std = _metric(summary, model_name, "accuracy")
                f1_mean, f1_std = _metric(summary, model_name, "macro_f1")
                kap_mean, kap_std = _metric(summary, model_name, "cohen_kappa")
                n1_mean, n1_std = _metric(summary, model_name, "per_class_f1_n1")
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "experiment_type": exp_type,
                        "label": label,
                        "model": model_name,
                        "accuracy_mean": acc_mean,
                        "accuracy_std": acc_std,
                        "macro_f1_mean": f1_mean,
                        "macro_f1_std": f1_std,
                        "cohen_kappa_mean": kap_mean,
                        "cohen_kappa_std": kap_std,
                        "per_class_f1_n1_mean": n1_mean,
                        "per_class_f1_n1_std": n1_std,
                        "summary_path": str(summary_path),
                    }
                )
        else:
            df = pd.read_csv(summary_path)
            for _, row in df.iterrows():
                rows.append(
                    {
                        "experiment_name": exp_name,
                        "experiment_type": exp_type,
                        "label": label,
                        "model": row.get("model"),
                        "accuracy_mean": row.get("accuracy"),
                        "accuracy_std": None,
                        "macro_f1_mean": row.get("macro_f1"),
                        "macro_f1_std": None,
                        "cohen_kappa_mean": row.get("cohen_kappa"),
                        "cohen_kappa_std": None,
                        "per_class_f1_n1_mean": row.get("per_class_f1_n1"),
                        "per_class_f1_n1_std": None,
                        "summary_path": str(summary_path),
                    }
                )

    rows.extend(_sleep_eegnet_summary(ROOT))
    out_df = pd.DataFrame(rows)

    baseline_row = out_df.loc[out_df["experiment_name"] == "sleep_eegnet_outputs_2013_fpz_cz"]
    if not baseline_row.empty:
        baseline = baseline_row.iloc[0]
        mask = out_df["experiment_name"].isin(
            ["sleep_edf_expanded", "sleep_edf_expanded_tuned", "sleep_edf_2013_fpzcz", "sleep_edf_2013_fpzcz_tuned"]
        )
        out_df.loc[mask, "gap_vs_sleep_eegnet_accuracy"] = (
            out_df.loc[mask, "accuracy_mean"].astype(float) - float(baseline["accuracy_mean"])
        )
        out_df.loc[mask, "gap_vs_sleep_eegnet_macro_f1"] = (
            out_df.loc[mask, "macro_f1_mean"].astype(float) - float(baseline["macro_f1_mean"])
        )
        out_df.loc[mask, "gap_vs_sleep_eegnet_kappa"] = (
            out_df.loc[mask, "cohen_kappa_mean"].astype(float) - float(baseline["cohen_kappa_mean"])
        )
        out_df.loc[mask, "gap_vs_sleep_eegnet_n1_f1"] = (
            out_df.loc[mask, "per_class_f1_n1_mean"].astype(float) - float(baseline["per_class_f1_n1_mean"])
        )

    out_csv = root / "staging_phase_e_summary.csv"
    out_md = root / "staging_phase_e_summary.md"
    out_df.to_csv(out_csv, index=False)

    if out_df.empty:
        md_lines.append("No experiment summaries were found.")
    else:
        for exp_name, block in out_df.groupby("experiment_name", sort=False):
            label = block["label"].iloc[0]
            md_lines.append(f"## {label}")
            for _, row in block.iterrows():
                line = (
                    f"- `{row['model']}`: accuracy={row['accuracy_mean']}, macro_f1={row['macro_f1_mean']}, "
                    f"kappa={row['cohen_kappa_mean']}, N1_F1={row['per_class_f1_n1_mean']}"
                )
                if pd.notna(row.get("gap_vs_sleep_eegnet_accuracy")):
                    line += (
                        f", gap_vs_sleep_eegnet_acc={row['gap_vs_sleep_eegnet_accuracy']}, "
                        f"gap_vs_sleep_eegnet_macro_f1={row['gap_vs_sleep_eegnet_macro_f1']}"
                    )
                md_lines.append(line)
            md_lines.append("")

    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
