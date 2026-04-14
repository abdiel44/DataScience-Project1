"""
Build a before/after summary for Fase D from raw EDA outputs and processed CSVs.

Outputs:
  - reports/phase_d_before_after_summary.csv
  - reports/phase_d_before_after_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from main import prepare_processed_df_for_eda  # noqa: E402


DATASETS: List[Dict[str, str]] = [
    {
        "dataset_task": "ISRUC",
        "target_col": "event_group",
        "raw_csv": "data/processed/isruc_sleep_raw.csv",
        "prep_csv": "data/processed/isruc_sleep_prep.csv",
        "eda_raw_dir": "reports/eda_raw/isruc_sleep_raw",
        "eda_processed_dir": "reports/eda_processed/isruc_sleep",
        "scaling_summary": "reports/scaling/isruc_sleep/scaling_summary.md",
    },
    {
        "dataset_task": "St Vincent",
        "target_col": "stage_mode",
        "raw_csv": "data/processed/st_vincent_apnea_raw.csv",
        "prep_csv": "data/processed/st_vincent_apnea_prep.csv",
        "eda_raw_dir": "reports/eda_raw/st_vincent_apnea_raw",
        "eda_processed_dir": "reports/eda_processed/st_vincent_apnea",
        "scaling_summary": "reports/scaling/st_vincent_apnea/scaling_summary.md",
    },
    {
        "dataset_task": "MIT-BIH sleep stages",
        "target_col": "sleep_stage",
        "raw_csv": "data/processed/mitbih_sleep_stages.csv",
        "prep_csv": "data/processed/mitbih_sleep_stages_prep.csv",
        "eda_raw_dir": "reports/eda_raw/mitbih_sleep_stages_raw",
        "eda_processed_dir": "reports/eda_processed/mitbih_sleep_stages",
        "scaling_summary": "reports/scaling/mitbih_sleep_stages/scaling_summary.md",
    },
    {
        "dataset_task": "MIT-BIH respiratory events",
        "target_col": "event_tokens",
        "raw_csv": "data/processed/mitbih_respiratory_events.csv",
        "prep_csv": "data/processed/mitbih_respiratory_events_prep.csv",
        "eda_raw_dir": "reports/eda_raw/mitbih_respiratory_events_raw",
        "eda_processed_dir": "reports/eda_processed/mitbih_respiratory_events",
        "scaling_summary": "reports/scaling/mitbih_respiratory_events/scaling_summary.md",
    },
    {
        "dataset_task": "SHHS sleep stages",
        "target_col": "sleep_stage",
        "raw_csv": "data/processed/shhs_sleep_stages.csv",
        "prep_csv": "data/processed/shhs_sleep_stages_prep.csv",
        "eda_raw_dir": "reports/eda_raw/shhs_sleep_stages_raw",
        "eda_processed_dir": "reports/eda_processed/shhs_sleep_stages",
        "scaling_summary": "reports/scaling/shhs_sleep_stages/scaling_summary.md",
    },
    {
        "dataset_task": "SHHS respiratory events",
        "target_col": "event_label",
        "raw_csv": "data/processed/shhs_respiratory_events.csv",
        "prep_csv": "data/processed/shhs_respiratory_events_prep.csv",
        "eda_raw_dir": "reports/eda_raw/shhs_respiratory_events_raw",
        "eda_processed_dir": "reports/eda_processed/shhs_respiratory_events",
        "scaling_summary": "reports/scaling/shhs_respiratory_events/scaling_summary.md",
    },
    {
        "dataset_task": "Sleep-EDF Expanded",
        "target_col": "sleep_stage",
        "raw_csv": "data/processed/sleep_edf_expanded_raw.csv",
        "prep_csv": "data/processed/sleep_edf_expanded_prep.csv",
        "eda_raw_dir": "reports/eda_raw/sleep_edf_expanded_raw",
        "eda_processed_dir": "reports/eda_processed/sleep_edf_expanded",
        "scaling_summary": "reports/scaling/sleep_edf_expanded/scaling_summary.md",
    },
]


def _load_target_counts(df: pd.DataFrame, target_col_raw: str) -> Dict[str, int]:
    prepared, target_col = prepare_processed_df_for_eda(df, target_col_raw=target_col_raw)
    counts = prepared[target_col].astype("string").fillna("<MISSING>").value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _counts_to_inline(counts: Dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def _balance_changed(raw_counts: Dict[str, int], processed_counts: Dict[str, int]) -> bool:
    return raw_counts != processed_counts


def _scaling_note(path: Path) -> str:
    if not path.is_file():
        return "No scaling summary found; likely no scaling applied."
    text = path.read_text(encoding="utf-8", errors="replace").lower()
    if "method: `standardize`" in text:
        return "Standardization applied."
    if "method: `minmax`" in text:
        return "Min-max scaling applied."
    return "Scaling summary present, but method was not parsed."


def main() -> None:
    rows: List[Dict[str, object]] = []
    md_lines: List[str] = ["# Phase D Before/After Summary", ""]

    for item in DATASETS:
        raw_csv = ROOT / item["raw_csv"]
        prep_csv = ROOT / item["prep_csv"]
        eda_raw_dir = ROOT / item["eda_raw_dir"]
        eda_processed_dir = ROOT / item["eda_processed_dir"]
        scaling_summary = ROOT / item["scaling_summary"]

        raw_df = pd.read_csv(raw_csv)
        prep_df = pd.read_csv(prep_csv)

        raw_counts = _load_target_counts(raw_df, item["target_col"])
        processed_counts = _load_target_counts(prep_df, item["target_col"])
        balance_changed = _balance_changed(raw_counts, processed_counts)
        scaling_note = _scaling_note(scaling_summary)

        rows.append(
            {
                "dataset_task": item["dataset_task"],
                "raw_rows": int(raw_df.shape[0]),
                "raw_cols": int(raw_df.shape[1]),
                "prep_rows": int(prep_df.shape[0]),
                "prep_cols": int(prep_df.shape[1]),
                "eda_raw_dir": item["eda_raw_dir"],
                "eda_processed_dir": item["eda_processed_dir"],
                "raw_target_fig": f"{item['eda_raw_dir']}/fig_target_distribution.png",
                "processed_target_fig": f"{item['eda_processed_dir']}/fig_target_distribution.png",
            }
        )

        md_lines.extend(
            [
                f"## {item['dataset_task']}",
                "",
                f"- Dimensionalidad: {raw_df.shape[1]} columnas en raw -> {prep_df.shape[1]} columnas en prep.",
                (
                    f"- Balance del target: {'cambió' if balance_changed else 'no cambió'} "
                    f"(raw: {_counts_to_inline(raw_counts)}; processed: {_counts_to_inline(processed_counts)})."
                ),
                f"- Escalado visible: {scaling_note}",
                f"- Figura target antes: `{item['eda_raw_dir']}/fig_target_distribution.png`",
                f"- Figura target después: `{item['eda_processed_dir']}/fig_target_distribution.png`",
                "",
            ]
        )

    out_df = pd.DataFrame(rows)
    csv_path = ROOT / "reports" / "phase_d_before_after_summary.csv"
    md_path = ROOT / "reports" / "phase_d_before_after_summary.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
