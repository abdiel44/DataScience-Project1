from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from pre_processing.class_balance import (
    ClassBalanceOptions,
    ClassBalanceReport,
    align_balance_options_to_snake_case,
    balance_dataframe,
    class_balance_options_from_dict,
    write_class_balance_report,
)
from pre_processing.cleaning import CleaningOptions, CleaningReport, clean_dataframe, to_snake_case, write_cleaning_artifacts
from pre_processing.eda import run_eda
from pre_processing.encoding import (
    EncodingReport,
    VariableEncodingSpec,
    align_spec_to_snake_case,
    encode_dataframe,
    load_spec_json,
    variable_encoding_spec_from_dict,
    write_encoding_report,
)
from pre_processing.dimensionality import (
    DimensionalityOptions,
    DimensionalityReport,
    align_dimensionality_options_to_snake_case,
    apply_dimensionality,
    dimensionality_options_from_dict,
    write_dimensionality_report,
)
from pre_processing.raw_loaders import SourceId, ingest_by_source_id
from pre_processing.wfdb_epoch_export import export_mitbih_two_csvs, export_shhs_two_csvs
from pre_processing.scaling import (
    ScalingOptions,
    ScalingReport,
    align_scaling_options_to_snake_case,
    scale_numeric_dataframe,
    scaling_options_from_dict,
    write_scaling_summary,
)

_SOURCE_CLI_TO_ID: Dict[str, SourceId] = {
    "isruc-sleep": "isruc_sleep",
    "st-vincent-apnea": "st_vincent_apnea",
    "sleep-edf-expanded": "sleep_edf_expanded",
    "mit-bih-psg": "mit_bih_psg",
    "shhs-psg": "shhs_psg",
}


def _parse_comma_separated(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if value is None or not str(value).strip():
        return None
    parts = tuple(p.strip() for p in str(value).split(",") if p.strip())
    return parts if parts else None


def _parse_balance_strategy(value: Optional[str]) -> Union[str, float, dict, None]:
    if value is None or not str(value).strip():
        return None
    s = str(value).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return float(s)
        except ValueError:
            return s


def _resolve_class_balance_options(
    args: argparse.Namespace,
    spec_data: Dict[str, Any],
) -> Optional[ClassBalanceOptions]:
    cli_m = getattr(args, "balance_method", None) or "none"
    if cli_m != "none":
        if not args.target_col:
            raise ValueError("--target-col is required when --balance-method is not none.")
        return align_balance_options_to_snake_case(
            ClassBalanceOptions(
                target_column=args.target_col,
                method=cli_m,  # type: ignore[arg-type]
                random_state=int(args.balance_random_state),
                sampling_strategy=_parse_balance_strategy(args.balance_strategy),
                smote_k_neighbors=args.smote_k_neighbors,
            )
        )
    opt = class_balance_options_from_dict(spec_data)
    if opt is None:
        return None
    if not opt.target_column.strip():
        if not args.target_col:
            raise ValueError(
                "class_balance_method in JSON requires balance_target_column/target_column or --target-col.",
            )
        opt = ClassBalanceOptions(
            target_column=args.target_col,
            method=opt.method,
            random_state=opt.random_state,
            sampling_strategy=opt.sampling_strategy,
            smote_k_neighbors=opt.smote_k_neighbors,
        )
    return align_balance_options_to_snake_case(opt)


def _resolve_scaling_options(
    args: argparse.Namespace,
    spec_data: Dict[str, Any],
) -> Optional[ScalingOptions]:
    """CLI `--scale-method` overrides JSON `numeric_scaling` when not `none`."""
    cli_method = getattr(args, "scale_method", None) or "none"
    if cli_method != "none":
        return align_scaling_options_to_snake_case(
            ScalingOptions(
                method=cli_method,  # type: ignore[arg-type]
                exclude_columns=_parse_comma_separated(args.scale_exclude) or (),
                target_column=args.target_col if args.target_col else None,
            )
        )
    opt = scaling_options_from_dict(spec_data)
    if opt is None:
        return None
    return align_scaling_options_to_snake_case(opt)


def _parse_pca_n_components(value: Optional[str]) -> Optional[Union[int, float]]:
    if value is None or not str(value).strip():
        return None
    s = str(value).strip()
    if "." in s:
        return float(s)
    return int(s)


def _resolve_dimensionality_options(
    args: argparse.Namespace,
    spec_data: Dict[str, Any],
) -> Optional[DimensionalityOptions]:
    """CLI `--dimensionality-method` overrides JSON when not `none`."""
    cli_method = getattr(args, "dimensionality_method", None) or "none"
    if cli_method != "none":
        if cli_method in ("lda", "select_k_best") and not args.target_col:
            raise ValueError(f"--target-col is required when --dimensionality-method is {cli_method!r}.")
        return align_dimensionality_options_to_snake_case(
            DimensionalityOptions(
                method=cli_method,  # type: ignore[arg-type]
                target_column=args.target_col if args.target_col else None,
                feature_exclude=_parse_comma_separated(args.dimensionality_exclude) or (),
                random_state=int(args.dimensionality_random_state),
                pca_n_components=_parse_pca_n_components(getattr(args, "pca_n_components", None)),
                lda_n_components=args.lda_n_components,
                variance_threshold_value=float(args.variance_threshold),
                select_k=int(args.select_k),
                select_score_func=args.select_score_func,  # type: ignore[arg-type]
            )
        )
    opt = dimensionality_options_from_dict(spec_data)
    if opt is None:
        return None
    if opt.method in ("lda", "select_k_best") and not opt.target_column:
        if not args.target_col:
            raise ValueError(
                "dimensionality_method in JSON requires dimensionality_target_column/target_column or --target-col.",
            )
        opt = DimensionalityOptions(
            method=opt.method,
            target_column=args.target_col,
            feature_exclude=opt.feature_exclude,
            random_state=opt.random_state,
            pca_n_components=opt.pca_n_components,
            lda_n_components=opt.lda_n_components,
            variance_threshold_value=opt.variance_threshold_value,
            select_k=opt.select_k,
            select_score_func=opt.select_score_func,
        )
    return align_dimensionality_options_to_snake_case(opt)


def step_clean_data(
    df: pd.DataFrame,
    *,
    options: CleaningOptions,
    write_report: bool,
    cleaning_outdir: Optional[Path],
    task_label: str,
) -> Tuple[pd.DataFrame, CleaningReport, Dict[str, str]]:
    """Step 1 — Data cleaning: transform raw tabular data (no CSV write; see main)."""
    clean_df, report = clean_dataframe(df, options)

    artifact_paths: Dict[str, str] = {}
    if write_report:
        out_dir = cleaning_outdir if cleaning_outdir is not None else Path("reports") / "cleaning" / task_label
        artifact_paths = write_cleaning_artifacts(out_dir, task=task_label, report=report)

    return clean_df, report, artifact_paths


def step_encode_variables(
    df: pd.DataFrame,
    *,
    spec: VariableEncodingSpec,
    write_report: bool,
    encoding_outdir: Optional[Path],
    task_label: str,
) -> Tuple[pd.DataFrame, EncodingReport, Dict[str, str]]:
    """Optional — Encode categoricals only (Topic 1)."""
    encoded, enc_report = encode_dataframe(df, spec)
    paths: Dict[str, str] = {}
    if write_report:
        out = encoding_outdir if encoding_outdir is not None else Path("reports") / "encoding" / task_label
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "encoding_summary.md"
        write_encoding_report(md_path, report=enc_report, spec=spec)
        paths["summary"] = str(md_path)
    return encoded, enc_report, paths


def step_scale_data(
    df: pd.DataFrame,
    *,
    options: Optional[ScalingOptions],
    write_report: bool,
    scaling_outdir: Optional[Path],
    task_label: str,
) -> Tuple[pd.DataFrame, Optional[ScalingReport], Dict[str, str]]:
    """Optional — Numeric scaling (Topic 5)."""
    if options is None:
        return df, None, {}
    scaled, report = scale_numeric_dataframe(df, options)
    paths: Dict[str, str] = {}
    if write_report:
        out = scaling_outdir if scaling_outdir is not None else Path("reports") / "scaling" / task_label
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "scaling_summary.md"
        write_scaling_summary(md_path, report=report, options=options)
        paths["summary"] = str(md_path)
    return scaled, report, paths


def step_class_balance_data(
    df: pd.DataFrame,
    *,
    options: Optional[ClassBalanceOptions],
    write_report: bool,
    balance_outdir: Optional[Path],
    task_label: str,
) -> Tuple[pd.DataFrame, Optional[ClassBalanceReport], Dict[str, str]]:
    """Optional — Class imbalance resampling (Topic 11)."""
    if options is None:
        return df, None, {}
    balanced, report = balance_dataframe(df, options)
    paths: Dict[str, str] = {}
    if write_report:
        out = balance_outdir if balance_outdir is not None else Path("reports") / "class_balance" / task_label
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "class_balance_summary.md"
        write_class_balance_report(md_path, report=report, options=options)
        paths["summary"] = str(md_path)
    return balanced, report, paths


def step_dimensionality_data(
    df: pd.DataFrame,
    *,
    options: Optional[DimensionalityOptions],
    write_report: bool,
    dimensionality_outdir: Optional[Path],
    task_label: str,
) -> Tuple[pd.DataFrame, Optional[DimensionalityReport], Dict[str, str]]:
    """Optional — Feature selection / extraction (Topic 6)."""
    if options is None or options.method == "none":
        return df, None, {}
    reduced, report = apply_dimensionality(df, options)
    paths: Dict[str, str] = {}
    if write_report:
        out = (
            dimensionality_outdir
            if dimensionality_outdir is not None
            else Path("reports") / "dimensionality" / task_label
        )
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "dimensionality_summary.md"
        write_dimensionality_report(md_path, report=report, options=options)
        paths["summary"] = str(md_path)
    return reduced, report, paths


def step_exploratory_data_analysis(
    df: pd.DataFrame,
    *,
    task: str,
    target_col_raw: str,
    eda_outdir: Optional[Path],
    top_n_plots: int,
    report_subdir: str = "eda",
) -> Dict[str, Any]:
    """EDA on cleaned-only data (default) or on the fully preprocessed dataframe (Phase D: report_subdir='eda_processed')."""
    task_slug = to_snake_case(task) if task else "default"
    output_dir = eda_outdir if eda_outdir is not None else Path("reports") / report_subdir / task_slug
    target_col = to_snake_case(target_col_raw)
    return run_eda(
        df=df,
        output_dir=output_dir,
        task=task,
        target_col=target_col,
        top_n=top_n_plots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preprocessing: cleaning -> encoding -> class balance -> scaling -> dimensionality (each optional) -> optional EDA.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        default=None,
        help="Path to input CSV file. Omit when using --source to build a table from data/raw.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["none", *_SOURCE_CLI_TO_ID.keys()],
        default="none",
        help="Build a tabular dataframe from a known dataset under --raw-root (then run the same pipeline as CSV).",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/raw",
        help="Directory containing downloaded dataset folders (default: data/raw).",
    )
    parser.add_argument(
        "--ingest-max-files",
        type=int,
        default=None,
        help="Optional cap on files scanned (ISRUC CSVs or sleep-edf PSG EDFs) for faster tests.",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="CSV has no header row; first row is data (columns become 0,1,2,... then normalized in cleaning).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path to output CSV file (required unless using --export-epochs only).",
    )
    parser.add_argument(
        "--export-epochs",
        type=str,
        choices=["none", "mit-bih-psg", "shhs-psg"],
        default="none",
        help="Write two 30 s epoch CSVs (sleep stages + respiratory/events) from WFDB under --raw-root, then exit.",
    )
    parser.add_argument(
        "--output-stages",
        type=str,
        default=None,
        help="With --export-epochs: path for sleep-stage epoch CSV.",
    )
    parser.add_argument(
        "--output-events",
        type=str,
        default=None,
        help="With --export-epochs: path for respiratory/event epoch CSV.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run EDA after the pipeline step (default: on cleaned data only; use --run-eda-processed for post-pipeline table).",
    )
    parser.add_argument(
        "--run-eda-processed",
        action="store_true",
        help="With --run-eda: run EDA on the final output (after encoding/balance/scaling/dimensionality). Default dir: reports/eda_processed/<task>.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task label for report paths (e.g. sleep, apnea, or any short name). Used with --run-eda and report subfolders.",
    )
    parser.add_argument("--target-col", type=str, help="Target column (EDA, cleaning, scaling exclusion).")
    parser.add_argument("--eda-outdir", type=str, help="Directory to save EDA tables, figures and markdown summary.")
    parser.add_argument("--top-n-plots", type=int, default=15, help="Max number of numeric features to plot.")
    parser.add_argument(
        "--dedupe-subset",
        type=str,
        help="Comma-separated column names for drop_duplicates subset (raw names; normalized like other columns).",
    )
    parser.add_argument(
        "--drop-cols-missing-pct",
        type=float,
        default=None,
        help="Drop feature columns with strictly more than this percentage of missing values (0-100). Target column is never dropped.",
    )
    parser.add_argument(
        "--drop-rows-target-missing",
        action="store_true",
        help="Remove rows where the target column is missing (requires --target-col).",
    )
    parser.add_argument(
        "--outlier-method",
        type=str,
        choices=["none", "tukey_winsorize"],
        default="none",
        help="Outlier handling on numeric features after imputation (Tukey fences, 1.5*IQR by default).",
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for Tukey fences when --outlier-method tukey_winsorize.",
    )
    parser.add_argument(
        "--outlier-exclude-cols",
        type=str,
        default=None,
        help="Comma-separated numeric columns to skip for winsorization (raw names).",
    )
    parser.add_argument(
        "--outlier-columns",
        type=str,
        default=None,
        help="Comma-separated numeric columns to winsorize; default is all numeric except exclusions and target.",
    )
    parser.add_argument(
        "--string-normalize-max-cardinality",
        type=int,
        default=50,
        help="Max unique values for string strip/lowercase normalization on object-like columns.",
    )
    parser.add_argument(
        "--numeric-coerce-min-parsed-ratio",
        type=float,
        default=0.9,
        help="Min ratio of parseable values to coerce object columns to numeric (0-1).",
    )
    parser.add_argument(
        "--write-cleaning-report",
        action="store_true",
        help="Write cleaning_summary.md and cleaning_log.csv under --cleaning-outdir.",
    )
    parser.add_argument(
        "--cleaning-outdir",
        type=str,
        default=None,
        help="Directory for cleaning artifacts (default: reports/cleaning/<task> or reports/cleaning/default).",
    )
    parser.add_argument(
        "--encoding-spec",
        type=str,
        default=None,
        help="JSON: encoding (Topic 1), optional scaling and class_balance_* keys when CLI defaults are none.",
    )
    parser.add_argument(
        "--write-encoding-report",
        action="store_true",
        help="Write encoding_summary.md under --encoding-outdir.",
    )
    parser.add_argument(
        "--encoding-outdir",
        type=str,
        default=None,
        help="Directory for encoding report (default: reports/encoding/<task> or reports/encoding/default).",
    )
    parser.add_argument(
        "--scale-method",
        type=str,
        choices=["none", "standardize", "minmax"],
        default="none",
        help="Numeric scaling (Topic 5). Overrides numeric_scaling in JSON when not 'none'.",
    )
    parser.add_argument(
        "--scale-exclude",
        type=str,
        default=None,
        help="Comma-separated columns to never scale (raw names). Target is always excluded when --target-col is set.",
    )
    parser.add_argument(
        "--write-scaling-report",
        action="store_true",
        help="Write scaling_summary.md under --scaling-outdir.",
    )
    parser.add_argument(
        "--scaling-outdir",
        type=str,
        default=None,
        help="Directory for scaling report (default: reports/scaling/<task> or reports/scaling/default).",
    )
    parser.add_argument(
        "--dimensionality-method",
        type=str,
        choices=["none", "pca", "lda", "variance_threshold", "select_k_best"],
        default="none",
        help="Feature selection / extraction (Topic 6). Overrides dimensionality_method in JSON when not 'none'.",
    )
    parser.add_argument(
        "--pca-n-components",
        type=str,
        default=None,
        help="PCA: int components or float in (0,1] for explained variance ratio; default in module is 0.95.",
    )
    parser.add_argument(
        "--lda-n-components",
        type=int,
        default=None,
        help="LDA: number of components (capped at n_classes - 1); default is max allowed.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.0,
        help="VarianceThreshold: minimum variance per feature (default 0.0).",
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=10,
        help="SelectKBest: k features to keep.",
    )
    parser.add_argument(
        "--select-score-func",
        type=str,
        choices=["auto", "f_classif", "f_regression"],
        default="auto",
        help="SelectKBest score function (auto: classification heuristic vs f_regression).",
    )
    parser.add_argument(
        "--dimensionality-exclude",
        type=str,
        default=None,
        help="Comma-separated feature columns to exclude from X (raw names).",
    )
    parser.add_argument(
        "--dimensionality-random-state",
        type=int,
        default=42,
        help="Random seed for PCA.",
    )
    parser.add_argument(
        "--write-dimensionality-report",
        action="store_true",
        help="Write dimensionality_summary.md under --dimensionality-outdir.",
    )
    parser.add_argument(
        "--dimensionality-outdir",
        type=str,
        default=None,
        help="Directory for dimensionality report (default: reports/dimensionality/<task> or default).",
    )
    parser.add_argument(
        "--balance-method",
        type=str,
        choices=["none", "random_under", "random_over", "smote"],
        default="none",
        help="Class imbalance resampling (Topic 11). Overrides class_balance_method in JSON when not 'none'.",
    )
    parser.add_argument(
        "--balance-random-state",
        type=int,
        default=42,
        help="Random seed for resampling / SMOTE.",
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        default=None,
        help='imblearn sampling_strategy: e.g. auto, 0.5, or JSON object like {"minority_class": 500}.',
    )
    parser.add_argument(
        "--smote-k-neighbors",
        type=int,
        default=None,
        help="SMOTE k_neighbors (use 1–3 if the minority class is small).",
    )
    parser.add_argument(
        "--write-class-balance-report",
        action="store_true",
        help="Write class_balance_summary.md under --class-balance-outdir.",
    )
    parser.add_argument(
        "--class-balance-outdir",
        type=str,
        default=None,
        help="Directory for class balance report (default: reports/class_balance/<task> or default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.export_epochs != "none":
        if not args.output_stages or not args.output_events:
            raise ValueError("--export-epochs requires both --output-stages and --output-events.")
        raw_root = Path(args.raw_root)
        if not raw_root.is_dir():
            raise FileNotFoundError(f"--raw-root not found: {raw_root}")
        out_st = Path(args.output_stages)
        out_ev = Path(args.output_events)
        mr = args.ingest_max_files
        if args.export_epochs == "mit-bih-psg":
            stats = export_mitbih_two_csvs(
                raw_root,
                out_st,
                out_ev,
                max_records=mr,
            )
        else:
            stats = export_shhs_two_csvs(
                raw_root,
                out_st,
                out_ev,
                max_records=mr,
            )
        print(
            f"Export ({args.export_epochs}): staging_rows={stats.n_staging_rows}, "
            f"event_rows={stats.n_event_rows}, skipped_other={stats.n_epochs_skipped_other}, "
            f"records={stats.n_records}",
        )
        print(f"  Wrote: {out_st}")
        print(f"  Wrote: {out_ev}")
        sys.exit(0)

    if not args.output:
        raise ValueError("--output is required unless using --export-epochs.")

    output_path = Path(args.output)

    if args.source != "none":
        raw_root = Path(args.raw_root)
        if not raw_root.is_dir():
            raise FileNotFoundError(f"--raw-root not found: {raw_root}")
        sid = _SOURCE_CLI_TO_ID[args.source]
        df, ingest_meta = ingest_by_source_id(sid, raw_root, max_files=args.ingest_max_files)
        print(
            f"Ingest ({args.source}): rows={len(df)}, files_used={ingest_meta.n_files_used}, "
            f"skipped={ingest_meta.n_files_skipped}",
        )
        for note in ingest_meta.notes:
            print(f"  Note: {note}")
    else:
        if not args.input:
            raise ValueError("Either --input or --source must be provided.")
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        df = pd.read_csv(input_path, header=None) if args.no_header else pd.read_csv(input_path)

    if args.drop_rows_target_missing and not args.target_col:
        raise ValueError("--drop-rows-target-missing requires --target-col.")

    cleaning_options = CleaningOptions(
        target_col=args.target_col,
        dedupe_subset=_parse_comma_separated(args.dedupe_subset),
        drop_cols_missing_pct=args.drop_cols_missing_pct,
        outlier_method=args.outlier_method,
        outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
        outlier_exclude_cols=_parse_comma_separated(args.outlier_exclude_cols) or (),
        outlier_columns=_parse_comma_separated(args.outlier_columns),
        drop_rows_target_missing=bool(args.drop_rows_target_missing),
        string_normalize_max_cardinality=int(args.string_normalize_max_cardinality),
        numeric_coerce_min_parsed_ratio=float(args.numeric_coerce_min_parsed_ratio),
    )

    task_label = args.task if args.task else "default"
    cleaning_dir_arg = Path(args.cleaning_outdir) if args.cleaning_outdir else None

    clean_df, report, cleaning_artifacts = step_clean_data(
        df,
        options=cleaning_options,
        write_report=bool(args.write_cleaning_report),
        cleaning_outdir=cleaning_dir_arg,
        task_label=task_label,
    )

    print("Step 1 (cleaning) finished")
    print(f"  Input rows: {report.input_rows}")
    print(f"  Output rows: {report.output_rows}")
    print(f"  Removed empty rows: {report.removed_empty_rows}")
    print(f"  Removed duplicates: {report.removed_duplicates}")
    print(f"  Removed target-missing rows: {report.removed_target_missing_rows}")
    if report.dropped_columns_high_missing:
        print(f"  Dropped columns (high missing %): {report.dropped_columns_high_missing}")

    if cleaning_artifacts:
        print(f"  Cleaning summary: {cleaning_artifacts['summary']}")
        print(f"  Cleaning log: {cleaning_artifacts['log']}")

    spec_data: Dict[str, Any] = {}
    if args.encoding_spec:
        spec_path = Path(args.encoding_spec)
        if not spec_path.exists():
            raise FileNotFoundError(f"Encoding spec not found: {spec_path}")
        spec_data = load_spec_json(spec_path)

    output_df = clean_df
    encoding_outdir_arg = Path(args.encoding_outdir) if args.encoding_outdir else None
    if args.encoding_spec:
        enc_spec = align_spec_to_snake_case(variable_encoding_spec_from_dict(spec_data))
        output_df, _enc_report, encoding_artifacts = step_encode_variables(
            clean_df,
            spec=enc_spec,
            write_report=bool(args.write_encoding_report),
            encoding_outdir=encoding_outdir_arg,
            task_label=task_label,
        )
        print("Encoding finished")
        if encoding_artifacts:
            print(f"  Encoding summary: {encoding_artifacts['summary']}")

    balance_opts = _resolve_class_balance_options(args, spec_data)
    balance_outdir_arg = Path(args.class_balance_outdir) if args.class_balance_outdir else None
    output_df, bal_report, balance_artifacts = step_class_balance_data(
        output_df,
        options=balance_opts,
        write_report=bool(args.write_class_balance_report),
        balance_outdir=balance_outdir_arg,
        task_label=task_label,
    )
    if bal_report is not None and bal_report.method != "none":
        print("Class balancing finished")
        print(f"  Method: {bal_report.method}; rows: {bal_report.rows_before} -> {bal_report.rows_after}")
        if balance_artifacts:
            print(f"  Class balance summary: {balance_artifacts['summary']}")

    scale_opts = _resolve_scaling_options(args, spec_data)
    scaling_outdir_arg = Path(args.scaling_outdir) if args.scaling_outdir else None
    output_df, scale_report, scaling_artifacts = step_scale_data(
        output_df,
        options=scale_opts,
        write_report=bool(args.write_scaling_report),
        scaling_outdir=scaling_outdir_arg,
        task_label=task_label,
    )
    if scale_report is not None:
        print("Scaling finished")
        print(f"  Method: {scale_report.method}; scaled columns: {len(scale_report.scaled_columns)}")
        if scaling_artifacts:
            print(f"  Scaling summary: {scaling_artifacts['summary']}")

    dim_opts = _resolve_dimensionality_options(args, spec_data)
    dimensionality_outdir_arg = Path(args.dimensionality_outdir) if args.dimensionality_outdir else None
    output_df, dim_report, dim_artifacts = step_dimensionality_data(
        output_df,
        options=dim_opts,
        write_report=bool(args.write_dimensionality_report),
        dimensionality_outdir=dimensionality_outdir_arg,
        task_label=task_label,
    )
    if dim_report is not None and dim_report.method != "none":
        print("Dimensionality finished")
        print(f"  Method: {dim_report.method}; output columns: {len(dim_report.output_columns)}")
        if dim_artifacts:
            print(f"  Dimensionality summary: {dim_artifacts['summary']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"  Saved file: {output_path}")

    if args.run_eda_processed and not args.run_eda:
        raise ValueError("--run-eda-processed requires --run-eda.")

    if args.run_eda:
        if not args.target_col:
            raise ValueError("--target-col is required when --run-eda is enabled.")
        if not args.task:
            raise ValueError("--task is required when --run-eda is enabled.")

        eda_df = output_df if args.run_eda_processed else clean_df
        report_subdir = "eda_processed" if args.run_eda_processed else "eda"
        eda_output_dir = Path(args.eda_outdir) if args.eda_outdir else None
        eda_results = step_exploratory_data_analysis(
            eda_df,
            task=args.task,
            target_col_raw=args.target_col,
            eda_outdir=eda_output_dir,
            top_n_plots=int(args.top_n_plots),
            report_subdir=report_subdir,
        )
        print("EDA finished")
        print(f"  EDA summary: {eda_results['summary']}")
        print(f"  EDA tables: {len(eda_results['tables'])} files")
        print(f"  EDA figures: {len(eda_results['figures'])} files")


if __name__ == "__main__":
    main()
