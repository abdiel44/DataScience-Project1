from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

OutlierMethod = Literal["none", "tukey_winsorize"]


def to_snake_case(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _object_like_column_names(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])]


def _normalize_category_value(value: Any) -> Any:
    if pd.isna(value):
        return value
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


@dataclass
class NumericCoercionSummary:
    column: str
    induced_nan_count: int
    parsed_non_null_count: int


@dataclass
class OutlierColumnSummary:
    column: str
    q1: float
    q3: float
    iqr: float
    lower_bound: float
    upper_bound: float
    n_winsorized_low: int
    n_winsorized_high: int


@dataclass
class CleaningReport:
    input_rows: int
    output_rows: int
    removed_empty_rows: int
    removed_duplicates: int
    removed_target_missing_rows: int
    dropped_columns_high_missing: List[str]
    missing_counts_before: Dict[str, int]
    numeric_coercions: List[NumericCoercionSummary]
    string_normalized_columns: List[str]
    dedupe_subset: Optional[List[str]]
    outlier_method: str
    outlier_summaries: List[OutlierColumnSummary] = field(default_factory=list)


@dataclass
class CleaningOptions:
    """Options for `clean_dataframe`. Column names refer to values after snake_case renaming."""

    target_col: Optional[str] = None
    dedupe_subset: Optional[Tuple[str, ...]] = None
    drop_cols_missing_pct: Optional[float] = None
    outlier_method: OutlierMethod = "none"
    outlier_iqr_multiplier: float = 1.5
    outlier_exclude_cols: Tuple[str, ...] = ()
    outlier_columns: Optional[Tuple[str, ...]] = None
    drop_rows_target_missing: bool = False
    string_normalize_max_cardinality: int = 50
    numeric_coerce_min_parsed_ratio: float = 0.9


def _resolve_target_key(target_col: Optional[str]) -> Optional[str]:
    if target_col is None or target_col == "":
        return None
    return to_snake_case(target_col)


def clean_dataframe(df: pd.DataFrame, options: Optional[CleaningOptions] = None) -> Tuple[pd.DataFrame, CleaningReport]:
    opts = options or CleaningOptions()
    input_rows = len(df)
    clean = df.copy()
    clean.columns = [to_snake_case(col) for col in clean.columns]

    target_key = _resolve_target_key(opts.target_col)
    if target_key is not None and target_key not in clean.columns:
        raise ValueError(f"Target column '{opts.target_col}' not found after normalizing names (expected '{target_key}').")

    missing_counts_before = {str(c): int(clean[c].isna().sum()) for c in clean.columns}

    dropped_columns_high_missing: List[str] = []
    if opts.drop_cols_missing_pct is not None:
        threshold = float(opts.drop_cols_missing_pct)
        if threshold < 0 or threshold > 100:
            raise ValueError("drop_cols_missing_pct must be between 0 and 100.")
        n = len(clean)
        if n > 0:
            to_drop: List[str] = []
            for col in clean.columns:
                if target_key is not None and col == target_key:
                    continue
                pct = 100.0 * float(clean[col].isna().sum()) / n
                if pct > threshold:
                    to_drop.append(col)
            if to_drop:
                clean = clean.drop(columns=to_drop)
                dropped_columns_high_missing = to_drop

    removed_target_missing_rows = 0
    if opts.drop_rows_target_missing and target_key is not None:
        before = len(clean)
        clean = clean.dropna(subset=[target_key])
        removed_target_missing_rows = before - len(clean)

    before_dropna = len(clean)
    clean = clean.dropna(how="all")
    removed_empty_rows = before_dropna - len(clean)

    numeric_coercions: List[NumericCoercionSummary] = []
    object_cols = _object_like_column_names(clean)
    for col in object_cols:
        ser = clean[col]
        non_null = int(ser.notna().sum())
        if non_null == 0:
            continue
        conv = pd.to_numeric(ser, errors="coerce")
        parsed_non_null = int(conv.notna().sum())
        ratio = parsed_non_null / non_null if non_null else 0.0
        induced = int((ser.notna() & conv.isna()).sum())
        if ratio >= opts.numeric_coerce_min_parsed_ratio and parsed_non_null > 0:
            clean[col] = conv
            numeric_coercions.append(
                NumericCoercionSummary(
                    column=col,
                    induced_nan_count=induced,
                    parsed_non_null_count=parsed_non_null,
                )
            )

    string_normalized_columns: List[str] = []
    object_cols_after = _object_like_column_names(clean)
    max_card = opts.string_normalize_max_cardinality
    for col in object_cols_after:
        n_unique = int(clean[col].nunique(dropna=True))
        if n_unique <= max_card:
            clean[col] = clean[col].map(_normalize_category_value)
            string_normalized_columns.append(col)

    subset_list: Optional[List[str]] = None
    if opts.dedupe_subset is not None:
        subset_list = [to_snake_case(c) for c in opts.dedupe_subset]
        missing = [c for c in subset_list if c not in clean.columns]
        if missing:
            raise ValueError(f"dedupe_subset columns not found: {missing}")
        before_dedup = len(clean)
        clean = clean.drop_duplicates(subset=subset_list)
        removed_duplicates = before_dedup - len(clean)
    else:
        before_dedup = len(clean)
        clean = clean.drop_duplicates()
        removed_duplicates = before_dedup - len(clean)

    numeric_columns = clean.select_dtypes(include=[np.number]).columns
    object_columns = clean.select_dtypes(exclude=[np.number]).columns

    for col in numeric_columns:
        median_value = clean[col].median()
        clean[col] = clean[col].fillna(median_value)

    for col in object_columns:
        mode = clean[col].mode(dropna=True)
        if not mode.empty:
            clean[col] = clean[col].fillna(mode.iloc[0])

    outlier_summaries: List[OutlierColumnSummary] = []
    if opts.outlier_method == "tukey_winsorize":
        k = float(opts.outlier_iqr_multiplier)
        exclude = {to_snake_case(c) for c in opts.outlier_exclude_cols}
        if target_key is not None:
            exclude.add(target_key)

        if opts.outlier_columns is not None:
            candidate_cols = [to_snake_case(c) for c in opts.outlier_columns]
        else:
            candidate_cols = clean.select_dtypes(include=[np.number]).columns.tolist()

        for col in candidate_cols:
            if col in exclude or col not in clean.columns:
                continue
            ser = clean[col]
            if not pd.api.types.is_numeric_dtype(ser):
                continue
            if ser.notna().sum() == 0:
                continue
            q1 = float(ser.quantile(0.25))
            q3 = float(ser.quantile(0.75))
            if not np.isfinite(q1) or not np.isfinite(q3):
                continue
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            below = ser < lower
            above = ser > upper
            n_low = int(below.sum())
            n_high = int(above.sum())
            clean[col] = ser.clip(lower=lower, upper=upper)
            outlier_summaries.append(
                OutlierColumnSummary(
                    column=col,
                    q1=q1,
                    q3=q3,
                    iqr=iqr,
                    lower_bound=lower,
                    upper_bound=upper,
                    n_winsorized_low=n_low,
                    n_winsorized_high=n_high,
                )
            )

    report = CleaningReport(
        input_rows=input_rows,
        output_rows=len(clean),
        removed_empty_rows=removed_empty_rows,
        removed_duplicates=removed_duplicates,
        removed_target_missing_rows=removed_target_missing_rows,
        dropped_columns_high_missing=dropped_columns_high_missing,
        missing_counts_before=missing_counts_before,
        numeric_coercions=numeric_coercions,
        string_normalized_columns=string_normalized_columns,
        dedupe_subset=subset_list,
        outlier_method=opts.outlier_method,
        outlier_summaries=outlier_summaries,
    )
    return clean, report


def write_cleaning_artifacts(output_dir: Path, *, task: str, report: CleaningReport) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "cleaning_summary.md"
    csv_path = output_dir / "cleaning_log.csv"

    top_missing = sorted(report.missing_counts_before.items(), key=lambda x: x[1], reverse=True)[:10]
    top_missing_lines = "\n".join(f"- `{col}`: {cnt}" for col, cnt in top_missing) or "- None"

    dropped_lines = (
        "\n".join(f"- `{c}`" for c in report.dropped_columns_high_missing)
        if report.dropped_columns_high_missing
        else "- None"
    )

    coercion_lines = (
        "\n".join(
            f"- `{c.column}`: induced NaN={c.induced_nan_count}, parsed={c.parsed_non_null_count}"
            for c in report.numeric_coercions
        )
        if report.numeric_coercions
        else "- None"
    )

    str_norm = (
        "\n".join(f"- `{c}`" for c in report.string_normalized_columns)
        if report.string_normalized_columns
        else "- None"
    )

    dedupe_desc = (
        f"subset={report.dedupe_subset}" if report.dedupe_subset is not None else "full row duplicate"
    )

    outlier_lines = ""
    if report.outlier_summaries:
        outlier_lines = "\n".join(
            f"- `{o.column}`: bounds [{o.lower_bound:.6g}, {o.upper_bound:.6g}], "
            f"winsorized low={o.n_winsorized_low}, high={o.n_winsorized_high} "
            f"(Q1={o.q1:.6g}, Q3={o.q3:.6g}, IQR={o.iqr:.6g})"
            for o in report.outlier_summaries
        )
    else:
        outlier_lines = "- No winsorization applied or no numeric columns processed."

    md_content = f"""# Data cleaning summary - {task}

## Row and column changes

- Input rows: {report.input_rows}
- Output rows: {report.output_rows}
- Removed fully empty rows: {report.removed_empty_rows}
- Removed duplicate rows ({dedupe_desc}): {report.removed_duplicates}
- Removed rows with missing target: {report.removed_target_missing_rows}

## Columns dropped (high missing %)

{dropped_lines}

## Missing values (top 10 columns, before imputation)

{top_missing_lines}

## Numeric coercion (object to numeric)

{coercion_lines}

## String normalization (strip, collapse spaces, lowercase)

{str_norm}

## Outliers ({report.outlier_method})

{outlier_lines}
"""
    md_path.write_text(md_content, encoding="utf-8")

    log_rows: List[Dict[str, Any]] = [
        {
            "metric": "input_rows",
            "value": report.input_rows,
            "detail": "",
        },
        {
            "metric": "output_rows",
            "value": report.output_rows,
            "detail": "",
        },
        {
            "metric": "removed_empty_rows",
            "value": report.removed_empty_rows,
            "detail": "",
        },
        {
            "metric": "removed_duplicates",
            "value": report.removed_duplicates,
            "detail": dedupe_desc,
        },
        {
            "metric": "removed_target_missing_rows",
            "value": report.removed_target_missing_rows,
            "detail": "",
        },
    ]
    for col in report.dropped_columns_high_missing:
        log_rows.append({"metric": "dropped_column_high_missing", "value": col, "detail": ""})
    for c in report.numeric_coercions:
        log_rows.append(
            {
                "metric": "numeric_coercion",
                "value": c.column,
                "detail": f"induced_nan={c.induced_nan_count},parsed={c.parsed_non_null_count}",
            }
        )
    for col in report.string_normalized_columns:
        log_rows.append({"metric": "string_normalized", "value": col, "detail": ""})
    for o in report.outlier_summaries:
        log_rows.append(
            {
                "metric": "outlier_winsorize",
                "value": o.column,
                "detail": (
                    f"lower={o.lower_bound:.6g},upper={o.upper_bound:.6g},"
                    f"n_low={o.n_winsorized_low},n_high={o.n_winsorized_high}"
                ),
            }
        )

    pd.DataFrame(log_rows).to_csv(csv_path, index=False)

    return {"summary": str(md_path), "log": str(csv_path)}
