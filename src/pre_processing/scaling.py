"""
Numeric feature scaling (Topic 5 — DataScienceTopics).

**Normalization (min-max):** X' = (X - X_min) / (X_max - X_min), typically in [0, 1].
Suited when magnitudes must be bounded and outliers are mild (e.g. KNN, neural nets).

**Standardization (z-score):** X' = (X - μ) / σ with population σ (ddof=0 here).
Suited with outliers or Gaussian-oriented models (e.g. SVM, logistic regression).

Tree-based models are often scale-invariant; skipping scaling is valid.

Ordinal integer codes or one-hot dummies may or may not be scaled — use
`exclude_columns` or `include_columns` to control which columns are transformed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

ScaleMethod = Literal["standardize", "minmax"]


def _to_snake_case(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


@dataclass
class ScalingOptions:
    """Per-column scaling: only numeric dtypes are considered."""

    method: ScaleMethod
    exclude_columns: Tuple[str, ...] = ()
    """Never scale these (e.g. IDs)."""

    target_column: Optional[str] = None
    """Never scale (supervised target left as-is)."""

    include_columns: Optional[Tuple[str, ...]] = None
    """If set, only these numeric columns are scaled (after exclude/target checks)."""


@dataclass
class ScalingReport:
    method: str
    scaled_columns: List[str]
    skipped_columns: List[str]


def scale_series(series: pd.Series, method: ScaleMethod) -> pd.Series:
    """Scale a single numeric series; NaNs unchanged; constant series → 0 on non-NaN."""
    vals = series.astype(float)
    mask = vals.notna()
    if mask.sum() == 0:
        return vals
    x = vals[mask]
    if method == "standardize":
        mu = float(x.mean())
        sigma = float(x.std(ddof=0))
        if sigma == 0.0:
            out = vals.copy()
            out.loc[mask] = 0.0
            return out
        out = vals.copy()
        out.loc[mask] = (x - mu) / sigma
        return out
    lo = float(x.min())
    hi = float(x.max())
    if hi == lo:
        out = vals.copy()
        out.loc[mask] = 0.0
        return out
    out = vals.copy()
    out.loc[mask] = (x - lo) / (hi - lo)
    return out


def scale_numeric_dataframe(df: pd.DataFrame, options: ScalingOptions) -> Tuple[pd.DataFrame, ScalingReport]:
    out = df.copy()
    exclude = set(options.exclude_columns)
    if options.target_column:
        exclude.add(options.target_column)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    scaled: List[str] = []
    skipped: List[str] = []

    for col in numeric_cols:
        if col in exclude:
            skipped.append(col)
            continue
        if options.include_columns is not None and col not in options.include_columns:
            skipped.append(col)
            continue
        out[col] = scale_series(out[col], options.method)
        scaled.append(col)

    report = ScalingReport(
        method=options.method,
        scaled_columns=scaled,
        skipped_columns=skipped,
    )
    return out, report


def scaling_options_from_dict(data: Mapping[str, Any]) -> Optional[ScalingOptions]:
    """
    Build options from a flat JSON/dict (legacy fields shared with encoding spec).

    Returns None if numeric_scaling is missing or null.
    """
    ns = data.get("numeric_scaling")
    if ns is None or ns == "":
        return None
    ns = str(ns).strip().lower()
    if ns == "none":
        return None
    if ns not in ("standardize", "minmax"):
        raise ValueError("numeric_scaling must be 'standardize' or 'minmax'")

    scale_exclude = tuple(str(c) for c in (data.get("scale_exclude") or ()))
    target = data.get("target_column")
    target_column = str(target) if target is not None and str(target) != "" else None
    inc = data.get("scale_include")
    include_columns: Optional[Tuple[str, ...]] = None
    if inc is not None and len(inc) > 0:
        if not isinstance(inc, (list, tuple)):
            raise TypeError("scale_include must be a list of column names")
        include_columns = tuple(str(x) for x in inc)

    return ScalingOptions(
        method=ns,  # type: ignore[arg-type]
        exclude_columns=scale_exclude,
        target_column=target_column,
        include_columns=include_columns,
    )


def align_scaling_options_to_snake_case(options: ScalingOptions) -> ScalingOptions:
    return ScalingOptions(
        method=options.method,
        exclude_columns=tuple(_to_snake_case(c) for c in options.exclude_columns),
        target_column=_to_snake_case(options.target_column) if options.target_column else None,
        include_columns=(
            tuple(_to_snake_case(c) for c in options.include_columns)
            if options.include_columns is not None
            else None
        ),
    )


def write_scaling_summary(path: Path, *, report: ScalingReport, options: ScalingOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Scaling summary (Topic 5)",
        "",
        f"- Method: `{report.method}`",
        f"- Scaled columns ({len(report.scaled_columns)}): `{report.scaled_columns}`",
        f"- Skipped columns ({len(report.skipped_columns)}): `{report.skipped_columns}`",
        "",
        "## Options",
        "",
        f"- exclude_columns: `{list(options.exclude_columns)}`",
        f"- target_column (excluded): `{options.target_column!r}`",
        f"- include_columns: `{list(options.include_columns) if options.include_columns else None}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
