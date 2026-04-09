"""
Variable encoding for tabular datasets (any schema).

Topic 1 (DataScienceTopics): categoricals are **nominal** (no order → one-hot)
or **ordinal** (ordered levels → integer ranks). For **normalization vs
standardization** of numeric features, use `scaling.py` (Topic 5).

Typical workflow: clean (`cleaning.py`) → encode (this module) → scale (`scaling.py`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _to_snake_case(text: str) -> str:
    """Same rules as `cleaning.to_snake_case` (local copy for `src.*` test imports)."""
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


@dataclass
class VariableEncodingSpec:
    """
    Declarative rules for a generic tabular dataset.

    Column names must match the dataframe (e.g. after cleaning / snake_case).
    """

    nominal_columns: Tuple[str, ...] = ()
    """Nominal (unordered) categoricals → one-hot columns (float 0/1)."""

    ordinal_columns: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    """Column name → category order from lowest to highest (integer codes 0..n-1)."""

    binary_columns: Tuple[str, ...] = ()
    """Binary-like columns → 0/1 (case-insensitive yes/no, true/false, 1/0)."""

    drop_first_dummy: bool = False
    """If True, drop first dummy per nominal column (reduces collinearity)."""

    unknown_ordinal_strategy: Literal["nan", "raise"] = "nan"
    """Behaviour for ordinal values not listed in the declared order."""


@dataclass
class EncodingReport:
    nominal_columns: List[str]
    ordinal_columns: List[str]
    binary_columns: List[str]
    added_dummy_columns: List[str]


_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "si", "sí"})
_FALSY = frozenset({"0", "false", "f", "no", "n"})


def _to_binary_numeric(series: pd.Series) -> pd.Series:
    def convert(v: Any) -> float:
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return float(bool(v))
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            if float(v) == 1.0:
                return 1.0
            if float(v) == 0.0:
                return 0.0
        s = str(v).strip().lower()
        if s in _TRUTHY:
            return 1.0
        if s in _FALSY:
            return 0.0
        raise ValueError(f"Cannot map binary value {v!r}")

    return series.map(convert).astype(float)


def _ordinal_integer_codes(
    series: pd.Series,
    order: Sequence[str],
    *,
    unknown: Literal["nan", "raise"],
) -> pd.Series:
    normalized_order = tuple(str(c).strip().lower() for c in order)
    mapping = {c: i for i, c in enumerate(normalized_order)}

    def code(v: Any) -> Any:
        if pd.isna(v):
            return np.nan
        key = str(v).strip().lower()
        if key not in mapping:
            if unknown == "raise":
                raise ValueError(f"Ordinal value {v!r} not in declared order {order!r}")
            return np.nan
        return float(mapping[key])

    return series.map(code).astype(float)


def encode_dataframe(df: pd.DataFrame, spec: VariableEncodingSpec) -> Tuple[pd.DataFrame, EncodingReport]:
    """
    Apply binary → ordinal → nominal one-hot. No numeric scaling (see `scaling.py`).
    """
    overlap = set(spec.nominal_columns) & set(spec.ordinal_columns.keys())
    if overlap:
        raise ValueError(f"Columns cannot be both nominal and ordinal: {sorted(overlap)}")

    out = df.copy()
    added_dummies: List[str] = []
    cols_before = set(out.columns)

    for col in spec.binary_columns:
        if col not in out.columns:
            raise ValueError(f"Binary column not found: {col!r}")
        out[col] = _to_binary_numeric(out[col])

    for col, order in spec.ordinal_columns.items():
        if col not in out.columns:
            raise ValueError(f"Ordinal column not found: {col!r}")
        if not order:
            raise ValueError(f"Ordinal order for {col!r} must be non-empty")
        out[col] = _ordinal_integer_codes(
            out[col], order, unknown=spec.unknown_ordinal_strategy
        )
        if out[col].isna().any():
            med = float(out[col].median())
            if np.isnan(med):
                med = 0.0
            out[col] = out[col].fillna(med)

    if spec.nominal_columns:
        for col in spec.nominal_columns:
            if col not in out.columns:
                raise ValueError(f"Nominal column not found: {col!r}")
        out = pd.get_dummies(
            out,
            columns=list(spec.nominal_columns),
            prefix=list(spec.nominal_columns),
            prefix_sep="_",
            drop_first=spec.drop_first_dummy,
            dtype=float,
        )
        added_dummies = sorted(c for c in out.columns if c not in cols_before)

    report = EncodingReport(
        nominal_columns=list(spec.nominal_columns),
        ordinal_columns=list(spec.ordinal_columns.keys()),
        binary_columns=list(spec.binary_columns),
        added_dummy_columns=added_dummies,
    )
    return out, report


def align_spec_to_snake_case(spec: VariableEncodingSpec) -> VariableEncodingSpec:
    """Align column names to cleaned dataframes (same snake_case as `cleaning.py`)."""
    return VariableEncodingSpec(
        nominal_columns=tuple(_to_snake_case(c) for c in spec.nominal_columns),
        ordinal_columns={_to_snake_case(k): v for k, v in spec.ordinal_columns.items()},
        binary_columns=tuple(_to_snake_case(c) for c in spec.binary_columns),
        drop_first_dummy=spec.drop_first_dummy,
        unknown_ordinal_strategy=spec.unknown_ordinal_strategy,
    )


def load_spec_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON object from disk (encoding + optional scaling keys in one file)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if text.startswith("\ufeff"):
        text = text[1:]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("Spec JSON root must be an object")
    return data


def variable_encoding_spec_from_dict(data: Mapping[str, Any]) -> VariableEncodingSpec:
    """Build encoding spec; ignores keys used only for scaling (`numeric_scaling`, etc.)."""
    nominal = tuple(data.get("nominal_columns") or ())
    ordinal_raw = data.get("ordinal_columns") or {}
    if not isinstance(ordinal_raw, dict):
        raise TypeError("ordinal_columns must be a JSON object / dict of column -> ordered list")
    ordinal: Dict[str, Tuple[str, ...]] = {}
    for k, v in ordinal_raw.items():
        if not isinstance(v, (list, tuple)):
            raise TypeError(f"ordinal_columns[{k!r}] must be a list of category strings")
        ordinal[str(k)] = tuple(str(x) for x in v)

    binary = tuple(data.get("binary_columns") or ())
    drop_first = bool(data.get("drop_first_dummy", False))
    unknown = data.get("unknown_ordinal_strategy", "nan")
    if unknown not in ("nan", "raise"):
        raise ValueError("unknown_ordinal_strategy must be 'nan' or 'raise'")

    return VariableEncodingSpec(
        nominal_columns=tuple(str(c) for c in nominal),
        ordinal_columns=ordinal,
        binary_columns=tuple(str(c) for c in binary),
        drop_first_dummy=drop_first,
        unknown_ordinal_strategy=unknown,  # type: ignore[arg-type]
    )


def load_variable_encoding_spec(path: Union[str, Path]) -> VariableEncodingSpec:
    """Load encoding-only spec from JSON (scaling keys in file are ignored here)."""
    return variable_encoding_spec_from_dict(load_spec_json(path))


def write_encoding_report(path: Path, *, report: EncodingReport, spec: VariableEncodingSpec) -> None:
    """Short Markdown summary for reports or coursework."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Variable encoding summary",
        "",
        "## Topic alignment",
        "",
        "- **Nominal** columns (no natural order): one-hot encoding → `added_dummy_columns`.",
        "- **Ordinal** columns: integer codes 0..n-1 following the declared order.",
        "- **Binary** columns: mapped to 0.0 / 1.0.",
        "- Numeric scaling: use `scaling.py` / `--scale-method` after encoding.",
        "",
        "## Applied rules",
        "",
        f"- Nominal: `{report.nominal_columns}`",
        f"- Ordinal: `{report.ordinal_columns}`",
        f"- Binary: `{report.binary_columns}`",
        f"- `drop_first_dummy`: {spec.drop_first_dummy}",
        "",
        "## New dummy columns",
        "",
    ]
    if report.added_dummy_columns:
        lines.extend(f"- `{c}`" for c in report.added_dummy_columns)
    else:
        lines.append("- (none)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
