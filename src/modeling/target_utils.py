"""Target reconstruction and normalization helpers for preprocessing/modeling."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import pandas as pd


def _to_snake_case(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _is_binary_indicator(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return False
    uniq = set(float(v) for v in vals.unique())
    return uniq.issubset({0.0, 1.0})


def target_dummy_columns(df: pd.DataFrame, target_col_raw: str) -> Tuple[str, ...]:
    """Return one-hot columns derived from `target_col_raw`, excluding similarly-prefixed numeric features."""
    target_col = _to_snake_case(target_col_raw)
    prefix = f"{target_col}_"
    cols = [c for c in df.columns if c.startswith(prefix) and _is_binary_indicator(df[c])]
    return tuple(cols)


_SLEEP_STAGE_AASM: Dict[str, str] = {
    "w": "W",
    "wake": "W",
    "sleep_stage_w": "W",
    "1": "N1",
    "n1": "N1",
    "sleep_stage_1": "N1",
    "2": "N2",
    "n2": "N2",
    "sleep_stage_2": "N2",
    "3": "N3",
    "4": "N3",
    "n3": "N3",
    "sleep_stage_3": "N3",
    "sleep_stage_4": "N3",
    "r": "REM",
    "rem": "REM",
    "sleep_stage_r": "REM",
}


def _sleep_stage_encoded_columns(df: pd.DataFrame, target_col: str) -> Tuple[str, ...]:
    prefix = f"{target_col}_"
    cols = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix):]
        if _to_snake_case(suffix) in _SLEEP_STAGE_AASM:
            cols.append(col)
    return tuple(cols)


def normalize_sleep_stage_label(value: Any) -> Any:
    if pd.isna(value):
        return value
    key = _to_snake_case(str(value))
    return _SLEEP_STAGE_AASM.get(key, value)


def normalize_sleep_stage_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_sleep_stage_label)


def ensure_target_column(
    df: pd.DataFrame,
    *,
    target_col_raw: str,
) -> Tuple[pd.DataFrame, str, Tuple[str, ...]]:
    """
    Ensure a target column exists. If the original target was one-hot encoded,
    reconstruct it and return the consumed dummy columns.
    """
    target_col = _to_snake_case(target_col_raw)
    if target_col in df.columns:
        return df, target_col, ()

    dummy_cols = target_dummy_columns(df, target_col_raw)
    if not dummy_cols and target_col == "sleep_stage":
        dummy_cols = _sleep_stage_encoded_columns(df, target_col)
    if not dummy_cols:
        raise ValueError(
            f"Target column '{target_col}' is not present and no one-hot columns like "
            f"'{target_col}_*' were found."
        )

    out = df.copy()
    dummy_frame = out[list(dummy_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = dummy_frame.idxmax(axis=1).str[len(f'{target_col}_'):]
    no_positive_mask = dummy_frame.max(axis=1) <= 0.0
    if no_positive_mask.any():
        labels = labels.astype("string")
        labels.loc[no_positive_mask] = "<UNASSIGNED>"
    out[target_col] = labels
    return out, target_col, dummy_cols
