"""
Stable subject / recording identifiers for subject-wise CV (PRD Phase E3).

Corpora use different column names (`record_id`, `recording_id`, …). This module
normalizes them to a single column when possible, or derives a proxy from `source_file`
(ISRUC-style paths).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd

DEFAULT_PREFER: Tuple[str, ...] = (
    "subject_unit_id",
    "subject_id",
    "record_id",
    "recording_id",
)


def subject_proxy_from_source_file(rel: str) -> str:
    """
    Derive a grouping key from a relative path (e.g. ISRUC ``source_file``).

    Uses the **parent directory path** (all components except the filename). That way
    paths like ``Subgroup_1/1/segment.csv`` map to ``Subgroup_1/1`` (disambiguates
    subject IDs reused across subgroups) instead of only ``Subgroup_1``.

    If the path has no directory, falls back to the file stem. Always confirm in the
    informe that this path matches the **clinical subject or recording unit** you
    intend for subject-wise CV for your exact download layout.
    """
    p = Path(str(rel).replace("\\", "/"))
    parts = p.parts
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return p.stem if p.stem else str(rel)


def ensure_subject_unit_column(
    df: pd.DataFrame,
    *,
    output_col: str = "subject_unit_id",
    prefer: Sequence[str] = DEFAULT_PREFER,
    derive_from_source_file: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Return a copy of ``df`` with ``output_col`` suitable for :func:`subject_wise_fold_indices`.

    Resolution order:

    1. If ``output_col`` already exists, has no all-NA, and ``overwrite`` is False — return as-is.
    2. First column from ``prefer`` that exists and is not entirely NA.
    3. If ``derive_from_source_file`` and ``source_file`` exists — map each row with
       :func:`subject_proxy_from_source_file`.
    4. Otherwise raise ``ValueError`` with an actionable message.

    Parameters
    ----------
    overwrite :
        If True, recompute ``output_col`` even when present.
    """
    out = df.copy()
    if not overwrite and output_col in out.columns and out[output_col].notna().all():
        return out

    for name in prefer:
        if name == output_col and not overwrite:
            continue
        if name in out.columns and out[name].notna().any():
            out[output_col] = out[name]
            return out

    if derive_from_source_file and "source_file" in out.columns:
        out[output_col] = out["source_file"].astype(str).map(subject_proxy_from_source_file)
        return out

    missing = ", ".join(prefer)
    raise ValueError(
        f"Cannot build '{output_col}': add one of [{missing}], or a 'source_file' column "
        "for path-based derivation (ISRUC). WFDB exports typically provide 'record_id'; "
        "St. Vincent / Sleep-EDF summaries use 'recording_id'."
    )
