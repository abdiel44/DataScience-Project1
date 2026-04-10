"""
Class imbalance handling for tabular supervised data (Topic 11 — DataScienceTopics).

Supports random under/over-sampling and SMOTE via imbalanced-learn. SMOTE builds
synthetic minority examples in numeric feature space; use after encoding so all
features are numeric. Rows produced by SMOTE are synthetic — not a substitute
for collecting real minority data or for EDA on the original distribution.

Topic 11 also recommends appropriate metrics (F1, AUC-PR), stratified splits,
and cost-sensitive learning at training time; see report Markdown.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

BalanceMethod = Literal["none", "random_under", "random_over", "smote"]


def _to_snake_case(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


@dataclass
class ClassBalanceOptions:
    """Target column name must match the dataframe (e.g. snake_case after cleaning)."""

    target_column: str
    method: BalanceMethod
    random_state: int = 42
    sampling_strategy: Any = None
    """Passed to imbalanced-learn; None uses the library default (usually 'auto')."""
    smote_k_neighbors: Optional[int] = None
    """If set, passed to SMOTE only (use 1–2 when the minority class is very small)."""


@dataclass
class ClassBalanceReport:
    method: str
    rows_before: int
    rows_after: int
    counts_before: Dict[str, int]
    counts_after: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
    # sklearn-style weights from pre-resample counts (Topic 11 cost-sensitive)
    class_weights_original: Dict[str, float] = field(default_factory=dict)


def _class_counts(series: pd.Series) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def compute_class_weights_for_cost_sensitive(y: pd.Series) -> Dict[str, float]:
    """
    sklearn-style balanced class weights: n_samples / (n_classes * count).
    Use as `class_weight` in many sklearn classifiers (Topic 11).
    """
    vc = y.value_counts()
    n = len(y)
    k = len(vc)
    if k == 0:
        return {}
    return {str(cls): float(n / (k * cnt)) for cls, cnt in vc.items()}


def _build_sampler(
    method: BalanceMethod,
    random_state: int,
    sampling_strategy: Any,
    smote_k_neighbors: Optional[int],
):
    kw: Dict[str, Any] = {"random_state": random_state}
    if sampling_strategy is not None:
        kw["sampling_strategy"] = sampling_strategy
    if method == "random_under":
        return RandomUnderSampler(**kw)
    if method == "random_over":
        return RandomOverSampler(**kw)
    if method == "smote":
        if smote_k_neighbors is not None:
            kw["k_neighbors"] = smote_k_neighbors
        return SMOTE(**kw)
    raise ValueError(f"Unknown balance method: {method!r}")


def _ensure_numeric_features_for_smote(X: pd.DataFrame) -> None:
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(
                f"SMOTE requires all feature columns to be numeric; '{col}' is not. "
                "Encode categoricals first (encoding.py) or drop non-numeric features."
            )
    if X.isna().any().any():
        raise ValueError("SMOTE cannot run with NaN in feature columns; fix missing values first.")


def balance_dataframe(
    df: pd.DataFrame,
    options: ClassBalanceOptions,
) -> Tuple[pd.DataFrame, ClassBalanceReport]:
    if options.method == "none":
        y0 = df[options.target_column]
        rep = ClassBalanceReport(
            method="none",
            rows_before=len(df),
            rows_after=len(df),
            counts_before=_class_counts(y0),
            counts_after=_class_counts(y0),
            class_weights_original=compute_class_weights_for_cost_sensitive(y0),
        )
        return df.copy(), rep

    target = options.target_column
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target!r}")

    y = df[target]
    X = df.drop(columns=[target])
    counts_before = _class_counts(y)
    class_weights_original = compute_class_weights_for_cost_sensitive(y)
    warnings: List[str] = []

    min_cls = y.value_counts().min()
    if options.method == "smote":
        _ensure_numeric_features_for_smote(X)
        k_def = 5
        if min_cls <= k_def:
            warnings.append(
                f"Smallest class count is {min_cls}; SMOTE default k_neighbors=5 may fail. "
                "Consider random_over, more data, or reducing k_neighbors."
            )

    sampler = _build_sampler(
        options.method,
        options.random_state,
        options.sampling_strategy,
        options.smote_k_neighbors,
    )
    try:
        X_res, y_res = sampler.fit_resample(X, y)
    except ValueError as e:
        raise ValueError(
            f"Resampling failed ({options.method!r}). Check class counts and feature validity. "
            f"Original error: {e}"
        ) from e

    out = pd.DataFrame(X_res, columns=X.columns)
    out[target] = y_res
    out.reset_index(drop=True, inplace=True)

    counts_after = _class_counts(out[target])
    report = ClassBalanceReport(
        method=options.method,
        rows_before=len(df),
        rows_after=len(out),
        counts_before=counts_before,
        counts_after=counts_after,
        warnings=warnings,
        class_weights_original=class_weights_original,
    )
    return out, report


def align_balance_options_to_snake_case(options: ClassBalanceOptions) -> ClassBalanceOptions:
    return ClassBalanceOptions(
        target_column=_to_snake_case(options.target_column),
        method=options.method,
        random_state=options.random_state,
        sampling_strategy=options.sampling_strategy,
        smote_k_neighbors=options.smote_k_neighbors,
    )


def class_balance_options_from_dict(data: Mapping[str, Any]) -> Optional[ClassBalanceOptions]:
    """Read optional keys from the same JSON as encoding/scaling."""
    raw = data.get("class_balance_method")
    if raw is None:
        raw = data.get("balance_method")
    if raw is None or str(raw).strip() == "" or str(raw).lower() == "none":
        return None

    method = str(raw).lower()
    if method not in ("random_under", "random_over", "smote"):
        raise ValueError(f"class_balance_method must be none, random_under, random_over, or smote; got {raw!r}")

    rs = data.get("balance_random_state", data.get("balance_seed", 42))
    ss = data.get("sampling_strategy")
    if isinstance(ss, str) and ss.strip().startswith("{"):
        ss = json.loads(ss)

    tc = data.get("balance_target_column") or data.get("target_column")
    target_column = str(tc).strip() if tc is not None and str(tc).strip() else ""
    kn = data.get("smote_k_neighbors")
    smote_k = int(kn) if kn is not None else None

    return ClassBalanceOptions(
        target_column=target_column,
        method=method,  # type: ignore[arg-type]
        random_state=int(rs),
        sampling_strategy=ss,
        smote_k_neighbors=smote_k,
    )


def write_class_balance_report(
    path: Path,
    *,
    report: ClassBalanceReport,
    options: ClassBalanceOptions,
    class_weights: Optional[Dict[str, float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Class balance summary (Topic 11)",
        "",
        "## What was applied",
        "",
        f"- Method: `{report.method}`",
        f"- Target column: `{options.target_column}`",
        f"- Rows: {report.rows_before} → {report.rows_after}",
        "",
        "## Class distribution",
        "",
        "### Before",
        "",
    ]
    for k, v in sorted(report.counts_before.items(), key=lambda x: x[0]):
        lines.append(f"- `{k}`: {v}")
    lines.extend(["", "### After", ""])
    for k, v in sorted(report.counts_after.items(), key=lambda x: x[0]):
        lines.append(f"- `{k}`: {v}")

    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in report.warnings)

    cw = class_weights if class_weights is not None else report.class_weights_original
    if cw:
        lines.extend(["", "## Suggested `class_weight` (from pre-resample counts, cost-sensitive)", ""])
        lines.extend(f"- `{k}`: {v:.6g}" for k, v in sorted(cw.items(), key=lambda x: x[0]))

    lines.extend(
        [
            "",
            "## Topic 11 recommendations (inform your modeling step)",
            "",
            "- Prefer **F1-score**, **AUC-ROC**, or **precision–recall** over accuracy alone on imbalanced tasks.",
            "- Use **stratified** train/validation/test splits to preserve class proportions.",
            "- **Cost-sensitive learning**: pass `class_weight` (or equivalent) in the classifier when not resampling.",
            "- **Ensembles** (e.g. Balanced Random Forest) are model choices, not part of this dataframe step.",
            "- **SMOTE** rows are synthetic; validate models on real held-out data.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
