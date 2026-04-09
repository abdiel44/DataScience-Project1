"""
Feature selection and extraction (Topic 6 — DataScienceTopics).

**Selection** keeps a subset of original columns (e.g. variance threshold, univariate scores).
**Extraction** builds a new space (PCA: max variance, unsupervised; LDA: class separation, supervised).

For PCA/LDA, scale features first when magnitudes differ (Topic 5). t-SNE/UMAP and wrapper
methods (RFE) are out of scope here; see the generated Markdown report for pointers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression

DimensionMethod = Literal["none", "pca", "lda", "variance_threshold", "select_k_best"]
ScoreFuncChoice = Literal["auto", "f_classif", "f_regression"]


def _to_snake_case(text: str) -> str:
    s = str(text).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


@dataclass
class DimensionalityOptions:
    method: DimensionMethod
    target_column: Optional[str] = None
    feature_exclude: Tuple[str, ...] = ()
    random_state: int = 42
    pca_n_components: Optional[Union[int, float]] = None
    """int = count; float in (0,1] = explained variance ratio (PCA only)."""
    lda_n_components: Optional[int] = None
    """None → min(n_features, n_classes - 1)."""
    variance_threshold_value: float = 0.0
    select_k: int = 10
    select_score_func: ScoreFuncChoice = "auto"


@dataclass
class DimensionalityReport:
    method: str
    input_feature_columns: List[str]
    output_columns: List[str]
    explained_variance_ratio: List[float] = field(default_factory=list)
    cumulative_explained_variance: Optional[float] = None
    selected_feature_scores: Dict[str, float] = field(default_factory=dict)
    dropped_constant_features: List[str] = field(default_factory=list)


def _numeric_feature_matrix(
    df: pd.DataFrame,
    *,
    target_column: Optional[str],
    feature_exclude: Tuple[str, ...],
) -> Tuple[pd.DataFrame, List[str]]:
    exclude = set(feature_exclude)
    if target_column:
        exclude.add(target_column)
    cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        raise ValueError("No numeric feature columns left after exclusions.")
    X = df[cols].copy()
    if X.isna().any().any():
        raise ValueError("Numeric features contain NaN; clean or impute before dimensionality reduction.")
    return X, cols


def _passthrough_columns(df: pd.DataFrame, feature_cols: List[str], target_column: Optional[str]) -> List[str]:
    return [
        c
        for c in df.columns
        if c not in feature_cols and c != target_column
    ]


def _is_classification_target(y: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y) or pd.api.types.is_bool_dtype(y):
        return True
    nu = int(y.nunique(dropna=True))
    n = int(y.notna().sum())
    if nu < 2:
        return False
    if pd.api.types.is_float_dtype(y):
        max_classes = min(25, max(2, n // 2))
        return nu <= max_classes
    return nu <= 50


def _resolve_score_func(y: pd.Series, choice: ScoreFuncChoice):
    if choice == "f_classif":
        return f_classif
    if choice == "f_regression":
        return f_regression
    if choice == "auto":
        return f_classif if _is_classification_target(y) else f_regression
    raise ValueError(f"Unknown score_func: {choice!r}")


def apply_dimensionality(
    df: pd.DataFrame,
    options: DimensionalityOptions,
) -> Tuple[pd.DataFrame, DimensionalityReport]:
    if options.method == "none":
        rep = DimensionalityReport(
            method="none",
            input_feature_columns=[],
            output_columns=list(df.columns),
        )
        return df.copy(), rep

    target_col = options.target_column
    if options.method in ("lda", "select_k_best"):
        if not target_col or target_col not in df.columns:
            raise ValueError(f"method={options.method!r} requires a valid target_column present in the dataframe.")

    X, feat_names = _numeric_feature_matrix(df, target_column=target_col, feature_exclude=options.feature_exclude)
    passthrough = _passthrough_columns(df, feat_names, target_col)
    y = df[target_col] if target_col and target_col in df.columns else None

    parts: List[pd.DataFrame] = []
    evr: List[float] = []
    cum_ev: Optional[float] = None
    scores: Dict[str, float] = {}
    dropped: List[str] = []
    out_feat_cols: List[str] = []

    if options.method == "pca":
        n_comp = options.pca_n_components
        if n_comp is None:
            n_comp = 0.95
        pca = PCA(n_components=n_comp, random_state=options.random_state)
        X_new = pca.fit_transform(X)
        n_out = X_new.shape[1]
        out_feat_cols = [f"pca_{i + 1}" for i in range(n_out)]
        evr = [float(x) for x in pca.explained_variance_ratio_]
        cum_ev = float(np.sum(pca.explained_variance_ratio_))
        parts.append(pd.DataFrame(X_new, columns=out_feat_cols, index=df.index))

    elif options.method == "lda":
        assert y is not None
        if not _is_classification_target(y):
            raise ValueError(
                "LDA requires a classification target (discrete labels). "
                "Use PCA or regression-style targets with select_k_best + f_regression.",
            )
        n_classes = int(y.nunique(dropna=True))
        if n_classes < 2:
            raise ValueError("LDA needs at least two classes.")
        max_comp = min(X.shape[1], n_classes - 1)
        if max_comp < 1:
            raise ValueError("LDA needs n_classes >= 2 and enough features (n_components cap is n_classes - 1).")
        n_lda = options.lda_n_components
        if n_lda is None:
            n_lda = max_comp
        n_lda = min(int(n_lda), max_comp)
        lda = LinearDiscriminantAnalysis(n_components=n_lda)
        X_new = lda.fit_transform(X, y)
        n_out = X_new.shape[1]
        out_feat_cols = [f"lda_{i + 1}" for i in range(n_out)]
        parts.append(pd.DataFrame(X_new, columns=out_feat_cols, index=df.index))

    elif options.method == "variance_threshold":
        vt = VarianceThreshold(threshold=options.variance_threshold_value)
        mask = vt.fit(X).get_support()
        dropped = [feat_names[i] for i in range(len(feat_names)) if not mask[i]]
        X_new = vt.transform(X)
        out_feat_cols = [feat_names[i] for i in range(len(feat_names)) if mask[i]]
        parts.append(pd.DataFrame(X_new, columns=out_feat_cols, index=df.index))

    elif options.method == "select_k_best":
        assert y is not None
        k = min(int(options.select_k), X.shape[1])
        if k < 1:
            raise ValueError("select_k must be >= 1.")
        score_fn = _resolve_score_func(y, options.select_score_func)
        skb = SelectKBest(score_fn, k=k)
        X_new = skb.fit_transform(X, y)
        support = skb.get_support(indices=True)
        out_feat_cols = [feat_names[i] for i in support]
        raw_scores = skb.scores_
        for i in support:
            scores[feat_names[i]] = float(raw_scores[i])
        parts.append(pd.DataFrame(X_new, columns=out_feat_cols, index=df.index))

    else:
        raise ValueError(f"Unknown method: {options.method!r}")

    if passthrough:
        parts.append(df[passthrough].copy())
    if target_col and target_col in df.columns:
        parts.append(df[[target_col]].copy())

    out = pd.concat(parts, axis=1)
    out.reset_index(drop=True, inplace=True)

    report = DimensionalityReport(
        method=options.method,
        input_feature_columns=feat_names,
        output_columns=list(out.columns),
        explained_variance_ratio=evr,
        cumulative_explained_variance=cum_ev,
        selected_feature_scores=scores,
        dropped_constant_features=dropped,
    )
    return out, report


def align_dimensionality_options_to_snake_case(options: DimensionalityOptions) -> DimensionalityOptions:
    tc = _to_snake_case(options.target_column) if options.target_column else None
    return DimensionalityOptions(
        method=options.method,
        target_column=tc,
        feature_exclude=tuple(_to_snake_case(c) for c in options.feature_exclude),
        random_state=options.random_state,
        pca_n_components=options.pca_n_components,
        lda_n_components=options.lda_n_components,
        variance_threshold_value=options.variance_threshold_value,
        select_k=options.select_k,
        select_score_func=options.select_score_func,
    )


def dimensionality_options_from_dict(data: Mapping[str, Any]) -> Optional[DimensionalityOptions]:
    raw = data.get("dimensionality_method")
    if raw is None or str(raw).strip() == "" or str(raw).lower() == "none":
        return None
    m = str(raw).lower()
    if m not in ("pca", "lda", "variance_threshold", "select_k_best"):
        raise ValueError(
            f"dimensionality_method must be none, pca, lda, variance_threshold, or select_k_best; got {raw!r}",
        )
    tc = data.get("dimensionality_target_column") or data.get("target_column")
    target = str(tc).strip() if tc is not None and str(tc).strip() else None
    ex = data.get("dimensionality_feature_exclude") or data.get("feature_exclude") or ()
    if not isinstance(ex, (list, tuple)):
        ex = ()
    rs = int(data.get("dimensionality_random_state", data.get("random_state", 42)))
    pca_nc = data.get("pca_n_components")
    if isinstance(pca_nc, str) and pca_nc.strip():
        try:
            pca_nc = float(pca_nc) if "." in pca_nc else int(pca_nc)
        except ValueError:
            pca_nc = None
    lda_nc = data.get("lda_n_components")
    lda_nc = int(lda_nc) if lda_nc is not None else None
    vt = float(data.get("variance_threshold", data.get("variance_threshold_value", 0.0)))
    sk = int(data.get("select_k", 10))
    sf = data.get("select_score_func", "auto")
    if sf not in ("auto", "f_classif", "f_regression"):
        raise ValueError(f"select_score_func must be auto, f_classif, or f_regression; got {sf!r}")

    return DimensionalityOptions(
        method=m,  # type: ignore[arg-type]
        target_column=target,
        feature_exclude=tuple(str(x) for x in ex),
        random_state=rs,
        pca_n_components=pca_nc,
        lda_n_components=lda_nc,
        variance_threshold_value=vt,
        select_k=sk,
        select_score_func=sf,  # type: ignore[arg-type]
    )


def write_dimensionality_report(path: Path, *, report: DimensionalityReport, options: DimensionalityOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dimensionality reduction summary (Topic 6)",
        "",
        "## Method applied",
        "",
        f"- `{report.method}`",
        f"- Input numeric feature columns ({len(report.input_feature_columns)}): `{report.input_feature_columns}`",
        f"- Output columns ({len(report.output_columns)}): `{report.output_columns}`",
        "",
    ]
    if report.explained_variance_ratio:
        lines.append("## PCA explained variance ratio")
        lines.append("")
        for i, r in enumerate(report.explained_variance_ratio, start=1):
            lines.append(f"- Component {i}: {r:.6g}")
        if report.cumulative_explained_variance is not None:
            lines.append(f"- Cumulative (shown components): {report.cumulative_explained_variance:.6g}")
        lines.append("")
    if report.dropped_constant_features:
        lines.extend(["## Dropped low-variance features", ""])
        lines.extend(f"- `{c}`" for c in report.dropped_constant_features)
        lines.append("")
    if report.selected_feature_scores:
        lines.extend(["## SelectKBest scores (selected features)", ""])
        for k, v in sorted(report.selected_feature_scores.items(), key=lambda x: x[0]):
            lines.append(f"- `{k}`: {v:.6g}")
        lines.append("")
    lines.extend(
        [
            "## Topic 6 notes",
            "",
            "- **PCA** — unsupervised; good for compression and decorrelation. Interpretability of components is limited.",
            "- **LDA** — supervised; needs discrete class labels; at most `n_classes - 1` components.",
            "- **Wrappers** (e.g. RFE) and **embedded** methods (e.g. Lasso) are typically tied to a specific model.",
            "- **t-SNE / UMAP** are mainly for 2D/3D visualization, not a default preprocessing export.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
