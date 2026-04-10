from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def _safe_filename(name: str) -> str:
    return "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in name)


def build_dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile_rows: List[Dict[str, Any]] = []
    total_rows = len(df)

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        profile_rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "missing_count": missing_count,
                "missing_pct": round((missing_count / total_rows) * 100, 4) if total_rows else 0.0,
                "n_unique": int(df[col].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(profile_rows).sort_values(["missing_count", "column"], ascending=[False, True])


def _compute_mode(series: pd.Series) -> float:
    mode = series.mode(dropna=True)
    if mode.empty:
        return np.nan
    return float(mode.iloc[0])


def _compute_skewness(values: np.ndarray) -> float:
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 3:
        return np.nan
    return float(stats.skew(valid_values, bias=False))


def _compute_kurtosis(values: np.ndarray) -> float:
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 4:
        return np.nan
    return float(stats.kurtosis(valid_values, fisher=True, bias=False))


def compute_descriptive_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        min_val = float(series.min()) if series.notna().any() else np.nan
        max_val = float(series.max()) if series.notna().any() else np.nan
        values = series.to_numpy(dtype=float)

        numeric_rows.append(
            {
                "column": col,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "mode": _compute_mode(series),
                "std": float(series.std(ddof=1)),
                "var": float(series.var(ddof=1)),
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val if not np.isnan(min_val) and not np.isnan(max_val) else np.nan,
                "iqr": q3 - q1,
                "skewness": _compute_skewness(values),
                "kurtosis": _compute_kurtosis(values),
            }
        )

    categorical_rows: List[Dict[str, Any]] = []
    for col in categorical_cols:
        value_counts = df[col].astype("string").fillna("<MISSING>").value_counts(dropna=False)
        total = int(value_counts.sum())
        for category, freq in value_counts.items():
            categorical_rows.append(
                {
                    "column": col,
                    "category": str(category),
                    "frequency": int(freq),
                    "proportion_pct": round((int(freq) / total) * 100, 4) if total else 0.0,
                }
            )

    numeric_df = pd.DataFrame(numeric_rows).sort_values("column") if numeric_rows else pd.DataFrame()
    categorical_df = (
        pd.DataFrame(categorical_rows).sort_values(["column", "frequency"], ascending=[True, False])
        if categorical_rows
        else pd.DataFrame()
    )
    return numeric_df, categorical_df


def compute_correlations(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not numeric_cols:
        return pd.DataFrame(), pd.DataFrame()

    numeric_df = df[numeric_cols]
    pearson = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    return pearson, spearman


def _top_numeric_columns(df: pd.DataFrame, top_n: int) -> List[str]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return []

    variances = numeric_df.var(numeric_only=True).sort_values(ascending=False)
    return variances.head(top_n).index.tolist()


def generate_plots(df: pd.DataFrame, target_col: str, output_dir: Path, top_n: int = 15) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: List[Path] = []

    selected_numeric_cols = _top_numeric_columns(df, top_n)
    for col in selected_numeric_cols:
        safe_col = _safe_filename(col)
        series = pd.to_numeric(df[col], errors="coerce")

        hist_path = output_dir / f"fig_hist_{safe_col}.png"
        plt.figure(figsize=(8, 4))
        sns.histplot(series.dropna(), bins=30, kde=True)
        plt.title(f"Histogram - {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()
        generated_paths.append(hist_path)

        box_path = output_dir / f"fig_box_{safe_col}.png"
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=series.dropna())
        plt.title(f"Box Plot - {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.savefig(box_path)
        plt.close()
        generated_paths.append(box_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr = df[numeric_cols].corr(method="pearson")
        heatmap_path = output_dir / "fig_corr_heatmap.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("Correlation Heatmap (Pearson)")
        plt.tight_layout()
        plt.savefig(heatmap_path)
        plt.close()
        generated_paths.append(heatmap_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column not found for target distribution plot: {target_col}")

    target_path = output_dir / "fig_target_distribution.png"
    tser = df[target_col]
    plt.figure(figsize=(8, 4))
    if pd.api.types.is_numeric_dtype(tser) and tser.nunique(dropna=True) > 25:
        sns.histplot(pd.to_numeric(tser, errors="coerce").dropna(), bins=40, kde=True)
        plt.title(f"Target Distribution - {target_col} (numeric)")
        plt.xlabel(target_col)
        plt.ylabel("Count")
    else:
        target_counts = tser.astype("string").fillna("<MISSING>").value_counts()
        sns.barplot(x=target_counts.index, y=target_counts.values)
        plt.title(f"Target Distribution - {target_col}")
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(target_path)
    plt.close()
    generated_paths.append(target_path)

    return generated_paths


def write_markdown_summary(
    output_path: Path,
    *,
    task: str,
    target_col: str,
    df: pd.DataFrame,
    profile_df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    figures: List[Path],
) -> None:
    missing_total = int(df.isna().sum().sum())
    n_rows, n_cols = df.shape
    numeric_count = len(df.select_dtypes(include=[np.number]).columns)
    categorical_count = n_cols - numeric_count

    top_missing = (
        profile_df[["column", "missing_count", "missing_pct"]]
        .sort_values("missing_count", ascending=False)
        .head(5)
        .to_dict(orient="records")
        if not profile_df.empty
        else []
    )
    top_missing_lines = "\n".join(
        f"- `{item['column']}`: {item['missing_count']} ({item['missing_pct']}%)" for item in top_missing
    )
    if not top_missing_lines:
        top_missing_lines = "- No missing values detected."

    tcol = df[target_col]
    if pd.api.types.is_numeric_dtype(tcol) and tcol.nunique(dropna=True) > 25:
        desc = tcol.describe()
        class_lines = "\n".join(f"- `{k}`: {float(v):.6g}" for k, v in desc.items())
    else:
        class_distribution = tcol.astype("string").fillna("<MISSING>").value_counts(normalize=True) * 100
        class_lines = "\n".join(f"- `{label}`: {value:.2f}%" for label, value in class_distribution.items())

    numeric_notes = ""
    if not numeric_df.empty:
        strongest_skew = numeric_df.reindex(numeric_df["skewness"].abs().sort_values(ascending=False).index).head(3)
        skew_lines = "\n".join(
            f"- `{row.column}`: skewness={row.skewness:.3f}, kurtosis={row.kurtosis:.3f}"
            for row in strongest_skew.itertuples()
        )
        numeric_notes = (
            "### Shape indicators (numeric)\n\n"
            "Variables with strongest asymmetry (absolute skewness):\n"
            f"{skew_lines}\n"
        )

    figure_lines = "\n".join(f"- `{path.name}`" for path in figures)

    content = f"""# EDA Summary - {task}

## Dataset overview

- Rows: {n_rows}
- Columns: {n_cols}
- Numeric columns: {numeric_count}
- Categorical columns: {categorical_count}
- Total missing values: {missing_total}
- Target column: `{target_col}`

## Missing values (top 5 columns)

{top_missing_lines}

## Target balance

{class_lines}

{numeric_notes}
## Generated figures

{figure_lines}
"""
    output_path.write_text(content, encoding="utf-8")


def run_eda(df: pd.DataFrame, output_dir: Path, task: str, target_col: str, top_n: int = 15) -> Dict[str, Any]:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is not present in dataframe columns.")

    output_dir.mkdir(parents=True, exist_ok=True)

    profile_df = build_dataset_profile(df)
    numeric_df, categorical_df = compute_descriptive_tables(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    pearson_corr, spearman_corr = compute_correlations(df, numeric_cols)

    profile_path = output_dir / "01_dataset_profile.csv"
    numeric_path = output_dir / "02_descriptive_numeric.csv"
    categorical_path = output_dir / "03_descriptive_categorical.csv"
    pearson_path = output_dir / "04_correlations_pearson.csv"
    spearman_path = output_dir / "05_correlations_spearman.csv"
    summary_path = output_dir / "eda_summary.md"

    profile_df.to_csv(profile_path, index=False)
    numeric_df.to_csv(numeric_path, index=False)
    categorical_df.to_csv(categorical_path, index=False)
    pearson_corr.to_csv(pearson_path, index=True)
    spearman_corr.to_csv(spearman_path, index=True)

    figures = generate_plots(df=df, target_col=target_col, output_dir=output_dir, top_n=top_n)
    write_markdown_summary(
        summary_path,
        task=task,
        target_col=target_col,
        df=df,
        profile_df=profile_df,
        numeric_df=numeric_df,
        categorical_df=categorical_df,
        figures=figures,
    )

    return {
        "tables": {
            "profile": str(profile_path),
            "numeric": str(numeric_path),
            "categorical": str(categorical_path),
            "pearson_corr": str(pearson_path),
            "spearman_corr": str(spearman_path),
        },
        "figures": [str(path) for path in figures],
        "summary": str(summary_path),
    }

