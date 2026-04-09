from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    CleaningOptions,
    clean_dataframe,
    write_cleaning_artifacts,
)


def test_tukey_winsorize_clips_extremes() -> None:
    df = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
            "sleep_stage": ["N1", "N1", "N2", "N2", "REM", "REM"],
        }
    )
    opts = CleaningOptions(
        target_col="sleep_stage",
        outlier_method="tukey_winsorize",
        outlier_iqr_multiplier=1.5,
    )
    clean, report = clean_dataframe(df, opts)
    assert report.outlier_method == "tukey_winsorize"
    assert len(report.outlier_summaries) == 1
    assert clean["feature"].max() < 100.0
    assert clean["feature"].min() >= 1.0


def test_drop_high_missing_column() -> None:
    df = pd.DataFrame(
        {
            "ok": [1.0, 2.0, 3.0],
            "mostly_missing": [1.0, np.nan, np.nan],
            "label": ["a", "b", "c"],
        }
    )
    clean, report = clean_dataframe(df, CleaningOptions(drop_cols_missing_pct=50.0))
    assert "mostly_missing" not in clean.columns
    assert "ok" in clean.columns
    assert "mostly_missing" in report.dropped_columns_high_missing


def test_drop_rows_target_missing() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", None, "c"]})
    clean, report = clean_dataframe(
        df,
        CleaningOptions(target_col="y", drop_rows_target_missing=True),
    )
    assert len(clean) == 2
    assert report.removed_target_missing_rows == 1


def test_dedupe_subset() -> None:
    df = pd.DataFrame(
        {
            "subject": [1, 1, 2],
            "window": [0, 0, 0],
            "value": [10, 99, 10],
        }
    )
    clean, report = clean_dataframe(df, CleaningOptions(dedupe_subset=("subject", "window")))
    assert len(clean) == 2
    assert report.removed_duplicates == 1
    assert report.dedupe_subset == ["subject", "window"]


def test_numeric_coercion_from_strings() -> None:
    df = pd.DataFrame({"num_text": ["1", "2", "3", "4", "5"]})
    clean, report = clean_dataframe(df, CleaningOptions(numeric_coerce_min_parsed_ratio=0.9))
    assert pd.api.types.is_numeric_dtype(clean["num_text"])
    assert len(report.numeric_coercions) == 1


def test_string_normalization_low_cardinality() -> None:
    df = pd.DataFrame({"cat": ["  Lima ", "lima", "LIMA"]})
    clean, report = clean_dataframe(df, CleaningOptions(string_normalize_max_cardinality=10))
    assert clean["cat"].nunique() == 1
    assert clean["cat"].iloc[0] == "lima"
    assert "cat" in report.string_normalized_columns


def test_write_cleaning_artifacts(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    _, report = clean_dataframe(df)
    paths = write_cleaning_artifacts(tmp_path / "c", task="test", report=report)
    assert Path(paths["summary"]).exists()
    assert Path(paths["log"]).exists()


def test_target_column_not_found_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="not found"):
        clean_dataframe(df, CleaningOptions(target_col="missing_target"))


def test_basic_cleanup_columns_duplicates_and_imputation() -> None:
    df = pd.DataFrame(
        {
            "Edad ": [20, 20, None, None],
            "Ciudad": ["Lima", "Lima", "Quito", None],
        }
    )
    clean_df, report = clean_dataframe(df)

    assert "edad" in clean_df.columns
    assert "ciudad" in clean_df.columns
    assert report.input_rows == 4
    assert report.output_rows == 2
    assert report.removed_duplicates == 1
    assert report.removed_empty_rows == 1
    assert clean_df["edad"].isna().sum() == 0
