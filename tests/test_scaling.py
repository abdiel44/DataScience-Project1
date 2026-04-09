from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.encoding import align_spec_to_snake_case, encode_dataframe, variable_encoding_spec_from_dict
from src.scaling import (
    ScalingOptions,
    align_scaling_options_to_snake_case,
    scale_numeric_dataframe,
    scale_series,
    scaling_options_from_dict,
    write_scaling_summary,
)


def test_minmax_range_zero_to_one() -> None:
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
    out, report = scale_numeric_dataframe(df, ScalingOptions(method="minmax"))
    assert report.scaled_columns == ["x"]
    assert float(out["x"].min()) == pytest.approx(0.0)
    assert float(out["x"].max()) == pytest.approx(1.0)


def test_standardize_mean_near_zero() -> None:
    df = pd.DataFrame({"x": [0.0, 10.0, 20.0]})
    out, report = scale_numeric_dataframe(df, ScalingOptions(method="standardize"))
    assert "x" in report.scaled_columns
    assert float(out["x"].mean()) == pytest.approx(0.0, abs=1e-9)


def test_target_column_excluded() -> None:
    df = pd.DataFrame({"f": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    out, report = scale_numeric_dataframe(
        df,
        ScalingOptions(method="standardize", target_column="y"),
    )
    assert "y" in report.skipped_columns
    assert "y" not in report.scaled_columns
    assert "f" in report.scaled_columns
    assert out["y"].tolist() == [0, 1, 0]


def test_constant_column_becomes_zero() -> None:
    s = scale_series(pd.Series([7.0, 7.0, 7.0]), "standardize")
    assert float(s.iloc[0]) == pytest.approx(0.0)


def test_nan_preserved() -> None:
    s = scale_series(pd.Series([1.0, np.nan, 3.0]), "minmax")
    assert np.isnan(s.iloc[1])


def test_include_columns_only() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    out, report = scale_numeric_dataframe(
        df,
        ScalingOptions(method="standardize", include_columns=("a",)),
    )
    assert "a" in report.scaled_columns
    assert "b" in report.skipped_columns
    assert out["b"].tolist() == [10.0, 20.0]


def test_scaling_options_from_dict() -> None:
    opt = scaling_options_from_dict(
        {
            "numeric_scaling": "standardize",
            "scale_exclude": ["id"],
            "target_column": "label",
        }
    )
    assert opt is not None
    aligned = align_scaling_options_to_snake_case(opt)
    assert aligned.method == "standardize"
    assert aligned.target_column == "label"


def test_scaling_options_from_dict_none() -> None:
    assert scaling_options_from_dict({"nominal_columns": []}) is None


def test_write_scaling_summary(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]})
    _, report = scale_numeric_dataframe(df, ScalingOptions(method="minmax"))
    opt = ScalingOptions(method="minmax")
    p = tmp_path / "s.md"
    write_scaling_summary(p, report=report, options=opt)
    assert p.exists()
    assert "minmax" in p.read_text(encoding="utf-8")


def test_encode_then_scale_integration() -> None:
    df = pd.DataFrame({"color": ["r", "b"], "f": [1.0, 3.0], "y": [0, 1]})
    spec_dict = {
        "nominal_columns": ["color"],
        "numeric_scaling": "standardize",
        "target_column": "y",
    }
    enc_spec = align_spec_to_snake_case(variable_encoding_spec_from_dict(spec_dict))
    encoded, _ = encode_dataframe(df, enc_spec)
    scale_opt = scaling_options_from_dict(spec_dict)
    assert scale_opt is not None
    scaled, rep = scale_numeric_dataframe(encoded, align_scaling_options_to_snake_case(scale_opt))
    assert any(c.startswith("color_") for c in rep.scaled_columns)
    assert "y" not in rep.scaled_columns
