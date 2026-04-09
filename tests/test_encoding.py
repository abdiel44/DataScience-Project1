from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.encoding import (
    VariableEncodingSpec,
    align_spec_to_snake_case,
    encode_dataframe,
    load_variable_encoding_spec,
    variable_encoding_spec_from_dict,
    write_encoding_report,
)


def test_nominal_one_hot() -> None:
    df = pd.DataFrame({"color": ["red", "blue", "red"], "x": [1.0, 2.0, 3.0]})
    spec = VariableEncodingSpec(nominal_columns=("color",))
    out, report = encode_dataframe(df, spec)
    assert "color_red" in out.columns or "color_blue" in out.columns
    assert "color" not in out.columns
    assert len(report.added_dummy_columns) >= 2


def test_ordinal_preserves_order_as_integers() -> None:
    df = pd.DataFrame({"size": ["s", "m", "l", "m"]})
    spec = VariableEncodingSpec(
        ordinal_columns={"size": ("s", "m", "l")},
    )
    out, report = encode_dataframe(df, spec)
    assert report.ordinal_columns == ["size"]
    assert out.loc[out["size"] == 0.0].shape[0] == 1  # s
    assert out.loc[out["size"] == 2.0].shape[0] == 1  # l


def test_nominal_and_ordinal_overlap_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    spec = VariableEncodingSpec(nominal_columns=("x",), ordinal_columns={"x": ("a", "b")})
    with pytest.raises(ValueError, match="both nominal and ordinal"):
        encode_dataframe(df, spec)


def test_binary_mapping() -> None:
    df = pd.DataFrame({"flag": ["yes", "no", "YES", False, True]})
    spec = VariableEncodingSpec(binary_columns=("flag",))
    out, _ = encode_dataframe(df, spec)
    assert set(out["flag"].dropna().unique()) <= {0.0, 1.0}


def test_unknown_ordinal_raises() -> None:
    df = pd.DataFrame({"size": ["s", "xl"]})
    spec = VariableEncodingSpec(
        ordinal_columns={"size": ("s", "m", "l")},
        unknown_ordinal_strategy="raise",
    )
    with pytest.raises(ValueError, match="not in declared order"):
        encode_dataframe(df, spec)


def test_load_spec_from_json(tmp_path: Path) -> None:
    p = tmp_path / "enc.json"
    p.write_text(
        """
        {
          "nominal_columns": ["City"],
          "ordinal_columns": {"rank": ["bronze", "silver", "gold"]},
          "numeric_scaling": null,
          "target_column": "label"
        }
        """.strip(),
        encoding="utf-8",
    )
    spec = load_variable_encoding_spec(p)
    assert "City" in spec.nominal_columns
    assert spec.ordinal_columns["rank"] == ("bronze", "silver", "gold")


def test_align_spec_snake_case() -> None:
    raw = variable_encoding_spec_from_dict(
        {
            "nominal_columns": ["Sleep Stage"],
            "ordinal_columns": {},
        }
    )
    aligned = align_spec_to_snake_case(raw)
    assert aligned.nominal_columns == ("sleep_stage",)


def test_write_encoding_report(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": ["x", "y"]})
    spec = VariableEncodingSpec(nominal_columns=("a",))
    _, report = encode_dataframe(df, spec)
    md = tmp_path / "enc.md"
    write_encoding_report(md, report=report, spec=spec)
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    assert "Nominal" in text
