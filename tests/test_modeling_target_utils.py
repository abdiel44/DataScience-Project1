from pathlib import Path

import pandas as pd

from modeling.target_utils import ensure_target_column, normalize_sleep_stage_series, target_dummy_columns


def test_target_dummy_columns_excludes_non_binary_prefixed_features() -> None:
    df = pd.DataFrame(
        {
            "sleep_stage_frac_w": [0.3, 0.7],
            "sleep_stage_w": [1.0, 0.0],
            "sleep_stage_n2": [0.0, 1.0],
        }
    )
    cols = target_dummy_columns(df, "sleep_stage")
    assert cols == ("sleep_stage_w", "sleep_stage_n2")


def test_ensure_target_column_reconstructs_from_one_hot() -> None:
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "sleep_stage_w": [1.0, 0.0],
            "sleep_stage_1": [0.0, 1.0],
        }
    )
    out, target, dummy_cols = ensure_target_column(df, target_col_raw="sleep_stage")
    assert target == "sleep_stage"
    assert dummy_cols == ("sleep_stage_w", "sleep_stage_1")
    assert out["sleep_stage"].tolist() == ["w", "1"]


def test_ensure_target_column_reconstructs_scaled_sleep_stage_columns() -> None:
    df = pd.DataFrame(
        {
            "sleep_stage_frac_n2": [0.4, 0.6],
            "sleep_stage_w": [1.5, -0.5],
            "sleep_stage_1": [-0.5, 1.8],
            "sleep_stage_2": [-0.5, -0.5],
        }
    )
    out, target, dummy_cols = ensure_target_column(df, target_col_raw="sleep_stage")
    assert target == "sleep_stage"
    assert dummy_cols == ("sleep_stage_w", "sleep_stage_1", "sleep_stage_2")
    assert out["sleep_stage"].tolist() == ["w", "1"]


def test_normalize_sleep_stage_series_maps_to_aasm() -> None:
    ser = pd.Series(["w", "1", "2", "3", "4", "r", "N2", "REM"])
    out = normalize_sleep_stage_series(ser)
    assert out.tolist() == ["W", "N1", "N2", "N3", "N3", "REM", "N2", "REM"]
