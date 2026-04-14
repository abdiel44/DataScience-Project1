"""Phase D: EDA on fully preprocessed dataframe (--run-eda-processed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from main import parse_args, prepare_processed_df_for_eda, step_exploratory_data_analysis


def test_step_eda_uses_eda_processed_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "target": ["x", "y"]})
    res = step_exploratory_data_analysis(
        df,
        task="t1",
        target_col_raw="target",
        eda_outdir=None,
        top_n_plots=3,
        report_subdir="eda_processed",
    )
    summary = Path(res["summary"]).resolve()
    assert summary.name == "eda_summary.md"
    assert "eda_processed" in str(summary)
    assert summary.is_relative_to(tmp_path)


def test_parse_args_run_eda_processed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "--input",
            "x.csv",
            "--output",
            "y.csv",
            "--task",
            "t",
            "--target-col",
            "c",
            "--run-eda",
            "--run-eda-processed",
        ],
    )
    args = parse_args()
    assert args.run_eda is True
    assert args.run_eda_processed is True


def test_run_eda_processed_without_run_eda_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inp = tmp_path / "in.csv"
    inp.write_text("a,target\n1,x\n2,y\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--task",
            "t",
            "--target-col",
            "target",
            "--run-eda-processed",
        ],
    )
    import main as main_module

    with pytest.raises(ValueError, match="--run-eda-processed requires --run-eda"):
        main_module.main()


def test_prepare_processed_df_for_eda_reconstructs_one_hot_target() -> None:
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "sleep_stage_w": [1.0, 0.0, 0.0],
            "sleep_stage_n2": [0.0, 1.0, 0.0],
            "sleep_stage_rem": [0.0, 0.0, 1.0],
        }
    )

    out, target = prepare_processed_df_for_eda(df, target_col_raw="sleep_stage")

    assert target == "sleep_stage"
    assert "sleep_stage" in out.columns
    assert out["sleep_stage"].tolist() == ["w", "n2", "rem"]


def test_prepare_processed_df_for_eda_keeps_existing_target() -> None:
    df = pd.DataFrame({"x": [1, 2], "target": ["a", "b"]})
    out, target = prepare_processed_df_for_eda(df, target_col_raw="target")
    assert target == "target"
    assert out.equals(df)
