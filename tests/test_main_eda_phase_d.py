"""Phase D: EDA on fully preprocessed dataframe (--run-eda-processed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from main import parse_args, step_exploratory_data_analysis


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
