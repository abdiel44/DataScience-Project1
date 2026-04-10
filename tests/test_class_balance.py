import pandas as pd
import pytest

from pre_processing.class_balance import (
    ClassBalanceOptions,
    align_balance_options_to_snake_case,
    balance_dataframe,
    class_balance_options_from_dict,
    compute_class_weights_for_cost_sensitive,
    write_class_balance_report,
)


def test_random_under_balances_to_minority() -> None:
    df = pd.DataFrame(
        {
            "f1": list(range(100)),
            "y": ["A"] * 90 + ["B"] * 10,
        }
    )
    out, rep = balance_dataframe(
        df,
        ClassBalanceOptions(target_column="y", method="random_under", random_state=0),
    )
    assert rep.rows_after == 20
    vc = out["y"].value_counts()
    assert int(vc["A"]) == 10
    assert int(vc["B"]) == 10


def test_random_over_balances_to_majority() -> None:
    df = pd.DataFrame(
        {
            "f1": list(range(20)),
            "y": ["A"] * 15 + ["B"] * 5,
        }
    )
    out, rep = balance_dataframe(
        df,
        ClassBalanceOptions(target_column="y", method="random_over", random_state=0),
    )
    assert rep.rows_after == 30
    vc = out["y"].value_counts()
    assert int(vc["A"]) == 15
    assert int(vc["B"]) == 15


def test_smote_increases_minority_numeric_only() -> None:
    df = pd.DataFrame(
        {
            "x": list(range(12)),
            "y": [0] * 9 + [1] * 3,
        }
    )
    out, rep = balance_dataframe(
        df,
        ClassBalanceOptions(
            target_column="y",
            method="smote",
            random_state=0,
            smote_k_neighbors=2,
        ),
    )
    assert rep.rows_after > rep.rows_before
    assert out["y"].value_counts().min() == out["y"].value_counts().max()


def test_smote_rejects_non_numeric_feature() -> None:
    df = pd.DataFrame({"x": ["a", "b", "c"], "y": [0, 0, 1]})
    with pytest.raises(ValueError, match="numeric"):
        balance_dataframe(df, ClassBalanceOptions(target_column="y", method="smote", random_state=0))


def test_compute_class_weights() -> None:
    y = pd.Series(["A", "A", "A", "B"])
    w = compute_class_weights_for_cost_sensitive(y)
    assert "A" in w and "B" in w
    assert w["B"] > w["A"]


def test_class_balance_options_from_dict() -> None:
    d = {"class_balance_method": "random_under", "target_column": "label", "balance_random_state": 1}
    opt = class_balance_options_from_dict(d)
    assert opt is not None
    assert opt.method == "random_under"
    aligned = align_balance_options_to_snake_case(opt)
    assert aligned.target_column == "label"


def test_write_class_balance_report(tmp_path) -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    _, rep = balance_dataframe(
        df,
        ClassBalanceOptions(target_column="y", method="none"),
    )
    p = tmp_path / "b.md"
    write_class_balance_report(
        p,
        report=rep,
        options=ClassBalanceOptions(target_column="y", method="none"),
    )
    text = p.read_text(encoding="utf-8")
    assert "Topic 11" in text
