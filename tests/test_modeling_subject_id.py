"""Tests for modeling.subject_id (canonical subject column for CV)."""

import pandas as pd
import pytest

from modeling.subject_id import ensure_subject_unit_column, subject_proxy_from_source_file


def test_subject_proxy_from_source_file_nested() -> None:
    assert subject_proxy_from_source_file("SubjectA/seg/foo.csv") == "SubjectA/seg"


def test_subject_proxy_subgroup_style() -> None:
    assert subject_proxy_from_source_file("Subgroup_2/3/clip.csv") == "Subgroup_2/3"


def test_subject_proxy_from_source_file_flat() -> None:
    assert subject_proxy_from_source_file("foo.csv") == "foo"


def test_ensure_from_record_id() -> None:
    df = pd.DataFrame({"record_id": ["r1", "r1", "r2"], "y": [0, 1, 0]})
    out = ensure_subject_unit_column(df)
    assert list(out["subject_unit_id"]) == ["r1", "r1", "r2"]


def test_ensure_from_source_file_derives() -> None:
    df = pd.DataFrame({"source_file": ["a/b.csv", "a/c.csv"], "y": [0, 1]})
    out = ensure_subject_unit_column(df)
    assert list(out["subject_unit_id"]) == ["a", "a"]


def test_ensure_from_source_file_nested_parent() -> None:
    df = pd.DataFrame({"source_file": ["g/s/t/x.csv", "g/s/t/y.csv"], "y": [0, 1]})
    out = ensure_subject_unit_column(df)
    assert list(out["subject_unit_id"]) == ["g/s/t", "g/s/t"]


def test_ensure_raises_when_no_hook() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
    with pytest.raises(ValueError, match="Cannot build"):
        ensure_subject_unit_column(df, derive_from_source_file=False)


def test_ensure_idempotent() -> None:
    df = pd.DataFrame({"subject_unit_id": ["s", "s"], "y": [0, 1]})
    out = ensure_subject_unit_column(df)
    assert list(out["subject_unit_id"]) == ["s", "s"]
