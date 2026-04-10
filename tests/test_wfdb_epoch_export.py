from pathlib import Path

import numpy as np
import pytest

from pre_processing.wfdb_epoch_export import (
    MITBIH_EVENT_CODES,
    _feature_dict_from_slice,
    export_mitbih_two_csvs,
    export_shhs_two_csvs,
    parse_mitbih_aux_note,
    route_mitbih_row,
)


def test_parse_mitbih_aux_note() -> None:
    assert parse_mitbih_aux_note("2 HA") == ("2", ("HA",))
    assert parse_mitbih_aux_note("4 LA LA") == ("4", ("LA", "LA"))
    assert parse_mitbih_aux_note("W") == ("W", ())
    assert parse_mitbih_aux_note("R H") == ("R", ("H",))


def test_route_mitbih_row() -> None:
    assert route_mitbih_row("2", ("HA",)) == (True, True)
    assert route_mitbih_row("2", ()) == (True, False)
    assert route_mitbih_row("W", ()) == (True, False)


def test_feature_dict_from_slice() -> None:
    p = np.arange(60, dtype=float).reshape(20, 3)
    d = _feature_dict_from_slice(p, ["A", "B", "C"], 0, 10)
    assert "a_mean" in d and "a_std" in d


def test_mitbih_event_codes_cover_readme() -> None:
    assert "HA" in MITBIH_EVENT_CODES
    assert "OA" in MITBIH_EVENT_CODES


@pytest.mark.integration
def test_export_mitbih_one_record_writes_csvs(tmp_path: Path) -> None:
    raw = Path("data/raw")
    mit = raw / "mit-bih-polysomnographic-database-1.0.0"
    if not (mit / "RECORDS").is_file():
        pytest.skip("MIT-BIH dataset not present under data/raw")
    out_s = tmp_path / "st.csv"
    out_e = tmp_path / "ev.csv"
    stats = export_mitbih_two_csvs(raw, out_s, out_e, max_records=1)
    assert stats.n_staging_rows > 0
    assert out_s.is_file() and out_e.is_file()


@pytest.mark.integration
def test_export_shhs_writes_csvs(tmp_path: Path) -> None:
    raw = Path("data/raw")
    shhs = raw / "sleep-heart-health-study-psg-database-1.0.0"
    if not (shhs / "RECORDS").is_file():
        pytest.skip("SHHS dataset not present under data/raw")
    out_s = tmp_path / "st.csv"
    out_e = tmp_path / "ev.csv"
    stats = export_shhs_two_csvs(raw, out_s, out_e)
    assert stats.n_staging_rows > 0
    assert out_s.is_file() and out_e.is_file()
