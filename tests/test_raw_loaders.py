from pathlib import Path

import pandas as pd

from pre_processing.raw_loaders import ingest_by_source_id, ingest_isruc_sleep


def test_isruc_ingest_two_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    seg = raw / "ISRUC-Sleep" / "Non_Events" / "batch"
    seg.mkdir(parents=True)
    p1 = seg / "S1_p10_Stagen2_Event1_Session1.csv"
    p2 = seg / "S1_p10_Stagen3_Event2_Session1.csv"
    cols = {f"ch{i}": [float(i), float(i + 1)] for i in range(5)}
    pd.DataFrame(cols).to_csv(p1, index=False)
    pd.DataFrame({f"ch{i}": [float(i)] for i in range(5)}).to_csv(p2, index=False)

    df, meta = ingest_isruc_sleep(raw)
    assert len(df) == 2
    assert meta.n_files_skipped == 0
    assert set(df["event_group"]) == {"non_event"}
    assert list(df["sleep_stage"]) == [2, 3]
    assert "ch0_mean" in df.columns
    assert "subject_unit_id" in df.columns
    assert df["subject_unit_id"].iloc[0] == "Non_Events/batch"


def test_ingest_dispatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0").mkdir(
        parents=True
    )
    stage = (
        raw / "st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0" / "ucddb001_stage.txt"
    )
    stage.write_text("0\n1\n2\n2\n5\n", encoding="utf-8")

    df, _ = ingest_by_source_id("st_vincent_apnea", raw)
    assert len(df) == 1
    assert df["recording_id"].iloc[0] == "ucddb001"
    assert "stage_2_frac" in df.columns
