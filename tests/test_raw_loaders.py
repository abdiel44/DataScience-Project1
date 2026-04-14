from pathlib import Path

import pandas as pd

from pre_processing.raw_loaders import (
    _add_temporal_context_features,
    _apply_sleep_edf_wake_trim,
    ingest_by_source_id,
    ingest_isruc_sleep,
)


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
    assert "eeg_mean" in df.columns
    assert "eeg_bandpower_delta" in df.columns
    assert "subject_unit_id" in df.columns
    assert df["subject_unit_id"].iloc[0] == "S1_p10"


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


def test_sleep_edf_wake_trim_keeps_sleep_with_30min_margin() -> None:
    rows = []
    for idx, stage in enumerate(["W", "W", "N1", "N2", "N3", "REM", "W", "W"]):
        rows.append({"epoch_index": idx, "sleep_stage": stage})
    trimmed, before_count, after_count = _apply_sleep_edf_wake_trim(rows, wake_edge_mins=1)
    assert before_count == 8
    assert after_count == 8
    assert trimmed[0]["recording_epochs_before_trim"] == 8
    assert trimmed[-1]["recording_epochs_after_trim"] == 8


def test_temporal_context_features_do_not_cross_recordings() -> None:
    df = pd.DataFrame(
        {
            "recording_id": ["r1", "r1", "r2", "r2"],
            "epoch_index": [0, 1, 0, 1],
            "eeg_mean": [1.0, 2.0, 10.0, 20.0],
            "eeg_bandpower_delta": [0.1, 0.2, 1.0, 2.0],
        }
    )
    out = _add_temporal_context_features(
        df,
        group_col="recording_id",
        order_col="epoch_index",
        feature_cols=["eeg_mean", "eeg_bandpower_delta"],
        lags=(1,),
        leads=(1,),
    )
    r1 = out[out["recording_id"] == "r1"].reset_index(drop=True)
    r2 = out[out["recording_id"] == "r2"].reset_index(drop=True)
    assert r1.loc[0, "eeg_mean_lag1"] == 1.0
    assert r1.loc[1, "eeg_mean_lag1"] == 1.0
    assert r2.loc[0, "eeg_mean_lag1"] == 10.0
    assert r2.loc[0, "eeg_mean_lead1"] == 20.0
