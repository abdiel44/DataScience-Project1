"""Prepare multitask apnea/staging metadata CSVs from existing processed corpora."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.target_utils import normalize_sleep_stage_series

_ST_VINCENT_STAGE_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "REM"}


def _prepare_sleep_edf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset_id"] = "sleep_edf_expanded"
    out["subject_unit_id"] = out["subject_id"].astype(str)
    out["recording_id"] = out["recording_id"].astype(str)
    out["sleep_stage"] = normalize_sleep_stage_series(out["sleep_stage"])
    out["apnea_binary"] = pd.NA
    out["eeg_channel_standardized"] = "EEG Fpz-Cz"
    out["severity_global"] = np.nan
    return out[
        [
            "dataset_id",
            "subject_unit_id",
            "recording_id",
            "epoch_index",
            "epoch_start_sec",
            "epoch_end_sec",
            "source_file",
            "eeg_channel_standardized",
            "apnea_binary",
            "sleep_stage",
            "severity_global",
        ]
    ].copy()


def _prepare_mitbih(stages: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = stages.copy()
    out["dataset_id"] = "mit_bih_psg"
    out["subject_unit_id"] = out["record_id"].astype(str)
    out["recording_id"] = out["record_id"].astype(str)
    out["source_file"] = out["record_id"].astype(str)
    out["sleep_stage"] = normalize_sleep_stage_series(out["sleep_stage"])
    out["eeg_channel_standardized"] = "EEG (C4-A1)"
    event_keys = set(
        events[["record_id", "epoch_index"]]
        .astype({"record_id": str, "epoch_index": int})
        .itertuples(index=False, name=None)
    )
    out["apnea_binary"] = [1 if (str(rec), int(ep)) in event_keys else 0 for rec, ep in zip(out["record_id"], out["epoch_index"])]
    out["severity_global"] = np.nan
    return out[
        [
            "dataset_id",
            "subject_unit_id",
            "recording_id",
            "epoch_index",
            "epoch_start_sec",
            "epoch_end_sec",
            "source_file",
            "eeg_channel_standardized",
            "apnea_binary",
            "sleep_stage",
            "severity_global",
        ]
    ].copy()


def _prepare_shhs(stages: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = stages.copy()
    out["dataset_id"] = "shhs_psg"
    out["record_id"] = out["record_id"].astype(str)
    out["subject_unit_id"] = out["record_id"]
    out["recording_id"] = out["record_id"]
    out["source_file"] = out["record_id"] + ".edf"
    out["sleep_stage"] = normalize_sleep_stage_series(out["sleep_stage"])
    out["eeg_channel_standardized"] = "EEG"
    resp_events = events[events["annotation_source"].astype(str).str.lower() == "resp"].copy()
    resp_keys = set(
        resp_events[["record_id", "epoch_start_sec", "epoch_end_sec"]]
        .assign(record_id=lambda x: x["record_id"].astype(str))
        .itertuples(index=False, name=None)
    )
    out["apnea_binary"] = [
        1 if (str(rec), float(start), float(end)) in resp_keys else 0
        for rec, start, end in zip(out["record_id"], out["epoch_start_sec"], out["epoch_end_sec"])
    ]
    out["severity_global"] = np.nan
    return out[
        [
            "dataset_id",
            "subject_unit_id",
            "recording_id",
            "epoch_index",
            "epoch_start_sec",
            "epoch_end_sec",
            "source_file",
            "eeg_channel_standardized",
            "apnea_binary",
            "sleep_stage",
            "severity_global",
        ]
    ].copy()


def _parse_hms(text: str) -> Optional[int]:
    m = re.match(r"^\s*(\d{2}):(\d{2}):(\d{2})\s*$", str(text))
    if not m:
        return None
    hh, mm, ss = (int(m.group(i)) for i in range(1, 4))
    return hh * 3600 + mm * 60 + ss


def _parse_st_vincent_events(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not re.match(r"^\d{2}:\d{2}:\d{2}\s+", line):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        start_sec = _parse_hms(parts[0])
        if start_sec is None:
            continue
        try:
            duration = int(float(parts[2]))
        except ValueError:
            continue
        rows.append({"start_sec": float(start_sec), "end_sec": float(start_sec + duration), "event_type": str(parts[1]).strip()})
    return rows


def _prepare_st_vincent(raw_root: Path) -> pd.DataFrame:
    base = raw_root / "st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
    rows: List[Dict[str, Any]] = []
    for stage_path in sorted(base.glob("ucddb*_stage.txt")):
        recording_id = stage_path.stem.replace("_stage", "")
        stage_vals = [int(x.strip()) for x in stage_path.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        events = _parse_st_vincent_events(base / f"{recording_id}_respevt.txt")
        for epoch_index, stage_raw in enumerate(stage_vals):
            start_sec = float(epoch_index * 30)
            end_sec = start_sec + 30.0
            apnea_binary = 0
            for ev in events:
                if ev["start_sec"] < end_sec and ev["end_sec"] > start_sec:
                    apnea_binary = 1
                    break
            rows.append(
                {
                    "dataset_id": "st_vincent_apnea",
                    "subject_unit_id": recording_id,
                    "recording_id": recording_id,
                    "epoch_index": epoch_index,
                    "epoch_start_sec": start_sec,
                    "epoch_end_sec": end_sec,
                    "source_file": f"{recording_id}_lifecard.edf",
                    "eeg_channel_standardized": "chan 1",
                    "apnea_binary": apnea_binary,
                    "sleep_stage": stage_raw,
                    "severity_global": np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["sleep_stage"] = normalize_sleep_stage_series(out["sleep_stage"])
    out["sleep_stage"] = pd.to_numeric(out["sleep_stage"], errors="coerce").map(_ST_VINCENT_STAGE_MAP).fillna(out["sleep_stage"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multitask apnea/staging metadata.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--raw-root", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--combined-output", type=str, default="apnea_multitask_combined.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sleep_edf = _prepare_sleep_edf(pd.read_csv(processed_dir / "sleep_edf_expanded_raw.csv"))
    mitbih = _prepare_mitbih(pd.read_csv(processed_dir / "mitbih_sleep_stages.csv"), pd.read_csv(processed_dir / "mitbih_respiratory_events.csv"))
    shhs = _prepare_shhs(pd.read_csv(processed_dir / "shhs_sleep_stages.csv"), pd.read_csv(processed_dir / "shhs_respiratory_events.csv"))
    st_vincent = _prepare_st_vincent(raw_root)

    outputs = {
        "sleep_edf_expanded_multitask.csv": sleep_edf,
        "mitbih_apnea_binary_multitask.csv": mitbih,
        "shhs_apnea_binary_multitask.csv": shhs,
        "st_vincent_apnea_binary_multitask.csv": st_vincent,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    combined = pd.concat([sleep_edf, mitbih, shhs, st_vincent], ignore_index=True)
    combined.to_csv(output_dir / args.combined_output, index=False)
    print(f"Wrote multitask metadata under {output_dir}")


if __name__ == "__main__":
    main()
