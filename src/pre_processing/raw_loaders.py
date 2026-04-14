"""
Ingest heterogeneous PSG/sleep corpora into one tabular dataframe per source.

Each loader returns rows suitable for supervised learning with a dataset-specific
target plus numeric features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from pre_processing.cleaning import to_snake_case
from pre_processing.epoch_signal_features import extract_epoch_signal_features
from pre_processing.wfdb_epoch_export import mitbih_staging_dataframe, shhs_staging_dataframe

SourceId = Literal[
    "isruc_sleep",
    "st_vincent_apnea",
    "sleep_edf_expanded",
    "sleep_edf_2013_fpzcz",
    "mit_bih_psg",
    "shhs_psg",
]


@dataclass(frozen=True)
class IngestResult:
    """Metadata returned alongside the dataframe."""

    source: SourceId
    n_files_used: int
    n_files_skipped: int
    notes: Tuple[str, ...] = ()


_STAGEN_RE = re.compile(r"Stagen(\d+)", re.IGNORECASE)
_ISRUC_SUBJECT_RE = re.compile(r"^(S\d+_p\d+(?:_\d+)?)", re.IGNORECASE)
_SLEEP_EDF_SUBJECT_RE = re.compile(r"^([A-Z]{2}\d{4})", re.IGNORECASE)

_SLEEP_EDF_STAGE_TO_AASM: Dict[str, str] = {
    "sleep stage w": "W",
    "sleep stage 1": "N1",
    "sleep stage 2": "N2",
    "sleep stage 3": "N3",
    "sleep stage 4": "N3",
    "sleep stage r": "REM",
}

_ISRUC_EEG_CHANNELS: Tuple[str, ...] = (
    "C4-M1",
    "C3-M2",
    "F4-M1",
    "F3-M2",
    "O2-M1",
    "O1-M2",
    "E1-M2",
    "E2-M1",
)
_SLEEP_EDF_EEG_CHANNELS: Tuple[str, ...] = ("EEG Fpz-Cz", "EEG Pz-Oz")
_SLEEP_EDF_FPZ_CZ_CHANNELS: Tuple[str, ...] = ("EEG Fpz-Cz",)
_ISRUC_DEFAULT_SFREQ = 200.0
_SLEEP_EDF_EPOCH_SEC = 30.0


def _parse_isruc_relative_path(rel: str) -> Tuple[str, Optional[int]]:
    """
    Derive coarse event group and sleep stage index from ISRUC path/name.

    Folders: Events/plm, Events/rem, Non_Events/...
    Filenames often contain Stagen{n}.
    """
    parts = Path(rel.replace("\\", "/")).parts
    event_group = "unknown"
    lower_parts = [p.lower() for p in parts]
    if any("plm" in p for p in lower_parts):
        event_group = "plm"
    elif any("rem" in p for p in lower_parts):
        event_group = "rem"
    elif any("non_event" in p for p in lower_parts):
        event_group = "non_event"

    m = _STAGEN_RE.search(rel)
    stage: Optional[int] = int(m.group(1)) if m else None
    return event_group, stage


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def _parse_isruc_subject_id(path: Path) -> str:
    m = _ISRUC_SUBJECT_RE.match(path.stem)
    if m:
        return m.group(1)
    return path.stem


def _choose_first_available(columns: List[str], preferred: Tuple[str, ...]) -> Optional[str]:
    colset = set(columns)
    for name in preferred:
        if name in colset:
            return name
    return None


def _first_numeric_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        ser = pd.to_numeric(df[col], errors="coerce")
        if ser.notna().sum() > 0:
            return str(col)
    return None


def _row_from_isruc_csv(path: Path, rel: str) -> Optional[Dict[str, Any]]:
    df = _safe_read_csv(path)
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        return None

    event_group, sleep_stage = _parse_isruc_relative_path(rel)
    if sleep_stage is None:
        return None

    eeg_col = _choose_first_available(df.columns.tolist(), _ISRUC_EEG_CHANNELS)
    if eeg_col is None:
        eeg_col = _first_numeric_column(df)
    if eeg_col is None:
        return None

    eeg_signal = pd.to_numeric(df[eeg_col], errors="coerce").to_numpy(dtype=float)
    eeg_features = extract_epoch_signal_features(eeg_signal, _ISRUC_DEFAULT_SFREQ, prefix="eeg")
    if not eeg_features:
        return None

    rel_norm = rel.replace("\\", "/")
    return {
        "source_file": rel_norm,
        "subject_unit_id": _parse_isruc_subject_id(Path(rel_norm)),
        "event_group": event_group,
        "sleep_stage": sleep_stage,
        "eeg_channel": to_snake_case(eeg_col),
        "sfreq_hz": _ISRUC_DEFAULT_SFREQ,
        "epoch_duration_sec": float(len(eeg_signal) / _ISRUC_DEFAULT_SFREQ),
        **eeg_features,
    }


def ingest_isruc_sleep(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Walk ISRUC CSV segments and emit one row per segment.

    The subject identifier is derived from the filename prefix (e.g. `S1_p33_1`),
    not the parent directory, so subject-wise CV can group by the real segment owner.
    A single EEG channel is selected per segment and exported through generic `eeg_*`
    features shared with other staging datasets.
    """
    base = raw_root / "ISRUC-Sleep"
    if not base.is_dir():
        raise FileNotFoundError(f"Expected ISRUC-Sleep under {raw_root!s}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    n = 0
    for csv_path in sorted(base.rglob("*.csv")):
        if max_files is not None and n >= max_files:
            break
        rel = str(csv_path.relative_to(base))
        row = _row_from_isruc_csv(csv_path, rel)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
        n += 1

    if not rows:
        raise ValueError("No usable ISRUC CSV rows produced (all files skipped or unreadable).")

    return pd.DataFrame(rows), IngestResult(
        source="isruc_sleep",
        n_files_used=len(rows),
        n_files_skipped=skipped,
        notes=(
            "One row per CSV segment; subject inferred from filename prefix.",
            "Single EEG channel exported as generic eeg_* features; assumed ISRUC sample rate = 200 Hz.",
        ),
    )


def ingest_st_vincent_apnea_stages(
    raw_root: Path,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Build one row per `ucddb*_stage.txt` integer hypnogram.

    Does not read EDF signals; only the stage text files present in the corpus.
    """
    base = raw_root / "st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0"
    if not base.is_dir():
        raise FileNotFoundError(f"Expected St Vincent apnea folder under {raw_root!s}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for stage_path in sorted(base.glob("ucddb*_stage.txt")):
        try:
            lines = stage_path.read_text(encoding="utf-8", errors="replace").splitlines()
            vals = [int(x.strip()) for x in lines if x.strip() != ""]
        except Exception:
            skipped += 1
            continue
        if not vals:
            skipped += 1
            continue
        arr = np.array(vals, dtype=int)
        rec_id = stage_path.stem.replace("_stage", "")
        vc = pd.Series(arr).value_counts(normalize=True)
        row: Dict[str, Any] = {
            "recording_id": rec_id,
            "n_epochs": int(len(arr)),
            "stage_mode": int(pd.Series(arr).mode().iloc[0]),
            "stage_median": float(np.median(arr)),
        }
        for k in range(6):
            row[f"stage_{k}_frac"] = float(vc.get(k, 0.0))
        rows.append(row)

    if not rows:
        raise ValueError("No ucddb*_stage.txt files produced rows.")

    return pd.DataFrame(rows), IngestResult(
        source="st_vincent_apnea",
        n_files_used=len(rows),
        n_files_skipped=skipped,
        notes=("Labels are epoch-level sleep stages; row-level targets are distributional summaries.",),
    )


def _sleep_edf_subject_id(recording_id: str) -> str:
    m = _SLEEP_EDF_SUBJECT_RE.match(recording_id)
    if m:
        return m.group(1)
    return recording_id[:-2] if len(recording_id) > 2 else recording_id


def _apply_sleep_edf_wake_trim(
    rows: List[Dict[str, Any]],
    *,
    wake_edge_mins: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    if not rows:
        return rows, 0, 0

    before_count = len(rows)
    wake_edge_epochs = int((wake_edge_mins * 60) / _SLEEP_EDF_EPOCH_SEC)
    non_wake_idx = [idx for idx, row in enumerate(rows) if str(row.get("sleep_stage")) != "W"]
    if not non_wake_idx:
        return rows, before_count, before_count

    start_idx = max(0, non_wake_idx[0] - wake_edge_epochs)
    end_idx = min(before_count - 1, non_wake_idx[-1] + wake_edge_epochs)
    trimmed = [dict(row) for row in rows[start_idx : end_idx + 1]]
    after_count = len(trimmed)
    for row in trimmed:
        row["recording_epochs_before_trim"] = before_count
        row["recording_epochs_after_trim"] = after_count
        row["wake_trim_minutes"] = wake_edge_mins
    return trimmed, before_count, after_count


def _add_temporal_context_features(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    feature_cols: List[str],
    lags: Tuple[int, ...] = (1, 2),
    leads: Tuple[int, ...] = (1, 2),
) -> pd.DataFrame:
    if df.empty or not feature_cols:
        return df

    out = df.sort_values([group_col, order_col]).copy()
    grouped = out.groupby(group_col, sort=False)
    first_vals = grouped[feature_cols].transform("first")
    last_vals = grouped[feature_cols].transform("last")

    for lag in sorted(set(int(x) for x in lags if int(x) > 0)):
        shifted = grouped[feature_cols].shift(lag).fillna(first_vals)
        shifted = shifted.rename(columns={col: f"{col}_lag{lag}" for col in feature_cols})
        out[shifted.columns] = shifted.to_numpy(dtype=float, copy=False)

    for lead in sorted(set(int(x) for x in leads if int(x) > 0)):
        shifted = grouped[feature_cols].shift(-lead).fillna(last_vals)
        shifted = shifted.rename(columns={col: f"{col}_lead{lead}" for col in feature_cols})
        out[shifted.columns] = shifted.to_numpy(dtype=float, copy=False)

    return out


def _iter_sleep_edf_pairs(base: Path) -> List[Tuple[Path, Path, str]]:
    pairs: List[Tuple[Path, Path, str]] = []
    for edf in sorted(base.rglob("*-PSG.edf")):
        rec_id = edf.name.replace("-PSG.edf", "")
        hyp_prefix = rec_id[:-1] if len(rec_id) > 1 else rec_id
        hypo_matches = sorted(edf.parent.glob(f"{hyp_prefix}*-Hypnogram.edf"))
        if not hypo_matches:
            continue
        pairs.append((edf, hypo_matches[0], rec_id))
    return pairs


def _sleep_edf_epoch_rows_for_recording(
    *,
    base: Path,
    edf: Path,
    hypnogram: Path,
    rec_id: str,
    preferred_channels: Tuple[str, ...],
    normalize_epoch: bool,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[float]]:
    try:
        import mne  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "sleep_edf ingestion requires `mne`. Install with: pip install mne",
        ) from e

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    eeg_channel = _choose_first_available(list(raw.ch_names), preferred_channels)
    if eeg_channel is None:
        return [], None, None

    sfreq = float(raw.info["sfreq"])
    signal = raw.get_data(picks=[eeg_channel])[0]
    ann = mne.read_annotations(hypnogram)

    subject_id = _sleep_edf_subject_id(rec_id)
    source_file = str(edf.relative_to(base)).replace("\\", "/")
    hyp_file = str(hypnogram.relative_to(base)).replace("\\", "/")
    rows: List[Dict[str, Any]] = []
    epoch_index = 0

    for desc, onset, duration in zip(ann.description, ann.onset, ann.duration):
        mapped = _SLEEP_EDF_STAGE_TO_AASM.get(str(desc).strip().lower())
        if mapped is None:
            continue
        n_epochs = int(np.floor(float(duration) / _SLEEP_EDF_EPOCH_SEC))
        for local_idx in range(n_epochs):
            start_sec = float(onset) + (local_idx * _SLEEP_EDF_EPOCH_SEC)
            end_sec = start_sec + _SLEEP_EDF_EPOCH_SEC
            start_sample = int(round(start_sec * sfreq))
            end_sample = int(round(end_sec * sfreq))
            if end_sample > len(signal) or start_sample >= end_sample:
                continue
            feats = extract_epoch_signal_features(
                signal[start_sample:end_sample],
                sfreq,
                prefix="eeg",
                normalize_epoch=normalize_epoch,
            )
            if not feats:
                continue
            rows.append(
                {
                    "recording_id": rec_id,
                    "subject_id": subject_id,
                    "epoch_index": epoch_index,
                    "epoch_start_sample": start_sample,
                    "epoch_end_sample": end_sample,
                    "epoch_start_sec": start_sec,
                    "epoch_end_sec": end_sec,
                    "sleep_stage": mapped,
                    "eeg_channel": to_snake_case(eeg_channel),
                    "sfreq_hz": sfreq,
                    "source_file": source_file,
                    "hypnogram_file": hyp_file,
                    **feats,
                }
            )
            epoch_index += 1
    return rows, eeg_channel, sfreq


def _ingest_sleep_edf_epochs(
    raw_root: Path,
    *,
    relative_dir: str,
    source_id: SourceId,
    preferred_channels: Tuple[str, ...],
    normalize_epoch: bool,
    wake_trim_mins: Optional[int],
    add_temporal_context: bool,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    base = raw_root / "sleep-edf-database-expanded-1.0.0" / relative_dir
    if not base.is_dir():
        raise FileNotFoundError(f"Expected {relative_dir} under sleep-edf-database-expanded-1.0.0 in {raw_root!s}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    n_used = 0

    for edf, hypnogram, rec_id in _iter_sleep_edf_pairs(base):
        if max_files is not None and n_used >= max_files:
            break
        try:
            recording_rows, eeg_channel, _sfreq = _sleep_edf_epoch_rows_for_recording(
                base=base,
                edf=edf,
                hypnogram=hypnogram,
                rec_id=rec_id,
                preferred_channels=preferred_channels,
                normalize_epoch=normalize_epoch,
            )
        except Exception:
            skipped += 1
            continue
        if not recording_rows or eeg_channel is None:
            skipped += 1
            continue

        if wake_trim_mins is not None:
            recording_rows, before_trim, after_trim = _apply_sleep_edf_wake_trim(
                recording_rows,
                wake_edge_mins=wake_trim_mins,
            )
            if after_trim == 0:
                skipped += 1
                continue
            for row in recording_rows:
                row["sleep_edf_variant"] = source_id
                row["sleep_edf_subset"] = relative_dir
        rows.extend(recording_rows)
        n_used += 1

    if not rows:
        raise ValueError("No Sleep-EDF epoch rows produced (check mne, EEG channel, and hypnogram layout).")

    df = pd.DataFrame(rows)
    if add_temporal_context:
        eeg_feature_cols = [
            c
            for c in df.columns
            if c.startswith("eeg_") and pd.api.types.is_numeric_dtype(df[c])
        ]
        df = _add_temporal_context_features(
            df,
            group_col="recording_id",
            order_col="epoch_index",
            feature_cols=eeg_feature_cols,
            lags=(1, 2),
            leads=(1, 2),
        )

    notes = [
        "One row per 30 s epoch from paired PSG/hypnogram EDF files.",
        f"Single EEG channel exported as generic eeg_* features; preferred channel(s) = {preferred_channels}.",
    ]
    if normalize_epoch:
        notes.append("Each epoch is normalized to zero mean and unit variance before feature extraction.")
    if wake_trim_mins is not None:
        notes.append(f"Wake trimming applied: sleep period plus {wake_trim_mins} min wake margins on both sides.")
    if add_temporal_context:
        notes.append("Temporal context appended with lag/lead features for t-2, t-1, t+1, t+2 within recording.")

    return df, IngestResult(
        source=source_id,
        n_files_used=n_used,
        n_files_skipped=skipped,
        notes=tuple(notes),
    )


def ingest_sleep_edf_expanded_epochs(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Export one row per 30 s epoch from paired Sleep-EDF PSG + hypnogram EDF files.

    Uses one preferred EEG channel and emits generic `eeg_*` features so the resulting
    tables can participate in cross-dataset staging experiments.
    """
    return _ingest_sleep_edf_epochs(
        raw_root,
        relative_dir=".",
        source_id="sleep_edf_expanded",
        preferred_channels=_SLEEP_EDF_EEG_CHANNELS,
        normalize_epoch=False,
        wake_trim_mins=None,
        add_temporal_context=False,
        max_files=max_files,
    )


def ingest_sleep_edf_2013_fpzcz(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Sleep-EDF 2013 cassette subset aligned to the SleepEEGNet problem definition.

    Rules:
    - subset = sleep-cassette (SC subjects)
    - channel = EEG Fpz-Cz only
    - 30 s epochs with stage 3/4 merged into N3
    - drop movement / unknown labels
    - keep sleep plus +/- 30 min wake
    - append lag/lead context features within each recording
    """
    return _ingest_sleep_edf_epochs(
        raw_root,
        relative_dir="sleep-cassette",
        source_id="sleep_edf_2013_fpzcz",
        preferred_channels=_SLEEP_EDF_FPZ_CZ_CHANNELS,
        normalize_epoch=True,
        wake_trim_mins=30,
        add_temporal_context=True,
        max_files=max_files,
    )


def ingest_mit_bih_psg(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    df, st = mitbih_staging_dataframe(raw_root, max_records=max_files, max_rows=None)
    return df, IngestResult(
        source="mit_bih_psg",
        n_files_used=st.n_records,
        n_files_skipped=0,
        notes=(
            "30 s epochs from WFDB annotator st; column sleep_stage.",
            "Export respiratory/events with: python src/main.py --export-epochs mit-bih-psg ...",
        ),
    )


def ingest_shhs_psg(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    df, st = shhs_staging_dataframe(raw_root, max_records=max_files, max_rows=None)
    return df, IngestResult(
        source="shhs_psg",
        n_files_used=st.n_records,
        n_files_skipped=0,
        notes=(
            "30 s epochs from hypn annotations; column sleep_stage.",
            "Respiratory/arousal rows: --export-epochs shhs-psg with --output-events.",
        ),
    )


_INGEST_FUNCS: Dict[SourceId, Callable[..., Tuple[pd.DataFrame, IngestResult]]] = {
    "isruc_sleep": ingest_isruc_sleep,
    "st_vincent_apnea": ingest_st_vincent_apnea_stages,
    "sleep_edf_expanded": ingest_sleep_edf_expanded_epochs,
    "sleep_edf_2013_fpzcz": ingest_sleep_edf_2013_fpzcz,
    "mit_bih_psg": ingest_mit_bih_psg,
    "shhs_psg": ingest_shhs_psg,
}


def ingest_by_source_id(
    source_id: SourceId,
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """Dispatch to the correct loader. `raw_root` should be `data/raw`."""
    fn = _INGEST_FUNCS[source_id]
    if source_id in ("sleep_edf_expanded", "sleep_edf_2013_fpzcz", "isruc_sleep", "mit_bih_psg", "shhs_psg"):
        return fn(raw_root, max_files=max_files)
    return fn(raw_root)


def list_supported_sources() -> Tuple[SourceId, ...]:
    return tuple(_INGEST_FUNCS.keys())
