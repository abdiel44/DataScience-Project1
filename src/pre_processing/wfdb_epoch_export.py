"""
Export 30 s epoch feature tables from WFDB PSG corpora (MIT-BIH, SHHS).

MIT-BIH: annotations in `.st` (annotator `st`); each entry marks a contiguous segment until the
next annotation (PhysioNet: labels the following interval; segments are ~30 s at 250 Hz).

SHHS: `hypn` = sleep stage per 30 s; `resp` / `arou` = respiratory and arousal events at
irregular times; event rows use the 30 s window aligned to floor(sample/30)*30 at fs=1.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb

from pre_processing.cleaning import to_snake_case

EPOCH_SEC = 30.0

# README.st event codes (token match after sleep-stage token).
MITBIH_EVENT_CODES = frozenset({"H", "HA", "OA", "X", "CA", "CAA", "L", "LA", "A", "MT"})

# Sleep stage tokens (first field of MIT-BIH aux_note).
MITBIH_SLEEP_STAGE_TOKENS = frozenset({"W", "R", "1", "2", "3", "4"})


@dataclass
class ExportStats:
    n_staging_rows: int
    n_event_rows: int
    n_epochs_skipped_other: int
    n_records: int


def _normalize_aux_token(t: str) -> str:
    return str(t).strip()


def parse_mitbih_aux_note(aux: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Split MIT-BIH `st` aux_note: first token = sleep stage (W,R,1-4); further tokens = events.
    """
    parts = [_normalize_aux_token(x) for x in str(aux).split() if _normalize_aux_token(x)]
    if not parts:
        return "unknown", ()
    stage = parts[0]
    events = tuple(p for p in parts[1:] if p in MITBIH_EVENT_CODES)
    return stage, events


def route_mitbih_row(stage: str, events: Tuple[str, ...]) -> Tuple[bool, bool]:
    """Return (include_in_staging, include_in_events)."""
    in_stage = stage in MITBIH_SLEEP_STAGE_TOKENS
    in_evt = len(events) > 0
    return in_stage, in_evt


@contextmanager
def _chdir(path: Path) -> Iterator[None]:
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _read_record_ids(records_file: Path) -> List[str]:
    lines = records_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _feature_dict_from_slice(
    p_signal: np.ndarray,
    sig_names: Sequence[str],
    start: int,
    end: int,
) -> Dict[str, float]:
    start = max(0, int(start))
    end = max(start, int(end))
    end = min(end, p_signal.shape[0])
    if end <= start:
        return {}
    sl = p_signal[start:end, :]
    out: Dict[str, float] = {}
    for j, name in enumerate(sig_names):
        col = np.asarray(sl[:, j], dtype=float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        base = to_snake_case(str(name))
        out[f"{base}_mean"] = float(np.mean(col))
        out[f"{base}_std"] = float(np.std(col, ddof=0))
    return out


def mitbih_dataset_dir(raw_root: Path) -> Path:
    p = raw_root / "mit-bih-polysomnographic-database-1.0.0"
    if not p.is_dir():
        raise FileNotFoundError(f"MIT-BIH folder not found under {raw_root}")
    return p


def shhs_dataset_dir(raw_root: Path) -> Path:
    p = raw_root / "sleep-heart-health-study-psg-database-1.0.0"
    if not p.is_dir():
        raise FileNotFoundError(f"SHHS folder not found under {raw_root}")
    return p


def iter_mitbih_epochs(
    dataset_dir: Path,
    *,
    max_records: Optional[int] = None,
    max_staging_rows: Optional[int] = None,
    max_event_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], ExportStats]:
    records_path = dataset_dir / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError(f"RECORDS not found: {records_path}")
    record_ids = _read_record_ids(records_path)
    if max_records is not None:
        record_ids = record_ids[: max(0, max_records)]

    staging_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    skipped_other = 0

    with _chdir(dataset_dir):
        for record_id in record_ids:
            rec = wfdb.rdrecord(record_id)
            ann = wfdb.rdann(record_id, "st")
            sig_names = list(rec.sig_name)
            for i in range(ann.ann_len):
                if max_staging_rows is not None and len(staging_rows) >= max_staging_rows:
                    break
                start = int(ann.sample[i])
                end = int(ann.sample[i + 1]) if i + 1 < ann.ann_len else int(rec.sig_len)
                aux = ann.aux_note[i] if ann.aux_note is not None else ""
                stage, events = parse_mitbih_aux_note(aux)
                inc_st, inc_ev = route_mitbih_row(stage, events)
                if not inc_st and not inc_ev:
                    skipped_other += 1
                    continue
                feats = _feature_dict_from_slice(np.asarray(rec.p_signal), sig_names, start, end)
                base_meta = {
                    "record_id": record_id,
                    "epoch_index": i,
                    "epoch_start_sample": start,
                    "epoch_end_sample": end,
                    "epoch_start_sec": start / float(rec.fs),
                    "epoch_end_sec": end / float(rec.fs),
                    "aux_raw": aux,
                }
                if inc_st:
                    row = {**base_meta, "sleep_stage": stage, **feats}
                    staging_rows.append(row)
                if inc_ev and (max_event_rows is None or (max_event_rows > 0 and len(event_rows) < max_event_rows)):
                    erow = {
                        **base_meta,
                        "sleep_stage": stage,
                        "event_tokens": " ".join(events),
                        **feats,
                    }
                    event_rows.append(erow)
            if max_staging_rows is not None and len(staging_rows) >= max_staging_rows:
                break

    stats = ExportStats(
        n_staging_rows=len(staging_rows),
        n_event_rows=len(event_rows),
        n_epochs_skipped_other=skipped_other,
        n_records=len(record_ids),
    )
    return staging_rows, event_rows, stats


def export_mitbih_two_csvs(
    raw_root: Path,
    out_sleep_csv: Path,
    out_event_csv: Path,
    *,
    max_records: Optional[int] = None,
    max_staging_rows: Optional[int] = None,
    max_event_rows: Optional[int] = None,
) -> ExportStats:
    d = mitbih_dataset_dir(raw_root)
    st_rows, ev_rows, stats = iter_mitbih_epochs(
        d,
        max_records=max_records,
        max_staging_rows=max_staging_rows,
        max_event_rows=max_event_rows,
    )
    out_sleep_csv.parent.mkdir(parents=True, exist_ok=True)
    out_event_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(st_rows).to_csv(out_sleep_csv, index=False)
    pd.DataFrame(ev_rows).to_csv(out_event_csv, index=False)
    return stats


def iter_shhs_epochs(
    dataset_dir: Path,
    *,
    max_records: Optional[int] = None,
    max_staging_rows: Optional[int] = None,
    max_event_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], ExportStats]:
    records_path = dataset_dir / "RECORDS"
    if not records_path.is_file():
        raise FileNotFoundError(f"RECORDS not found: {records_path}")
    record_ids = _read_record_ids(records_path)
    if max_records is not None:
        record_ids = record_ids[: max(0, max_records)]

    staging_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    skipped_other = 0

    with _chdir(dataset_dir):
        for record_id in record_ids:
            rec = wfdb.rdrecord(record_id)
            sig_names = list(rec.sig_name)
            p_signal = np.asarray(rec.p_signal)
            fs = float(rec.fs)
            hypn = wfdb.rdann(record_id, "hypn")
            for i in range(hypn.ann_len):
                if max_staging_rows is not None and len(staging_rows) >= max_staging_rows:
                    break
                end = int(hypn.sample[i])
                start = end - int(round(EPOCH_SEC * fs))
                if start < 0:
                    start = 0
                label = hypn.aux_note[i] if hypn.aux_note is not None else ""
                label = _normalize_aux_token(label)
                feats = _feature_dict_from_slice(p_signal, sig_names, start, end)
                staging_rows.append(
                    {
                        "record_id": record_id,
                        "epoch_index": i,
                        "epoch_start_sample": start,
                        "epoch_end_sample": end,
                        "epoch_start_sec": start / fs,
                        "epoch_end_sec": end / fs,
                        "sleep_stage": label,
                        **feats,
                    }
                )

            if max_event_rows is not None and max_event_rows <= 0:
                continue
            for suffix, kind in (("resp", "resp"), ("arou", "arou")):
                try:
                    ann = wfdb.rdann(record_id, suffix)
                except Exception:
                    continue
                for j in range(ann.ann_len):
                    if max_event_rows is not None and len(event_rows) >= max_event_rows:
                        break
                    s = int(ann.sample[j])
                    win_start = (s // int(EPOCH_SEC * fs)) * int(EPOCH_SEC * fs)
                    win_end = win_start + int(EPOCH_SEC * fs)
                    if win_end > p_signal.shape[0]:
                        win_end = p_signal.shape[0]
                    if win_start >= win_end:
                        skipped_other += 1
                        continue
                    aux = ann.aux_note[j] if ann.aux_note is not None else ""
                    feats = _feature_dict_from_slice(p_signal, sig_names, win_start, win_end)
                    event_rows.append(
                        {
                            "record_id": record_id,
                            "annotation_source": kind,
                            "annotation_index": j,
                            "event_center_sample": s,
                            "epoch_start_sample": win_start,
                            "epoch_end_sample": win_end,
                            "epoch_start_sec": win_start / fs,
                            "epoch_end_sec": win_end / fs,
                            "event_label": str(aux).strip(),
                            **feats,
                        }
                    )

    stats = ExportStats(
        n_staging_rows=len(staging_rows),
        n_event_rows=len(event_rows),
        n_epochs_skipped_other=skipped_other,
        n_records=len(record_ids),
    )
    return staging_rows, event_rows, stats


def export_shhs_two_csvs(
    raw_root: Path,
    out_sleep_csv: Path,
    out_event_csv: Path,
    *,
    max_records: Optional[int] = None,
    max_staging_rows: Optional[int] = None,
    max_event_rows: Optional[int] = None,
) -> ExportStats:
    d = shhs_dataset_dir(raw_root)
    st_rows, ev_rows, stats = iter_shhs_epochs(
        d,
        max_records=max_records,
        max_staging_rows=max_staging_rows,
        max_event_rows=max_event_rows,
    )
    out_sleep_csv.parent.mkdir(parents=True, exist_ok=True)
    out_event_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(st_rows).to_csv(out_sleep_csv, index=False)
    pd.DataFrame(ev_rows).to_csv(out_event_csv, index=False)
    return stats


def mitbih_staging_dataframe(
    raw_root: Path,
    *,
    max_records: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, ExportStats]:
    """Build staging-only dataframe (for --source mit-bih-psg) without writing CSV."""
    d = mitbih_dataset_dir(raw_root)
    rows, _, stats = iter_mitbih_epochs(
        d,
        max_records=max_records,
        max_staging_rows=max_rows,
        max_event_rows=0,
    )
    return pd.DataFrame(rows), stats


def shhs_staging_dataframe(
    raw_root: Path,
    *,
    max_records: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, ExportStats]:
    d = shhs_dataset_dir(raw_root)
    rows, _, stats = iter_shhs_epochs(
        d,
        max_records=max_records,
        max_staging_rows=max_rows,
        max_event_rows=0,
    )
    return pd.DataFrame(rows), stats
