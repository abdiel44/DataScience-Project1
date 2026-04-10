"""
Ingest heterogeneous PSG/sleep corpora into **one tabular DataFrame** per source so that
`cleaning`, `encoding`, `scaling`, etc. can run on a standard CSV-like schema.

Each loader returns rows suitable for supervised learning (derived labels + numeric features).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from pre_processing.cleaning import to_snake_case
from pre_processing.wfdb_epoch_export import mitbih_staging_dataframe, shhs_staging_dataframe

SourceId = Literal[
    "isruc_sleep",
    "st_vincent_apnea",
    "sleep_edf_expanded",
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


def _parse_isruc_relative_path(rel: str) -> Tuple[str, Optional[int]]:
    """
    Derive coarse event group and sleep stage index from ISRUC file path/name.

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


def _row_from_isruc_csv(path: Path, rel: str) -> Optional[Dict[str, Any]]:
    df = _safe_read_csv(path)
    if df is None or df.shape[1] < 5:
        return None
    event_group, sleep_stage = _parse_isruc_relative_path(rel)
    rel_norm = rel.replace("\\", "/")
    p = Path(rel_norm)
    parts = p.parts
    # CV grouping: directory path under ISRUC-Sleep (excludes filename). Captures Subgroup_k/subject/…
    # layouts and avoids collapsing everything under a single top folder (e.g. Non_Events/batch vs Non_Events/other).
    if len(parts) >= 2:
        subject_unit_id = "/".join(parts[:-1])
    else:
        subject_unit_id = p.stem
    out: Dict[str, Any] = {
        "source_file": rel_norm,
        "subject_unit_id": subject_unit_id,
        "event_group": event_group,
        "sleep_stage": sleep_stage,
    }
    for col in df.columns:
        key = to_snake_case(str(col))
        ser = pd.to_numeric(df[col], errors="coerce")
        if ser.notna().sum() == 0:
            continue
        out[f"{key}_mean"] = float(ser.mean())
        out[f"{key}_std"] = float(ser.std(ddof=0))
    return out


def ingest_isruc_sleep(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Walk `ISRUC-Sleep` CSV segments (one file = one short epoch window) and aggregate each file
    to **one row**: per-channel mean/std plus `event_group` and `sleep_stage` parsed from paths.

    Column names from different montages do not align across rows; missing channels are left NaN
    (handled by later cleaning / modeling).
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

    out_df = pd.DataFrame(rows)
    return out_df, IngestResult(
        source="isruc_sleep",
        n_files_used=len(rows),
        n_files_skipped=skipped,
        notes=("One row per CSV segment; features are channel mean/std; labels from path/filename.",),
    )


def ingest_st_vincent_apnea_stages(
    raw_root: Path,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Build one row per `ucddb*_stage.txt` integer hypnogram (epoch labels 0–5 as in PhysioNet).

    Does not read EDF signals — only the stage text files present in the corpus.
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


def ingest_sleep_edf_expanded_summary(
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    One row per *PSG* EDF recording with basic signal metadata (requires `mne`).

    Hypnogram annotations are summarized when a matching `*Hypnogram.edf` exists.
    """
    try:
        import mne  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "sleep_edf_expanded ingestion requires `mne`. Install with: pip install mne",
        ) from e

    base = raw_root / "sleep-edf-database-expanded-1.0.0"
    if not base.is_dir():
        raise FileNotFoundError(f"Expected sleep-edf-database-expanded-1.0.0 under {raw_root!s}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    n = 0
    for edf in sorted(base.glob("*-PSG.edf")):
        if max_files is not None and n >= max_files:
            break
        stem = edf.name.replace("-PSG.edf", "")
        hypo_matches = sorted(base.glob(f"{stem}*-Hypnogram.edf"))
        rec_id = stem
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
            duration = float(raw.times[-1]) if len(raw.times) else 0.0
            nchan = len(raw.ch_names)
            sfreq = float(raw.info["sfreq"])
        except Exception:
            skipped += 1
            continue

        row: Dict[str, Any] = {
            "recording_id": rec_id,
            "duration_sec": duration,
            "n_channels": nchan,
            "sfreq_first": sfreq,
        }
        if hypo_matches:
            try:
                h = mne.io.read_raw_edf(hypo_matches[0], preload=True, verbose="ERROR")
                if h.annotations is not None and len(h.annotations) > 0:
                    descs = list(h.annotations.description)
                    vc = pd.Series(descs).value_counts(normalize=True)
                    for lab, p in vc.items():
                        row[f"hypno_frac_{to_snake_case(str(lab))}"] = float(p)
            except Exception:
                pass

        rows.append(row)
        n += 1

    if not rows:
        raise ValueError("No PSG EDF rows produced (check mne and file layout).")

    return pd.DataFrame(rows), IngestResult(
        source="sleep_edf_expanded",
        n_files_used=len(rows),
        n_files_skipped=skipped,
        notes=("Signal-level summary only; extend with your own epoching if needed.",),
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
            "30 s epochs from WFDB annotator st; column sleep_stage. "
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
            "30 s epochs from hypn annotations; column sleep_stage. "
            "Respiratory/arousal rows: --export-epochs shhs-psg with --output-events.",
        ),
    )


_INGEST_FUNCS: Dict[SourceId, Callable[..., Tuple[pd.DataFrame, IngestResult]]] = {
    "isruc_sleep": ingest_isruc_sleep,
    "st_vincent_apnea": ingest_st_vincent_apnea_stages,
    "sleep_edf_expanded": ingest_sleep_edf_expanded_summary,
    "mit_bih_psg": ingest_mit_bih_psg,
    "shhs_psg": ingest_shhs_psg,
}


def ingest_by_source_id(
    source_id: SourceId,
    raw_root: Path,
    *,
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, IngestResult]:
    """
    Dispatch to the correct loader. `raw_root` should be `data/raw` (parent of each dataset folder).
    """
    fn = _INGEST_FUNCS[source_id]
    if source_id == "sleep_edf_expanded":
        return fn(raw_root, max_files=max_files)
    if source_id == "isruc_sleep":
        return fn(raw_root, max_files=max_files)
    if source_id in ("mit_bih_psg", "shhs_psg"):
        return fn(raw_root, max_files=max_files)
    return fn(raw_root)


def list_supported_sources() -> Tuple[SourceId, ...]:
    return tuple(_INGEST_FUNCS.keys())
