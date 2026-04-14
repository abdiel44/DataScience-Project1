from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modeling.waveform_io import load_waveform_record
from pre_processing.epoch_signal_features import extract_epoch_signal_features

STORE_VERSION = 1
EPOCH_STORE_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "epoch_store_relpath",
    "epoch_store_row",
    "sample_hz",
    "samples_per_epoch",
    "channel_name_used",
    "store_version",
)


def normalize_input_mode(value: Any) -> str:
    mode = str(value or "raw").strip().lower()
    if mode not in {"raw", "epoch_store"}:
        raise ValueError("dataset.input_mode must be 'raw' or 'epoch_store'.")
    return mode


def read_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format {suffix!r} for {path}. Use .csv or .parquet.")


def write_table_file(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported table format {suffix!r} for {path}. Use .csv or .parquet.")


def validate_epoch_store_manifest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    missing = [col for col in EPOCH_STORE_REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(
            f"Epoch-store manifest is missing columns {missing!r}. "
            "Run scripts/materialize_epoch_store.py first."
        )
    out["epoch_store_relpath"] = out["epoch_store_relpath"].astype(str)
    out["epoch_store_row"] = pd.to_numeric(out["epoch_store_row"], errors="raise").astype(int)
    out["sample_hz"] = pd.to_numeric(out["sample_hz"], errors="raise")
    out["samples_per_epoch"] = pd.to_numeric(out["samples_per_epoch"], errors="raise").astype(int)
    out["store_version"] = pd.to_numeric(out["store_version"], errors="raise").astype(int)
    return out


def read_epoch_store_manifest(path: Path) -> pd.DataFrame:
    return validate_epoch_store_manifest(read_table_file(path))


def recording_store_relpath(dataset_id: str, recording_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(recording_id).strip())
    return Path(str(dataset_id)) / f"{safe}.npy"


def recording_meta_relpath(store_relpath: Path) -> Path:
    return store_relpath.with_suffix(".json")


def _extract_epoch_from_waveform(
    signal: np.ndarray,
    sfreq: float,
    *,
    start_sec: float,
    end_sec: float,
    samples_per_epoch: int,
) -> np.ndarray:
    start_idx = int(round(float(start_sec) * float(sfreq)))
    end_idx = int(round(float(end_sec) * float(sfreq)))
    epoch = signal[start_idx:end_idx].astype(np.float32, copy=False)
    if len(epoch) > samples_per_epoch:
        epoch = epoch[:samples_per_epoch]
    elif len(epoch) < samples_per_epoch:
        pad = samples_per_epoch - len(epoch)
        epoch = np.pad(epoch, (0, pad), mode="edge" if len(epoch) > 0 else "constant")
    return epoch.astype(np.float32, copy=False)


def _write_recording_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)


def _read_recording_metadata(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _validate_existing_store(
    *,
    store_path: Path,
    meta_path: Path,
    dataset_id: str,
    recording_id: str,
    channel_expected: str,
    sample_hz: float,
    epoch_seconds: float,
    n_epochs: int,
    samples_per_epoch: int,
) -> Dict[str, Any]:
    if not store_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Epoch store or its metadata sidecar is missing.")
    arr = np.load(store_path, mmap_mode="r")
    if tuple(arr.shape) != (n_epochs, samples_per_epoch):
        raise ValueError(
            f"Existing store {store_path} has shape {tuple(arr.shape)}, expected {(n_epochs, samples_per_epoch)}."
        )
    meta = _read_recording_metadata(meta_path)
    checks = {
        "dataset_id": str(meta.get("dataset_id")) == str(dataset_id),
        "recording_id": str(meta.get("recording_id")) == str(recording_id),
        "sample_hz": abs(float(meta.get("sample_hz", -1.0)) - float(sample_hz)) < 1e-6,
        "epoch_seconds": abs(float(meta.get("epoch_seconds", -1.0)) - float(epoch_seconds)) < 1e-6,
        "samples_per_epoch": int(meta.get("samples_per_epoch", -1)) == int(samples_per_epoch),
        "store_version": int(meta.get("store_version", -1)) == int(STORE_VERSION),
    }
    if not all(checks.values()):
        raise ValueError(f"Existing store metadata mismatch for {store_path}: {checks}")
    channel_used = str(meta.get("channel_name_used", "")).strip()
    if channel_used and channel_expected and channel_used != channel_expected:
        raise ValueError(
            f"Existing store channel mismatch for {store_path}: got {channel_used!r}, expected {channel_expected!r}."
        )
    return meta


def materialize_epoch_store(
    metadata_df: pd.DataFrame,
    *,
    store_root: Path,
    manifest_path: Path,
    raw_root: Path,
    dataset_cfg: Mapping[str, Any],
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
    force: bool = False,
    skip_existing: bool = True,
) -> pd.DataFrame:
    if metadata_df.empty:
        raise ValueError("metadata_df is empty; nothing to materialize.")
    store_root = Path(store_root)
    manifest_path = Path(manifest_path)
    store_root.mkdir(parents=True, exist_ok=True)
    epoch_seconds = float(dataset_cfg.get("epoch_seconds", 30.0))
    samples_per_epoch_default = int(round(float(dataset_cfg.get("sample_hz", 100.0)) * epoch_seconds))
    out_rows: List[Dict[str, Any]] = []
    ordered = metadata_df.sort_values(["dataset_id", "recording_id", "epoch_index"]).reset_index(drop=True)

    for (dataset_id, recording_id), group in ordered.groupby(["dataset_id", "recording_id"], sort=False):
        group = group.sort_values("epoch_index").reset_index(drop=True)
        first = group.iloc[0].to_dict()
        store_relpath = recording_store_relpath(str(dataset_id), str(recording_id))
        store_path = store_root / store_relpath
        meta_path = store_root / recording_meta_relpath(store_relpath)
        channel_expected = str(first.get("eeg_channel_standardized", "")).strip()

        if store_path.is_file() and not force and skip_existing:
            meta = _validate_existing_store(
                store_path=store_path,
                meta_path=meta_path,
                dataset_id=str(dataset_id),
                recording_id=str(recording_id),
                channel_expected=channel_expected,
                sample_hz=float(dataset_cfg.get("sample_hz", 100.0)),
                epoch_seconds=epoch_seconds,
                n_epochs=len(group),
                samples_per_epoch=samples_per_epoch_default,
            )
            sample_hz = float(meta["sample_hz"])
            samples_per_epoch = int(meta["samples_per_epoch"])
            channel_name_used = str(meta.get("channel_name_used", channel_expected))
        else:
            if signal_loader is None:
                record = load_waveform_record(first, raw_root=raw_root, dataset_cfg=dataset_cfg)
                signal = record.signal
                sample_hz = float(record.sfreq)
                channel_name_used = str(record.channel_name)
                source_path = str(record.source_path)
            else:
                signal, sample_hz = signal_loader(first, raw_root, dataset_cfg)
                signal = np.asarray(signal, dtype=np.float32)
                sample_hz = float(sample_hz)
                channel_name_used = channel_expected or str(first.get("channel_name_used", ""))
                source_path = str(first.get("source_file", ""))
            samples_per_epoch = int(round(sample_hz * epoch_seconds))
            epochs = np.stack(
                [
                    _extract_epoch_from_waveform(
                        signal,
                        sample_hz,
                        start_sec=float(row["epoch_start_sec"]),
                        end_sec=float(row["epoch_end_sec"]),
                        samples_per_epoch=samples_per_epoch,
                    )
                    for row in group.to_dict(orient="records")
                ],
                axis=0,
            ).astype(np.float32, copy=False)
            store_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(store_path, epochs, allow_pickle=False)
            _write_recording_metadata(
                meta_path,
                {
                    "dataset_id": str(dataset_id),
                    "recording_id": str(recording_id),
                    "sample_hz": float(sample_hz),
                    "epoch_seconds": float(epoch_seconds),
                    "samples_per_epoch": int(samples_per_epoch),
                    "channel_name_used": channel_name_used,
                    "source_path": source_path,
                    "store_version": int(STORE_VERSION),
                    "n_epochs": int(len(group)),
                },
            )

        for epoch_row, row in enumerate(group.to_dict(orient="records")):
            out_row = dict(row)
            out_row["epoch_store_relpath"] = store_relpath.as_posix()
            out_row["epoch_store_row"] = int(epoch_row)
            out_row["sample_hz"] = float(sample_hz)
            out_row["samples_per_epoch"] = int(samples_per_epoch)
            out_row["channel_name_used"] = channel_name_used
            out_row["store_version"] = int(STORE_VERSION)
            out_rows.append(out_row)

    manifest_df = validate_epoch_store_manifest(pd.DataFrame(out_rows))
    write_table_file(manifest_df, manifest_path)
    return manifest_df


def export_epoch_store_features(
    manifest_df: pd.DataFrame,
    *,
    store_root: Path,
    output_path: Path,
    feature_prefix: str = "eeg",
    normalize_epoch: bool = False,
) -> pd.DataFrame:
    manifest = validate_epoch_store_manifest(manifest_df)
    store_root = Path(store_root)
    output_rows: List[Dict[str, Any]] = []
    cache: OrderedDict[str, np.ndarray] = OrderedDict()
    for relpath, group in manifest.groupby("epoch_store_relpath", sort=False):
        path = store_root / str(relpath)
        arr = np.load(path, mmap_mode="r")
        cache[str(relpath)] = arr
        sample_hz = float(group["sample_hz"].iloc[0])
        for row in group.to_dict(orient="records"):
            epoch_idx = int(row["epoch_store_row"])
            feat = extract_epoch_signal_features(
                np.asarray(arr[epoch_idx], dtype=np.float32),
                sfreq=sample_hz,
                prefix=feature_prefix,
                normalize_epoch=normalize_epoch,
            )
            out_row = dict(row)
            out_row.update(feat)
            output_rows.append(out_row)
    features_df = pd.DataFrame(output_rows)
    write_table_file(features_df, output_path)
    return features_df
