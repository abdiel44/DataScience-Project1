from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import resample


@dataclass(frozen=True)
class WaveformRecord:
    signal: np.ndarray
    sfreq: float
    channel_name: str
    source_path: str


def _match_one_signal_channel(ch_names: Sequence[str], requested: str) -> str:
    if requested in ch_names:
        return requested
    lower = {str(c).lower(): str(c) for c in ch_names}
    req_low = str(requested).lower()
    if req_low in lower:
        return lower[req_low]
    req_norm = "".join(ch for ch in req_low if ch.isalnum())
    for c in ch_names:
        cand = "".join(ch for ch in str(c).lower() if ch.isalnum())
        if cand == req_norm:
            return str(c)
    raise KeyError(f"Signal channel {requested!r} not found in {list(ch_names)!r}.")


def match_signal_channel(ch_names: Sequence[str], requested: Any) -> str:
    if isinstance(requested, (list, tuple)):
        errors = []
        for item in requested:
            try:
                return _match_one_signal_channel(ch_names, str(item))
            except KeyError as exc:
                errors.append(str(exc))
        eeg_candidates = [str(c) for c in ch_names if str(c).strip().lower().startswith("eeg")]
        if eeg_candidates:
            return eeg_candidates[0]
        raise KeyError("; ".join(errors) if errors else f"No compatible signal channel found in {list(ch_names)!r}.")
    return _match_one_signal_channel(ch_names, str(requested))


def resample_signal(signal: np.ndarray, orig_hz: float, target_hz: float) -> np.ndarray:
    if abs(float(orig_hz) - float(target_hz)) < 1e-6:
        return signal.astype(np.float32, copy=False)
    n_target = int(round(len(signal) * float(target_hz) / float(orig_hz)))
    if n_target <= 0:
        raise ValueError("Cannot resample signal to a non-positive number of samples.")
    return resample(signal, n_target).astype(np.float32, copy=False)


def _sleep_edf_signal_path(row: Mapping[str, Any], raw_root: Path, dataset_cfg: Mapping[str, Any]) -> Path:
    dataset_dir = str(dataset_cfg.get("dataset_dirname", "sleep-edf-database-expanded-1.0.0"))
    source_rel = str(row["source_file"]).replace("\\", "/")
    subset = str(row.get("sleep_edf_subset", "")).strip()
    base = raw_root / dataset_dir
    if subset:
        return base / subset / source_rel
    return base / source_rel


def _dataset_dir(dataset_id: str, raw_root: Path, dataset_cfg: Mapping[str, Any]) -> Path:
    dirname_map = dict(dataset_cfg.get("dataset_dirnames", {}) or {})
    default_map = {
        "sleep_edf_expanded": "sleep-edf-database-expanded-1.0.0",
        "mit_bih_psg": "mit-bih-polysomnographic-database-1.0.0",
        "shhs_psg": "sleep-heart-health-study-psg-database-1.0.0",
        "st_vincent_apnea": "st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0",
    }
    dirname = str(dirname_map.get(dataset_id, default_map.get(dataset_id, dataset_id)))
    return raw_root / dirname


def _dataset_channel(
    dataset_id: str,
    dataset_cfg: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    fallback_channel: str,
) -> Any:
    channel_map = dict(dataset_cfg.get("signal_channels", {}) or {})
    row_channel = str(row.get("eeg_channel_standardized", "")).strip()
    configured = channel_map.get(dataset_id)
    if isinstance(configured, (list, tuple)):
        if row_channel and row_channel.lower() != "nan":
            return [row_channel, *[str(x) for x in configured if str(x) != row_channel]]
        return [str(x) for x in configured]
    if row_channel and row_channel.lower() != "nan":
        return str(channel_map.get(dataset_id, row_channel))
    if dataset_id == "sleep_edf_expanded":
        return str(dataset_cfg.get("signal_channel", channel_map.get(dataset_id, fallback_channel)))
    return str(channel_map.get(dataset_id, fallback_channel))


def load_waveform_record(
    row: Mapping[str, Any],
    *,
    raw_root: Path,
    dataset_cfg: Mapping[str, Any],
    default_dataset_id: str = "sleep_edf_expanded",
) -> WaveformRecord:
    dataset_id = str(row.get("dataset_id", default_dataset_id))
    target_hz = float(dataset_cfg.get("sample_hz", row.get("sfreq_hz", 100.0)))

    if dataset_id == "mit_bih_psg":
        import wfdb

        dataset_dir = _dataset_dir(dataset_id, raw_root, dataset_cfg)
        record_id = str(row["recording_id"])
        channel = _dataset_channel(
            dataset_id,
            dataset_cfg,
            row,
            fallback_channel="EEG (C4-A1)",
        )
        record_path = (dataset_dir / record_id).resolve()
        rec = wfdb.rdrecord(str(record_path))
        pick = match_signal_channel(list(rec.sig_name), channel)
        idx = list(rec.sig_name).index(pick)
        signal = np.asarray(rec.p_signal[:, idx], dtype=np.float32)
        resampled = resample_signal(signal, float(rec.fs), target_hz)
        return WaveformRecord(signal=resampled, sfreq=target_hz, channel_name=pick, source_path=str(record_path))

    try:
        import mne  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError("mne is required for waveform loading.") from exc

    if dataset_id == "sleep_edf_expanded":
        path = _sleep_edf_signal_path(row, raw_root, dataset_cfg)
        channel = _dataset_channel(
            dataset_id,
            dataset_cfg,
            row,
            fallback_channel="EEG Fpz-Cz",
        )
    elif dataset_id == "shhs_psg":
        dataset_dir = _dataset_dir(dataset_id, raw_root, dataset_cfg)
        source_rel = str(row.get("source_file", f"{row['recording_id']}.edf")).replace("\\", "/")
        path = dataset_dir / source_rel
        channel = _dataset_channel(dataset_id, dataset_cfg, row, fallback_channel="EEG")
    elif dataset_id == "st_vincent_apnea":
        dataset_dir = _dataset_dir(dataset_id, raw_root, dataset_cfg)
        source_rel = str(row.get("source_file", f"{row['recording_id']}_lifecard.edf")).replace("\\", "/")
        path = dataset_dir / source_rel
        channel = _dataset_channel(dataset_id, dataset_cfg, row, fallback_channel="chan 1")
    else:
        raise ValueError(f"Unsupported dataset_id {dataset_id!r} for waveform loading.")

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    pick = match_signal_channel(list(raw.ch_names), channel)
    signal = raw.get_data(picks=[pick])[0].astype(np.float32, copy=False)
    resampled = resample_signal(signal, float(raw.info["sfreq"]), target_hz)
    return WaveformRecord(signal=resampled, sfreq=target_hz, channel_name=pick, source_path=str(path))
