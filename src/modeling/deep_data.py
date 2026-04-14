"""Waveform sequence datasets for deep Phase E experiments."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modeling.epoch_store import normalize_input_mode
from modeling.target_utils import normalize_sleep_stage_series
from modeling.waveform_io import load_waveform_record

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class when torch is unavailable."""

        pass


DEFAULT_STAGE_ORDER: Tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")


def _read_table_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format {suffix!r} for {path}. Use .csv or .parquet.")


def load_sleep_edf_recording(
    row: Mapping[str, Any],
    *,
    raw_root: Path,
    dataset_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, float]:
    """Load one Sleep-EDF recording into memory and optionally resample it."""
    record = load_waveform_record(row, raw_root=raw_root, dataset_cfg=dataset_cfg, default_dataset_id="sleep_edf_expanded")
    return record.signal, float(record.sfreq)


def prepare_sequence_metadata(
    df: pd.DataFrame,
    *,
    target_col: str,
    subject_col: str,
    recording_col: str,
    order_col: str,
    label_subset: Optional[Sequence[str]] = None,
    subject_fraction: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Normalize labels and optionally subsample subjects for dev runs."""
    out = df.copy()
    if target_col == "sleep_stage":
        out[target_col] = normalize_sleep_stage_series(out[target_col])
    out = out.dropna(subset=[target_col, subject_col, recording_col, order_col, "source_file"])
    if label_subset:
        allowed = {str(x) for x in label_subset}
        out = out[out[target_col].astype(str).isin(allowed)].copy()
        if out.empty:
            raise ValueError(f"label_subset={sorted(allowed)!r} removed every row from {target_col!r}.")
    if subject_fraction < 1.0:
        if not 0 < subject_fraction <= 1.0:
            raise ValueError("subject_fraction must be in (0, 1].")
        subjects = out[subject_col].astype(str).drop_duplicates().to_numpy()
        n_keep = max(1, int(np.floor(len(subjects) * subject_fraction)))
        rng = np.random.default_rng(random_seed)
        chosen = set(rng.choice(subjects, size=n_keep, replace=False).tolist())
        out = out[out[subject_col].astype(str).isin(chosen)].copy()
    out = out.sort_values([recording_col, order_col]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No rows left after metadata preparation.")
    return out


def build_sequence_index(
    df: pd.DataFrame,
    *,
    recording_col: str,
    order_col: str,
    sequence_length: int,
) -> List[np.ndarray]:
    """Build fixed-length centered windows without crossing recording boundaries."""
    if sequence_length <= 0 or sequence_length % 2 == 0:
        raise ValueError("sequence_length must be a positive odd integer.")
    ordered = df.sort_values([recording_col, order_col]).reset_index(drop=True)
    windows: List[np.ndarray] = []
    half = sequence_length // 2
    for _, group in ordered.groupby(recording_col, sort=False):
        idx = group.index.to_list()
        last = len(idx) - 1
        for center in range(len(idx)):
            seq = [idx[min(max(center + delta, 0), last)] for delta in range(-half, half + 1)]
            windows.append(np.asarray(seq, dtype=np.int64))
    return windows


class WaveformSequenceDataset(Dataset):
    """Sequence dataset over epoch-level metadata with on-demand waveform extraction."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        *,
        sequence_indices: Sequence[np.ndarray],
        target_col: str,
        subject_col: str,
        recording_col: str,
        raw_root: Path,
        dataset_cfg: Mapping[str, Any],
        label_to_index: Mapping[str, int],
        signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
    ) -> None:
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("torch is required to use WaveformSequenceDataset.")
        self.metadata = metadata_df.reset_index(drop=True).copy()
        self.sequence_indices = [np.asarray(idx, dtype=np.int64) for idx in sequence_indices]
        self.target_col = str(target_col)
        self.subject_col = str(subject_col)
        self.recording_col = str(recording_col)
        self.raw_root = Path(raw_root)
        self.dataset_cfg = dict(dataset_cfg)
        self.label_to_index = {str(k): int(v) for k, v in label_to_index.items()}
        self.signal_loader = signal_loader
        self.sequence_length = int(self.dataset_cfg.get("sequence_length", len(self.sequence_indices[0])))
        self.center_row_indices = [int(idx[len(idx) // 2]) for idx in self.sequence_indices]
        self.sample_hz = float(self.dataset_cfg.get("sample_hz", 100.0))
        self.epoch_seconds = float(self.dataset_cfg.get("epoch_seconds", 30.0))
        self.samples_per_epoch = int(round(self.sample_hz * self.epoch_seconds))
        self.max_recordings_in_memory = int(self.dataset_cfg.get("max_recordings_in_memory", 4))
        self.normalize_each_epoch = bool(self.dataset_cfg.get("normalize_each_epoch", True))
        self.input_mode = normalize_input_mode(self.dataset_cfg.get("input_mode", "raw"))
        self.epoch_store_root = Path(str(self.dataset_cfg.get("epoch_store_root", ""))) if self.input_mode == "epoch_store" else None
        if self.input_mode == "epoch_store":
            if self.epoch_store_root is None or not str(self.epoch_store_root):
                raise ValueError("dataset.epoch_store_root is required when dataset.input_mode=epoch_store.")
            missing = [col for col in ("epoch_store_relpath", "epoch_store_row") if col not in self.metadata.columns]
            if missing:
                raise ValueError(
                    f"Epoch-store mode requires manifest columns {missing!r}. "
                    "Run scripts/materialize_epoch_store.py first."
                )
        self._recording_cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()
        self.sequence_label_indices = [
            self.label_to_index[str(self.metadata.iloc[row_idx][self.target_col])] for row_idx in self.center_row_indices
        ]
        self.sample_recording_ids = [str(self.metadata.iloc[row_idx][self.recording_col]) for row_idx in self.center_row_indices]

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def _cache_key(self, row: Mapping[str, Any]) -> str:
        return str(row[self.recording_col])

    def _load_recording(self, row: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
        key = self._cache_key(row)
        cached = self._recording_cache.get(key)
        if cached is not None:
            self._recording_cache.move_to_end(key)
            return cached
        if self.input_mode == "epoch_store":
            if self.epoch_store_root is None:
                raise ValueError("epoch_store_root is not configured.")
            relpath = str(row["epoch_store_relpath"]).replace("\\", "/")
            signal = np.load(self.epoch_store_root / relpath, mmap_mode="r")
            sfreq = float(row.get("sample_hz", self.sample_hz))
        elif self.signal_loader is None:
            signal, sfreq = load_sleep_edf_recording(row, raw_root=self.raw_root, dataset_cfg=self.dataset_cfg)
        else:
            signal, sfreq = self.signal_loader(row, self.raw_root, self.dataset_cfg)
        self._recording_cache[key] = (signal, float(sfreq))
        while len(self._recording_cache) > self.max_recordings_in_memory:
            self._recording_cache.popitem(last=False)
        return self._recording_cache[key]

    def _extract_epoch(self, signal: np.ndarray, sfreq: float, row: Mapping[str, Any]) -> np.ndarray:
        if self.input_mode == "epoch_store":
            epoch = np.asarray(signal[int(row["epoch_store_row"])], dtype=np.float32)
            if len(epoch) != self.samples_per_epoch:
                if len(epoch) > self.samples_per_epoch:
                    epoch = epoch[: self.samples_per_epoch]
                else:
                    pad_width = self.samples_per_epoch - len(epoch)
                    epoch = np.pad(epoch, (0, pad_width), mode="edge" if len(epoch) > 0 else "constant")
            if self.normalize_each_epoch:
                mean = float(epoch.mean())
                std = float(epoch.std())
                epoch = epoch - mean
                if std > 1e-6:
                    epoch = epoch / std
            return epoch.astype(np.float32, copy=False)
        start_sec = float(row["epoch_start_sec"])
        end_sec = float(row["epoch_end_sec"])
        start_idx = int(round(start_sec * sfreq))
        end_idx = int(round(end_sec * sfreq))
        epoch = signal[start_idx:end_idx].astype(np.float32, copy=False)
        if len(epoch) > self.samples_per_epoch:
            epoch = epoch[: self.samples_per_epoch]
        elif len(epoch) < self.samples_per_epoch:
            pad_width = self.samples_per_epoch - len(epoch)
            epoch = np.pad(epoch, (0, pad_width), mode="edge" if len(epoch) > 0 else "constant")
        if self.normalize_each_epoch:
            mean = float(epoch.mean())
            std = float(epoch.std())
            epoch = epoch - mean
            if std > 1e-6:
                epoch = epoch / std
        return epoch.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_rows = self.sequence_indices[idx]
        center_pos = len(seq_rows) // 2
        center_row = self.metadata.iloc[int(seq_rows[center_pos])].to_dict()
        signal, sfreq = self._load_recording(center_row)
        epochs = []
        for row_idx in seq_rows:
            row = self.metadata.iloc[int(row_idx)].to_dict()
            epochs.append(self._extract_epoch(signal, sfreq, row))
        x = np.stack(epochs, axis=0).astype(np.float32, copy=False)
        label = str(center_row[self.target_col])
        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(self.label_to_index[label], dtype=torch.long),
            "label": label,
            "subject_id": str(center_row[self.subject_col]),
            "recording_id": str(center_row[self.recording_col]),
            "center_epoch_index": int(center_row.get("epoch_index", center_pos)),
        }


def read_deep_metadata(path: Path) -> pd.DataFrame:
    return _read_table_file(path)
