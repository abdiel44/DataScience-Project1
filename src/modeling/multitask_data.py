"""Metadata prep and waveform datasets for apnea/no-apnea multitask deep experiments."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from modeling.epoch_store import normalize_input_mode
from modeling.target_utils import normalize_sleep_stage_series
from modeling.waveform_io import load_waveform_record

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


DEFAULT_STAGE_ORDER: Tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
_ST_VINCENT_STAGE_MAP: Dict[int, str] = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "N3",
    5: "REM",
}


def read_multitask_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported metadata format {suffix!r}. Use .csv or .parquet.")


def standardize_multitask_metadata(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_unit_id",
    recording_col: str = "recording_id",
    order_col: str = "epoch_index",
    subject_fraction: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Normalize and validate metadata shared by multitask apnea experiments."""
    out = df.copy()
    if "dataset_id" not in out.columns:
        raise ValueError("Metadata must include dataset_id.")
    if "apnea_binary" not in out.columns:
        out["apnea_binary"] = pd.NA
    if "sleep_stage" not in out.columns:
        out["sleep_stage"] = pd.NA
    if "eeg_channel_standardized" not in out.columns:
        out["eeg_channel_standardized"] = pd.NA
    if "severity_global" not in out.columns:
        out["severity_global"] = np.nan

    if "subject_unit_id" not in out.columns and subject_col in out.columns:
        out["subject_unit_id"] = out[subject_col]
    if "recording_id" not in out.columns and recording_col in out.columns:
        out["recording_id"] = out[recording_col]
    if "epoch_index" not in out.columns and order_col in out.columns:
        out["epoch_index"] = out[order_col]

    out["sleep_stage"] = normalize_sleep_stage_series(out["sleep_stage"])
    st_mask = out["dataset_id"].astype(str).eq("st_vincent_apnea") & out["sleep_stage"].notna()
    if st_mask.any():
        vals = pd.to_numeric(out.loc[st_mask, "sleep_stage"], errors="coerce")
        mapped = vals.map(_ST_VINCENT_STAGE_MAP)
        out.loc[st_mask, "sleep_stage"] = mapped.fillna(out.loc[st_mask, "sleep_stage"]).astype("string")

    out["label_mask_apnea"] = out["apnea_binary"].notna().astype(int)
    out["label_mask_stage"] = out["sleep_stage"].notna().astype(int)
    out = out[(out["label_mask_apnea"] > 0) | (out["label_mask_stage"] > 0)].copy()
    if out.empty:
        raise ValueError("No rows left after removing samples without apnea or staging labels.")

    for col in ("epoch_start_sec", "epoch_end_sec", "epoch_index", "dataset_id", "subject_unit_id", "recording_id"):
        if col not in out.columns:
            raise ValueError(f"Metadata must include {col!r}.")
    out = out.dropna(subset=["dataset_id", "subject_unit_id", "recording_id", "epoch_start_sec", "epoch_end_sec"])
    out["apnea_binary"] = pd.to_numeric(out["apnea_binary"], errors="coerce")
    out["epoch_index"] = pd.to_numeric(out["epoch_index"], errors="coerce").fillna(0).astype(int)
    out["epoch_start_sec"] = pd.to_numeric(out["epoch_start_sec"], errors="coerce")
    out["epoch_end_sec"] = pd.to_numeric(out["epoch_end_sec"], errors="coerce")
    out["subject_unit_id"] = out["subject_unit_id"].astype(str)
    out["recording_id"] = out["recording_id"].astype(str)
    out["dataset_id"] = out["dataset_id"].astype(str)
    out = out.sort_values(["dataset_id", "recording_id", "epoch_index"]).reset_index(drop=True)

    if subject_fraction < 1.0:
        if not 0 < subject_fraction <= 1.0:
            raise ValueError("subject_fraction must be in (0, 1].")
        subjects = out["subject_unit_id"].drop_duplicates().to_numpy()
        n_keep = max(1, int(np.floor(len(subjects) * subject_fraction)))
        rng = np.random.default_rng(random_seed)
        keep = set(rng.choice(subjects, size=n_keep, replace=False).tolist())
        out = out[out["subject_unit_id"].isin(keep)].copy()
        if out.empty:
            raise ValueError("subject_fraction removed every subject from metadata.")
    return out.reset_index(drop=True)


def build_sequence_index(
    df: pd.DataFrame,
    *,
    recording_col: str = "recording_id",
    order_col: str = "epoch_index",
    sequence_length: int,
) -> List[np.ndarray]:
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


def load_recording_waveform(
    row: Mapping[str, Any],
    *,
    raw_root: Path,
    dataset_cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, float]:
    record = load_waveform_record(row, raw_root=raw_root, dataset_cfg=dataset_cfg)
    return record.signal, float(record.sfreq)


class MultiTaskWaveformDataset(Dataset):
    """Waveform sequence dataset with masked apnea/stage labels."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        *,
        sequence_indices: Sequence[np.ndarray],
        raw_root: Path,
        dataset_cfg: Mapping[str, Any],
        stage_label_to_index: Mapping[str, int],
        signal_loader: Optional[Any] = None,
    ) -> None:
        if torch is None:  # pragma: no cover - runtime guard
            raise RuntimeError("torch is required to use MultiTaskWaveformDataset.")
        self.metadata = metadata_df.reset_index(drop=True).copy()
        self.sequence_indices = [np.asarray(x, dtype=np.int64) for x in sequence_indices]
        self.raw_root = Path(raw_root)
        self.dataset_cfg = dict(dataset_cfg)
        self.stage_label_to_index = {str(k): int(v) for k, v in stage_label_to_index.items()}
        self.signal_loader = signal_loader
        self.samples_per_epoch = int(
            round(float(self.dataset_cfg.get("sample_hz", 100.0)) * float(self.dataset_cfg.get("epoch_seconds", 30.0)))
        )
        self.normalize_each_epoch = bool(self.dataset_cfg.get("normalize_each_epoch", True))
        self.max_recordings_in_memory = int(self.dataset_cfg.get("max_recordings_in_memory", 4))
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
        self.sample_recording_ids = [
            str(self.metadata.iloc[int(seq[len(seq) // 2])]["recording_id"]) for seq in self.sequence_indices
        ]

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def _cache_key(self, row: Mapping[str, Any]) -> str:
        return f"{row['dataset_id']}::{row['recording_id']}"

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
            sfreq = float(row.get("sample_hz", self.dataset_cfg.get("sample_hz", 100.0)))
        elif self.signal_loader is None:
            signal, sfreq = load_recording_waveform(row, raw_root=self.raw_root, dataset_cfg=self.dataset_cfg)
        else:
            signal, sfreq = self.signal_loader(row, self.raw_root, self.dataset_cfg)
        self._recording_cache[key] = (signal, float(sfreq))
        while len(self._recording_cache) > self.max_recordings_in_memory:
            self._recording_cache.popitem(last=False)
        return self._recording_cache[key]

    def _extract_epoch(self, signal: np.ndarray, sfreq: float, row: Mapping[str, Any]) -> np.ndarray:
        if self.input_mode == "epoch_store":
            epoch = np.asarray(signal[int(row["epoch_store_row"])], dtype=np.float32)
            if len(epoch) > self.samples_per_epoch:
                epoch = epoch[: self.samples_per_epoch]
            elif len(epoch) < self.samples_per_epoch:
                pad = self.samples_per_epoch - len(epoch)
                epoch = np.pad(epoch, (0, pad), mode="edge" if len(epoch) > 0 else "constant")
            if self.normalize_each_epoch:
                mean = float(epoch.mean())
                std = float(epoch.std())
                epoch = epoch - mean
                if std > 1e-6:
                    epoch = epoch / std
            return epoch.astype(np.float32, copy=False)
        start_idx = int(round(float(row["epoch_start_sec"]) * sfreq))
        end_idx = int(round(float(row["epoch_end_sec"]) * sfreq))
        epoch = signal[start_idx:end_idx].astype(np.float32, copy=False)
        if len(epoch) > self.samples_per_epoch:
            epoch = epoch[: self.samples_per_epoch]
        elif len(epoch) < self.samples_per_epoch:
            pad = self.samples_per_epoch - len(epoch)
            epoch = np.pad(epoch, (0, pad), mode="edge" if len(epoch) > 0 else "constant")
        if self.normalize_each_epoch:
            mean = float(epoch.mean())
            std = float(epoch.std())
            epoch = epoch - mean
            if std > 1e-6:
                epoch = epoch / std
        return epoch.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_rows = self.sequence_indices[idx]
        center_row = self.metadata.iloc[int(seq_rows[len(seq_rows) // 2])].to_dict()
        signal, sfreq = self._load_recording(center_row)
        epochs = []
        for row_idx in seq_rows:
            row = self.metadata.iloc[int(row_idx)].to_dict()
            epochs.append(self._extract_epoch(signal, sfreq, row))
        x = np.stack(epochs, axis=0).astype(np.float32, copy=False)

        apnea_mask = int(center_row.get("label_mask_apnea", 0))
        stage_mask = int(center_row.get("label_mask_stage", 0))
        apnea_target = float(center_row["apnea_binary"]) if apnea_mask else 0.0
        stage_label = str(center_row["sleep_stage"]) if stage_mask else ""
        stage_target = self.stage_label_to_index[stage_label] if stage_mask else -100
        return {
            "x": torch.from_numpy(x),
            "apnea_target": torch.tensor(apnea_target, dtype=torch.float32),
            "apnea_mask": torch.tensor(apnea_mask, dtype=torch.float32),
            "stage_target": torch.tensor(stage_target, dtype=torch.long),
            "stage_mask": torch.tensor(stage_mask, dtype=torch.float32),
            "dataset_id": str(center_row["dataset_id"]),
            "subject_id": str(center_row["subject_unit_id"]),
            "recording_id": str(center_row["recording_id"]),
            "center_epoch_index": int(center_row.get("epoch_index", 0)),
        }
