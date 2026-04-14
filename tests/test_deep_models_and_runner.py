"""Smoke tests for the deep Phase E runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from modeling.epoch_store import materialize_epoch_store
from modeling.deep_data import WaveformSequenceDataset, build_sequence_index
from modeling.deep_models import build_supervised_model
from modeling.deep_runner import run_cv


def _signal_bank() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t = np.linspace(0, 1, 120, endpoint=False, dtype=np.float32)
    for idx, rec in enumerate(["r0", "r1", "r2", "r3"]):
        out[rec] = np.sin(2 * np.pi * (idx + 1) * t).astype(np.float32)
    return out


def _signal_loader(row: Mapping[str, Any], _raw_root: Path, _cfg: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    bank = _signal_bank()
    return bank[str(row["recording_id"])], 10.0


def _failing_loader(_row: Mapping[str, Any], _raw_root: Path, _cfg: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    raise AssertionError("raw waveform loader should not be called in epoch_store mode")


@pytest.fixture()
def tiny_deep_df() -> pd.DataFrame:
    rows = []
    labels = ["W", "N1", "N2", "N3"]
    for rec_idx, rec in enumerate(["r0", "r1", "r2", "r3"]):
        for epoch_idx in range(4):
            rows.append(
                {
                    "recording_id": rec,
                    "subject_id": f"s{rec_idx}",
                    "source_file": f"{rec}.edf",
                    "epoch_index": epoch_idx,
                    "epoch_start_sec": float(epoch_idx * 3),
                    "epoch_end_sec": float((epoch_idx + 1) * 3),
                    "sleep_stage": labels[(rec_idx + epoch_idx) % len(labels)],
                }
            )
    return pd.DataFrame(rows)


def test_build_sequence_index_keeps_recording_boundaries(tiny_deep_df: pd.DataFrame) -> None:
    seq = build_sequence_index(tiny_deep_df, recording_col="recording_id", order_col="epoch_index", sequence_length=3)
    assert len(seq) == len(tiny_deep_df)
    first = seq[0].tolist()
    assert first == [0, 0, 1]
    boundary = seq[3].tolist()
    assert boundary == [2, 3, 3]


def test_waveform_sequence_dataset_returns_expected_shape(tiny_deep_df: pd.DataFrame, tmp_path: Path) -> None:
    seq = build_sequence_index(tiny_deep_df, recording_col="recording_id", order_col="epoch_index", sequence_length=3)
    ds = WaveformSequenceDataset(
        tiny_deep_df,
        sequence_indices=seq,
        target_col="sleep_stage",
        subject_col="subject_id",
        recording_col="recording_id",
        raw_root=tmp_path,
        dataset_cfg={"sequence_length": 3, "sample_hz": 10, "epoch_seconds": 3, "normalize_each_epoch": True},
        label_to_index={"W": 0, "N1": 1, "N2": 2, "N3": 3},
        signal_loader=_signal_loader,
    )
    item = ds[0]
    assert tuple(item["x"].shape) == (3, 30)
    assert int(item["y"]) in {0, 1, 2, 3}


def test_waveform_sequence_dataset_epoch_store_avoids_raw_loader(tiny_deep_df: pd.DataFrame, tmp_path: Path) -> None:
    meta = tiny_deep_df.copy()
    meta["dataset_id"] = "sleep_edf_expanded"
    manifest = materialize_epoch_store(
        meta,
        store_root=tmp_path / "store",
        manifest_path=tmp_path / "manifest.parquet",
        raw_root=tmp_path,
        dataset_cfg={"sample_hz": 10, "epoch_seconds": 3, "signal_channel": "EEG Fpz-Cz"},
        signal_loader=_signal_loader,
    )
    seq = build_sequence_index(manifest, recording_col="recording_id", order_col="epoch_index", sequence_length=3)
    ds = WaveformSequenceDataset(
        manifest,
        sequence_indices=seq,
        target_col="sleep_stage",
        subject_col="subject_id",
        recording_col="recording_id",
        raw_root=tmp_path,
        dataset_cfg={
            "input_mode": "epoch_store",
            "epoch_store_root": str(tmp_path / "store"),
            "sequence_length": 3,
            "sample_hz": 10,
            "epoch_seconds": 3,
            "normalize_each_epoch": True,
        },
        label_to_index={"W": 0, "N1": 1, "N2": 2, "N3": 3},
        signal_loader=_failing_loader,
    )
    item = ds[0]
    assert tuple(item["x"].shape) == (3, 30)


def test_build_supervised_model_forward_smoke() -> None:
    model = build_supervised_model(
        {
            "type": "conformer",
            "sequence_length": 3,
            "embedding_dim": 32,
            "conformer_blocks": 1,
            "attention_heads": 4,
            "ffn_dim": 64,
            "conv_kernel_size": 7,
            "dropout": 0.1,
        },
        num_classes=4,
    )
    x = torch.randn(2, 3, 30)
    logits = model(x)
    assert tuple(logits.shape) == (2, 4)


def test_run_cv_deep_cnn_smoke(tiny_deep_df: pd.DataFrame, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "deep_cnn_smoke",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_id",
        "recording_column": "recording_id",
        "target_column": "sleep_stage",
        "task": "multiclass",
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "dataset": {
            "raw_root": str(tmp_path),
            "sequence_length": 3,
            "sample_hz": 10,
            "epoch_seconds": 3,
            "order_column": "epoch_index",
            "normalize_each_epoch": True,
            "label_order": ["W", "N1", "N2", "N3"],
        },
        "model": {"type": "cnn", "embedding_dim": 32, "dropout": 0.1},
        "ssl": {"enabled": False},
        "train": {
            "subject_fraction": 1.0,
            "val_subject_fraction": 0.5,
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "early_stopping_patience": 2,
            "class_weight": "balanced",
        },
        "augmentations": {"enabled": False},
        "device": {"preferred": "cpu"},
        "output": {"root": str(tmp_path / "exp"), "save_models": True, "save_fold_models": True, "save_final_model": True},
    }
    out = tmp_path / "exp" / "deep_cnn_smoke"
    run_cv(cfg, tiny_deep_df, out, signal_loader=_signal_loader)
    assert (out / "metrics_per_fold.csv").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "models" / "cnn_final.pt").is_file()


def test_run_cv_deep_conformer_ssl_smoke(tiny_deep_df: pd.DataFrame, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "deep_ssl_smoke",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_id",
        "recording_column": "recording_id",
        "target_column": "sleep_stage",
        "task": "multiclass",
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "dataset": {
            "raw_root": str(tmp_path),
            "sequence_length": 3,
            "sample_hz": 10,
            "epoch_seconds": 3,
            "order_column": "epoch_index",
            "normalize_each_epoch": True,
            "label_order": ["W", "N1", "N2", "N3"],
        },
        "model": {
            "type": "conformer",
            "embedding_dim": 32,
            "conformer_blocks": 1,
            "attention_heads": 4,
            "ffn_dim": 64,
            "conv_kernel_size": 7,
            "dropout": 0.1,
        },
        "ssl": {"enabled": True, "epochs": 1, "projection_dim": 16, "temperature": 0.1, "lr": 1e-3},
        "train": {
            "subject_fraction": 1.0,
            "val_subject_fraction": 0.5,
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "early_stopping_patience": 2,
            "class_weight": "balanced",
        },
        "augmentations": {
            "enabled": True,
            "gaussian_noise_std": 0.01,
            "amplitude_scale_min": 0.95,
            "amplitude_scale_max": 1.05,
            "time_mask_fraction": 0.1,
            "time_mask_count": 1,
            "frequency_dropout_fraction": 0.05,
            "frequency_dropout_prob": 0.5,
        },
        "device": {"preferred": "cpu"},
        "output": {"root": str(tmp_path / "exp"), "save_models": True, "save_fold_models": True, "save_final_model": True},
    }
    out = tmp_path / "exp" / "deep_ssl_smoke"
    run_cv(cfg, tiny_deep_df, out, signal_loader=_signal_loader)
    metrics = pd.read_csv(out / "metrics_per_fold.csv")
    assert "ssl_pretrain_seconds" in metrics.columns
    assert (out / "models" / "conformer_ssl_final.pt").is_file()


def test_run_cv_deep_epoch_store_smoke(tiny_deep_df: pd.DataFrame, tmp_path: Path) -> None:
    meta = tiny_deep_df.copy()
    meta["dataset_id"] = "sleep_edf_expanded"
    manifest = materialize_epoch_store(
        meta,
        store_root=tmp_path / "store",
        manifest_path=tmp_path / "manifest.parquet",
        raw_root=tmp_path,
        dataset_cfg={"sample_hz": 10, "epoch_seconds": 3, "signal_channel": "EEG Fpz-Cz"},
        signal_loader=_signal_loader,
    )
    cfg = {
        "experiment_name": "deep_epoch_store_smoke",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_id",
        "recording_column": "recording_id",
        "target_column": "sleep_stage",
        "task": "multiclass",
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "dataset": {
            "input_mode": "epoch_store",
            "raw_root": str(tmp_path),
            "epoch_store_root": str(tmp_path / "store"),
            "epoch_store_manifest": str(tmp_path / "manifest.parquet"),
            "sequence_length": 3,
            "sample_hz": 10,
            "epoch_seconds": 3,
            "order_column": "epoch_index",
            "normalize_each_epoch": True,
            "label_order": ["W", "N1", "N2", "N3"],
        },
        "model": {"type": "cnn", "embedding_dim": 32, "dropout": 0.1},
        "ssl": {"enabled": False},
        "train": {
            "batching_strategy": "recording_blocked",
            "subject_fraction": 1.0,
            "val_subject_fraction": 0.5,
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "mixed_precision": False,
            "early_stopping_patience": 2,
            "class_weight": "balanced",
            "log_every_batches": 1,
        },
        "augmentations": {"enabled": False},
        "device": {"preferred": "cpu"},
        "output": {"root": str(tmp_path / "exp"), "save_models": True, "save_fold_models": True, "save_final_model": True},
    }
    out = tmp_path / "exp" / "deep_epoch_store_smoke"
    run_cv(cfg, manifest, out, signal_loader=_failing_loader)
    assert (out / "metrics_per_fold.csv").is_file()
    assert (out / "models" / "cnn_final.pt").is_file()
