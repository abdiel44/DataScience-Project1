"""Smoke tests for multitask apnea/staging runner and metadata helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from modeling.epoch_store import export_epoch_store_features, materialize_epoch_store
from modeling.multitask_data import MultiTaskWaveformDataset, build_sequence_index, standardize_multitask_metadata
from modeling.multitask_models import build_multitask_model
from modeling.multitask_runner import run_cv


def _signal_bank() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    t = np.linspace(0, 1, 120, endpoint=False, dtype=np.float32)
    for idx, rec in enumerate(["rec0", "rec1", "rec2", "rec3"]):
        out[rec] = np.sin(2 * np.pi * (idx + 1) * t).astype(np.float32)
    return out


def _signal_loader(row: Mapping[str, Any], _raw_root: Path, _cfg: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    return _signal_bank()[str(row["recording_id"])], 10.0


def _failing_loader(_row: Mapping[str, Any], _raw_root: Path, _cfg: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    raise AssertionError("raw waveform loader should not be called in epoch_store mode")


@pytest.fixture()
def tiny_multitask_df() -> pd.DataFrame:
    rows = []
    stage_labels = ["W", "N1", "N2", "N3"]
    for rec_idx, rec in enumerate(["rec0", "rec1", "rec2", "rec3"]):
        for epoch_idx in range(4):
            rows.append(
                {
                    "dataset_id": "mit_bih_psg" if rec_idx < 2 else "sleep_edf_expanded",
                    "subject_unit_id": f"s{rec_idx}",
                    "recording_id": rec,
                    "epoch_index": epoch_idx,
                    "epoch_start_sec": float(epoch_idx * 3),
                    "epoch_end_sec": float((epoch_idx + 1) * 3),
                    "source_file": f"{rec}.edf",
                    "eeg_channel_standardized": "EEG",
                    "apnea_binary": int((rec_idx + epoch_idx) % 2 == 0) if rec_idx < 2 else np.nan,
                    "sleep_stage": stage_labels[(rec_idx + epoch_idx) % len(stage_labels)],
                }
            )
    return pd.DataFrame(rows)


def test_standardize_multitask_metadata_adds_masks(tiny_multitask_df: pd.DataFrame) -> None:
    out = standardize_multitask_metadata(tiny_multitask_df)
    assert "label_mask_apnea" in out.columns
    assert "label_mask_stage" in out.columns
    assert int(out["label_mask_stage"].sum()) == len(out)
    assert int(out["label_mask_apnea"].sum()) > 0


def test_multitask_dataset_returns_masks(tiny_multitask_df: pd.DataFrame, tmp_path: Path) -> None:
    df = standardize_multitask_metadata(tiny_multitask_df)
    seq = build_sequence_index(df, recording_col="recording_id", order_col="epoch_index", sequence_length=3)
    ds = MultiTaskWaveformDataset(
        df,
        sequence_indices=seq,
        raw_root=tmp_path,
        dataset_cfg={"sequence_length": 3, "sample_hz": 10, "epoch_seconds": 3, "normalize_each_epoch": True},
        stage_label_to_index={"W": 0, "N1": 1, "N2": 2, "N3": 3},
        signal_loader=_signal_loader,
    )
    item = ds[0]
    assert tuple(item["x"].shape) == (3, 30)
    assert float(item["stage_mask"]) in {0.0, 1.0}
    assert float(item["apnea_mask"]) in {0.0, 1.0}


def test_multitask_dataset_epoch_store_avoids_raw_loader(tiny_multitask_df: pd.DataFrame, tmp_path: Path) -> None:
    manifest = materialize_epoch_store(
        tiny_multitask_df,
        store_root=tmp_path / "store",
        manifest_path=tmp_path / "manifest.parquet",
        raw_root=tmp_path,
        dataset_cfg={"sample_hz": 10, "epoch_seconds": 3},
        signal_loader=_signal_loader,
    )
    df = standardize_multitask_metadata(manifest)
    seq = build_sequence_index(df, recording_col="recording_id", order_col="epoch_index", sequence_length=3)
    ds = MultiTaskWaveformDataset(
        df,
        sequence_indices=seq,
        raw_root=tmp_path,
        dataset_cfg={
            "input_mode": "epoch_store",
            "epoch_store_root": str(tmp_path / "store"),
            "sequence_length": 3,
            "sample_hz": 10,
            "epoch_seconds": 3,
            "normalize_each_epoch": True,
        },
        stage_label_to_index={"W": 0, "N1": 1, "N2": 2, "N3": 3},
        signal_loader=_failing_loader,
    )
    item = ds[0]
    assert tuple(item["x"].shape) == (3, 30)


def test_build_multitask_model_forward_smoke() -> None:
    model = build_multitask_model(
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
        stage_num_classes=4,
    )
    x = torch.randn(2, 3, 30)
    out = model(x)
    assert tuple(out["apnea_logits"].shape) == (2,)
    assert tuple(out["stage_logits"].shape) == (2, 4)


def test_run_cv_multitask_smoke(tiny_multitask_df: pd.DataFrame, tmp_path: Path) -> None:
    cfg = {
        "experiment_name": "multitask_smoke",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "recording_column": "recording_id",
        "task": "multitask_apnea_stage",
        "cv": {"n_splits": 2, "stratify": True, "shuffle": True},
        "dataset": {
            "raw_root": str(tmp_path),
            "sequence_length": 3,
            "sample_hz": 10,
            "epoch_seconds": 3,
            "order_column": "epoch_index",
            "normalize_each_epoch": True,
        },
        "labels": {"stage_order": ["W", "N1", "N2", "N3"]},
        "model": {
            "type": "conformer",
            "embedding_dim": 32,
            "conformer_blocks": 1,
            "attention_heads": 4,
            "ffn_dim": 64,
            "conv_kernel_size": 7,
            "dropout": 0.1,
        },
        "transfer": {"enabled": False},
        "multitask": {"apnea_loss_weight": 1.0, "stage_loss_weight": 0.5},
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
        },
        "augmentations": {"enabled": False},
        "device": {"preferred": "cpu"},
        "output": {"root": str(tmp_path / "exp"), "save_models": True, "save_fold_models": True, "save_final_model": True},
    }
    out = tmp_path / "exp" / "multitask_smoke"
    run_cv(cfg, tiny_multitask_df, out, signal_loader=_signal_loader)
    metrics = pd.read_csv(out / "metrics_per_fold.csv")
    assert "apnea_accuracy" in metrics.columns
    assert "stage_macro_f1" in metrics.columns
    assert (out / "models" / "conformer_multitask_final.pt").is_file()


def test_run_cv_multitask_epoch_store_smoke(tiny_multitask_df: pd.DataFrame, tmp_path: Path) -> None:
    manifest = materialize_epoch_store(
        tiny_multitask_df,
        store_root=tmp_path / "store",
        manifest_path=tmp_path / "manifest.parquet",
        raw_root=tmp_path,
        dataset_cfg={"sample_hz": 10, "epoch_seconds": 3},
        signal_loader=_signal_loader,
    )
    features = export_epoch_store_features(
        manifest,
        store_root=tmp_path / "store",
        output_path=tmp_path / "features.parquet",
    )
    assert "eeg_mean" in features.columns
    cfg = {
        "experiment_name": "multitask_epoch_store_smoke",
        "random_seed": 0,
        "train_csv": "n/a",
        "cross_dataset": False,
        "subject_column": "subject_unit_id",
        "recording_column": "recording_id",
        "task": "multitask_apnea_stage",
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
        },
        "labels": {"stage_order": ["W", "N1", "N2", "N3"]},
        "model": {
            "type": "conformer",
            "embedding_dim": 32,
            "conformer_blocks": 1,
            "attention_heads": 4,
            "ffn_dim": 64,
            "conv_kernel_size": 7,
            "dropout": 0.1,
        },
        "transfer": {"enabled": False},
        "multitask": {"apnea_loss_weight": 1.0, "stage_loss_weight": 0.5},
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
            "log_every_batches": 1,
        },
        "augmentations": {"enabled": False},
        "device": {"preferred": "cpu"},
        "output": {"root": str(tmp_path / "exp"), "save_models": True, "save_fold_models": True, "save_final_model": True},
    }
    out = tmp_path / "exp" / "multitask_epoch_store_smoke"
    run_cv(cfg, manifest, out, signal_loader=_failing_loader)
    metrics = pd.read_csv(out / "metrics_per_fold.csv")
    assert "apnea_accuracy" in metrics.columns
    assert (out / "models" / "conformer_multitask_final.pt").is_file()
