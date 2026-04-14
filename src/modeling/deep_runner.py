"""Deep Phase E runner: waveform sleep staging with CNN/Conformer and optional SSL."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.utils.class_weight import compute_class_weight

from modeling.artifacts import save_confusion_matrix_figure, save_predictions_dataframe, write_model_registry
from modeling.batching import RecordingBatchSampler
from modeling.cv_split import SubjectFoldConfig, subject_wise_fold_indices
from modeling.deep_data import DEFAULT_STAGE_ORDER, WaveformSequenceDataset, build_sequence_index, prepare_sequence_metadata
from modeling.deep_models import (
    build_ssl_model,
    build_supervised_model,
    count_trainable_parameters,
    load_pretrained_encoder_weights,
)
from modeling.epoch_store import EPOCH_STORE_REQUIRED_COLUMNS, normalize_input_mode, read_epoch_store_manifest
from modeling.metrics import fold_metrics_summary, multiclass_sleep_metrics
from modeling.path_utils import resolve_path_any
from modeling.subject_id import ensure_subject_unit_column
from modeling.train_runner import load_config, read_table_file, resolve_csv_path

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None or F is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("Deep runner requires `torch`. Install it in the project .venv before running.")


def _make_grad_scaler(*, enabled: bool) -> Any:
    _require_torch()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(*, enabled: bool):
    _require_torch()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


def select_device(cfg: Mapping[str, Any]) -> "torch.device":
    _require_torch()
    preferred = str((cfg.get("device", {}) or {}).get("preferred", "cuda")).lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _output_settings(cfg: Mapping[str, Any]) -> Tuple[bool, bool, bool]:
    out = cfg.get("output", {}) or {}
    return bool(out.get("save_models", True)), bool(out.get("save_fold_models", True)), bool(
        out.get("save_final_model", True)
    )


def _save_torch_bundle(path: Path, bundle: Mapping[str, Any]) -> Path:
    _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(bundle), path)
    return path


def _checkpoint_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024)) if path.is_file() else 0.0


def _registry_row(
    *,
    out_dir: Path,
    artifact_path: Path,
    experiment_name: str,
    dataset_origin: str,
    algorithm: str,
    artifact_type: str,
    training_mode: str,
    class_labels: Sequence[str],
    fold: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "dataset_origin": dataset_origin,
        "algorithm": algorithm,
        "artifact_type": artifact_type,
        "training_mode": training_mode,
        "fold": fold,
        "path": str(artifact_path.relative_to(out_dir)),
        "class_labels": list(class_labels),
    }


def _build_class_labels(df: pd.DataFrame, target_col: str, label_order: Sequence[str]) -> List[str]:
    present = {str(x) for x in df[target_col].astype(str).unique()}
    ordered = [str(x) for x in label_order if str(x) in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _split_train_val_subjects(
    df: pd.DataFrame,
    *,
    subject_col: str,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = df[subject_col].astype(str).drop_duplicates().tolist()
    if len(subjects) <= 1 or val_fraction <= 0.0:
        return df.copy(), df.iloc[0:0].copy()
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    n_val = min(n_val, len(subjects) - 1)
    rng = np.random.default_rng(seed)
    val_subjects = set(rng.choice(np.asarray(subjects), size=n_val, replace=False).tolist())
    train_df = df[~df[subject_col].astype(str).isin(val_subjects)].copy()
    val_df = df[df[subject_col].astype(str).isin(val_subjects)].copy()
    if train_df.empty or val_df.empty:
        return df.copy(), df.iloc[0:0].copy()
    return train_df, val_df


def _loader_settings(cfg: Mapping[str, Any], *, shuffle: bool, input_mode: str) -> Dict[str, Any]:
    _require_torch()
    train_cfg = cfg.get("train", {}) or {}
    default_workers = 4 if input_mode == "epoch_store" else 0
    num_workers = int(train_cfg.get("num_workers", default_workers))
    settings = {
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "drop_last": False,
    }
    if num_workers > 0:
        settings["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        settings["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return settings


def _batching_strategy(cfg: Mapping[str, Any], *, input_mode: str) -> str:
    train_cfg = cfg.get("train", {}) or {}
    default = "recording_blocked" if input_mode == "epoch_store" else "random"
    strategy = str(train_cfg.get("batching_strategy", default)).strip().lower()
    if strategy not in {"random", "recording_blocked"}:
        raise ValueError("train.batching_strategy must be 'random' or 'recording_blocked'.")
    return strategy


def _make_loader(
    dataset: Any,
    cfg: Mapping[str, Any],
    *,
    shuffle: bool,
    seed: int,
) -> "DataLoader":
    _require_torch()
    input_mode = normalize_input_mode((cfg.get("dataset", {}) or {}).get("input_mode", "raw"))
    settings = _loader_settings(cfg, shuffle=shuffle, input_mode=input_mode)
    strategy = _batching_strategy(cfg, input_mode=input_mode)
    if shuffle and strategy == "recording_blocked" and getattr(dataset, "sample_recording_ids", None):
        batch_size = int(settings.pop("batch_size"))
        drop_last = bool(settings.pop("drop_last", False))
        settings.pop("shuffle", None)
        settings["batch_sampler"] = RecordingBatchSampler(
            list(getattr(dataset, "sample_recording_ids")),
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=True,
            seed=seed,
        )
    return DataLoader(dataset, **settings)


def _class_weights_tensor(labels: Sequence[int], *, num_classes: int, mode: str, device: "torch.device") -> Optional["torch.Tensor"]:
    _require_torch()
    if str(mode).lower() != "balanced":
        return None
    y = np.asarray(labels, dtype=int)
    classes = np.arange(num_classes, dtype=int)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _sequence_mask(x: "torch.Tensor", max_fraction: float, count: int) -> "torch.Tensor":
    if max_fraction <= 0 or count <= 0:
        return x
    out = x.clone()
    t_len = out.size(-1)
    mask_len = max(1, int(round(t_len * max_fraction)))
    for b in range(out.size(0)):
        for _ in range(count):
            start = random.randint(0, max(0, t_len - mask_len))
            out[b, :, start : start + mask_len] = 0.0
    return out


def _frequency_dropout(x: "torch.Tensor", fraction: float, prob: float) -> "torch.Tensor":
    if fraction <= 0 or prob <= 0 or random.random() > prob:
        return x
    out = x.clone()
    fft = torch.fft.rfft(out, dim=-1)
    width = max(1, int(round(fft.size(-1) * fraction)))
    start = random.randint(0, max(0, fft.size(-1) - width))
    fft[..., start : start + width] = 0
    return torch.fft.irfft(fft, n=out.size(-1), dim=-1)


def apply_sequence_augmentations(x: "torch.Tensor", aug_cfg: Mapping[str, Any]) -> "torch.Tensor":
    _require_torch()
    if not bool(aug_cfg.get("enabled", False)):
        return x
    out = x
    noise_std = float(aug_cfg.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        out = out + (torch.randn_like(out) * noise_std)
    scale_min = float(aug_cfg.get("amplitude_scale_min", 1.0))
    scale_max = float(aug_cfg.get("amplitude_scale_max", 1.0))
    if abs(scale_min - 1.0) > 1e-6 or abs(scale_max - 1.0) > 1e-6:
        scales = torch.empty((out.size(0), 1, 1), device=out.device).uniform_(scale_min, scale_max)
        out = out * scales
    out = _sequence_mask(
        out,
        max_fraction=float(aug_cfg.get("time_mask_fraction", 0.0)),
        count=int(aug_cfg.get("time_mask_count", 0)),
    )
    return _frequency_dropout(
        out,
        fraction=float(aug_cfg.get("frequency_dropout_fraction", 0.0)),
        prob=float(aug_cfg.get("frequency_dropout_prob", 0.0)),
    )


def nt_xent_loss(z1: "torch.Tensor", z2: "torch.Tensor", temperature: float) -> "torch.Tensor":
    _require_torch()
    if z1.size(0) < 2:
        return torch.zeros((), device=z1.device, dtype=z1.dtype)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
    logits = torch.matmul(z, z.T) / temperature
    mask = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(mask, float("-inf"))
    batch = z1.size(0)
    positives = torch.cat([torch.diag(logits, batch), torch.diag(logits, -batch)], dim=0)
    denom = torch.logsumexp(logits, dim=1)
    return -(positives - denom).mean()


def _predict_loader(
    model: "nn.Module",
    loader: "DataLoader",
    *,
    device: "torch.device",
    class_labels: Sequence[str],
) -> Dict[str, Any]:
    _require_torch()
    model.eval()
    y_true_idx: List[int] = []
    y_pred_idx: List[int] = []
    y_score: List[float] = []
    subject_ids: List[str] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(prob, dim=-1)
            y_true_idx.extend(batch["y"].cpu().numpy().astype(int).tolist())
            y_pred_idx.extend(pred.cpu().numpy().astype(int).tolist())
            y_score.extend(torch.max(prob, dim=-1).values.cpu().numpy().astype(float).tolist())
            subject_ids.extend([str(x) for x in batch["subject_id"]])
    y_true = [class_labels[int(i)] for i in y_true_idx]
    y_pred = [class_labels[int(i)] for i in y_pred_idx]
    return {
        "y_true_idx": np.asarray(y_true_idx, dtype=int),
        "y_pred_idx": np.asarray(y_pred_idx, dtype=int),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "subject_ids": subject_ids,
    }


def _validation_metrics(
    model: "nn.Module",
    loader: "DataLoader",
    *,
    device: "torch.device",
    class_labels: Sequence[str],
) -> Dict[str, float]:
    pred = _predict_loader(model, loader, device=device, class_labels=class_labels)
    metrics = multiclass_sleep_metrics(pred["y_true"], pred["y_pred"], labels=class_labels)
    out = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
    }
    for label, score in metrics["per_class_f1"].items():
        out[f"per_class_f1_{str(label).lower()}"] = float(score)
    return out


def _fit_supervised_model(
    *,
    cfg: Mapping[str, Any],
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: Optional["DataLoader"],
    class_labels: Sequence[str],
    device: "torch.device",
) -> Dict[str, Any]:
    _require_torch()
    train_cfg = cfg.get("train", {}) or {}
    aug_cfg = cfg.get("augmentations", {}) or {}
    model.to(device)
    class_weights = _class_weights_tensor(
        list(getattr(train_loader.dataset, "sequence_label_indices", [])),
        num_classes=len(class_labels),
        mode=str(train_cfg.get("class_weight", "balanced")),
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(enabled=amp_enabled)
    max_epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    log_every_batches = int(train_cfg.get("log_every_batches", 100))
    best_score = float("-inf")
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = -1
    bad_epochs = 0
    epoch_seconds: List[float] = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[train] start epochs={max_epochs} batches_per_epoch={len(train_loader)} "
        f"val={'yes' if val_loader is not None and len(val_loader.dataset) > 0 else 'no'} device={device}",
        flush=True,
    )

    for epoch in range(max_epochs):
        model.train()
        epoch_start = time.perf_counter()
        running_loss = 0.0
        seen_batches = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            x = apply_sequence_augmentations(x, aug_cfg)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(enabled=amp_enabled):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            seen_batches += 1
            if log_every_batches > 0 and (
                batch_idx == 1 or batch_idx % log_every_batches == 0 or batch_idx == len(train_loader)
            ):
                elapsed = time.perf_counter() - epoch_start
                mean_loss = running_loss / max(seen_batches, 1)
                batches_per_sec = batch_idx / max(elapsed, 1e-6)
                samples_per_sec = int((batch_idx * int(x.size(0))) / max(elapsed, 1e-6))
                extra = f" first_batch_sec={elapsed:.1f}" if batch_idx == 1 else ""
                print(
                    f"[train] epoch={epoch + 1}/{max_epochs} "
                    f"batch={batch_idx}/{len(train_loader)} "
                    f"loss_running={mean_loss:.4f} "
                    f"elapsed_sec={elapsed:.1f} "
                    f"batches_per_sec={batches_per_sec:.2f} "
                    f"samples_per_sec={samples_per_sec}{extra}",
                    flush=True,
                )
        epoch_seconds.append(time.perf_counter() - epoch_start)
        avg_train_loss = running_loss / max(seen_batches, 1)

        if val_loader is not None and len(val_loader.dataset) > 0:
            val_metrics = _validation_metrics(model, val_loader, device=device, class_labels=class_labels)
            score = float(val_metrics["macro_f1"])
            scheduler.step(score)
            print(
                f"[train] epoch={epoch + 1}/{max_epochs} loss={avg_train_loss:.4f} "
                f"val_macro_f1={score:.4f} seconds={epoch_seconds[-1]:.1f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        else:
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(
                f"[train] epoch={epoch + 1}/{max_epochs} loss={avg_train_loss:.4f} "
                f"seconds={epoch_seconds[-1]:.1f}",
                flush=True,
            )

    train_seconds = time.perf_counter() - start
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = max(0, len(epoch_seconds) - 1)
    model.load_state_dict(best_state, strict=True)
    max_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "model": model,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "avg_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
        "max_vram_mb": max_vram_mb,
    }


def _fit_ssl_pretraining(
    *,
    cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    train_loader: "DataLoader",
    device: "torch.device",
) -> Dict[str, Any]:
    _require_torch()
    ssl_cfg = cfg.get("ssl", {}) or {}
    aug_cfg = cfg.get("augmentations", {}) or {}
    ssl_model = build_ssl_model(dict(model_cfg), dict(ssl_cfg))
    ssl_model.to(device)
    optimizer = torch.optim.AdamW(
        ssl_model.parameters(),
        lr=float(ssl_cfg.get("lr", 1e-4)),
        weight_decay=float(ssl_cfg.get("weight_decay", 1e-2)),
    )
    amp_enabled = bool(cfg.get("train", {}).get("mixed_precision", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(enabled=amp_enabled)
    max_epochs = int(ssl_cfg.get("epochs", 10))
    best_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    start = time.perf_counter()
    for _epoch in range(max_epochs):
        ssl_model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            v1 = apply_sequence_augmentations(x, aug_cfg)
            v2 = apply_sequence_augmentations(x, aug_cfg)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(enabled=amp_enabled):
                z1 = ssl_model(v1)
                z2 = ssl_model(v2)
                loss = nt_xent_loss(z1, z2, temperature=float(ssl_cfg.get("temperature", 0.1)))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses)) if losses else float("inf")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                "epoch_encoder_state_dict": {
                    k: v.detach().cpu() for k, v in ssl_model.epoch_encoder.state_dict().items()
                },
                "temporal_encoder_state_dict": {
                    k: v.detach().cpu() for k, v in ssl_model.temporal_encoder.state_dict().items()
                },
            }
    return {
        "checkpoint": best_state or {},
        "pretrain_seconds": time.perf_counter() - start,
        "best_ssl_loss": best_loss,
    }


def _bundle_for_checkpoint(
    *,
    model: "nn.Module",
    cfg: Mapping[str, Any],
    class_labels: Sequence[str],
    model_name: str,
    artifact_kind: str,
    training_mode: str,
    fold: Optional[int],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    bundle = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": dict(cfg.get("model", {}) or {}),
        "dataset_config": dict(cfg.get("dataset", {}) or {}),
        "train_config": dict(cfg.get("train", {}) or {}),
        "ssl_config": dict(cfg.get("ssl", {}) or {}),
        "class_labels": list(class_labels),
        "target_column": str(cfg["target_column"]),
        "subject_column": str(cfg["subject_column"]),
        "recording_column": str(cfg.get("recording_column", "recording_id")),
        "random_seed": int(cfg.get("random_seed", 42)),
        "model_name": model_name,
        "artifact_kind": artifact_kind,
        "training_mode": training_mode,
        "fold": fold,
    }
    if extra:
        bundle.update(dict(extra))
    return bundle


def _metrics_row(
    *,
    model_name: str,
    fold: Any,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    class_labels: Sequence[str],
    n_parameters: int,
    train_seconds: float,
    avg_epoch_seconds: float,
    max_vram_mb: float,
    checkpoint_size_mb: float,
    best_epoch: int,
    ssl_seconds: float = 0.0,
) -> Dict[str, Any]:
    metrics = multiclass_sleep_metrics(y_true, y_pred, labels=class_labels)
    row: Dict[str, Any] = {
        "model": model_name,
        "fold": fold,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "n_parameters": int(n_parameters),
        "train_seconds": float(train_seconds),
        "avg_epoch_seconds": float(avg_epoch_seconds),
        "max_vram_mb": float(max_vram_mb),
        "checkpoint_size_mb": float(checkpoint_size_mb),
        "best_epoch": int(best_epoch),
        "ssl_pretrain_seconds": float(ssl_seconds),
    }
    for label, score in metrics["per_class_f1"].items():
        row[f"per_class_f1_{str(label).lower()}"] = float(score)
    return row


def _write_summary(out_dir: Path, metrics_rows: Sequence[Mapping[str, Any]], model_names: Sequence[str]) -> None:
    metric_keys = sorted(
        {
            str(key)
            for row in metrics_rows
            for key, value in row.items()
            if key not in {"model", "fold"} and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    summary: Dict[str, Any] = {}
    for model_name in model_names:
        rows = [r for r in metrics_rows if str(r.get("model")) == model_name and str(r.get("fold")) != "final"]
        if rows:
            summary[model_name] = fold_metrics_summary(rows, metric_keys)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_per_fold.csv", index=False)


def _make_datasets_and_loaders(
    *,
    cfg: Mapping[str, Any],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    class_labels: Sequence[str],
    raw_root: Path,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]],
) -> Dict[str, Any]:
    _require_torch()
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    subject_col = str(cfg["subject_column"])
    recording_col = str(cfg.get("recording_column", "recording_id"))
    order_col = str(dataset_cfg.get("order_column", "epoch_index"))
    target_col = str(cfg["target_column"])
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}

    def build(df_part: pd.DataFrame) -> WaveformSequenceDataset:
        seq_index = build_sequence_index(
            df_part,
            recording_col=recording_col,
            order_col=order_col,
            sequence_length=int(dataset_cfg.get("sequence_length", 9)),
        )
        return WaveformSequenceDataset(
            df_part,
            sequence_indices=seq_index,
            target_col=target_col,
            subject_col=subject_col,
            recording_col=recording_col,
            raw_root=raw_root,
            dataset_cfg=dataset_cfg,
            label_to_index=label_to_index,
            signal_loader=signal_loader,
        )

    ds_train = build(df_train)
    ds_val = build(df_val) if not df_val.empty else None
    ds_test = build(df_test) if df_test is not None else None
    return {
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "loaders": {
            "train": _make_loader(ds_train, cfg, shuffle=True, seed=int(cfg.get("random_seed", 42))),
            "val": _make_loader(ds_val, cfg, shuffle=False, seed=int(cfg.get("random_seed", 42))) if ds_val is not None else None,
            "test": _make_loader(ds_test, cfg, shuffle=False, seed=int(cfg.get("random_seed", 42))) if ds_test is not None else None,
        },
    }


def run_cv(
    cfg: Mapping[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    *,
    train_csv_path: Optional[Path] = None,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> None:
    _require_torch()
    if str(cfg.get("task", "multiclass")) != "multiclass":
        raise ValueError("Deep runner currently supports only task=multiclass for sleep staging.")
    seed = int(cfg.get("random_seed", 42))
    set_random_seed(seed)
    subject_col = str(cfg["subject_column"])
    target_col = str(cfg["target_column"])
    recording_col = str(cfg.get("recording_column", "recording_id"))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    ssl_cfg = dict(cfg.get("ssl", {}) or {})
    input_mode = normalize_input_mode(dataset_cfg.get("input_mode", "raw"))
    model_name = f"{str(model_cfg.get('type', 'conformer')).lower()}{'_ssl' if bool(ssl_cfg.get('enabled', False)) else ''}"
    raw_root = Path(str(dataset_cfg.get("raw_root", ".")))

    df = ensure_subject_unit_column(df, overwrite=False)
    df = prepare_sequence_metadata(
        df,
        target_col=target_col,
        subject_col=subject_col,
        recording_col=recording_col,
        order_col=str(dataset_cfg.get("order_column", "epoch_index")),
        label_subset=cfg.get("label_subset"),
        subject_fraction=float((cfg.get("train", {}) or {}).get("subject_fraction", 1.0)),
        random_seed=seed,
    )
    class_labels = _build_class_labels(df, target_col, dataset_cfg.get("label_order", DEFAULT_STAGE_ORDER))
    cv_cfg = cfg.get("cv", {}) or {}
    fold_conf = SubjectFoldConfig(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        random_state=seed,
        stratify=bool(cv_cfg.get("stratify", True)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
    )
    if df[subject_col].astype(str).nunique() < fold_conf.n_splits:
        raise ValueError(
            f"Deep subject-wise CV requires at least {fold_conf.n_splits} subjects in {subject_col!r}."
        )

    metrics_rows: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []
    save_models, save_fold_models, save_final_model = _output_settings(cfg)
    device = select_device(cfg)
    models_dir = out_dir / "models"
    training_mode = "ssl_tuned" if bool(ssl_cfg.get("enabled", False)) else "supervised"
    print(
        f"[run_cv] rows={len(df)} subjects={df[subject_col].astype(str).nunique()} "
        f"classes={class_labels} device={device} model={model_name} "
        f"input_mode={input_mode} batching={_batching_strategy(cfg, input_mode=input_mode)} "
        f"recordings={df[recording_col].astype(str).nunique()}",
        flush=True,
    )

    for fold_id, (train_idx, test_idx) in enumerate(
        subject_wise_fold_indices(df, subject_col=subject_col, y=df[target_col].astype(str).values, config=fold_conf)
    ):
        train_outer = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, val_df = _split_train_val_subjects(
            train_outer,
            subject_col=subject_col,
            val_fraction=float((cfg.get("train", {}) or {}).get("val_subject_fraction", 0.15)),
            seed=seed + fold_id,
        )
        print(
            f"[run_cv] fold={fold_id + 1}/{fold_conf.n_splits} train_rows={len(train_df)} "
            f"val_rows={len(val_df)} test_rows={len(test_df)}",
            flush=True,
        )
        data = _make_datasets_and_loaders(
            cfg=cfg,
            df_train=train_df,
            df_val=val_df,
            df_test=test_df,
            class_labels=class_labels,
            raw_root=raw_root,
            signal_loader=signal_loader,
        )
        ssl_info = {"checkpoint": {}, "pretrain_seconds": 0.0, "best_ssl_loss": None}
        if bool(ssl_cfg.get("enabled", False)):
            if str(model_cfg.get("type", "conformer")).lower() != "conformer":
                raise ValueError("SSL is only supported with model.type=conformer in this v1 runner.")
            ssl_info = _fit_ssl_pretraining(
                cfg=cfg,
                model_cfg={**model_cfg, "sequence_length": int(dataset_cfg.get("sequence_length", 9))},
                train_loader=data["loaders"]["train"],
                device=device,
            )
        model = build_supervised_model(
            {**model_cfg, "sequence_length": int(dataset_cfg.get("sequence_length", 9))},
            num_classes=len(class_labels),
        )
        if ssl_info["checkpoint"]:
            load_pretrained_encoder_weights(model, ssl_info["checkpoint"])
        fitted = _fit_supervised_model(
            cfg=cfg,
            model=model,
            train_loader=data["loaders"]["train"],
            val_loader=data["loaders"]["val"],
            class_labels=class_labels,
            device=device,
        )
        pred = _predict_loader(fitted["model"], data["loaders"]["test"], device=device, class_labels=class_labels)
        save_predictions_dataframe(
            out_dir / "predictions" / f"{model_name}_fold{fold_id}.csv",
            y_true=pred["y_true"],
            y_pred=pred["y_pred"],
            y_score=pred["y_score"],
            subject_id=pred["subject_ids"],
            fold_id=fold_id,
            extra_columns={"model": [model_name] * len(pred["y_true"])},
        )
        save_confusion_matrix_figure(
            pred["y_true"],
            pred["y_pred"],
            out_dir / "figures" / f"cm_{model_name}_fold{fold_id}.png",
            labels=class_labels,
            title=f"{model_name} fold {fold_id}",
        )
        checkpoint_size_mb = 0.0
        if save_models and save_fold_models:
            fold_path = _save_torch_bundle(
                models_dir / "folds" / f"{model_name}_fold{fold_id}.pt",
                _bundle_for_checkpoint(
                    model=fitted["model"],
                    cfg=cfg,
                    class_labels=class_labels,
                    model_name=model_name,
                    artifact_kind="fold",
                    training_mode=training_mode,
                    fold=fold_id,
                    extra=ssl_info["checkpoint"],
                ),
            )
            checkpoint_size_mb = _checkpoint_size_mb(fold_path)
            registry_rows.append(
                _registry_row(
                    out_dir=out_dir,
                    artifact_path=fold_path,
                    experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                    dataset_origin=str(train_csv_path or cfg.get("train_csv", "")),
                    algorithm=model_name,
                    artifact_type="fold",
                    training_mode=training_mode,
                    class_labels=class_labels,
                    fold=fold_id,
                )
            )
        metrics_rows.append(
            _metrics_row(
                model_name=model_name,
                fold=fold_id,
                y_true=pred["y_true"],
                y_pred=pred["y_pred"],
                class_labels=class_labels,
                n_parameters=count_trainable_parameters(fitted["model"]),
                train_seconds=float(fitted["train_seconds"]),
                avg_epoch_seconds=float(fitted["avg_epoch_seconds"]),
                max_vram_mb=float(fitted["max_vram_mb"]),
                checkpoint_size_mb=checkpoint_size_mb,
                best_epoch=int(fitted["best_epoch"]),
                ssl_seconds=float(ssl_info["pretrain_seconds"]),
            )
        )

    if save_models and save_final_model:
        train_df, val_df = _split_train_val_subjects(
            df,
            subject_col=subject_col,
            val_fraction=float((cfg.get("train", {}) or {}).get("val_subject_fraction", 0.15)),
            seed=seed + 999,
        )
        data = _make_datasets_and_loaders(
            cfg=cfg,
            df_train=train_df,
            df_val=val_df,
            df_test=None,
            class_labels=class_labels,
            raw_root=raw_root,
            signal_loader=signal_loader,
        )
        ssl_info = {"checkpoint": {}, "pretrain_seconds": 0.0, "best_ssl_loss": None}
        if bool(ssl_cfg.get("enabled", False)):
            ssl_info = _fit_ssl_pretraining(
                cfg=cfg,
                model_cfg={**model_cfg, "sequence_length": int(dataset_cfg.get("sequence_length", 9))},
                train_loader=data["loaders"]["train"],
                device=device,
            )
        final_model = build_supervised_model(
            {**model_cfg, "sequence_length": int(dataset_cfg.get("sequence_length", 9))},
            num_classes=len(class_labels),
        )
        if ssl_info["checkpoint"]:
            load_pretrained_encoder_weights(final_model, ssl_info["checkpoint"])
        fitted = _fit_supervised_model(
            cfg=cfg,
            model=final_model,
            train_loader=data["loaders"]["train"],
            val_loader=data["loaders"]["val"],
            class_labels=class_labels,
            device=device,
        )
        final_path = _save_torch_bundle(
            models_dir / f"{model_name}_final.pt",
            _bundle_for_checkpoint(
                model=fitted["model"],
                cfg=cfg,
                class_labels=class_labels,
                model_name=model_name,
                artifact_kind="final",
                training_mode=training_mode,
                fold=None,
                extra=ssl_info["checkpoint"],
            ),
        )
        registry_rows.append(
            _registry_row(
                out_dir=out_dir,
                artifact_path=final_path,
                experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                dataset_origin=str(train_csv_path or cfg.get("train_csv", "")),
                algorithm=model_name,
                artifact_type="final",
                training_mode=training_mode,
                class_labels=class_labels,
            )
        )

    _write_summary(out_dir, metrics_rows, [model_name])
    if save_models and registry_rows:
        write_model_registry(models_dir / "model_registry.json", registry_rows)
    print(f"Wrote deep metrics_per_fold.csv and summary.json under {out_dir}")


def run_cross_dataset(
    cfg: Mapping[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    train_csv_path: Optional[Path] = None,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> None:
    _require_torch()
    seed = int(cfg.get("random_seed", 42))
    set_random_seed(seed)
    subject_col = str(cfg["subject_column"])
    target_col = str(cfg["target_column"])
    recording_col = str(cfg.get("recording_column", "recording_id"))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    model_cfg = dict(cfg.get("model", {}) or {})
    ssl_cfg = dict(cfg.get("ssl", {}) or {})
    input_mode = normalize_input_mode(dataset_cfg.get("input_mode", "raw"))
    model_name = f"{str(model_cfg.get('type', 'conformer')).lower()}{'_ssl' if bool(ssl_cfg.get('enabled', False)) else ''}"
    raw_root = Path(str(dataset_cfg.get("raw_root", ".")))
    eval_raw_root = Path(str(dataset_cfg.get("eval_raw_root", raw_root)))
    training_mode = "ssl_tuned" if bool(ssl_cfg.get("enabled", False)) else "supervised"

    train_df = prepare_sequence_metadata(
        ensure_subject_unit_column(train_df, overwrite=False),
        target_col=target_col,
        subject_col=subject_col,
        recording_col=recording_col,
        order_col=str(dataset_cfg.get("order_column", "epoch_index")),
        label_subset=cfg.get("label_subset"),
        random_seed=seed,
    )
    eval_df = prepare_sequence_metadata(
        ensure_subject_unit_column(eval_df, overwrite=False),
        target_col=target_col,
        subject_col=subject_col,
        recording_col=recording_col,
        order_col=str(dataset_cfg.get("order_column", "epoch_index")),
        label_subset=cfg.get("label_subset"),
        random_seed=seed,
    )
    class_labels = _build_class_labels(train_df, target_col, dataset_cfg.get("label_order", DEFAULT_STAGE_ORDER))
    if not set(eval_df[target_col].astype(str).unique()).issubset(set(class_labels)):
        raise ValueError("Cross-dataset eval contains labels absent from train. Use label_subset.")
    train_fit, val_df = _split_train_val_subjects(
        train_df,
        subject_col=subject_col,
        val_fraction=float((cfg.get("train", {}) or {}).get("val_subject_fraction", 0.15)),
        seed=seed,
    )
    train_data = _make_datasets_and_loaders(
        cfg=cfg,
        df_train=train_fit,
        df_val=val_df,
        df_test=None,
        class_labels=class_labels,
        raw_root=raw_root,
        signal_loader=signal_loader,
    )
    eval_data = _make_datasets_and_loaders(
        cfg=cfg,
        df_train=eval_df,
        df_val=eval_df.iloc[0:0].copy(),
        df_test=eval_df,
        class_labels=class_labels,
        raw_root=eval_raw_root,
        signal_loader=signal_loader,
    )
    device = select_device(cfg)
    print(
        f"[run_cross] train_rows={len(train_df)} eval_rows={len(eval_df)} "
        f"classes={class_labels} device={device} model={model_name} "
        f"input_mode={input_mode} batching={_batching_strategy(cfg, input_mode=input_mode)}",
        flush=True,
    )
    ssl_info = {"checkpoint": {}, "pretrain_seconds": 0.0, "best_ssl_loss": None}
    if bool(ssl_cfg.get("enabled", False)):
        ssl_info = _fit_ssl_pretraining(
            cfg=cfg,
            model_cfg={**model_cfg, "sequence_length": int(dataset_cfg.get("sequence_length", 9))},
            train_loader=train_data["loaders"]["train"],
            device=device,
        )
    model = build_supervised_model(
        {**model_cfg, "sequence_length": int(dataset_cfg.get("sequence_length", 9))},
        num_classes=len(class_labels),
    )
    if ssl_info["checkpoint"]:
        load_pretrained_encoder_weights(model, ssl_info["checkpoint"])
    fitted = _fit_supervised_model(
        cfg=cfg,
        model=model,
        train_loader=train_data["loaders"]["train"],
        val_loader=train_data["loaders"]["val"],
        class_labels=class_labels,
        device=device,
    )
    pred = _predict_loader(fitted["model"], eval_data["loaders"]["test"], device=device, class_labels=class_labels)
    save_predictions_dataframe(
        out_dir / "predictions" / f"{model_name}_cross_eval.csv",
        y_true=pred["y_true"],
        y_pred=pred["y_pred"],
        y_score=pred["y_score"],
        subject_id=pred["subject_ids"],
        fold_id=-1,
        extra_columns={"model": [model_name] * len(pred["y_true"])},
    )
    save_confusion_matrix_figure(
        pred["y_true"],
        pred["y_pred"],
        out_dir / "figures" / f"cm_{model_name}_cross_eval.png",
        labels=class_labels,
        title=f"{model_name} cross-dataset",
    )
    save_models, _save_fold_models, save_final_model = _output_settings(cfg)
    checkpoint_size_mb = 0.0
    registry_rows: List[Dict[str, Any]] = []
    if save_models and save_final_model:
        final_path = _save_torch_bundle(
            out_dir / "models" / f"{model_name}_final.pt",
            _bundle_for_checkpoint(
                model=fitted["model"],
                cfg=cfg,
                class_labels=class_labels,
                model_name=model_name,
                artifact_kind="final",
                training_mode=training_mode,
                fold=None,
                extra=ssl_info["checkpoint"],
            ),
        )
        checkpoint_size_mb = _checkpoint_size_mb(final_path)
        registry_rows.append(
            _registry_row(
                out_dir=out_dir,
                artifact_path=final_path,
                experiment_name=str(cfg.get("experiment_name", out_dir.name)),
                dataset_origin=str(train_csv_path or cfg.get("train_csv", "")),
                algorithm=model_name,
                artifact_type="final",
                training_mode=training_mode,
                class_labels=class_labels,
            )
        )
        write_model_registry(out_dir / "models" / "model_registry.json", registry_rows)
    row = _metrics_row(
        model_name=model_name,
        fold="cross_eval",
        y_true=pred["y_true"],
        y_pred=pred["y_pred"],
        class_labels=class_labels,
        n_parameters=count_trainable_parameters(fitted["model"]),
        train_seconds=float(fitted["train_seconds"]),
        avg_epoch_seconds=float(fitted["avg_epoch_seconds"]),
        max_vram_mb=float(fitted["max_vram_mb"]),
        checkpoint_size_mb=checkpoint_size_mb,
        best_epoch=int(fitted["best_epoch"]),
        ssl_seconds=float(ssl_info["pretrain_seconds"]),
    )
    pd.DataFrame([row]).to_csv(out_dir / "metrics_cross_eval.csv", index=False)
    print(f"Wrote deep cross-dataset metrics under {out_dir}")


def run_experiment(
    config_path: Path,
    *,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> None:
    _require_torch()
    cfg = load_config(config_path)
    print(f"[run] config={config_path}", flush=True)
    out_root = Path((cfg.get("output", {}) or {}).get("root", "reports/experiments"))
    out_dir = out_root / str(cfg.get("experiment_name", "deep_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    input_mode = normalize_input_mode(dataset_cfg.get("input_mode", "raw"))
    if input_mode == "raw":
        dataset_cfg["raw_root"] = str(resolve_path_any(str(dataset_cfg.get("raw_root", "data/raw")), config_path, expect_dir=True))
    else:
        if not dataset_cfg.get("epoch_store_root") or not dataset_cfg.get("epoch_store_manifest"):
            raise ValueError(
                "dataset.input_mode=epoch_store requires dataset.epoch_store_root and "
                "dataset.epoch_store_manifest. Run scripts/materialize_epoch_store.py first."
            )
        dataset_cfg["epoch_store_root"] = str(
            resolve_path_any(str(dataset_cfg["epoch_store_root"]), config_path, expect_dir=True)
        )
        manifest_path = resolve_csv_path(str(dataset_cfg["epoch_store_manifest"]), config_path)
        dataset_cfg["epoch_store_manifest"] = str(manifest_path)
        if dataset_cfg.get("eval_epoch_store_manifest"):
            dataset_cfg["eval_epoch_store_manifest"] = str(
                resolve_csv_path(str(dataset_cfg["eval_epoch_store_manifest"]), config_path)
            )
    cfg = dict(cfg)
    cfg["dataset"] = dataset_cfg
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    train_path = resolve_csv_path(str(cfg["train_csv"]), config_path)
    print(f"[run] train_csv={train_path}", flush=True)
    if input_mode == "epoch_store":
        manifest_path = Path(str(dataset_cfg["epoch_store_manifest"]))
        print(f"[run] input_mode=epoch_store epoch_store_root={dataset_cfg['epoch_store_root']}", flush=True)
        print(f"[run] epoch_store_manifest={manifest_path}", flush=True)
        train_df = read_epoch_store_manifest(manifest_path)
        print(f"[run] epoch_store_shape={train_df.shape}", flush=True)
        train_origin = manifest_path
    else:
        train_df = read_table_file(train_path)
        print(f"[run] train_shape={train_df.shape}", flush=True)
        train_origin = train_path
    if bool(cfg.get("cross_dataset", False)):
        if input_mode == "epoch_store" and dataset_cfg.get("eval_epoch_store_manifest"):
            eval_path = Path(str(dataset_cfg["eval_epoch_store_manifest"]))
            print(f"[run] eval_epoch_store_manifest={eval_path}", flush=True)
            eval_df = read_epoch_store_manifest(eval_path)
            print(f"[run] eval_epoch_store_shape={eval_df.shape}", flush=True)
        else:
            eval_path = resolve_csv_path(str(cfg["eval_csv"]), config_path)
            print(f"[run] eval_csv={eval_path}", flush=True)
            eval_df = read_table_file(eval_path)
            if input_mode == "epoch_store":
                missing = [col for col in EPOCH_STORE_REQUIRED_COLUMNS if col not in eval_df.columns]
                if missing:
                    raise ValueError(
                        "Cross-dataset epoch_store mode requires dataset.eval_epoch_store_manifest "
                        f"or an eval_csv already materialized. Missing columns: {missing!r}."
                    )
                eval_df = read_epoch_store_manifest(eval_path)
            print(f"[run] eval_shape={eval_df.shape}", flush=True)
        run_cross_dataset(cfg, train_df, eval_df, out_dir, train_csv_path=train_origin, signal_loader=signal_loader)
    else:
        run_cv(cfg, train_df, out_dir, train_csv_path=train_origin, signal_loader=signal_loader)


def main(argv: Optional[Sequence[str]] = None) -> None:
    _require_torch()
    parser = argparse.ArgumentParser(description="Deep Phase E runner for waveform sleep staging.")
    parser.add_argument("--config", required=True, type=str, help="Path to deep experiment YAML.")
    args = parser.parse_args(argv)
    run_experiment(Path(args.config))


if __name__ == "__main__":
    main()
