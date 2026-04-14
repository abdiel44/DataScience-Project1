"""Deep multitask runner for apnea/no-apnea with optional staging transfer and evaluation."""

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

from modeling.artifacts import (
    save_confusion_matrix_figure,
    save_predictions_dataframe,
    save_roc_curve_figure,
    write_model_registry,
)
from modeling.batching import RecordingBatchSampler
from modeling.cv_split import SubjectFoldConfig, subject_wise_fold_indices
from modeling.epoch_store import EPOCH_STORE_REQUIRED_COLUMNS, normalize_input_mode, read_epoch_store_manifest
from modeling.metrics import apnea_binary_metrics, fold_metrics_summary, multiclass_sleep_metrics
from modeling.multitask_data import (
    DEFAULT_STAGE_ORDER,
    MultiTaskWaveformDataset,
    build_sequence_index,
    read_multitask_metadata,
    standardize_multitask_metadata,
)
from modeling.multitask_models import build_multitask_model, load_encoder_weights_from_checkpoint
from modeling.path_utils import resolve_path_any
from modeling.subject_id import ensure_subject_unit_column
from modeling.train_runner import load_config, resolve_csv_path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - runtime guard
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:  # pragma: no cover
        raise RuntimeError("torch is required for multitask deep runner.")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


def _loader_settings(cfg: Mapping[str, Any], *, shuffle: bool, input_mode: str) -> Dict[str, Any]:
    _require_torch()
    train_cfg = cfg.get("train", {}) or {}
    default_workers = 4 if input_mode == "epoch_store" else 0
    num_workers = int(train_cfg.get("num_workers", default_workers))
    out = {
        "batch_size": int(train_cfg.get("batch_size", 8)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "drop_last": False,
    }
    if num_workers > 0:
        out["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        out["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 2))
    return out


def _batching_strategy(cfg: Mapping[str, Any], *, input_mode: str) -> str:
    train_cfg = cfg.get("train", {}) or {}
    default = "recording_blocked" if input_mode == "epoch_store" else "random"
    strategy = str(train_cfg.get("batching_strategy", default)).strip().lower()
    if strategy not in {"random", "recording_blocked"}:
        raise ValueError("train.batching_strategy must be 'random' or 'recording_blocked'.")
    return strategy


def _make_loader(dataset: Any, cfg: Mapping[str, Any], *, shuffle: bool, seed: int) -> "DataLoader":
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


def _select_device(cfg: Mapping[str, Any]) -> "torch.device":
    _require_torch()
    preferred = str((cfg.get("device", {}) or {}).get("preferred", "cuda")).lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_stage_labels(df: pd.DataFrame, stage_order: Sequence[str]) -> List[str]:
    present = {str(x) for x in df.loc[df["label_mask_stage"] > 0, "sleep_stage"].astype(str).unique()}
    ordered = [str(x) for x in stage_order if str(x) in present]
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
    if len(subjects) <= 1 or val_fraction <= 0:
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


def _sequence_augmentations(x: "torch.Tensor", aug_cfg: Mapping[str, Any]) -> "torch.Tensor":
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
    return out


def _make_datasets_and_loaders(
    *,
    cfg: Mapping[str, Any],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    stage_labels: Sequence[str],
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> Dict[str, Any]:
    _require_torch()
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    raw_root = Path(dataset_cfg["raw_root"])
    stage_map = {str(lbl): idx for idx, lbl in enumerate(stage_labels)}
    seq_len = int(dataset_cfg.get("sequence_length", 9))
    order_col = str(dataset_cfg.get("order_column", "epoch_index"))

    def build(df_part: pd.DataFrame) -> MultiTaskWaveformDataset:
        seq_index = build_sequence_index(df_part, recording_col="recording_id", order_col=order_col, sequence_length=seq_len)
        return MultiTaskWaveformDataset(
            df_part,
            sequence_indices=seq_index,
            raw_root=raw_root,
            dataset_cfg=dataset_cfg,
            stage_label_to_index=stage_map,
            signal_loader=signal_loader,
        )

    ds_train = build(df_train)
    ds_val = build(df_val) if not df_val.empty else None
    ds_test = build(df_test) if df_test is not None and not df_test.empty else None
    return {
        "datasets": {"train": ds_train, "val": ds_val, "test": ds_test},
        "loaders": {
            "train": _make_loader(ds_train, cfg, shuffle=True, seed=int(cfg.get("random_seed", 42))),
            "val": _make_loader(ds_val, cfg, shuffle=False, seed=int(cfg.get("random_seed", 42))) if ds_val is not None else None,
            "test": _make_loader(ds_test, cfg, shuffle=False, seed=int(cfg.get("random_seed", 42))) if ds_test is not None else None,
        },
    }


def _maybe_load_transfer_checkpoint(model: "nn.Module", cfg: Mapping[str, Any], *, device: "torch.device") -> None:
    transfer_cfg = cfg.get("transfer", {}) or {}
    if not bool(transfer_cfg.get("enabled", False)):
        return
    ckpt_path = Path(str(transfer_cfg["checkpoint"]))
    checkpoint = torch.load(ckpt_path, map_location=device)
    load_encoder_weights_from_checkpoint(model, checkpoint)
    if bool(transfer_cfg.get("freeze_encoder", False)):
        for name, param in model.named_parameters():
            if name.startswith("epoch_encoder.") or name.startswith("temporal_encoder."):
                param.requires_grad = False


def _compute_loss(
    outputs: Mapping[str, "torch.Tensor"],
    batch: Mapping[str, "torch.Tensor"],
    *,
    apnea_weight: float,
    stage_weight: float,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    apnea_logits = outputs["apnea_logits"]
    stage_logits = outputs["stage_logits"]
    apnea_mask = batch["apnea_mask"] > 0
    stage_mask = batch["stage_mask"] > 0
    loss = torch.zeros((), device=apnea_logits.device, dtype=apnea_logits.dtype)
    metrics = {"loss_apnea": 0.0, "loss_stage": 0.0}
    if apnea_mask.any():
        apnea_loss = nn.functional.binary_cross_entropy_with_logits(
            apnea_logits[apnea_mask],
            batch["apnea_target"][apnea_mask],
        )
        loss = loss + (apnea_weight * apnea_loss)
        metrics["loss_apnea"] = float(apnea_loss.detach().cpu())
    if stage_mask.any():
        stage_loss = nn.functional.cross_entropy(stage_logits[stage_mask], batch["stage_target"][stage_mask])
        loss = loss + (stage_weight * stage_loss)
        metrics["loss_stage"] = float(stage_loss.detach().cpu())
    return loss, metrics


def _predict_loader(
    model: "nn.Module",
    loader: "DataLoader",
    *,
    device: "torch.device",
    stage_labels: Sequence[str],
) -> Dict[str, Any]:
    _require_torch()
    model.eval()
    apnea_true: List[int] = []
    apnea_pred: List[int] = []
    apnea_score: List[float] = []
    stage_true: List[str] = []
    stage_pred: List[str] = []
    stage_score: List[float] = []
    stage_mask_out: List[int] = []
    apnea_mask_out: List[int] = []
    dataset_ids: List[str] = []
    subject_ids: List[str] = []
    recording_ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            outputs = model(x)
            apnea_prob = torch.sigmoid(outputs["apnea_logits"])
            apnea_hat = (apnea_prob >= 0.5).to(torch.int64)
            stage_prob = torch.softmax(outputs["stage_logits"], dim=-1)
            stage_hat = torch.argmax(stage_prob, dim=-1)

            batch_apnea_mask = batch["apnea_mask"].cpu().numpy().astype(int)
            batch_stage_mask = batch["stage_mask"].cpu().numpy().astype(int)
            apnea_mask_out.extend(batch_apnea_mask.tolist())
            stage_mask_out.extend(batch_stage_mask.tolist())
            dataset_ids.extend([str(x) for x in batch["dataset_id"]])
            subject_ids.extend([str(x) for x in batch["subject_id"]])
            recording_ids.extend([str(x) for x in batch["recording_id"]])
            apnea_true.extend(batch["apnea_target"].cpu().numpy().astype(int).tolist())
            apnea_pred.extend(apnea_hat.cpu().numpy().astype(int).tolist())
            apnea_score.extend(apnea_prob.cpu().numpy().astype(float).tolist())

            batch_stage_true = batch["stage_target"].cpu().numpy().astype(int)
            batch_stage_pred = stage_hat.cpu().numpy().astype(int)
            batch_stage_prob = stage_prob.max(dim=-1).values.cpu().numpy().astype(float)
            for idx, stage_enabled in enumerate(batch_stage_mask.tolist()):
                if int(stage_enabled):
                    stage_true.append(stage_labels[int(batch_stage_true[idx])])
                    stage_pred.append(stage_labels[int(batch_stage_pred[idx])])
                    stage_score.append(float(batch_stage_prob[idx]))
                else:
                    stage_true.append("")
                    stage_pred.append("")
                    stage_score.append(np.nan)
    return {
        "apnea_true": apnea_true,
        "apnea_pred": apnea_pred,
        "apnea_score": apnea_score,
        "apnea_mask": apnea_mask_out,
        "stage_true": stage_true,
        "stage_pred": stage_pred,
        "stage_score": stage_score,
        "stage_mask": stage_mask_out,
        "dataset_id": dataset_ids,
        "subject_id": subject_ids,
        "recording_id": recording_ids,
    }


def _metrics_from_predictions(pred: Mapping[str, Any], *, stage_labels: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    apnea_mask = np.asarray(pred["apnea_mask"], dtype=int) > 0
    if apnea_mask.any():
        yt = np.asarray(pred["apnea_true"], dtype=int)[apnea_mask]
        yp = np.asarray(pred["apnea_pred"], dtype=int)[apnea_mask]
        ys = np.asarray(pred["apnea_score"], dtype=float)[apnea_mask]
        for k, v in apnea_binary_metrics(yt, yp, y_score_positive=ys).items():
            out[f"apnea_{k}"] = v
    else:
        out["apnea_accuracy"] = None
        out["apnea_sensitivity"] = None
        out["apnea_specificity"] = None
        out["apnea_auc_roc"] = None
        out["apnea_n_samples"] = 0

    stage_mask = np.asarray(pred["stage_mask"], dtype=int) > 0
    if stage_mask.any():
        yt_stage = [str(pred["stage_true"][i]) for i in range(len(stage_mask)) if stage_mask[i]]
        yp_stage = [str(pred["stage_pred"][i]) for i in range(len(stage_mask)) if stage_mask[i]]
        mm = multiclass_sleep_metrics(yt_stage, yp_stage, labels=stage_labels)
        out["stage_accuracy"] = mm["accuracy"]
        out["stage_macro_f1"] = mm["macro_f1"]
        out["stage_cohen_kappa"] = mm["cohen_kappa"]
    else:
        out["stage_accuracy"] = None
        out["stage_macro_f1"] = None
        out["stage_cohen_kappa"] = None
    return out


def _validation_score(metrics: Mapping[str, Any]) -> float:
    vals = []
    for key in ("apnea_auc_roc", "apnea_accuracy", "stage_macro_f1"):
        val = metrics.get(key)
        if val is None:
            continue
        vals.append(float(val))
    return float(np.mean(vals)) if vals else 0.0


def _fit_model(
    *,
    cfg: Mapping[str, Any],
    model: "nn.Module",
    train_loader: "DataLoader",
    val_loader: Optional["DataLoader"],
    stage_labels: Sequence[str],
    device: "torch.device",
) -> Dict[str, Any]:
    _require_torch()
    train_cfg = cfg.get("train", {}) or {}
    aug_cfg = cfg.get("augmentations", {}) or {}
    multitask_cfg = cfg.get("multitask", {}) or {}
    apnea_weight = float(multitask_cfg.get("apnea_loss_weight", 1.0))
    stage_weight = float(multitask_cfg.get("stage_loss_weight", 1.0))
    log_every_batches = int(train_cfg.get("log_every_batches", 100))

    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    amp_enabled = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(enabled=amp_enabled)
    max_epochs = int(train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("early_stopping_patience", 5))

    best_state: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_epoch = -1
    bad_epochs = 0
    epoch_seconds: List[float] = []
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[train] start epochs={max_epochs} batches_per_epoch={len(train_loader)} "
        f"val={'yes' if val_loader is not None and len(val_loader.dataset) > 0 else 'no'} "
        f"device={device} apnea_w={apnea_weight} stage_w={stage_weight}",
        flush=True,
    )

    for epoch in range(max_epochs):
        model.train()
        epoch_start = time.perf_counter()
        batch_losses: List[float] = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = _sequence_augmentations(batch["x"].to(device, non_blocking=True), aug_cfg)
            apnea_target = batch["apnea_target"].to(device, non_blocking=True)
            apnea_mask = batch["apnea_mask"].to(device, non_blocking=True)
            stage_target = batch["stage_target"].to(device, non_blocking=True)
            stage_mask = batch["stage_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(enabled=amp_enabled):
                outputs = model(x)
                loss, _ = _compute_loss(
                    outputs,
                    {
                        "apnea_target": apnea_target,
                        "apnea_mask": apnea_mask,
                        "stage_target": stage_target,
                        "stage_mask": stage_mask,
                    },
                    apnea_weight=apnea_weight,
                    stage_weight=stage_weight,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.detach().cpu()))
            if log_every_batches > 0 and (
                batch_idx == 1
                or batch_idx % log_every_batches == 0
                or batch_idx == len(train_loader)
            ):
                elapsed = time.perf_counter() - epoch_start
                mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
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
        avg_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        if val_loader is not None and len(val_loader.dataset) > 0:
            pred = _predict_loader(model, val_loader, device=device, stage_labels=stage_labels)
            val_metrics = _metrics_from_predictions(pred, stage_labels=stage_labels)
            score = _validation_score(val_metrics)
            scheduler.step(score)
            print(
                f"[train] epoch={epoch + 1}/{max_epochs} loss={avg_train_loss:.4f} "
                f"apnea_auc={val_metrics.get('apnea_auc_roc')} stage_f1={val_metrics.get('stage_macro_f1')} "
                f"score={score:.4f} seconds={epoch_seconds[-1]:.1f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
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

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_epoch = max(0, len(epoch_seconds) - 1)
    model.load_state_dict(best_state, strict=True)
    max_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    return {
        "model": model,
        "best_epoch": best_epoch,
        "train_seconds": time.perf_counter() - start_time,
        "avg_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else 0.0,
        "max_vram_mb": max_vram_mb,
    }


def _output_settings(cfg: Mapping[str, Any]) -> Tuple[bool, bool, bool]:
    out = cfg.get("output", {}) or {}
    return bool(out.get("save_models", True)), bool(out.get("save_fold_models", True)), bool(out.get("save_final_model", True))


def _checkpoint_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024)) if path.is_file() else 0.0


def _save_torch_bundle(path: Path, bundle: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(bundle), path)
    return path


def _bundle_for_checkpoint(
    *,
    model: "nn.Module",
    cfg: Mapping[str, Any],
    stage_labels: Sequence[str],
    model_name: str,
    artifact_kind: str,
    fold: Optional[int],
) -> Dict[str, Any]:
    return {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": dict(cfg.get("model", {}) or {}),
        "dataset_config": dict(cfg.get("dataset", {}) or {}),
        "train_config": dict(cfg.get("train", {}) or {}),
        "transfer_config": dict(cfg.get("transfer", {}) or {}),
        "multitask_config": dict(cfg.get("multitask", {}) or {}),
        "stage_labels": list(stage_labels),
        "model_name": model_name,
        "artifact_kind": artifact_kind,
        "fold": fold,
    }


def _metrics_row(
    *,
    model_name: str,
    fold: Any,
    pred: Mapping[str, Any],
    stage_labels: Sequence[str],
    fitted: Mapping[str, Any],
    checkpoint_size_mb: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"model": model_name, "fold": fold}
    row.update(_metrics_from_predictions(pred, stage_labels=stage_labels))
    row["train_seconds"] = float(fitted["train_seconds"])
    row["avg_epoch_seconds"] = float(fitted["avg_epoch_seconds"])
    row["max_vram_mb"] = float(fitted["max_vram_mb"])
    row["checkpoint_size_mb"] = float(checkpoint_size_mb)
    row["best_epoch"] = int(fitted["best_epoch"])
    return row


def _write_summary(out_dir: Path, metrics_rows: Sequence[Mapping[str, Any]], model_names: Sequence[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics_per_fold.csv", index=False)
    metric_keys = [
        "apnea_accuracy",
        "apnea_sensitivity",
        "apnea_specificity",
        "apnea_auc_roc",
        "stage_accuracy",
        "stage_macro_f1",
        "stage_cohen_kappa",
        "train_seconds",
    ]
    summary = {
        "models": list(model_names),
        "folds": list(metrics_rows),
        "aggregate": fold_metrics_summary(metrics_rows, metric_keys),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


def run_cv(
    cfg: Mapping[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    *,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> None:
    _require_torch()
    seed = int(cfg.get("random_seed", 42))
    set_random_seed(seed)
    device = _select_device(cfg)
    subject_col = str(cfg.get("subject_column", "subject_unit_id"))
    df = ensure_subject_unit_column(df, output_col=subject_col, overwrite=False)
    df = standardize_multitask_metadata(
        df,
        subject_col=subject_col,
        recording_col=str(cfg.get("recording_column", "recording_id")),
        order_col=str((cfg.get("dataset", {}) or {}).get("order_column", "epoch_index")),
        subject_fraction=float((cfg.get("train", {}) or {}).get("subject_fraction", 1.0)),
        random_seed=seed,
    )
    stage_labels = _build_stage_labels(df, (cfg.get("labels", {}) or {}).get("stage_order", DEFAULT_STAGE_ORDER))
    model_cfg = {**dict(cfg.get("model", {}) or {}), "sequence_length": int((cfg.get("dataset", {}) or {}).get("sequence_length", 9))}
    model_name = f"{str(model_cfg.get('type', 'conformer')).lower()}_multitask"
    input_mode = normalize_input_mode((cfg.get("dataset", {}) or {}).get("input_mode", "raw"))
    cv_cfg = cfg.get("cv", {}) or {}
    fold_conf = SubjectFoldConfig(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        random_state=seed,
        stratify=bool(cv_cfg.get("stratify", True)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
    )
    metrics_rows: List[Dict[str, Any]] = []
    registry_rows: List[Dict[str, Any]] = []
    save_models, save_fold_models, save_final_model = _output_settings(cfg)
    models_dir = out_dir / "models"
    print(
        f"[run_cv] rows={len(df)} subjects={df[subject_col].astype(str).nunique()} "
        f"stage_labels={stage_labels} device={device} model={model_name} "
        f"input_mode={input_mode} batching={_batching_strategy(cfg, input_mode=input_mode)} "
        f"recordings={df['recording_id'].astype(str).nunique()}",
        flush=True,
    )

    for fold_id, (train_idx, test_idx) in enumerate(
        subject_wise_fold_indices(df, subject_col=subject_col, y=df["subject_unit_id"].astype(str).values, config=fold_conf)
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
            f"[run_cv] fold={fold_id + 1}/{fold_conf.n_splits} "
            f"train_rows={len(train_df)} val_rows={len(val_df)} test_rows={len(test_df)}",
            flush=True,
        )
        data = _make_datasets_and_loaders(cfg=cfg, df_train=train_df, df_val=val_df, df_test=test_df, stage_labels=stage_labels, signal_loader=signal_loader)
        model = build_multitask_model(model_cfg, stage_num_classes=len(stage_labels))
        _maybe_load_transfer_checkpoint(model, cfg, device=device)
        fitted = _fit_model(cfg=cfg, model=model, train_loader=data["loaders"]["train"], val_loader=data["loaders"]["val"], stage_labels=stage_labels, device=device)
        pred = _predict_loader(fitted["model"], data["loaders"]["test"], device=device, stage_labels=stage_labels)

        apnea_mask = np.asarray(pred["apnea_mask"], dtype=int) > 0
        stage_mask = np.asarray(pred["stage_mask"], dtype=int) > 0
        save_predictions_dataframe(
            out_dir / "predictions" / f"{model_name}_fold{fold_id}.csv",
            y_true=np.asarray(pred["apnea_true"], dtype=object),
            y_pred=np.asarray(pred["apnea_pred"], dtype=object),
            y_score=pred["apnea_score"],
            subject_id=pred["subject_id"],
            fold_id=fold_id,
            extra_columns={
                "dataset_id": pred["dataset_id"],
                "recording_id": pred["recording_id"],
                "apnea_mask": pred["apnea_mask"],
                "stage_mask": pred["stage_mask"],
                "stage_true": pred["stage_true"],
                "stage_pred": pred["stage_pred"],
                "stage_score": pred["stage_score"],
            },
        )
        if apnea_mask.any():
            save_confusion_matrix_figure(
                np.asarray(pred["apnea_true"], dtype=int)[apnea_mask],
                np.asarray(pred["apnea_pred"], dtype=int)[apnea_mask],
                out_dir / "figures" / f"cm_apnea_{model_name}_fold{fold_id}.png",
                labels=[0, 1],
                title=f"apnea {model_name} fold {fold_id}",
            )
            save_roc_curve_figure(
                np.asarray(pred["apnea_true"], dtype=int)[apnea_mask],
                np.asarray(pred["apnea_score"], dtype=float)[apnea_mask],
                out_dir / "figures" / f"roc_apnea_{model_name}_fold{fold_id}.png",
                title=f"ROC apnea {model_name} fold {fold_id}",
            )
        if stage_mask.any():
            yt_stage = [pred["stage_true"][i] for i in range(len(stage_mask)) if stage_mask[i]]
            yp_stage = [pred["stage_pred"][i] for i in range(len(stage_mask)) if stage_mask[i]]
            save_confusion_matrix_figure(
                yt_stage,
                yp_stage,
                out_dir / "figures" / f"cm_stage_{model_name}_fold{fold_id}.png",
                labels=stage_labels,
                title=f"stage {model_name} fold {fold_id}",
            )
        checkpoint_size_mb = 0.0
        if save_models and save_fold_models:
            fold_path = _save_torch_bundle(
                models_dir / "folds" / f"{model_name}_fold{fold_id}.pt",
                _bundle_for_checkpoint(model=fitted["model"], cfg=cfg, stage_labels=stage_labels, model_name=model_name, artifact_kind="fold", fold=fold_id),
            )
            checkpoint_size_mb = _checkpoint_size_mb(fold_path)
            registry_rows.append(
                {
                    "experiment_name": str(cfg.get("experiment_name", out_dir.name)),
                    "dataset_origin": str(cfg.get("train_csv", "")),
                    "algorithm": model_name,
                    "artifact_type": "fold",
                    "training_mode": "multitask",
                    "fold": fold_id,
                    "path": str(fold_path.relative_to(out_dir)),
                    "class_labels": list(stage_labels),
                }
            )
        metrics_rows.append(_metrics_row(model_name=model_name, fold=fold_id, pred=pred, stage_labels=stage_labels, fitted=fitted, checkpoint_size_mb=checkpoint_size_mb))
        print(
            f"[run_cv] fold={fold_id + 1} done "
            f"apnea_acc={metrics_rows[-1].get('apnea_accuracy')} "
            f"stage_f1={metrics_rows[-1].get('stage_macro_f1')}",
            flush=True,
        )

    if save_models and save_final_model:
        train_df, val_df = _split_train_val_subjects(
            df,
            subject_col=subject_col,
            val_fraction=float((cfg.get("train", {}) or {}).get("val_subject_fraction", 0.15)),
            seed=seed + 999,
        )
        data = _make_datasets_and_loaders(cfg=cfg, df_train=train_df, df_val=val_df, df_test=None, stage_labels=stage_labels, signal_loader=signal_loader)
        final_model = build_multitask_model(model_cfg, stage_num_classes=len(stage_labels))
        _maybe_load_transfer_checkpoint(final_model, cfg, device=device)
        fitted = _fit_model(cfg=cfg, model=final_model, train_loader=data["loaders"]["train"], val_loader=data["loaders"]["val"], stage_labels=stage_labels, device=device)
        final_path = _save_torch_bundle(
            models_dir / f"{model_name}_final.pt",
            _bundle_for_checkpoint(model=fitted["model"], cfg=cfg, stage_labels=stage_labels, model_name=model_name, artifact_kind="final", fold=None),
        )
        registry_rows.append(
            {
                "experiment_name": str(cfg.get("experiment_name", out_dir.name)),
                "dataset_origin": str(cfg.get("train_csv", "")),
                "algorithm": model_name,
                "artifact_type": "final",
                "training_mode": "multitask",
                "fold": None,
                "path": str(final_path.relative_to(out_dir)),
                "class_labels": list(stage_labels),
            }
        )

    _write_summary(out_dir, metrics_rows, [model_name])
    if save_models and registry_rows:
        write_model_registry(models_dir / "model_registry.json", registry_rows)
    print(f"[run_cv] wrote metrics and summary under {out_dir}", flush=True)


def run_cross_dataset(
    cfg: Mapping[str, Any],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    out_dir: Path,
    *,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> None:
    _require_torch()
    seed = int(cfg.get("random_seed", 42))
    set_random_seed(seed)
    device = _select_device(cfg)
    subject_col = str(cfg.get("subject_column", "subject_unit_id"))
    train_df = ensure_subject_unit_column(train_df, output_col=subject_col, overwrite=False)
    eval_df = ensure_subject_unit_column(eval_df, output_col=subject_col, overwrite=False)
    dataset_cfg = cfg.get("dataset", {}) or {}
    train_df = standardize_multitask_metadata(train_df, subject_col=subject_col, recording_col=str(cfg.get("recording_column", "recording_id")), order_col=str(dataset_cfg.get("order_column", "epoch_index")), random_seed=seed)
    eval_df = standardize_multitask_metadata(eval_df, subject_col=subject_col, recording_col=str(cfg.get("recording_column", "recording_id")), order_col=str(dataset_cfg.get("order_column", "epoch_index")), random_seed=seed)
    stage_labels = _build_stage_labels(train_df, (cfg.get("labels", {}) or {}).get("stage_order", DEFAULT_STAGE_ORDER))
    model_cfg = {**dict(cfg.get("model", {}) or {}), "sequence_length": int(dataset_cfg.get("sequence_length", 9))}
    model_name = f"{str(model_cfg.get('type', 'conformer')).lower()}_multitask"
    input_mode = normalize_input_mode(dataset_cfg.get("input_mode", "raw"))

    train_fit, val_df = _split_train_val_subjects(train_df, subject_col=subject_col, val_fraction=float((cfg.get("train", {}) or {}).get("val_subject_fraction", 0.15)), seed=seed)
    train_data = _make_datasets_and_loaders(cfg=cfg, df_train=train_fit, df_val=val_df, df_test=None, stage_labels=stage_labels, signal_loader=signal_loader)
    eval_data = _make_datasets_and_loaders(cfg=cfg, df_train=eval_df, df_val=eval_df.iloc[0:0].copy(), df_test=eval_df, stage_labels=stage_labels, signal_loader=signal_loader)

    model = build_multitask_model(model_cfg, stage_num_classes=len(stage_labels))
    print(
        f"[run_cross] train_rows={len(train_df)} eval_rows={len(eval_df)} "
        f"stage_labels={stage_labels} device={device} model={model_name} "
        f"input_mode={input_mode} batching={_batching_strategy(cfg, input_mode=input_mode)}",
        flush=True,
    )
    _maybe_load_transfer_checkpoint(model, cfg, device=device)
    fitted = _fit_model(cfg=cfg, model=model, train_loader=train_data["loaders"]["train"], val_loader=train_data["loaders"]["val"], stage_labels=stage_labels, device=device)
    pred = _predict_loader(fitted["model"], eval_data["loaders"]["test"], device=device, stage_labels=stage_labels)

    save_predictions_dataframe(
        out_dir / "predictions" / f"{model_name}_cross_eval.csv",
        y_true=np.asarray(pred["apnea_true"], dtype=object),
        y_pred=np.asarray(pred["apnea_pred"], dtype=object),
        y_score=pred["apnea_score"],
        subject_id=pred["subject_id"],
        fold_id=-1,
        extra_columns={
            "dataset_id": pred["dataset_id"],
            "recording_id": pred["recording_id"],
            "apnea_mask": pred["apnea_mask"],
            "stage_mask": pred["stage_mask"],
            "stage_true": pred["stage_true"],
            "stage_pred": pred["stage_pred"],
            "stage_score": pred["stage_score"],
        },
    )
    apnea_mask = np.asarray(pred["apnea_mask"], dtype=int) > 0
    if apnea_mask.any():
        save_confusion_matrix_figure(
            np.asarray(pred["apnea_true"], dtype=int)[apnea_mask],
            np.asarray(pred["apnea_pred"], dtype=int)[apnea_mask],
            out_dir / "figures" / f"cm_apnea_{model_name}_cross_eval.png",
            labels=[0, 1],
            title=f"apnea {model_name} cross_eval",
        )
        save_roc_curve_figure(
            np.asarray(pred["apnea_true"], dtype=int)[apnea_mask],
            np.asarray(pred["apnea_score"], dtype=float)[apnea_mask],
            out_dir / "figures" / f"roc_apnea_{model_name}_cross_eval.png",
            title=f"ROC apnea {model_name} cross_eval",
        )
    row = _metrics_row(model_name=model_name, fold="cross_eval", pred=pred, stage_labels=stage_labels, fitted=fitted, checkpoint_size_mb=0.0)
    pd.DataFrame([row]).to_csv(out_dir / "metrics_cross_eval.csv", index=False)
    print(f"[run_cross] wrote cross metrics under {out_dir}", flush=True)


def run_experiment(
    config_path: Path,
    *,
    signal_loader: Optional[Callable[[Mapping[str, Any], Path, Mapping[str, Any]], Tuple[np.ndarray, float]]] = None,
) -> None:
    _require_torch()
    cfg = load_config(config_path)
    print(f"[run] config={config_path}", flush=True)
    out_root = Path((cfg.get("output", {}) or {}).get("root", "reports/experiments"))
    out_dir = out_root / str(cfg.get("experiment_name", "multitask_run"))
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
        dataset_cfg["epoch_store_manifest"] = str(
            resolve_csv_path(str(dataset_cfg["epoch_store_manifest"]), config_path)
        )
        if dataset_cfg.get("eval_epoch_store_manifest"):
            dataset_cfg["eval_epoch_store_manifest"] = str(
                resolve_csv_path(str(dataset_cfg["eval_epoch_store_manifest"]), config_path)
            )
    cfg = dict(cfg)
    cfg["dataset"] = dataset_cfg
    if input_mode == "raw":
        print(f"[run] raw_root={dataset_cfg['raw_root']}", flush=True)
    else:
        print(f"[run] input_mode=epoch_store epoch_store_root={dataset_cfg['epoch_store_root']}", flush=True)
    with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    train_path = resolve_csv_path(str(cfg["train_csv"]), config_path)
    if input_mode == "epoch_store":
        manifest_path = Path(str(dataset_cfg["epoch_store_manifest"]))
        train_df = read_epoch_store_manifest(manifest_path)
        print(f"[run] train_csv={train_path}", flush=True)
        print(f"[run] epoch_store_manifest={manifest_path} shape={train_df.shape}", flush=True)
        train_origin = manifest_path
    else:
        train_df = read_multitask_metadata(train_path)
        print(f"[run] train_csv={train_path} shape={train_df.shape}", flush=True)
        train_origin = train_path
    if bool(cfg.get("cross_dataset", False)):
        if input_mode == "epoch_store" and dataset_cfg.get("eval_epoch_store_manifest"):
            eval_path = Path(str(dataset_cfg["eval_epoch_store_manifest"]))
            eval_df = read_epoch_store_manifest(eval_path)
            print(f"[run] eval_epoch_store_manifest={eval_path} shape={eval_df.shape}", flush=True)
        else:
            eval_path = resolve_csv_path(str(cfg["eval_csv"]), config_path)
            eval_df = read_multitask_metadata(eval_path)
            if input_mode == "epoch_store":
                missing = [col for col in EPOCH_STORE_REQUIRED_COLUMNS if col not in eval_df.columns]
                if missing:
                    raise ValueError(
                        "Cross-dataset epoch_store mode requires dataset.eval_epoch_store_manifest "
                        f"or an eval_csv already materialized. Missing columns: {missing!r}."
                    )
                eval_df = read_epoch_store_manifest(eval_path)
            print(f"[run] eval_csv={eval_path} shape={eval_df.shape}", flush=True)
        run_cross_dataset(cfg, train_df, eval_df, out_dir, signal_loader=signal_loader)
    else:
        run_cv(cfg, train_df, out_dir, signal_loader=signal_loader)


def main(argv: Optional[Sequence[str]] = None) -> None:
    _require_torch()
    parser = argparse.ArgumentParser(description="Run multitask apnea/staging deep experiments.")
    parser.add_argument("--config", required=True, type=str, help="Path to multitask experiment YAML.")
    args = parser.parse_args(argv)
    run_experiment(Path(args.config))
