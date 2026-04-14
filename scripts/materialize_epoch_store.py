"""Materialize a reusable epoch store (.npy per recording) from experiment metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.epoch_store import export_epoch_store_features, materialize_epoch_store, read_table_file
from modeling.path_utils import resolve_path_any
from modeling.subject_id import ensure_subject_unit_column
from modeling.train_runner import load_config, resolve_csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a fast epoch store from raw waveform metadata.")
    parser.add_argument("--config", required=True, type=str, help="Experiment YAML with dataset settings.")
    parser.add_argument("--metadata", type=str, default="", help="Override metadata CSV/parquet path. Defaults to train_csv.")
    parser.add_argument(
        "--mode",
        choices=("waveforms", "features", "both"),
        default="waveforms",
        help="Write waveform store only, features only from an existing manifest, or both.",
    )
    parser.add_argument("--manifest-out", type=str, default="", help="Override epoch-store manifest output path.")
    parser.add_argument("--output-root", type=str, default="", help="Override epoch-store root directory.")
    parser.add_argument("--features-output", type=str, default="", help="Output table for tabular EEG features.")
    parser.add_argument("--dataset-id", type=str, default="", help="Fallback dataset_id when metadata lacks that column.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing store files.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not reuse validated existing store files.")
    parser.add_argument(
        "--feature-normalize-epoch",
        action="store_true",
        help="Normalize each epoch before feature extraction.",
    )
    return parser.parse_args()


def _prepare_metadata(df: pd.DataFrame, *, cfg: Dict[str, Any], dataset_id_fallback: str) -> pd.DataFrame:
    out = df.copy()
    subject_col = str(cfg.get("subject_column", "subject_unit_id"))
    recording_col = str(cfg.get("recording_column", "recording_id"))
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    if "dataset_id" not in out.columns:
        out["dataset_id"] = dataset_id_fallback
    if "subject_unit_id" not in out.columns:
        out = ensure_subject_unit_column(out, output_col="subject_unit_id", overwrite=False)
        if "subject_unit_id" not in out.columns and subject_col in out.columns:
            out["subject_unit_id"] = out[subject_col]
    if "recording_id" not in out.columns and recording_col in out.columns:
        out["recording_id"] = out[recording_col]
    order_col = str(dataset_cfg.get("order_column", "epoch_index"))
    if "epoch_index" not in out.columns and order_col in out.columns:
        out["epoch_index"] = out[order_col]
    if "eeg_channel_standardized" not in out.columns:
        channel = str(dataset_cfg.get("signal_channel", "EEG"))
        out["eeg_channel_standardized"] = channel
    required = ["dataset_id", "recording_id", "epoch_index", "epoch_start_sec", "epoch_end_sec"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise ValueError(f"Metadata is missing columns required for materialization: {missing!r}")
    return out.sort_values(["dataset_id", "recording_id", "epoch_index"]).reset_index(drop=True)


def _default_manifest_path(config_path: Path, cfg: Dict[str, Any]) -> Path:
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    if dataset_cfg.get("epoch_store_manifest"):
        raw = Path(str(dataset_cfg["epoch_store_manifest"]))
        if raw.is_absolute():
            return raw.resolve()
        cand_cwd = (Path.cwd() / raw).resolve()
        cand_cfg = (config_path.resolve().parent / raw).resolve()
        return cand_cwd if cand_cwd.parent.exists() else cand_cfg
    out_root = ROOT / "data" / "processed"
    return (out_root / f"{cfg.get('experiment_name', 'epoch_store')}_epoch_store.parquet").resolve()


def _default_store_root(config_path: Path, cfg: Dict[str, Any]) -> Path:
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    if dataset_cfg.get("epoch_store_root"):
        try:
            return resolve_path_any(str(dataset_cfg["epoch_store_root"]), config_path, expect_dir=True)
        except FileNotFoundError:
            raw = Path(str(dataset_cfg["epoch_store_root"]))
            if raw.is_absolute():
                return raw
            return (ROOT / raw).resolve()
    return (ROOT / "data" / "processed" / "epoch_store" / str(cfg.get("experiment_name", "epoch_store"))).resolve()


def _default_features_path(manifest_path: Path) -> Path:
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        return manifest_path.with_name(f"{manifest_path.stem}_features.csv")
    return manifest_path.with_name(f"{manifest_path.stem}_features.parquet")


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    dataset_cfg = dict(cfg.get("dataset", {}) or {})

    metadata_path = resolve_csv_path(args.metadata or str(cfg["train_csv"]), config_path)
    manifest_path = (
        Path(args.manifest_out).resolve()
        if args.manifest_out
        else _default_manifest_path(config_path, cfg)
    )
    store_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _default_store_root(config_path, cfg)
    )
    features_output = (
        Path(args.features_output).resolve()
        if args.features_output
        else _default_features_path(manifest_path)
    )

    raw_root = Path(str(dataset_cfg.get("raw_root", "data/raw")))
    if args.mode in {"waveforms", "both"}:
        raw_root = resolve_path_any(str(raw_root), config_path, expect_dir=True)

    dataset_id_fallback = args.dataset_id or str(dataset_cfg.get("dataset_id", "sleep_edf_expanded"))
    print(f"[materialize] config={config_path}", flush=True)
    print(f"[materialize] metadata={metadata_path}", flush=True)
    print(f"[materialize] mode={args.mode}", flush=True)
    print(f"[materialize] store_root={store_root}", flush=True)
    print(f"[materialize] manifest={manifest_path}", flush=True)

    if args.mode in {"waveforms", "both"}:
        metadata_df = read_table_file(metadata_path)
        prepared = _prepare_metadata(metadata_df, cfg=cfg, dataset_id_fallback=dataset_id_fallback)
        manifest_df = materialize_epoch_store(
            prepared,
            store_root=store_root,
            manifest_path=manifest_path,
            raw_root=raw_root,
            dataset_cfg=dataset_cfg,
            force=bool(args.force),
            skip_existing=not bool(args.no_skip_existing),
        )
        print(f"[materialize] manifest_rows={len(manifest_df)}", flush=True)
    else:
        manifest_df = read_table_file(manifest_path)

    if args.mode in {"features", "both"}:
        print(f"[materialize] features_output={features_output}", flush=True)
        features_df = export_epoch_store_features(
            manifest_df,
            store_root=store_root,
            output_path=features_output,
            normalize_epoch=bool(args.feature_normalize_epoch),
        )
        print(f"[materialize] feature_rows={len(features_df)}", flush=True)


if __name__ == "__main__":
    main()
