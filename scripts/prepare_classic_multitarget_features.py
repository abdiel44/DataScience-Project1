"""Export canonical EEG tabular features for classic multitarget experiments from an epoch store manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.epoch_store import export_epoch_store_features, read_epoch_store_manifest
from modeling.path_utils import resolve_path_any
from modeling.train_runner import load_config, resolve_csv_path


def _default_output_path(config_path: Path, cfg: dict) -> Path:
    train_csv = Path(str(cfg.get("train_csv", "")))
    if train_csv.suffix.lower() == ".csv":
        return (config_path.resolve().parents[1] / "data" / "processed" / f"{train_csv.stem}_classic.parquet").resolve()
    if train_csv.suffix.lower() == ".parquet":
        return train_csv.with_name(f"{train_csv.stem}_classic.parquet").resolve()
    name = str(cfg.get("experiment_name", "classic_multitarget")).replace(".yaml", "")
    return (ROOT / "data" / "processed" / f"{name}_classic.parquet").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EEG feature tables for classic multitarget apnea/staging runs.")
    parser.add_argument("--config", required=True, type=str, help="Path to experiment YAML.")
    parser.add_argument("--manifest", default="", type=str, help="Override epoch-store manifest path.")
    parser.add_argument("--store-root", default="", type=str, help="Override epoch-store root directory.")
    parser.add_argument("--output", default="", type=str, help="Output .csv/.parquet path for tabular EEG features.")
    parser.add_argument("--normalize-each-epoch", action="store_true", help="Normalize each epoch before feature extraction.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    dataset_cfg = dict(cfg.get("dataset", {}) or {})
    manifest_path = resolve_csv_path(
        args.manifest or str(dataset_cfg.get("epoch_store_manifest", "")),
        config_path,
    )
    if not manifest_path:
        raise ValueError("Missing manifest path. Set dataset.epoch_store_manifest or pass --manifest.")
    store_root = (
        Path(args.store_root).resolve()
        if args.store_root
        else resolve_path_any(str(dataset_cfg.get("epoch_store_root", "")), config_path, expect_dir=True)
    )
    if not store_root:
        raise ValueError("Missing store root. Set dataset.epoch_store_root or pass --store-root.")
    output_path = Path(args.output).resolve() if args.output else _default_output_path(config_path, cfg)

    manifest_df = read_epoch_store_manifest(manifest_path)
    print(f"[prepare_classic] config={config_path}", flush=True)
    print(f"[prepare_classic] manifest={manifest_path} rows={len(manifest_df)}", flush=True)
    print(f"[prepare_classic] store_root={store_root}", flush=True)
    print(f"[prepare_classic] output={output_path}", flush=True)

    features_df = export_epoch_store_features(
        manifest_df,
        store_root=store_root,
        output_path=output_path,
        feature_prefix="eeg",
        normalize_epoch=bool(args.normalize_each_epoch),
    )
    print(f"[prepare_classic] feature_rows={len(features_df)}", flush=True)


if __name__ == "__main__":
    main()
