"""
Materialize a raw/tabular CSV from a supported `--source` loader without applying
the preprocessing pipeline from `src/main.py`.

Examples:
  python scripts/export_source_raw.py --source isruc-sleep --raw-root data/raw --output data/processed/isruc_sleep_raw.csv
  python scripts/export_source_raw.py --source st-vincent-apnea --raw-root data/raw --output data/processed/st_vincent_apnea_raw.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_processing.raw_loaders import ingest_by_source_id  # noqa: E402

_SOURCE_CLI_TO_ID = {
    "isruc-sleep": "isruc_sleep",
    "st-vincent-apnea": "st_vincent_apnea",
    "sleep-edf-expanded": "sleep_edf_expanded",
    "sleep-edf-2013-fpzcz": "sleep_edf_2013_fpzcz",
    "mit-bih-psg": "mit_bih_psg",
    "shhs-psg": "shhs_psg",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export raw/tabular CSV from a supported source loader.")
    p.add_argument("--source", required=True, choices=sorted(_SOURCE_CLI_TO_ID.keys()), type=str)
    p.add_argument("--raw-root", default="data/raw", type=str, help="Parent folder containing dataset directories.")
    p.add_argument("--output", required=True, type=str, help="CSV path to write.")
    p.add_argument("--max-files", default=None, type=int, help="Optional cap on source files/records.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"--raw-root not found: {raw_root}")

    source_id = _SOURCE_CLI_TO_ID[args.source]
    df, meta = ingest_by_source_id(source_id, raw_root, max_files=args.max_files)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"Exported raw source ({args.source}): rows={len(df)}, "
        f"files_used={meta.n_files_used}, skipped={meta.n_files_skipped}"
    )
    for note in meta.notes:
        print(f"  Note: {note}")
    print(f"  Wrote: {output_path}")


if __name__ == "__main__":
    main()
