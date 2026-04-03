from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline import preprocess_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Data Science preprocessing pipeline.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output CSV file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    clean_df, report = preprocess_dataframe(df)
    clean_df.to_csv(output_path, index=False)

    print("Pipeline finished successfully")
    print(f"Input rows: {report.input_rows}")
    print(f"Output rows: {report.output_rows}")
    print(f"Removed empty rows: {report.removed_empty_rows}")
    print(f"Removed duplicates: {report.removed_duplicates}")
    print(f"Saved file: {output_path}")


if __name__ == "__main__":
    main()

