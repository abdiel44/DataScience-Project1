"""
EDA sobre un CSV tabular sin ejecutar el pipeline de limpieza de main.py (Fase B — PRD).

Uso (desde la raíz del repositorio):
  python scripts/raw_eda.py --input data/processed/mitbih_sleep_stages.csv --target-col sleep_stage --task mitbih_sleep_raw

Salidas por defecto: reports/eda_raw/<task>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pre_processing.cleaning import to_snake_case  # noqa: E402
from pre_processing.eda import run_eda  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA exploratorio sobre CSV (datos brutos/tabular exportado).")
    p.add_argument("--input", required=True, type=str, help="Ruta al CSV.")
    p.add_argument("--target-col", required=True, type=str, help="Columna objetivo (se normaliza a snake_case).")
    p.add_argument("--task", required=True, type=str, help="Etiqueta para carpetas e informe (ej. mitbih_sleep_raw).")
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directorio de salida (default: reports/eda_raw/<task>).",
    )
    p.add_argument("--top-n-plots", type=int, default=15, help="Máx. features numéricas para histogramas/boxplots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [to_snake_case(str(c)) for c in df.columns]
    target_key = to_snake_case(args.target_col)
    if target_key not in df.columns:
        cols_preview = list(df.columns)[:30]
        raise ValueError(
            f"Tras normalizar nombres, la columna objetivo '{target_key}' no está en las columnas: {cols_preview}"
        )

    out = Path(args.outdir) if args.outdir else ROOT / "reports" / "eda_raw" / to_snake_case(args.task)
    res = run_eda(df, output_dir=out, task=args.task, target_col=target_key, top_n=int(args.top_n_plots))
    print("EDA (raw/tabular) terminado.")
    print(f"  Resumen: {res['summary']}")
    print(f"  Tablas: {len(res['tables'])}")
    print(f"  Figuras: {len(res['figures'])}")


if __name__ == "__main__":
    main()
