"""
Phase E: ejecutar CV o cross-dataset según YAML.

  python scripts/run_phase_e_cv.py --config config/experiment_train.example.yaml

Requiere cwd = raíz del repo (o ajustar rutas en el YAML a absolutas).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.train_runner import main

if __name__ == "__main__":
    main()
