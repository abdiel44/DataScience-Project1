"""
Phase E deep runner: waveform sleep staging with CNN/Conformer (+ optional SSL).

  python scripts/run_phase_e_deep.py --config config/experiments/sleep_edf_2013_fpzcz_deep_conformer.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.deep_runner import main

if __name__ == "__main__":
    main()
