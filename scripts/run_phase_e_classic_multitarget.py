"""Run the classic multitarget EEG runner."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modeling.classic_multitarget_runner import main


if __name__ == "__main__":
    main()
