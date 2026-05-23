"""Alias for run_phase2_baseline.py (spec name: run_baseline.py)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase2_baseline import main

if __name__ == "__main__":
    raise SystemExit(main())
