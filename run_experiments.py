#!/usr/bin/env python3
"""
Entry point.

    python run_experiments.py --all
    python run_experiments.py --config configs/ensemble_dpr.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from duq.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
