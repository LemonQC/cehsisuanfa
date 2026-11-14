#!/usr/bin/env python3
"""Stable CLI entrypoint for the read-only B11/B12 asset audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agcofl.baseline_asset_audit import main  # noqa: E402


if __name__ == "__main__":
    main()
