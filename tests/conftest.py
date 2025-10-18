"""Pytest configuration helpers."""

import sys
from pathlib import Path

# Ensure the project package under src/ is importable without installation.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
