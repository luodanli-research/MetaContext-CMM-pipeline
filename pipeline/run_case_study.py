#!/usr/bin/env python3
"""End-to-end MetaContext-CMM pipeline for the Zenodo case_study/ dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from _orchestrator import main as run_pipeline
from presets import CASE_STUDY


def main() -> int:
    return run_pipeline(CASE_STUDY)


if __name__ == "__main__":
    raise SystemExit(main())
