#!/usr/bin/env python3
"""End-to-end MetaContext-CMM pipeline for the shipped example/ tutorial.

One output tree under ``example/outputs``. Tradeoff sensitivity solves only
fraction 1. Medium and reaction use the same cooperative-tradeoff fraction.

    python -u pipeline/run_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from _orchestrator import main as run_pipeline
from presets import EXAMPLE


def main() -> int:
    argv = sys.argv
    if "--tradeoffs" not in argv:
        argv.extend(["--tradeoffs", "1"])
    if "--solve-tradeoff" not in argv:
        argv.extend(["--solve-tradeoff", "1"])
    return run_pipeline(EXAMPLE)


if __name__ == "__main__":
    raise SystemExit(main())
