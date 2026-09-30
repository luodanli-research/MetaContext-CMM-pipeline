#!/usr/bin/env python3
"""Case-study pipeline launcher for pre-solve reruns (isolated run root).

Copy of ``run_case_study.py`` — leave the original defaults on
``case_study/outputs``. Pass ``--output-root`` / ``--analysis-dir`` at an
isolated tree (reuse prior medium/tradeoff fluxes; do **not** pass ``--force``).

Example (add reaction after medium+tradeoff; omit incomplete bound 21 so
medium REUSE instead of re-solving the known SW55/bound21 failure)::

    RUN=case_study/outputs/tradeoff1
    python pipeline/run_case_study_presolve_rerun.py \\
      --output-root \"$RUN\" --analysis-dir \"$RUN/04_analysis\" \\
      --sensitivity medium tradeoff reaction \\
      --medium-bounds 1000 619 383 237 146 90 56 34 13 8 5 \\
      --metrics eai arb ac
"""

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
