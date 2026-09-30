#!/usr/bin/env python3
"""Case-study pipeline for the published tradeoff1 and tradeoff0.5 layouts.

Points at ``case_study/outputs/tradeoff1`` or ``tradeoff0.5``. Both trees
reuse the pre-solve context models. Medium fluxes are the phosphate-and-iron
list and are plotted from the existing directory; this script does not solve
medium again. Tradeoff sensitivity and reaction reuse files already in the
selected tree. ``tradeoff0.5`` links its tradeoff sweep to ``tradeoff1``.

Examples::

    python -u pipeline/run_case_study.py --solve-tradeoff 1
    python -u pipeline/run_case_study.py --solve-tradeoff 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from _orchestrator import main as run_pipeline
from presets import CASE_STUDY

MEDIUM_LIST = Path("case_study/inputs/04_medium_sensitivity_list.csv")
PUBLISHED_ROOTS = {
    1.0: Path("case_study/outputs/tradeoff1"),
    0.5: Path("case_study/outputs/tradeoff0.5"),
}


def _ensure_flag(argv: list[str], flag: str, values: list[str]) -> None:
    if flag not in argv:
        argv.extend([flag, *values])


def _flag_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {flag}")
    return argv[index + 1]


def _published_fraction(raw: str | None) -> float:
    if raw is None:
        return 1.0
    try:
        value = float(raw)
    except ValueError as error:
        raise SystemExit(
            "--solve-tradeoff must be 1 or 0.5 for the published case-study trees."
        ) from error
    for candidate in PUBLISHED_ROOTS:
        if abs(value - candidate) <= 1e-12:
            return candidate
    raise SystemExit(
        "--solve-tradeoff must be 1 or 0.5 for the published case-study trees."
    )


def main() -> int:
    argv = sys.argv
    fraction = _published_fraction(_flag_value(argv, "--solve-tradeoff"))
    _ensure_flag(argv, "--solve-tradeoff", [f"{fraction:g}"])
    _ensure_flag(argv, "--output-root", [str(PUBLISHED_ROOTS[fraction])])
    root = Path(_flag_value(argv, "--output-root") or str(PUBLISHED_ROOTS[fraction]))
    _ensure_flag(argv, "--analysis-dir", [str(root / "04_analysis")])
    _ensure_flag(argv, "--medium-reaction-list", [str(MEDIUM_LIST)])
    _ensure_flag(argv, "--sensitivity", ["tradeoff", "reaction"])
    _ensure_flag(
        argv,
        "--prior-medium-dir",
        [str(root / "03_simulation" / "sensitivity_medium")],
    )
    _ensure_flag(argv, "--solve-timeout", ["1800"])
    _ensure_flag(argv, "--metrics", ["eai", "arb", "ac"])
    return run_pipeline(CASE_STUDY)


if __name__ == "__main__":
    raise SystemExit(main())
