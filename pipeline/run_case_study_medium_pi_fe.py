#!/usr/bin/env python3
"""Case-study pipeline copy: medium sensitivity includes phosphate and iron.

Copy of ``run_case_study_presolve_rerun.py``. Two differences:

- ``--medium-reaction-list`` defaults to
  ``case_study/inputs/04_medium_sensitivity_list.csv``
  (published list plus ``EX_pi_m``; ``EX_fe2_m`` and ``EX_fe3_m`` stay).
- ``--output-root`` / ``--analysis-dir`` default to a new run tree.

Before the orchestrator starts, formal context and tradeoff outputs are
symlinked from the existing metacontext run so those stages REUSE.
``sensitivity_medium`` and ``04_analysis`` are real directories in the new
tree. Reaction sampling is not linked and is not in the default command.

Example::

    python -u pipeline/run_case_study_medium_pi_fe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from _orchestrator import main as run_pipeline
from presets import CASE_STUDY

SOURCE_RUN = Path("case_study/outputs/tradeoff1")
RUN = Path("case_study/outputs/tradeoff1")
MEDIUM_LIST = Path("case_study/inputs/04_medium_sensitivity_list.csv")

REUSE_LINKS = (
    "01_baseline",
    "02_context",
    "03_simulation/context",
    "03_simulation/sensitivity_tradeoff",
)


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


def _ensure_dir_link(link: Path, source: Path) -> None:
    if not source.is_dir():
        raise SystemExit(f"Missing source directory to reuse: {source}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == source.resolve():
            return
        raise SystemExit(
            f"{link} already points at {link.resolve()}, not {source.resolve()}"
        )
    if link.exists():
        raise SystemExit(f"{link} exists and is not a symlink; refuse to replace.")
    link.symlink_to(source.resolve(), target_is_directory=True)


def prepare_run_root(run_root: Path, source_run: Path) -> None:
    """Link completed formal/tradeoff outputs. Leave medium and analysis new."""

    run_root = run_root.expanduser()
    source_run = source_run.expanduser()
    if not run_root.is_absolute():
        run_root = REPO_ROOT / run_root
    if not source_run.is_absolute():
        source_run = REPO_ROOT / source_run
    run_root = run_root.resolve()
    source_run = source_run.resolve()
    published = str(run_root).replace("\\", "/")
    if published.rstrip("/").endswith("case_study/outputs") or "/case_study/outputs/" in published + "/":
        return

    if run_root == source_run or source_run in run_root.parents:
        raise SystemExit(
            "Refusing to prepare a run root inside the source run. "
            f"run_root={run_root} source={source_run}"
        )
    text = str(run_root).replace("\\", "/")
    if text.rstrip("/").endswith("case_study/outputs") or "/case_study/outputs/" in text + "/":
        raise SystemExit(f"Refusing to prepare inside case_study/outputs: {run_root}")

    for relative in REUSE_LINKS:
        _ensure_dir_link(run_root / relative, source_run / relative)

    medium_dir = run_root / "03_simulation" / "sensitivity_medium"
    if medium_dir.is_symlink():
        raise SystemExit(
            f"{medium_dir} is a symlink. Medium sensitivity must be a real directory."
        )
    medium_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    argv = sys.argv
    _ensure_flag(argv, "--medium-reaction-list", [str(MEDIUM_LIST)])
    _ensure_flag(argv, "--output-root", [str(RUN)])
    _ensure_flag(argv, "--analysis-dir", [str(RUN / "04_analysis")])
    _ensure_flag(argv, "--sensitivity", ["medium", "tradeoff"])
    _ensure_flag(
        argv,
        "--medium-bounds",
        ["1000", "619", "383", "237", "146", "90", "56", "34", "13", "8", "5"],
    )
    _ensure_flag(argv, "--solve-timeout", ["1800"])
    _ensure_flag(argv, "--metrics", ["eai", "arb", "ac"])

    run_root = Path(_flag_value(argv, "--output-root") or str(RUN))
    prepare_run_root(run_root, SOURCE_RUN)
    return run_pipeline(CASE_STUDY)


if __name__ == "__main__":
    raise SystemExit(main())
