#!/usr/bin/env python3
"""Case-study copy: medium+reaction at tradeoff 0.5, with phosphate and iron.

Does not modify ``run_case_study.py`` or ``run_case_study_presolve_rerun.py``.

- Medium list: ``case_study/inputs/04_medium_sensitivity_list.csv``.
- Medium and reaction solves use cooperative tradeoff 0.5.
- The default command does not rerun medium or tradeoff sensitivity.
- Metric plots read this run's finished ``sensitivity_medium`` directory and
  the previous run's ``sensitivity_tradeoff`` directory.

Formal baseline, context pickles, and formal fluxes are symlinked read-only
from the finished metacontext run. Medium, reaction, and analysis are new
directories.

Example::

    python -u pipeline/run_case_study_pi_fe_tradeoff05.py
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
RUN = Path("case_study/outputs/tradeoff0.5")
MEDIUM_LIST = Path("case_study/inputs/04_medium_sensitivity_list.csv")
PRIOR_TRADEOFF = SOURCE_RUN / "03_simulation" / "sensitivity_tradeoff"
PRIOR_MEDIUM = RUN / "03_simulation" / "sensitivity_medium"

REUSE_LINKS = (
    "01_baseline",
    "02_context",
    "03_simulation/context",
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
    """Link formal outputs. Leave medium, reaction, and analysis as new dirs."""

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

    for relative in (
        "03_simulation/sensitivity_medium",
        "03_simulation/sensitivity_reaction",
        "03_simulation/sensitivity_tradeoff",
    ):
        path = run_root / relative
        if path.is_symlink():
            raise SystemExit(
                f"{path} is a symlink. This run must not write through "
                "the finished tradeoff/medium/reaction tree."
            )


def main() -> int:
    argv = sys.argv
    _ensure_flag(argv, "--medium-reaction-list", [str(MEDIUM_LIST)])
    _ensure_flag(argv, "--output-root", [str(RUN)])
    _ensure_flag(argv, "--analysis-dir", [str(RUN / "04_analysis")])
    _ensure_flag(argv, "--sensitivity", ["reaction"])
    _ensure_flag(
        argv,
        "--medium-bounds",
        ["1000", "619", "383", "237", "146", "90", "56", "34", "13", "8", "5"],
    )
    _ensure_flag(argv, "--solve-tradeoff", ["0.5"])
    _ensure_flag(argv, "--prior-medium-dir", [str(PRIOR_MEDIUM)])
    _ensure_flag(argv, "--prior-tradeoff-dir", [str(PRIOR_TRADEOFF)])
    _ensure_flag(argv, "--solve-timeout", ["1800"])
    _ensure_flag(argv, "--metrics", ["eai", "arb", "ac"])

    run_root = Path(_flag_value(argv, "--output-root") or str(RUN))
    prepare_run_root(run_root, SOURCE_RUN)
    return run_pipeline(CASE_STUDY)


if __name__ == "__main__":
    raise SystemExit(main())
