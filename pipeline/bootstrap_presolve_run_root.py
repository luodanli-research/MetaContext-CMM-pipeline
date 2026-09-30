#!/usr/bin/env python3
"""Bootstrap an isolated pre-solve run root (no model solves).

Creates::

    case_study/runs/presolve_unified_YYYYMMDD/
    case_study/runs/presolve_unified_YYYYMMDD_analysis/

- Symlinks ``01_baseline`` to the published baseline (read-only reuse).
- **Copies** ``context_bounds.csv`` and QC CSVs into ``02_context/`` as
  regular files (never symlinks bounds — ``simulate_community_model`` writes
  ``*-ctx.pickle`` beside the bounds path's lexical parent).

Does not write under ``case_study/outputs/``.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


CONTEXT_TABLES = (
    "context_bounds.csv",
    "context_species.csv",
    "context_expression.csv",
    "context_summary.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case-study-root",
        type=Path,
        default=Path("case_study"),
        help="Case-study data root (default: case_study).",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Run stamp YYYYMMDD or YYYYMMDD_HHMM (default: today).",
    )
    parser.add_argument(
        "--source-output",
        type=Path,
        default=None,
        help="Published outputs tree to copy/symlink from (default: <root>/outputs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_root = args.case_study_root.expanduser().resolve()
    source = (
        args.source_output.expanduser().resolve()
        if args.source_output is not None
        else case_root / "outputs"
    )
    stamp = args.stamp or datetime.now().strftime("%Y%m%d")
    run_root = case_root / "runs" / f"presolve_unified_{stamp}"
    analysis_root = case_root / "runs" / f"presolve_unified_{stamp}_analysis"

    if run_root.resolve() == source.resolve() or source in run_root.resolve().parents:
        raise SystemExit(
            "Refusing to bootstrap into the published outputs tree. "
            f"run_root={run_root} source={source}"
        )

    src_baseline = source / "01_baseline"
    src_context = source / "02_context"
    if not src_baseline.is_dir():
        raise SystemExit(f"Missing baseline directory: {src_baseline}")
    if not (src_context / "context_bounds.csv").is_file():
        raise SystemExit(f"Missing context_bounds.csv under {src_context}")

    run_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    (run_root / "03_simulation" / "context").mkdir(parents=True, exist_ok=True)

    baseline_link = run_root / "01_baseline"
    if baseline_link.is_symlink() or baseline_link.exists():
        if not baseline_link.is_symlink():
            raise SystemExit(
                f"{baseline_link} exists and is not a symlink; refuse to replace."
            )
        baseline_link.unlink()
    baseline_link.symlink_to(src_baseline.resolve(), target_is_directory=True)

    context_dir = run_root / "02_context"
    context_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in CONTEXT_TABLES:
        src = src_context / name
        if not src.is_file():
            continue
        dst = context_dir / name
        shutil.copy2(src, dst)
        if dst.is_symlink():
            raise SystemExit(f"Refusing symlink context table: {dst}")
        copied.append(name)

    if "context_bounds.csv" not in copied:
        raise SystemExit("context_bounds.csv was not copied.")

    print(f"[OK] run root: {run_root}")
    print(f"[OK] analysis root: {analysis_root}")
    print(f"[OK] baseline symlink -> {src_baseline.resolve()}")
    print(f"[OK] copied context tables: {', '.join(copied)}")
    print(
        "\nExample pipeline invocation:\n"
        f"  python pipeline/run_case_study.py \\\n"
        f"    --output-root {run_root} \\\n"
        f"    --analysis-dir {analysis_root} \\\n"
        f"    --sample SW46 --sensitivity tradeoff --tradeoffs 1.0\n"
    )
    print(
        "Note: baseline directory is symlinked read-only for loads; "
        "formal/sensitivity must not write into 01_baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
