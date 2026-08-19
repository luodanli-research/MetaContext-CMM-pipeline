#!/usr/bin/env python3
"""Run cooperative-tradeoff fraction sensitivity simulations.

Tradeoff fraction is a MICOM runtime parameter, not a model-constraint change.
Unlike medium or reaction sensitivity, successive fractions do not need a fresh
community for isolation: each sample loads bsl once, or rebuilds ctx once from
``bsl pickle -> medium -> context_bounds.csv``, then reuses that community for
every pending fraction. Each fraction is solved with a scalar
``cooperative_tradeoff`` call (not MICOM's multi-fraction vector API).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _sensitivity_simulation import (
    find_baseline_cmm,
    find_context_file,
    load_fresh_context_community,
    load_bsl_community,
    make_sensitivity_progress,
    require_file,
    require_directory,
    resolve_samples,
    run_tradeoff,
    save_solution,
    solution_complete,
    write_table_atomic,
)


# Case-study defaults; users can supply any unique values in (0, 1].
DEFAULT_TRADEOFFS = tuple(round(value / 100, 2) for value in range(5, 101, 5))


def label_fraction(value: float) -> str:
    """Folder/file tag for a tradeoff fraction (1 → tradeoff1, 0.1 → tradeoff0.1)."""

    return f"tradeoff{float(value):g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run bsl and ctx cooperative-tradeoff sensitivity. Tradeoff "
            "fraction is a runtime parameter, so each sample loads the bsl "
            "pickle once, or rebuilds ctx once from the bsl pickle plus "
            "medium and context_bounds.csv, then reuses that community across "
            "all requested fractions. Each fraction uses a scalar "
            "cooperative_tradeoff call. Does not load <sample>-ctx.pickle."
        )
    )
    parser.add_argument(
        "--sample",
        nargs="+",
        required=True,
        help="One or more sample IDs to process.",
    )
    parser.add_argument(
        "--baseline-cmm-dir",
        type=Path,
        required=True,
        help="Directory with <sample>-bsl.pickle (or -ori.pickle) models.",
    )
    parser.add_argument(
        "--medium-file",
        type=Path,
        required=True,
        help="Medium table used when rebuilding ctx (ignored for bsl mode).",
    )
    parser.add_argument(
        "--context-cmm-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing context_bounds.csv used to rebuild ctx. "
            "Existing <sample>-ctx.pickle files in this directory are ignored."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Final output directory for this sensitivity run.",
    )
    parser.add_argument(
        "--tradeoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_TRADEOFFS),
        help="Tradeoff values in (0, 1]; defaults to 0.05, 0.10, ..., 1.00.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun fractions even when flux outputs already exist.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    samples = resolve_samples(args.sample)
    baseline_cmm_dir = require_directory(
        args.baseline_cmm_dir, "Baseline CMM directory"
    )
    medium_file = require_file(args.medium_file, "Medium file")
    context_cmm_dir = require_directory(
        args.context_cmm_dir, "Context CMM directory"
    )
    output_dir = args.output_dir.expanduser().resolve()
    fractions = list(dict.fromkeys(float(value) for value in args.tradeoffs))
    if any(
        not np.isfinite(value) or value <= 0 or value > 1
        for value in fractions
    ):
        raise ValueError("Every --tradeoffs value must be in (0, 1].")

    for sample in samples:
        find_baseline_cmm(baseline_cmm_dir, sample)
        find_context_file(context_cmm_dir, sample)

    manifest_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    n_jobs = len(samples) * len(fractions)

    with make_sensitivity_progress() as progress:
        tasks = {
            "bsl": progress.add_task(
                "tradeoff bsl", total=n_jobs, status="preparing"
            ),
            "ctx": progress.add_task(
                "tradeoff ctx", total=n_jobs, status="preparing"
            ),
        }
        for sample in samples:
            for mode in ("bsl", "ctx"):
                task_id = tasks[mode]
                community = None
                medium_summary = context_summary = None
                batch_failed = False
                batch_error: BaseException | None = None
                for fraction in fractions:
                    scenario = label_fraction(fraction)
                    prefix = (
                        output_dir
                        / scenario
                        / f"{sample}-{scenario}-{mode}"
                    )
                    progress.update(
                        task_id,
                        description=f"[{sample}/{scenario}/{mode}]",
                        status="running",
                    )
                    if not args.force and solution_complete(prefix):
                        manifest_rows.append(
                            {
                                "sample": sample,
                                "sensitivity_type": "tradeoff",
                                "scenario": scenario,
                                "parameter_value": fraction,
                                "mode": mode,
                                "status": "reused",
                                "batch_fraction_count": len(fractions),
                            }
                        )
                        progress.update(
                            task_id, advance=1, status="skipped"
                        )
                        continue
                    if batch_failed:
                        assert batch_error is not None
                        failure_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario,
                                "parameter_value": fraction,
                                "mode": mode,
                                "error_type": type(batch_error).__name__,
                                "error_message": str(batch_error),
                            }
                        )
                        progress.update(
                            task_id, advance=1, status="failed"
                        )
                        continue
                    try:
                        # One community per sample/mode for all fractions.
                        if community is None:
                            if mode == "bsl":
                                community = load_bsl_community(
                                    baseline_cmm_dir, sample
                                )
                            else:
                                (
                                    community,
                                    medium_summary,
                                    context_summary,
                                ) = load_fresh_context_community(
                                    baseline_cmm_dir,
                                    medium_file,
                                    context_cmm_dir,
                                    sample,
                                )
                        solution = run_tradeoff(community, fraction)
                        save_solution(solution, prefix)
                        manifest_row: dict[str, object] = {
                            "sample": sample,
                            "sensitivity_type": "tradeoff",
                            "scenario": scenario,
                            "parameter_value": fraction,
                            "mode": mode,
                            "status": "computed",
                            "batch_fraction_count": len(fractions),
                        }
                        if medium_summary is not None:
                            manifest_row.update(
                                {
                                    "medium_positive_applied": medium_summary[
                                        "applied_positive_rows"
                                    ],
                                    "context_growth_constraints": (
                                        context_summary[
                                            "growth_constraints"
                                        ]
                                    ),
                                    "context_gpr_constraints": (
                                        context_summary["gpr_constraints"]
                                    ),
                                }
                            )
                        manifest_rows.append(manifest_row)
                        progress.update(
                            task_id, advance=1, status="complete"
                        )
                    except Exception as error:
                        batch_failed = True
                        batch_error = error
                        failure_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario,
                                "parameter_value": fraction,
                                "mode": mode,
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                            }
                        )
                        progress.console.print(
                            f"[{sample}/{mode}] failed: {error}"
                        )
                        progress.update(
                            task_id, advance=1, status="failed"
                        )

    write_table_atomic(
        pd.DataFrame(manifest_rows),
        output_dir / "00_sensitivity_tradeoff_manifest.csv",
    )
    write_table_atomic(
        pd.DataFrame(failure_rows),
        output_dir / "00_sensitivity_tradeoff_failures.csv",
    )
    if failure_rows:
        failure_file = output_dir / "00_sensitivity_tradeoff_failures.csv"
        print(
            f"[WARN] {len(failure_rows)} tradeoff-sensitivity jobs failed; "
            f"continuing with completed results. See {failure_file}"
        )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
