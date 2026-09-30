#!/usr/bin/env python3
"""Run cooperative-tradeoff fraction sensitivity simulations.

Tradeoff fraction is a MICOM runtime parameter, not a model-constraint change.
Each pending ``(sample, mode, fraction)`` job loads an independent community
instance, solves once, then discards the object:

* ``bsl``: load ``<sample>-bsl.pickle`` from ``--baseline-cmm-dir``
* ``ctx``: load pre-solve ``<sample>-ctx.pickle`` from ``--context-cmm-dir``

There is no silent rebuild from medium/bounds when the ctx pickle is missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _sensitivity_simulation import (
    adopt_failure_table,
    find_baseline_cmm,
    find_context_cmm,
    load_bsl_community,
    load_context_community,
    make_sensitivity_progress,
    record_failure,
    recorded_failure,
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
            "Run bsl and ctx cooperative-tradeoff sensitivity. Each fraction "
            "loads an independent community (bsl pickle or pre-solve ctx "
            "pickle) and solves with a scalar cooperative_tradeoff call."
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
        default=None,
        help=(
            "Unused; kept for CLI compatibility with older callers. Ctx mode "
            "loads pre-solve <sample>-ctx.pickle and does not rebuild from medium."
        ),
    )
    parser.add_argument(
        "--context-cmm-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing pre-solve <sample>-ctx.pickle written by "
            "formal simulation before the first ctx solve. Missing pickles "
            "raise an error (no silent rebuild)."
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
        help="Rerun fractions even when a flux file or a recorded failure already exists.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    samples = resolve_samples(args.sample)
    baseline_cmm_dir = require_directory(
        args.baseline_cmm_dir, "Baseline CMM directory"
    )
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
    if not args.force:
        adopt_failure_table(
            output_dir,
            "00_sensitivity_tradeoff_failures.csv",
        )

    for sample in samples:
        find_baseline_cmm(baseline_cmm_dir, sample)
        find_context_cmm(context_cmm_dir, sample)

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
                                "community_source": (
                                    "bsl_pickle" if mode == "bsl" else "ctx_pickle"
                                ),
                                "fresh_load": False,
                            }
                        )
                        progress.update(
                            task_id, advance=1, status="skipped"
                        )
                        continue
                    recorded = None if args.force else recorded_failure(prefix)
                    if recorded is not None:
                        error_type, error_message = recorded
                        failure_rows.append(
                            {
                                "sample": sample,
                                "scenario": scenario,
                                "parameter_value": fraction,
                                "mode": mode,
                                "error_type": error_type,
                                "error_message": error_message,
                            }
                        )
                        manifest_rows.append(
                            {
                                "sample": sample,
                                "sensitivity_type": "tradeoff",
                                "scenario": scenario,
                                "parameter_value": fraction,
                                "mode": mode,
                                "status": "skipped_failed",
                                "batch_fraction_count": len(fractions),
                                "community_source": (
                                    "bsl_pickle" if mode == "bsl" else "ctx_pickle"
                                ),
                                "fresh_load": False,
                            }
                        )
                        progress.update(
                            task_id, advance=1, status="skipped_failed"
                        )
                        continue
                    community = None
                    try:
                        if mode == "bsl":
                            community = load_bsl_community(
                                baseline_cmm_dir, sample
                            )
                            community_source = "bsl_pickle"
                        else:
                            community = load_context_community(
                                context_cmm_dir, sample
                            )
                            community_source = "ctx_pickle"
                        solution = run_tradeoff(community, fraction)
                        save_solution(solution, prefix)
                        manifest_rows.append(
                            {
                                "sample": sample,
                                "sensitivity_type": "tradeoff",
                                "scenario": scenario,
                                "parameter_value": fraction,
                                "mode": mode,
                                "status": "computed",
                                "batch_fraction_count": len(fractions),
                                "community_source": community_source,
                                "fresh_load": True,
                            }
                        )
                        progress.update(
                            task_id, advance=1, status="complete"
                        )
                    except Exception as error:
                        record_failure(
                            prefix, type(error).__name__, str(error)
                        )
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
                            f"[{sample}/{mode}/{scenario}] failed: {error}"
                        )
                        progress.update(
                            task_id, advance=1, status="failed"
                        )
                    finally:
                        if community is not None:
                            del community

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
