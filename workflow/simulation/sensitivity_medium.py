#!/usr/bin/env python3
"""Run medium-bound sensitivity via ``*-ctx.pickle`` load + medium edit.

Constraint-loading contract:

1. If ``1000`` is in ``--bounds``: copy formal ``{sample}-ctx-flux.csv`` into
   ``bound1000/``, or solve the loaded ctx CMM unchanged when no flux exists.
2. For every other bound: load ``{sample}-ctx.pickle``, copy
   ``community.medium``, compress listed exchanges that are currently 1000 to
   the scenario bound, assign, then cooperative tradeoff.

Note: compressing medium on a loaded ctx pickle applies medium *after*
context. A previous fresh-rebuild implementation is archived under ``bak/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _sensitivity_simulation import (
    DEFAULT_MEDIUM_BOUND,
    bounds_include_default,
    community_medium_series,
    compress_listed_medium,
    copy_file_atomic,
    ensure_bound1000_from_formal,
    find_baseline_cmm,
    find_context_cmm,
    load_bsl_community,
    load_context_community,
    make_sensitivity_progress,
    read_one_column_list,
    require_file,
    require_directory,
    resolve_samples,
    run_tradeoff,
    save_solution,
    solution_complete,
    write_table_atomic,
)


DEFAULT_BOUNDS = (
    5.0,
    10.0,
    50.0,
    100.0,
    200.0,
    400.0,
    600.0,
    800.0,
    1000.0,
)
DEFAULT_BOUND = DEFAULT_MEDIUM_BOUND


def label_bound(value: float) -> str:
    return f"bound{value:g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ctx medium-bound sensitivity from <sample>-ctx.pickle. "
            "bound1000 reuses formal {sample}-ctx-flux.csv or solves ctx "
            "unchanged; other bounds compress listed community.medium "
            "exchanges that are currently 1000."
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
        help="Directory with <sample>-bsl.pickle (used to materialize bsl fluxes).",
    )
    parser.add_argument(
        "--medium-file",
        type=Path,
        required=True,
        help="Integrated medium CSV with a sample_id column.",
    )
    parser.add_argument(
        "--context-cmm-dir",
        type=Path,
        required=True,
        help="Directory with <sample>-ctx.pickle starting models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Final output directory for this sensitivity run.",
    )
    parser.add_argument(
        "--reaction-list",
        type=Path,
        required=True,
        help="CSV listing community exchange reactions to compress.",
    )
    parser.add_argument(
        "--baseline-flux-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with <sample>-bsl-flux.csv and formal "
            "<sample>-ctx-flux.csv (typically 03_simulation/context/)."
        ),
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs="+",
        default=list(DEFAULT_BOUNDS),
        help=(
            "Uptake bounds in (0, 1000]; defaults to "
            "5 10 50 100 200 400 600 800 1000."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun scenarios even when flux outputs already exist.",
    )
    return parser


def resolve_baseline_fluxes(
    directory: Path | None,
    samples: list[str],
) -> dict[str, Path]:
    if directory is None:
        return {}
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            f"Baseline flux directory does not exist: {root}"
        )
    resolved: dict[str, Path] = {}
    for sample in samples:
        flux_file = root / f"{sample}-bsl-flux.csv"
        if not flux_file.is_file():
            raise FileNotFoundError(
                f"Baseline flux file is missing for {sample}: {flux_file}"
            )
        if flux_file.stat().st_size == 0:
            raise ValueError(f"Baseline flux file is empty: {flux_file}")
        resolved[sample] = flux_file.resolve()
    return resolved


def flux_path(prefix: Path) -> Path:
    return prefix.with_name(prefix.name + "-flux.csv")


def is_default_bound(value: float) -> bool:
    return bool(np.isclose(float(value), DEFAULT_BOUND))


def run(args: argparse.Namespace) -> None:
    samples = resolve_samples(args.sample)
    baseline_cmm_dir = require_directory(
        args.baseline_cmm_dir, "Baseline CMM directory"
    )
    require_file(args.medium_file, "Medium file")
    context_cmm_dir = require_directory(
        args.context_cmm_dir, "Context CMM directory"
    )
    reactions = read_one_column_list(args.reaction_list)
    output_dir = args.output_dir.expanduser().resolve()
    bounds = list(dict.fromkeys(float(value) for value in args.bounds))
    if any(
        not np.isfinite(value) or value <= 0 or value > DEFAULT_BOUND
        for value in bounds
    ):
        raise ValueError("Every --bounds value must be in (0, 1000].")

    for sample in samples:
        find_baseline_cmm(baseline_cmm_dir, sample)
        find_context_cmm(context_cmm_dir, sample)
    baseline_fluxes = resolve_baseline_fluxes(
        args.baseline_flux_dir, samples
    )
    first_scenario = label_bound(bounds[0])
    include_bound1000 = bounds_include_default(bounds)
    compress_bounds = [b for b in bounds if not is_default_bound(b)]

    n_bsl = len(samples)
    n_ctx = len(samples) * (
        int(include_bound1000) + len(compress_bounds)
    )

    with make_sensitivity_progress() as progress:
        bsl_task = progress.add_task(
            "medium bsl",
            total=n_bsl,
            status="preparing",
        )
        for sample in samples:
            first_prefix = (
                output_dir
                / first_scenario
                / f"{sample}-{first_scenario}-bsl"
            )
            progress.update(
                bsl_task,
                description=f"[{sample}/{first_scenario}/bsl]",
                status="running",
            )
            if args.force or not solution_complete(first_prefix):
                if sample in baseline_fluxes:
                    copy_file_atomic(
                        baseline_fluxes[sample],
                        flux_path(first_prefix),
                    )
                    status = "reused input flux"
                else:
                    community = load_bsl_community(baseline_cmm_dir, sample)
                    solution = run_tradeoff(community, fraction=1.0)
                    save_solution(solution, first_prefix)
                    status = "solved"
            else:
                status = "skipped"

            first_flux = flux_path(first_prefix)
            for bound in bounds[1:]:
                scenario = label_bound(bound)
                prefix = output_dir / scenario / f"{sample}-{scenario}-bsl"
                if not args.force and solution_complete(prefix):
                    continue
                copy_file_atomic(first_flux, flux_path(prefix))
            progress.update(bsl_task, advance=1, status=status)

        manifest_rows: list[dict[str, object]] = []
        failure_rows: list[dict[str, object]] = []
        ctx_task = progress.add_task(
            "medium ctx",
            total=n_ctx,
            status="preparing",
        )
        for sample in samples:
            if include_bound1000:
                scenario = label_bound(DEFAULT_BOUND)
                prefix = (
                    output_dir / scenario / f"{sample}-{scenario}-ctx"
                )
                progress.update(
                    ctx_task,
                    description=f"[{sample}/{scenario}]",
                    status="running",
                )
                try:
                    result = ensure_bound1000_from_formal(
                        sample,
                        context_cmm_dir=context_cmm_dir,
                        baseline_flux_dir=args.baseline_flux_dir,
                        output_prefix=prefix,
                        force=args.force,
                    )
                    manifest_rows.append(
                        {
                            "sample": sample,
                            "sensitivity_type": "medium",
                            "scenario": scenario,
                            "parameter_value": DEFAULT_BOUND,
                            "mode": "ctx",
                            "status": result["status"],
                            "listed_reactions": len(reactions),
                            "formal_flux_source": result.get(
                                "formal_flux_source"
                            ),
                            "flux_file": result.get("flux_file"),
                        }
                    )
                    progress.update(
                        ctx_task,
                        advance=1,
                        status=str(result["status"]),
                    )
                except Exception as error:
                    failure_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario,
                            "parameter_value": DEFAULT_BOUND,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    progress.console.print(
                        f"[{sample}/{scenario}] failed: {error}"
                    )
                    progress.update(ctx_task, advance=1, status="failed")

            for bound in compress_bounds:
                scenario = label_bound(bound)
                prefix = (
                    output_dir / scenario / f"{sample}-{scenario}-ctx"
                )
                progress.update(
                    ctx_task,
                    description=f"[{sample}/{scenario}]",
                    status="running",
                )
                if not args.force and solution_complete(prefix):
                    manifest_rows.append(
                        {
                            "sample": sample,
                            "sensitivity_type": "medium",
                            "scenario": scenario,
                            "parameter_value": bound,
                            "mode": "ctx",
                            "status": "reused",
                            "listed_reactions": len(reactions),
                        }
                    )
                    progress.update(ctx_task, advance=1, status="skipped")
                    continue
                try:
                    community = load_context_community(
                        context_cmm_dir, sample
                    )
                    medium, stats = compress_listed_medium(
                        community_medium_series(community),
                        reactions,
                        bound,
                    )
                    community.medium = medium
                    solution = run_tradeoff(community, fraction=1.0)
                    save_solution(solution, prefix)
                    manifest_rows.append(
                        {
                            "sample": sample,
                            "sensitivity_type": "medium",
                            "scenario": scenario,
                            "parameter_value": bound,
                            "mode": "ctx",
                            "status": "computed",
                            "listed_reactions": len(reactions),
                            "applied_reactions": stats["applied"],
                            "not_in_medium": stats["not_present"],
                            "not_default_1000": stats["not_default"],
                            "flux_file": flux_path(prefix).name,
                        }
                    )
                    progress.update(ctx_task, advance=1, status="complete")
                except Exception as error:
                    failure_rows.append(
                        {
                            "sample": sample,
                            "scenario": scenario,
                            "parameter_value": bound,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    progress.console.print(
                        f"[{sample}/{scenario}] failed: {error}"
                    )
                    progress.update(ctx_task, advance=1, status="failed")

    write_table_atomic(
        pd.DataFrame(manifest_rows),
        output_dir / "00_sensitivity_medium_manifest.csv",
    )
    write_table_atomic(
        pd.DataFrame(failure_rows),
        output_dir / "00_sensitivity_medium_failures.csv",
    )
    if failure_rows:
        failure_file = output_dir / "00_sensitivity_medium_failures.csv"
        print(
            f"[WARN] {len(failure_rows)} medium-sensitivity jobs failed; "
            f"continuing with completed results. See {failure_file}"
        )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
