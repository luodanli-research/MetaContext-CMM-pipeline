#!/usr/bin/env python3
"""Generate and simulate deterministic RIPTiDe reaction-bound sensitivity.

Aligned with the parent ANA3-1 / ANA3-2 workflow:

* generate: sample intervals from ``context_bounds.csv`` (RIPTiDe FVA role);
  use ``<sample>-ctx.pickle`` only for community reaction membership.
* simulate: each realization ``load_pickle(ctx)`` → apply LHS bounds →
  cooperative tradeoff (no per-realization bsl/medium/context rebuild).

Each sample/realization solve runs in an isolated child process so a native
solver crash (for example SIGSEGV / exit -11) fails only that job and the
parent can retry or continue.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from _sensitivity_simulation import (
    find_context_cmm,
    find_context_file,
    load_context_community,
    load_sample_context_data,
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


SIGN_EPS = 1e-12
FILE_PREFIX = "LHS_sample"
SCRIPT_PATH = Path(__file__).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LHS reaction bounds and simulate each realization by "
            "loading <sample>-ctx.pickle from --context-cmm-dir. Simulation "
            "is realization-major: for each Sxxx, all requested samples are "
            "solved before the next realization."
        )
    )
    parser.add_argument(
        "--sample",
        nargs="+",
        required=True,
        help="One or more sample IDs to process.",
    )
    parser.add_argument(
        "--context-cmm-dir",
        type=Path,
        required=True,
        help=(
            "Directory with <sample>-ctx.pickle (starting model) and "
            "context_bounds.csv (RIPTiDe intervals for LHS generation)."
        ),
    )
    parser.add_argument(
        "--baseline-cmm-dir",
        type=Path,
        default=None,
        help="Unused; kept for CLI compatibility with older callers.",
    )
    parser.add_argument(
        "--medium-file",
        type=Path,
        default=None,
        help="Unused; kept for CLI compatibility with older callers.",
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
        default=None,
        help="CSV listing GEM reaction families eligible for LHS perturbation.",
    )
    parser.add_argument(
        "--stage",
        choices=("generate", "simulate", "all"),
        default="all",
        help="Run LHS generation, simulation, or both (default: all).",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=100,
        help="Number of LHS realizations to generate/simulate (default: 100).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=5.0,
        help=(
            "Percentage of eligible reaction families perturbed in each "
            "realization; must be in (0, 100)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for deterministic LHS sampling (default: 42).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Solver retries after transient failures (default: 2).",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=1.0,
        help="Seconds to wait between solver retries (default: 1.0).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Parallel sample workers within each realization (default: 1). "
            "Each job runs in an isolated child process."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate LHS tables and rerun simulations even if outputs exist.",
    )
    # Internal worker entry used by simulate_one_isolated (not a public CLI).
    parser.add_argument(
        "--_simulate-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_realization-id",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_result-file",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _exit_label(returncode: int) -> str:
    """Human-readable label for a child return code (signals are negative)."""

    if returncode < 0:
        signum = -returncode
        try:
            return signal.Signals(signum).name
        except ValueError:
            return f"signal_{signum}"
    return f"exit_{returncode}"


def classify_interval(lb: float, ub: float) -> str:
    if not np.isfinite(lb) or not np.isfinite(ub) or ub < lb:
        return "invalid"
    if lb > SIGN_EPS and ub > SIGN_EPS:
        return "pos"
    if lb < -SIGN_EPS and ub < -SIGN_EPS:
        return "neg"
    if lb < -SIGN_EPS and ub > SIGN_EPS:
        return "cross"
    return "invalid"


def perturbed_bounds(
    interval_type: str,
    lower_bound: float,
    upper_bound: float,
    quantile: float,
) -> tuple[float, float] | None:
    if interval_type == "pos":
        return 0.0, lower_bound + quantile * (upper_bound - lower_bound)
    if interval_type == "neg":
        return lower_bound + quantile * (upper_bound - lower_bound), 0.0
    if interval_type == "cross":
        return (
            lower_bound * (1.0 - quantile),
            upper_bound * (1.0 - quantile),
        )
    return None


def lhs_1d(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 0:
        return np.array([], dtype=float)
    cuts = np.linspace(0.0, 1.0, size + 1)
    values = cuts[:-1] + rng.random(size) * np.diff(cuts)
    rng.shuffle(values)
    return values


def eligible_reactions(
    community,
    reaction_list: list[str],
    sample_context: pd.DataFrame,
) -> pd.DataFrame:
    """Use RIPTiDe intervals from context_bounds; membership from ctx pickle."""

    selected = set(reaction_list)
    community_reactions = {str(reaction.id) for reaction in community.reactions}
    rows = []
    for row in sample_context.itertuples(index=False):
        base_id = str(row.gem_rxn_id)
        mag_id = str(row.mag_id)
        if base_id not in selected:
            continue
        reaction_id = f"{base_id}__{mag_id}"
        if reaction_id not in community_reactions:
            continue
        lb = float(row.lb)
        ub = float(row.ub)
        interval_type = classify_interval(lb, ub)
        if interval_type == "invalid":
            continue
        rows.append(
            {
                "reaction_id": base_id,
                "mag_id": mag_id,
                "cmm_reaction_id": reaction_id,
                "fva_lb": lb,
                "fva_ub": ub,
                "interval_type": interval_type,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError(
            "No listed reaction family has a non-zero valid interval in "
            "the context community."
        )
    family_counts = (
        table.groupby("reaction_id")["mag_id"]
        .nunique()
        .rename("n_mags_with_reaction")
    )
    return table.merge(family_counts, on="reaction_id", how="left")


def generate_realization(
    eligible: pd.DataFrame,
    realization_id: int,
    perturb_fraction: float,
    base_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = sorted(eligible["reaction_id"].unique())
    count = min(
        len(families),
        max(1, int(math.ceil(len(families) * perturb_fraction))),
    )
    rng = np.random.default_rng(base_seed + realization_id)
    chosen = rng.choice(families, size=count, replace=False).tolist()
    family_quantiles = dict(zip(chosen, lhs_1d(count, rng)))
    output_rows = []
    detail_rows = []
    for row in eligible[eligible["reaction_id"].isin(chosen)].itertuples(
        index=False
    ):
        quantile = float(family_quantiles[row.reaction_id])
        bounds = perturbed_bounds(
            row.interval_type,
            float(row.fva_lb),
            float(row.fva_ub),
            quantile,
        )
        if bounds is None:
            continue
        lb, ub = bounds
        output_rows.append(
            {"cmm_reaction_id": row.cmm_reaction_id, "lb": lb, "ub": ub}
        )
        detail_rows.append(
            {
                "realization_id": realization_id,
                "seed": base_seed + realization_id,
                "reaction_id": row.reaction_id,
                "mag_id": row.mag_id,
                "cmm_reaction_id": row.cmm_reaction_id,
                "shared_quantile": quantile,
                "fva_lb": row.fva_lb,
                "fva_ub": row.fva_ub,
                "interval_type": row.interval_type,
                "n_mags_with_reaction": row.n_mags_with_reaction,
                "sampled_lb": lb,
                "sampled_ub": ub,
            }
        )
    return pd.DataFrame(output_rows), pd.DataFrame(detail_rows)


def generate_for_sample(
    sample: str,
    context_dir: Path,
    lhs_root: Path,
    reactions: list[str],
    n_realizations: int,
    perturb_fraction: float,
    seed: int,
    force: bool,
) -> None:
    # Parent ANA3-1: load ctx for membership; intervals from RIPTiDe table.
    community = load_context_community(context_dir, sample)
    sample_context = load_sample_context_data(context_dir, sample)
    eligible = eligible_reactions(community, reactions, sample_context)
    sample_dir = lhs_root / sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    detail_frames = []

    # S000 is the mandatory unchanged-bound control through the same pipeline.
    control = sample_dir / f"{FILE_PREFIX}_S000.csv"
    if force or not control.is_file() or control.stat().st_size == 0:
        write_table_atomic(
            pd.DataFrame(columns=["cmm_reaction_id", "lb", "ub"]),
            control,
        )
    manifest_rows.append(
        {
            "sample": sample,
            "realization_id": 0,
            "seed": np.nan,
            "fraction_percent": 0.0,
            "eligible_reaction_families": eligible[
                "reaction_id"
            ].nunique(),
            "selected_reaction_families": 0,
            "selected_cmm_reactions": 0,
            "output_file": control.name,
        }
    )

    for realization_id in range(1, n_realizations + 1):
        output = sample_dir / f"{FILE_PREFIX}_S{realization_id:03d}.csv"
        if output.is_file() and output.stat().st_size > 0 and not force:
            continue
        bounds, detail = generate_realization(
            eligible,
            realization_id,
            perturb_fraction,
            seed,
        )
        write_table_atomic(bounds, output)
        manifest_rows.append(
            {
                "sample": sample,
                "realization_id": realization_id,
                "seed": seed + realization_id,
                "fraction_percent": perturb_fraction * 100.0,
                "eligible_reaction_families": eligible[
                    "reaction_id"
                ].nunique(),
                "selected_reaction_families": detail[
                    "reaction_id"
                ].nunique(),
                "selected_cmm_reactions": len(bounds),
                "output_file": output.name,
            }
        )
        detail_frames.append(detail)
    write_table_atomic(
        pd.DataFrame(manifest_rows),
        sample_dir / "LHS_sample_manifest.csv",
    )
    detail_table = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )
    write_table_atomic(detail_table, sample_dir / "LHS_sample_detail.csv")


def apply_bounds(community, table: pd.DataFrame) -> dict[str, int]:
    applied = missing = invalid = 0
    for row in table.itertuples(index=False):
        lb = float(row.lb)
        ub = float(row.ub)
        if not np.isfinite(lb) or not np.isfinite(ub) or ub < lb:
            invalid += 1
            continue
        try:
            reaction = community.reactions.get_by_id(str(row.cmm_reaction_id))
        except KeyError:
            missing += 1
            continue
        reaction.bounds = (lb, ub)
        applied += 1
    return {"applied": applied, "missing": missing, "invalid": invalid}


def classify_error(error: Exception) -> str:
    message = str(error).lower()
    if "infeasible" in message:
        return "infeasible"
    if "unbounded" in message:
        return "unbounded"
    if "license" in message:
        return "solver_license"
    return type(error).__name__


def discover_realization_ids(lhs_root: Path, samples: list[str]) -> list[int]:
    """Union of LHS realization IDs across samples, sorted ascending."""

    realization_ids: set[int] = set()
    for sample in samples:
        sample_input = lhs_root / sample
        if not sample_input.is_dir():
            raise NotADirectoryError(
                f"LHS input directory is missing: {sample_input}"
            )
        for lhs_file in sample_input.glob(f"{FILE_PREFIX}_S*.csv"):
            realization_text = lhs_file.stem.rsplit("_S", 1)[-1]
            if realization_text.isdigit():
                realization_ids.add(int(realization_text))
    if not realization_ids:
        raise FileNotFoundError(
            f"No {FILE_PREFIX}_S*.csv files found under {lhs_root}"
        )
    return sorted(realization_ids)


def _scenario_paths(
    sample: str,
    realization_id: int,
    lhs_root: Path,
    simulation_dir: Path,
) -> tuple[str, Path, Path]:
    scenario = f"S{realization_id:03d}"
    lhs_file = lhs_root / sample / f"{FILE_PREFIX}_{scenario}.csv"
    prefix = simulation_dir / scenario / f"{sample}-reaction-{scenario}-ctx"
    return scenario, lhs_file, prefix


def simulate_one_attempt(
    sample: str,
    realization_id: int,
    context_dir: Path,
    lhs_root: Path,
    simulation_dir: Path,
) -> dict[str, object]:
    """Run one sample/realization solve. Raise on validation or solver errors."""

    scenario, lhs_file, prefix = _scenario_paths(
        sample, realization_id, lhs_root, simulation_dir
    )
    if not lhs_file.is_file():
        raise FileNotFoundError(f"LHS bounds file is missing: {lhs_file}")

    community = load_context_community(context_dir, sample)
    bounds = pd.read_csv(lhs_file)
    required = {"cmm_reaction_id", "lb", "ub"}
    if not required.issubset(bounds.columns):
        raise ValueError(
            f"{lhs_file} lacks columns: "
            + ", ".join(sorted(required - set(bounds.columns)))
        )
    stats = apply_bounds(community, bounds)
    solution = run_tradeoff(community, fraction=1.0)
    outputs = save_solution(solution, prefix)
    return {
        "sample": sample,
        "realization_id": realization_id,
        "scenario": scenario,
        "status": "success",
        "growth_rate": float(solution.growth_rate),
        "n_applied": stats["applied"],
        "n_missing": stats["missing"],
        "n_invalid": stats["invalid"],
        "flux_file": outputs["flux"].name,
    }


def simulate_one(
    sample: str,
    realization_id: int,
    context_dir: Path,
    lhs_root: Path,
    simulation_dir: Path,
    max_retries: int,
    retry_wait: float,
    force: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Simulate one sample/realization in-process. Prefer simulate_one_isolated."""

    scenario, _lhs_file, prefix = _scenario_paths(
        sample, realization_id, lhs_root, simulation_dir
    )
    if not force and solution_complete(prefix):
        return (
            {
                "sample": sample,
                "realization_id": realization_id,
                "scenario": scenario,
                "status": "reused",
                "flux_file": f"{prefix.name}-flux.csv",
            },
            [],
        )

    error_rows: list[dict[str, object]] = []
    for attempt in range(max_retries + 1):
        try:
            return (
                simulate_one_attempt(
                    sample,
                    realization_id,
                    context_dir,
                    lhs_root,
                    simulation_dir,
                ),
                error_rows,
            )
        except Exception as error:
            error_rows.append(
                {
                    "sample": sample,
                    "realization_id": realization_id,
                    "attempt": attempt,
                    "error_type": classify_error(error),
                    "error_message": str(error),
                }
            )
            if attempt < max_retries:
                time.sleep(retry_wait)
    return (
        {
            "sample": sample,
            "realization_id": realization_id,
            "scenario": scenario,
            "status": "failed",
        },
        error_rows,
    )


def _run_simulate_worker(args: argparse.Namespace) -> int:
    """Child-process entry: one solve attempt, JSON result on --_result-file."""

    if args._realization_id is None or args._result_file is None:
        raise ValueError("Worker requires --_realization-id and --_result-file.")
    if len(args.sample) != 1:
        raise ValueError("Worker requires exactly one --sample.")

    sample = args.sample[0]
    context_dir = args.context_cmm_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    lhs_root = output_dir / "01_lhs"
    result_path = args._result_file.expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        summary = simulate_one_attempt(
            sample,
            args._realization_id,
            context_dir,
            lhs_root,
            output_dir,
        )
        payload = {"ok": True, "summary": summary}
    except Exception as error:
        payload = {
            "ok": False,
            "error_type": classify_error(error),
            "error_message": str(error),
        }

    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, result_path)
    return 0


def simulate_one_isolated(
    sample: str,
    realization_id: int,
    context_dir: Path,
    lhs_root: Path,
    simulation_dir: Path,
    max_retries: int,
    retry_wait: float,
    force: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Simulate one job in a child process; survive native solver crashes."""

    scenario, _lhs_file, prefix = _scenario_paths(
        sample, realization_id, lhs_root, simulation_dir
    )
    if not force and solution_complete(prefix):
        return (
            {
                "sample": sample,
                "realization_id": realization_id,
                "scenario": scenario,
                "status": "reused",
                "flux_file": f"{prefix.name}-flux.csv",
            },
            [],
        )

    error_rows: list[dict[str, object]] = []
    for attempt in range(max_retries + 1):
        with tempfile.TemporaryDirectory(prefix="reaction_job_") as tmp:
            result_file = Path(tmp) / "result.json"
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--_simulate-worker",
                "--sample",
                sample,
                "--_realization-id",
                str(realization_id),
                "--context-cmm-dir",
                str(context_dir),
                "--output-dir",
                str(simulation_dir),
                "--_result-file",
                str(result_file),
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                error_rows.append(
                    {
                        "sample": sample,
                        "realization_id": realization_id,
                        "attempt": attempt,
                        "error_type": _exit_label(completed.returncode),
                        "error_message": (
                            f"Isolated worker terminated with return code "
                            f"{completed.returncode}"
                        ),
                    }
                )
            elif not result_file.is_file():
                error_rows.append(
                    {
                        "sample": sample,
                        "realization_id": realization_id,
                        "attempt": attempt,
                        "error_type": "missing_result",
                        "error_message": "Worker exited 0 without a result file.",
                    }
                )
            else:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
                if payload.get("ok"):
                    return payload["summary"], error_rows
                error_rows.append(
                    {
                        "sample": sample,
                        "realization_id": realization_id,
                        "attempt": attempt,
                        "error_type": payload.get("error_type", "worker_error"),
                        "error_message": payload.get("error_message", ""),
                    }
                )
        if attempt < max_retries:
            time.sleep(retry_wait)

    return (
        {
            "sample": sample,
            "realization_id": realization_id,
            "scenario": scenario,
            "status": "failed",
        },
        error_rows,
    )


def write_sample_summaries(
    simulation_dir: Path,
    summary_by_sample: dict[str, list[dict[str, object]]],
    error_by_sample: dict[str, list[dict[str, object]]],
) -> None:
    for sample, rows in summary_by_sample.items():
        write_table_atomic(
            pd.DataFrame(rows),
            simulation_dir / f"00_reaction_simulation_summary_{sample}.csv",
        )
        write_table_atomic(
            pd.DataFrame(error_by_sample[sample]),
            simulation_dir / f"00_reaction_error_summary_{sample}.csv",
        )


def simulate_all_samples(
    samples: list[str],
    context_dir: Path,
    lhs_root: Path,
    simulation_dir: Path,
    max_retries: int,
    retry_wait: float,
    force: bool,
    jobs: int = 1,
) -> None:
    """Simulate realization-major: finish all samples for Sxxx, then next.

    Every sample/realization job is launched in an isolated child process so a
    native solver crash cannot abort the remaining schedule.
    """

    if jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    realization_ids = discover_realization_ids(lhs_root, samples)
    summary_by_sample = {sample: [] for sample in samples}
    error_by_sample = {sample: [] for sample in samples}
    workers = min(jobs, len(samples))
    total = len(realization_ids) * len(samples)

    with make_sensitivity_progress() as progress:
        task_id = progress.add_task(
            "reaction simulate",
            total=total,
            status="preparing",
        )
        for realization_id in realization_ids:
            scenario = f"S{realization_id:03d}"
            results: dict[
                str, tuple[dict[str, object], list[dict[str, object]]]
            ] = {}
            if workers == 1:
                for sample in samples:
                    progress.update(
                        task_id,
                        description=f"[{sample}/{scenario}]",
                        status="running",
                    )
                    results[sample] = simulate_one_isolated(
                        sample,
                        realization_id,
                        context_dir,
                        lhs_root,
                        simulation_dir,
                        max_retries,
                        retry_wait,
                        force,
                    )
                    status = str(results[sample][0].get("status", "done"))
                    progress.update(task_id, advance=1, status=status)
            else:
                progress.update(
                    task_id,
                    description=f"[{scenario} ×{len(samples)} jobs={workers}]",
                    status="running",
                )
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    future_map = {
                        pool.submit(
                            simulate_one_isolated,
                            sample,
                            realization_id,
                            context_dir,
                            lhs_root,
                            simulation_dir,
                            max_retries,
                            retry_wait,
                            force,
                        ): sample
                        for sample in samples
                    }
                    for future in as_completed(future_map):
                        sample = future_map[future]
                        results[sample] = future.result()
                        status = str(results[sample][0].get("status", "done"))
                        progress.update(
                            task_id,
                            description=f"[{sample}/{scenario}]",
                            advance=1,
                            status=status,
                        )
            for sample in samples:
                summary_row, error_rows = results[sample]
                summary_by_sample[sample].append(summary_row)
                error_by_sample[sample].extend(error_rows)
            # Flush after each realization so partial progress is inspectable.
            write_sample_summaries(
                simulation_dir, summary_by_sample, error_by_sample
            )


def run(args: argparse.Namespace) -> None:
    samples = resolve_samples(args.sample)
    context_cmm_dir = require_directory(
        args.context_cmm_dir, "Context CMM directory"
    )
    output_dir = args.output_dir.expanduser().resolve()
    if args.n_realizations < 1:
        raise ValueError("--n-realizations must be at least 1.")
    if not np.isfinite(args.fraction) or not 0 < args.fraction < 100:
        raise ValueError("--fraction must be a percentage in (0, 100).")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    if args.stage in {"generate", "all"} and args.reaction_list is None:
        raise ValueError("--reaction-list is required for LHS generation.")
    perturb_fraction = float(args.fraction) / 100.0
    for sample in samples:
        find_context_cmm(context_cmm_dir, sample)
    lhs_root = output_dir / "01_lhs"
    if args.stage in {"generate", "all"}:
        reactions = read_one_column_list(args.reaction_list)
        find_context_file(context_cmm_dir, samples[0])
        with make_sensitivity_progress() as progress:
            task_id = progress.add_task(
                "reaction generate",
                total=len(samples),
                status="preparing",
            )
            for sample in samples:
                progress.update(
                    task_id,
                    description=f"[{sample}/LHS]",
                    status="running",
                )
                generate_for_sample(
                    sample,
                    context_cmm_dir,
                    lhs_root,
                    reactions,
                    args.n_realizations,
                    perturb_fraction,
                    args.seed,
                    args.force,
                )
                progress.update(task_id, advance=1, status="complete")
    if args.stage in {"simulate", "all"}:
        simulate_all_samples(
            samples,
            context_cmm_dir,
            lhs_root,
            output_dir,
            args.max_retries,
            args.retry_wait,
            args.force,
            jobs=args.jobs,
        )


def main() -> None:
    args = build_parser().parse_args()
    if args._simulate_worker:
        raise SystemExit(_run_simulate_worker(args))
    run(args)


if __name__ == "__main__":
    main()
