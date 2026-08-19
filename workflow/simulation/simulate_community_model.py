#!/usr/bin/env python3
"""Run baseline and context-specific MICOM community simulations.

This pipeline always runs two independent cooperative-tradeoff simulations:

``bsl``
    The baseline ``<sample>-bsl.pickle`` community without additional
    environmental or transcriptomic constraints.

``ctx``
    A freshly loaded baseline community with both the sample-specific medium
    and sample-specific RIPTiDe reaction bounds applied.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from micom import load_pickle
from micom.community import Community
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from _cli_utils import configure_logging


LOGGER = logging.getLogger("cmm.simulate")
MIN_GROWTH = 1e-7
TRADEOFF_FRACTION = 1.0
PFBA = True
# After an infeasible first solve, retry with these solver tolerances.
# Feasibility is set together with optimality: probes on consensus/SW50
# showed feasibility (not MICOM atol/rtol alone) rescues the solve.
INFEASIBLE_TOLERANCE_RETRIES = (1e-7, 1e-5)

REQUIRED_MEDIUM_COLUMNS = {"reaction", "flux", "sample_id"}
REQUIRED_CONTEXT_COLUMNS = {
    "gem_rxn_id",
    "lb",
    "ub",
    "sample_id",
    "mag_id",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed bsl and context-specific MICOM cooperative-tradeoff "
            "simulations for one sample."
        )
    )
    parser.add_argument(
        "--sample",
        required=True,
        help="Sample identifier, for example SW60.",
    )
    parser.add_argument(
        "--baseline-cmm-dir",
        dest="baseline_cmm_dir",
        required=True,
        type=Path,
        help="Directory containing <sample>-bsl.pickle.",
    )
    parser.add_argument(
        "--context-file",
        dest="context_file",
        required=True,
        type=Path,
        help=(
            "Integrated RIPTiDe context-bounds CSV. The constrained community "
            "pickle is written beside this file."
        ),
    )
    parser.add_argument(
        "--medium-file",
        dest="medium_file",
        required=True,
        type=Path,
        help="Integrated medium CSV containing a sample_id column.",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        type=Path,
        help=(
            "Directory in which to write bsl and ctx simulation result tables. "
            "The ctx model is written beside --context-file."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs from MICOM and its dependencies.",
    )
    return parser.parse_args(argv)


def load_baseline_community(
    baseline_cmm_dir: Path,
    sample: str,
) -> Community:
    """Load a fresh baseline community for one simulation mode."""
    model_path = baseline_cmm_dir.resolve() / f"{sample}-bsl.pickle"
    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline community not found: {model_path}")
    LOGGER.debug("Loading baseline community: %s", model_path)
    return load_pickle(model_path)


def load_sample_medium(medium_file: Path, sample: str) -> pd.Series:
    """Load and validate the sample-specific environmental medium."""
    medium_file = medium_file.resolve()
    if not medium_file.is_file():
        raise FileNotFoundError(f"Medium file not found: {medium_file}")

    medium_table = pd.read_csv(medium_file)
    missing_columns = REQUIRED_MEDIUM_COLUMNS.difference(medium_table.columns)
    if missing_columns:
        raise ValueError(
            "Missing required medium column(s): "
            + ", ".join(sorted(missing_columns))
        )

    sample_medium = medium_table.loc[
        medium_table["sample_id"].eq(sample),
        ["reaction", "flux"],
    ].copy()
    if sample_medium.empty:
        available_samples = sorted(
            medium_table["sample_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            f"Sample {sample!r} is absent from {medium_file}. "
            f"Available samples: {', '.join(available_samples)}"
        )
    if sample_medium["reaction"].isna().any():
        raise ValueError(f"Missing reaction identifiers for sample {sample}.")
    if sample_medium["reaction"].duplicated().any():
        duplicates = sorted(
            sample_medium.loc[
                sample_medium["reaction"].duplicated(keep=False), "reaction"
            ].astype(str).unique()
        )
        raise ValueError(
            f"Duplicate medium reactions for sample {sample}: "
            + ", ".join(duplicates[:10])
        )

    sample_medium["flux"] = pd.to_numeric(
        sample_medium["flux"], errors="coerce"
    )
    invalid_mask = sample_medium["flux"].isna() | ~np.isfinite(
        sample_medium["flux"]
    )
    if invalid_mask.any():
        invalid_reactions = sample_medium.loc[invalid_mask, "reaction"].astype(str)
        raise ValueError(
            f"Non-numeric or non-finite medium flux for sample {sample}: "
            + ", ".join(invalid_reactions.head(10))
        )
    if (sample_medium["flux"] < 0).any():
        negative_reactions = sample_medium.loc[
            sample_medium["flux"] < 0, "reaction"
        ].astype(str)
        raise ValueError(
            f"Negative medium flux for sample {sample}: "
            + ", ".join(negative_reactions.head(10))
        )

    return sample_medium.set_index("reaction")["flux"].astype(float)


def load_sample_context(context_file: Path, sample: str) -> pd.DataFrame:
    """Load and validate sample-specific RIPTiDe reaction bounds."""
    context_file = context_file.resolve()
    if not context_file.is_file():
        raise FileNotFoundError(f"Context file not found: {context_file}")

    context_table = pd.read_csv(context_file)
    missing_columns = REQUIRED_CONTEXT_COLUMNS.difference(context_table.columns)
    if missing_columns:
        raise ValueError(
            "Missing required context column(s): "
            + ", ".join(sorted(missing_columns))
        )

    sample_context = context_table.loc[
        context_table["sample_id"].eq(sample),
        ["gem_rxn_id", "lb", "ub", "mag_id"],
    ].copy()
    if sample_context.empty:
        available_samples = sorted(
            context_table["sample_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            f"Sample {sample!r} is absent from {context_file}. "
            f"Available samples: {', '.join(available_samples)}"
        )
    if sample_context[["gem_rxn_id", "mag_id"]].isna().any().any():
        raise ValueError(
            f"Missing gem_rxn_id or mag_id values for sample {sample}."
        )
    if sample_context.duplicated(["mag_id", "gem_rxn_id"]).any():
        duplicates = sample_context.loc[
            sample_context.duplicated(
                ["mag_id", "gem_rxn_id"], keep=False
            ),
            ["mag_id", "gem_rxn_id"],
        ]
        preview = ", ".join(
            f"{row.mag_id}/{row.gem_rxn_id}"
            for row in duplicates.head(10).itertuples(index=False)
        )
        raise ValueError(f"Duplicate context bounds: {preview}")

    for column in ("lb", "ub"):
        sample_context[column] = pd.to_numeric(
            sample_context[column], errors="coerce"
        )
    invalid_mask = (
        sample_context[["lb", "ub"]].isna().any(axis=1)
        | ~np.isfinite(sample_context[["lb", "ub"]]).all(axis=1)
    )
    if invalid_mask.any():
        invalid_rows = sample_context.loc[
            invalid_mask, ["mag_id", "gem_rxn_id"]
        ]
        preview = ", ".join(
            f"{row.mag_id}/{row.gem_rxn_id}"
            for row in invalid_rows.head(10).itertuples(index=False)
        )
        raise ValueError(f"Invalid context bounds: {preview}")
    invalid_order = sample_context["lb"] > sample_context["ub"]
    if invalid_order.any():
        invalid_rows = sample_context.loc[
            invalid_order, ["mag_id", "gem_rxn_id"]
        ]
        preview = ", ".join(
            f"{row.mag_id}/{row.gem_rxn_id}"
            for row in invalid_rows.head(10).itertuples(index=False)
        )
        raise ValueError(f"Context lower bound exceeds upper bound: {preview}")

    return sample_context


def context_bounds(
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float] | None:
    """Convert RIPTiDe FVA bounds for community context simulation."""
    if lower_bound >= 0 and 0 < upper_bound < 1000:
        return 0.0, upper_bound
    if -1000 < lower_bound < 0 and upper_bound <= 0:
        return lower_bound, 0.0
    if -1000 < lower_bound < 0 < upper_bound < 1000:
        return lower_bound, upper_bound
    if lower_bound == 0 and upper_bound == 0:
        return 0.0, 0.0
    return None


def apply_context_bounds(
    community: Community,
    sample_context: pd.DataFrame,
) -> dict[str, int]:
    """Apply RIPTiDe context bounds to a community."""
    community_taxa = {str(taxon) for taxon in community.taxa}
    summary = {
        "context_rows": len(sample_context),
        "noncommunity_rows": 0,
        "exchange_or_sink_rows": 0,
        "growth_constraints": 0,
        "gpr_constraints": 0,
        "unchanged_rows": 0,
    }

    for row in sample_context.itertuples(index=False):
        mag_id = str(row.mag_id)
        gem_rxn_id = str(row.gem_rxn_id)
        if mag_id not in community_taxa:
            summary["noncommunity_rows"] += 1
            continue
        if gem_rxn_id.startswith(("EX_", "sink_")):
            summary["exchange_or_sink_rows"] += 1
            continue

        community_rxn_id = f"{gem_rxn_id}__{mag_id}"
        try:
            reaction = community.reactions.get_by_id(community_rxn_id)
        except KeyError as error:
            raise KeyError(
                f"Context reaction is absent from community: {community_rxn_id}"
            ) from error

        lower_bound = float(row.lb)
        upper_bound = float(row.ub)
        if reaction.id.startswith("Growth"):
            reaction.bounds = (0.0, upper_bound)
            summary["growth_constraints"] += 1
            continue
        if not reaction.gene_reaction_rule:
            summary["unchanged_rows"] += 1
            continue

        new_bounds = context_bounds(lower_bound, upper_bound)
        if new_bounds is None:
            summary["unchanged_rows"] += 1
            continue
        reaction.bounds = new_bounds
        summary["gpr_constraints"] += 1

    return summary


def apply_sample_medium(
    community: Community,
    sample_medium: pd.Series,
) -> dict[str, object]:
    """Apply an environmental medium and report unavailable exchanges."""
    community.medium = sample_medium
    applied_medium = pd.Series(community.medium, dtype=float)
    requested_positive = sample_medium[sample_medium > 0]
    unavailable = sorted(
        set(requested_positive.index).difference(applied_medium.index)
    )
    return {
        "requested_rows": len(sample_medium),
        "requested_positive_rows": len(requested_positive),
        "applied_positive_rows": len(applied_medium),
        "unavailable_reactions": unavailable,
    }


def _looks_infeasible(error: BaseException) -> bool:
    """Return True for solver failures that warrant a tolerance retry."""
    text = f"{type(error).__name__}: {error}".lower()
    markers = (
        "infeasible",
        "could not get community growth rate",
        "could not get optimum",
        "incomplete solution",
        "no solution exists",
        "1217",
    )
    return any(marker in text for marker in markers)


def _set_solver_tolerances(community: Community, tolerance: float) -> None:
    """Set CPLEX/optlang feasibility and optimality tolerances together."""
    community.solver.configuration.tolerances.feasibility = float(tolerance)
    community.solver.configuration.tolerances.optimality = float(tolerance)


def solve_cooperative_tradeoff(
    community: Community,
    *,
    min_growth: float = MIN_GROWTH,
    fraction: float | Sequence[float] = TRADEOFF_FRACTION,
    fluxes: bool = True,
    pfba: bool = PFBA,
    mode: str = "simulation",
):
    """Run cooperative tradeoff, retrying infeasible solves with looser/tighter tol.

    Attempt order:
    1. current solver tolerances (MICOM default feasibility ``1e-6``);
    2. feasibility/optimality ``1e-7``;
    3. feasibility/optimality ``1e-5``.
    """
    community.solver.configuration.verbosity = 0
    attempts: list[float | None] = [None, *INFEASIBLE_TOLERANCE_RETRIES]
    last_error: BaseException | None = None

    for attempt_index, tolerance in enumerate(attempts, start=1):
        if tolerance is not None:
            LOGGER.warning(
                "[%s] infeasible on attempt %d/%d; retrying with "
                "solver feasibility/optimality=%g.",
                mode,
                attempt_index,
                len(attempts),
                tolerance,
            )
            _set_solver_tolerances(community, tolerance)

        try:
            kwargs = {
                "min_growth": min_growth,
                "fraction": fraction,
                "fluxes": fluxes,
                "pfba": pfba,
            }
            if tolerance is not None:
                kwargs["atol"] = float(tolerance)
                kwargs["rtol"] = float(tolerance)
            solution = community.cooperative_tradeoff(**kwargs)
            if (
                solution is None
                or getattr(solution, "fluxes", None) is None
                or getattr(solution, "members", None) is None
            ):
                # Vector tradeoff returns a DataFrame; accept that shape too.
                if isinstance(solution, pd.DataFrame):
                    return solution
                raise RuntimeError(
                    f"{mode} simulation returned an incomplete solution."
                )
            if tolerance is not None:
                LOGGER.info(
                    "[%s] recovered after tolerance retry (%g).",
                    mode,
                    tolerance,
                )
            return solution
        except Exception as error:
            last_error = error
            if (
                not _looks_infeasible(error)
                or attempt_index >= len(attempts)
            ):
                raise
            LOGGER.warning(
                "[%s] attempt %d/%d failed (%s); trying next tolerance.",
                mode,
                attempt_index,
                len(attempts),
                error,
            )

    assert last_error is not None
    raise last_error


def run_cooperative_tradeoff(community: Community, mode: str):
    """Run the fixed cooperative-tradeoff simulation."""
    LOGGER.debug(
        "Running %s cooperative tradeoff: min_growth=%g, fraction=%g, pfba=%s.",
        mode,
        MIN_GROWTH,
        TRADEOFF_FRACTION,
        PFBA,
    )
    return solve_cooperative_tradeoff(
        community,
        min_growth=MIN_GROWTH,
        fraction=TRADEOFF_FRACTION,
        fluxes=True,
        pfba=PFBA,
        mode=mode,
    )


def atomic_to_csv(table: pd.DataFrame, output_path: Path) -> None:
    """Write a DataFrame atomically."""
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        table.to_csv(temporary_path, index=True)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def save_solution(solution, sample: str, mode: str, out_dir: Path) -> None:
    """Save one MICOM solution as its complete flux table."""
    atomic_to_csv(solution.fluxes, out_dir / f"{sample}-{mode}-flux.csv")


def save_context_model(
    community: Community,
    sample: str,
    context_dir: Path,
) -> Path:
    """Save the constrained context community atomically."""
    context_dir.mkdir(parents=True, exist_ok=True)
    output_path = context_dir / f"{sample}-ctx.pickle"
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        community.to_pickle(temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    abundance_path = context_dir / f"{sample}-abundances.csv"
    abundance_table = (
        community.abundances.rename("abundance")
        .rename_axis("taxon")
        .reset_index()
    )
    temporary_abundance = abundance_path.with_name(f".{abundance_path.name}.tmp")
    try:
        abundance_table.to_csv(temporary_abundance, index=False)
        os.replace(temporary_abundance, abundance_path)
    except Exception:
        temporary_abundance.unlink(missing_ok=True)
        raise
    return output_path


def run_pipeline(
    sample: str,
    baseline_cmm_dir: Path,
    context_file: Path,
    medium_file: Path,
    out_dir: Path,
) -> None:
    """Run independent bsl and fully constrained ctx simulations."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    context_dir = context_file.resolve().parent
    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[step]}"),
    )

    with Progress(*progress_columns) as progress:
        task = progress.add_task(
            f"[{sample}] Simulation",
            total=6,
            step="loading bsl CMM",
        )
        bsl_community = load_baseline_community(baseline_cmm_dir, sample)
        bsl_taxa = len(bsl_community.taxa)
        progress.update(task, advance=1, step="solving bsl")

        bsl_solution = run_cooperative_tradeoff(bsl_community, mode="bsl")
        progress.update(task, advance=1, step="saving bsl")
        save_solution(bsl_solution, sample, mode="bsl", out_dir=out_dir)
        bsl_growth = float(bsl_solution.growth_rate)
        del bsl_solution
        del bsl_community
        progress.update(task, advance=1, step="loading ctx CMM")

        ctx_community = load_baseline_community(baseline_cmm_dir, sample)
        sample_medium = load_sample_medium(medium_file, sample)
        medium_summary = apply_sample_medium(ctx_community, sample_medium)
        sample_context = load_sample_context(context_file, sample)
        context_summary = apply_context_bounds(ctx_community, sample_context)
        progress.update(task, advance=1, step="solving ctx")

        ctx_solution = run_cooperative_tradeoff(ctx_community, mode="ctx")
        progress.update(task, advance=1, step="saving ctx")
        save_solution(ctx_solution, sample, mode="ctx", out_dir=out_dir)
        ctx_growth = float(ctx_solution.growth_rate)
        context_model_path = save_context_model(
            ctx_community,
            sample,
            context_dir,
        )
        progress.update(task, advance=1, step="complete")

    LOGGER.info(
        (
            "[%s] Simulation complete: %d taxa; bsl growth=%.10g; "
            "ctx growth=%.10g; medium=%d/%d positive exchanges applied; "
            "context=%d growth + %d GPR constraints; results=%s; model=%s"
        ),
        sample,
        bsl_taxa,
        bsl_growth,
        ctx_growth,
        medium_summary["applied_positive_rows"],
        medium_summary["requested_positive_rows"],
        context_summary["growth_constraints"],
        context_summary["gpr_constraints"],
        out_dir,
        context_model_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    run_pipeline(
        sample=args.sample,
        baseline_cmm_dir=args.baseline_cmm_dir,
        context_file=args.context_file,
        medium_file=args.medium_file,
        out_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
