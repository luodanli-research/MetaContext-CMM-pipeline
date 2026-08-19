"""Shared utilities for MICOM sensitivity simulations."""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from micom import load_pickle
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


MIN_GROWTH = 1e-7
PFBA = True
DEFAULT_MEDIUM_BOUND = 1000.0
MEDIUM_BOUND_ATOL = 1e-9


def sensitivity_progress_columns() -> tuple:
    """Shared rich Progress columns for medium / tradeoff / reaction."""

    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
    )


def make_sensitivity_progress() -> Progress:
    """Create a Progress instance matching sensitivity_medium.py."""

    return Progress(*sensitivity_progress_columns())


def _load_formal_simulation_module():
    """Load the formal simulator so sensitivity runs use identical rules."""

    path = (
        Path(__file__).resolve().parents[1]
        / "simulation"
        / "simulate_community_model.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Formal simulation module is missing: {path}")
    spec = importlib.util.spec_from_file_location(
        "_micom310_formal_simulation",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import formal simulation module: {path}")
    module = importlib.util.module_from_spec(spec)
    module_dir = str(path.parent)
    inserted = module_dir not in sys.path
    if inserted:
        sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(module_dir)
    return module


FORMAL_SIMULATION = _load_formal_simulation_module()


def resolve_samples(values: Iterable[str]) -> list[str]:
    samples = list(dict.fromkeys(str(value) for value in values))
    if not samples:
        raise ValueError("At least one --sample is required.")
    return samples


def require_directory(path: str | Path, label: str) -> Path:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} does not exist: {directory}")
    return directory


def require_file(path: str | Path, label: str) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {file_path}")
    return file_path


def find_baseline_cmm(baseline_cmm_dir: str | Path, sample: str) -> Path:
    directory = require_directory(
        baseline_cmm_dir, "Baseline CMM directory"
    )
    candidates = []
    for pattern in (f"{sample}-bsl.pickle", f"{sample}-ori.pickle"):
        candidate = directory / pattern
        if candidate.is_file():
            candidates.append(candidate)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one bsl/ori CMM for {sample} in {directory}; "
            f"found {len(candidates)}."
        )
    return candidates[0]


def find_context_cmm(simulation_dir: str | Path, sample: str) -> Path:
    directory = require_directory(simulation_dir, "Simulation directory")
    candidate = directory / f"{sample}-ctx.pickle"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Context community pickle is missing: {candidate}"
        )
    return candidate


def find_context_file(context_dir: str | Path, sample: str) -> Path:
    directory = require_directory(context_dir, "Context directory")
    candidate = directory / "context_bounds.csv"
    if not candidate.is_file():
        raise FileNotFoundError(f"Context table is missing: {candidate}")
    return candidate


def find_formal_flux(
    simulation_dir: str | Path,
    sample: str,
    mode: str,
) -> Path:
    directory = require_directory(simulation_dir, "Simulation directory")
    candidate = directory / f"{sample}-{mode}-flux.csv"
    if not candidate.is_file():
        raise FileNotFoundError(f"Formal {mode} flux is missing: {candidate}")
    return candidate


def locate_formal_ctx_flux(
    sample: str,
    *,
    context_cmm_dir: str | Path,
    baseline_flux_dir: str | Path | None = None,
) -> Path | None:
    """Locate formal ``{sample}-ctx-flux.csv`` beside the ctx pickle or in the
    paired simulation flux directory.

    Formal naming is ``{sample}-ctx-flux.csv`` (not
    ``{sample}-bound…-ctx-flux.csv``). Prefer the ctx-model folder first, then
    ``baseline_flux_dir`` (pipeline passes ``03_simulation/<filter>/``).
    """

    candidates: list[Path] = [
        Path(context_cmm_dir).expanduser().resolve()
        / f"{sample}-ctx-flux.csv"
    ]
    if baseline_flux_dir is not None:
        candidates.append(
            Path(baseline_flux_dir).expanduser().resolve()
            / f"{sample}-ctx-flux.csv"
        )
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def community_medium_series(community) -> pd.Series:
    return pd.Series(dict(community.medium), dtype=float)


def compress_listed_medium(
    community_medium: pd.Series,
    reactions: list[str],
    bound: float,
    *,
    default_bound: float = DEFAULT_MEDIUM_BOUND,
    atol: float = MEDIUM_BOUND_ATOL,
) -> tuple[pd.Series, dict[str, int]]:
    """Copy a CMM medium and compress listed exchanges that are currently 1000.

    Only reactions present on ``community_medium`` with uptake ≈ ``default_bound``
    are rewritten to ``bound``. Missing or non-1000 listed compounds are left
    unchanged (they are not invented from the CSV).
    """

    updated = community_medium.copy()
    applied = not_present = not_default = 0
    for reaction in reactions:
        original = updated.get(reaction, np.nan)
        if pd.isna(original):
            not_present += 1
            continue
        if not np.isclose(float(original), default_bound, atol=atol):
            not_default += 1
            continue
        updated.loc[reaction] = bound
        applied += 1
    return updated, {
        "applied": applied,
        "not_present": not_present,
        "not_default": not_default,
    }


# Backward-compatible alias (historical name operated on a medium Series).
prepare_bound_medium = compress_listed_medium


def bounds_include_default(
    bounds: Iterable[float],
    *,
    default_bound: float = DEFAULT_MEDIUM_BOUND,
    atol: float = MEDIUM_BOUND_ATOL,
) -> bool:
    return any(np.isclose(float(b), default_bound, atol=atol) for b in bounds)


def ensure_bound1000_from_formal(
    sample: str,
    *,
    context_cmm_dir: str | Path,
    baseline_flux_dir: str | Path | None,
    output_prefix: Path,
    force: bool = False,
) -> dict[str, object]:
    """Materialize bound1000 from formal ctx flux copy or by solving ctx CMM.

    Preference order:
    1. copy ``{sample}-ctx-flux.csv`` from the ctx-model folder or
       ``baseline_flux_dir``;
    2. otherwise load ``{sample}-ctx.pickle`` and solve cooperative tradeoff
       without changing medium.
    """

    flux_out = output_prefix.with_name(output_prefix.name + "-flux.csv")
    if not force and flux_out.is_file() and flux_out.stat().st_size > 0:
        return {
            "status": "reused",
            "flux_file": flux_out.name,
        }

    formal_flux = locate_formal_ctx_flux(
        sample,
        context_cmm_dir=context_cmm_dir,
        baseline_flux_dir=baseline_flux_dir,
    )
    if formal_flux is not None:
        copy_file_atomic(formal_flux, flux_out)
        return {
            "status": "reused_formal_flux",
            "formal_flux_source": str(formal_flux),
            "flux_file": flux_out.name,
        }

    community = load_context_community(context_cmm_dir, sample)
    try:
        solution = run_tradeoff(community, fraction=1.0)
        save_solution(solution, output_prefix)
    finally:
        del community
    return {
        "status": "computed_from_formal_ctx",
        "flux_file": flux_out.name,
    }


def load_bsl_community(baseline_cmm_dir: str | Path, sample: str):
    return load_pickle(find_baseline_cmm(baseline_cmm_dir, sample))


def load_context_community(simulation_dir: str | Path, sample: str):
    return load_pickle(find_context_cmm(simulation_dir, sample))


def load_sample_medium_data(
    medium_file: str | Path,
    sample: str,
) -> pd.Series:
    """Load medium using the formal simulation's validation."""

    return FORMAL_SIMULATION.load_sample_medium(
        require_file(medium_file, "Medium file"),
        sample,
    )


def load_sample_context_data(
    context_dir: str | Path,
    sample: str,
) -> pd.DataFrame:
    """Load RIPTiDe context bounds for one sample from context_bounds.csv."""

    return FORMAL_SIMULATION.load_sample_context(
        find_context_file(context_dir, sample),
        sample,
    )


def build_fresh_context_community(
    baseline_cmm_dir: str | Path,
    sample: str,
    sample_medium: pd.Series,
    sample_context: pd.DataFrame,
):
    """Build ctx strictly as bsl CMM -> medium -> context bounds."""

    community = load_bsl_community(baseline_cmm_dir, sample)
    medium_summary = FORMAL_SIMULATION.apply_sample_medium(
        community,
        sample_medium,
    )
    context_summary = FORMAL_SIMULATION.apply_context_bounds(
        community,
        sample_context,
    )
    return community, medium_summary, context_summary


def load_fresh_context_community(
    baseline_cmm_dir: str | Path,
    medium_file: str | Path,
    context_dir: str | Path,
    sample: str,
):
    """Build a fresh formal-equivalent context community."""

    medium = load_sample_medium_data(medium_file, sample)
    context = load_sample_context_data(context_dir, sample)
    return build_fresh_context_community(
        baseline_cmm_dir, sample, medium, context
    )


def run_tradeoff(community, fraction: float):
    solution = FORMAL_SIMULATION.solve_cooperative_tradeoff(
        community,
        min_growth=MIN_GROWTH,
        fraction=float(fraction),
        fluxes=True,
        pfba=PFBA,
        mode=f"tradeoff/{fraction:g}",
    )
    if (
        solution is None
        or getattr(solution, "fluxes", None) is None
        or getattr(solution, "members", None) is None
    ):
        raise RuntimeError("MICOM returned an incomplete solution.")
    return solution


def run_tradeoff_many(
    community,
    fractions: Iterable[float],
) -> dict[float, object]:
    """Solve all fractions in one MICOM cooperative-tradeoff call."""

    values = list(dict.fromkeys(float(value) for value in fractions))
    if not values:
        raise ValueError("At least one tradeoff fraction is required.")
    result = FORMAL_SIMULATION.solve_cooperative_tradeoff(
        community,
        min_growth=MIN_GROWTH,
        fraction=np.asarray(values, dtype=float),
        fluxes=True,
        pfba=PFBA,
        mode="tradeoff_batch",
    )
    if not isinstance(result, pd.DataFrame):
        if len(values) != 1:
            raise RuntimeError("MICOM returned one solution for multiple fractions.")
        return {values[0]: result}

    solutions: dict[float, object] = {}
    for row in result.itertuples(index=False):
        solution = row.solution
        if (
            solution is None
            or solution.fluxes is None
            or solution.members is None
        ):
            raise RuntimeError(
                f"Incomplete solution for fraction {row.tradeoff}."
            )
        solutions[float(row.tradeoff)] = solution
    missing = [
        value
        for value in values
        if not any(np.isclose(value, observed) for observed in solutions)
    ]
    if missing:
        raise RuntimeError(f"MICOM omitted tradeoff fractions: {missing}")
    return solutions


def _atomic_dataframe(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=True)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def copy_file_atomic(source: str | Path, target: str | Path) -> Path:
    """Copy one existing file atomically."""

    source_path = require_file(source, "Source file")
    target_path = Path(target).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, target_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target_path


def save_solution(solution, output_prefix: Path) -> dict[str, Path]:
    outputs = {
        "flux": output_prefix.with_name(output_prefix.name + "-flux.csv"),
    }
    _atomic_dataframe(solution.fluxes, outputs["flux"])
    return outputs


def solution_complete(output_prefix: Path) -> bool:
    path = output_prefix.with_name(output_prefix.name + "-flux.csv")
    return path.is_file() and path.stat().st_size > 0


def read_one_column_list(path: str | Path) -> list[str]:
    file_path = require_file(path, "Reaction list")
    with file_path.open(encoding="utf-8-sig") as handle:
        values = [line.strip().split("\t", 1)[0] for line in handle]
    values = [value for value in values if value]
    if not values:
        raise ValueError(f"Reaction list is empty: {file_path}")
    return list(dict.fromkeys(values))


def write_table_atomic(table: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output
