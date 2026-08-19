"""Shared single-dataset and batch-dataset discovery for metric workflows.

Every metric treats the immediate subdirectories physically present under a
batch path as its members. Simulation parameters and producer manifests never
restrict analysis discovery.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CTX_SUFFIX = "-ctx-flux.csv"
BSL_SUFFIX = "-bsl-flux.csv"
FILTER_METHOD_NAMES = {
    "0.01",
    "consensus",
    "discordance",
    "fixed",
    "gates",
    "gmm",
    "kde-valley",
    "kmeans",
    "mad",
    "mad1.5",
    "mad2",
    "mad2.5",
    "mad3",
    "min-clear",
    "otsu",
    "q5",
    "q10",
    "q15",
    "q20",
    "quantile",
}


@dataclass(frozen=True)
class Dataset:
    """One user-supplied metric dataset."""

    kind: str
    path: Path
    label: str
    sensitivity_type: str | None = None


@dataclass(frozen=True)
class FluxJob:
    """One sample/member flux pair discovered inside a dataset."""

    dataset: str
    input_type: str
    member: str
    sample: str
    ctx_flux: Path
    bsl_flux: Path | None
    sensitivity_type: str | None = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_context: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--single-dir",
        nargs="+",
        type=Path,
        default=[],
        help=(
            "One or more directories containing flux files directly. "
            "Each directory name becomes its dataset label."
        ),
    )
    parser.add_argument(
        "--batch-dir",
        nargs="+",
        type=Path,
        default=[],
        help=(
            "One or more directories whose immediate subdirectories contain "
            "flux files. The parent name is the dataset label and each child "
            "name is the member label. All immediate child directories are "
            "analyzed; simulation gradient arguments and manifests are ignored."
        ),
    )
    parser.add_argument(
        "--sample",
        nargs="+",
        help="Optional sample IDs; otherwise discover them from ctx flux names.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Final directory for this metric run. The caller is responsible "
            "for creating any metric or analysis-group hierarchy."
        ),
    )
    if require_context:
        parser.add_argument(
            "--context-cmm-dir",
            required=True,
            type=Path,
            help=(
                "Directory with <sample>-ctx.pickle (and optional "
                "<sample>-abundances.csv sidecars used by metrics_ac)."
            ),
        )
    return parser


def analysis_output_dir(args: argparse.Namespace) -> Path:
    """Return the caller-supplied final analysis directory."""

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    return output


def _infer_sensitivity_type(path: Path) -> str | None:
    """Infer a standard sensitivity type from path or member names."""

    for part in reversed(path.parts):
        lowered = part.lower()
        if "reaction" in lowered or "perturbation" in lowered:
            return "reaction"
        if "tradeoff" in lowered:
            return "tradeoff"
        if "medium" in lowered:
            return "medium"
    children = [
        child.name.lower()
        for child in path.iterdir()
        if child.is_dir()
    ]
    if children and all(name.startswith("bound") for name in children):
        return "medium"
    if children and all(name.startswith("tradeoff") for name in children):
        return "tradeoff"
    if len(FILTER_METHOD_NAMES.intersection(children)) >= 2:
        return "filter"
    return None


def _resolve_directories(paths: Iterable[Path], kind: str) -> list[Dataset]:
    datasets: list[Dataset] = []
    for input_path in paths:
        path = input_path.expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"{kind} dataset does not exist: {path}")
        datasets.append(
            Dataset(
                kind=kind,
                path=path,
                label=path.name,
                sensitivity_type=(
                    _infer_sensitivity_type(path) if kind == "batch" else None
                ),
            )
        )
    return datasets


def parse_datasets(args: argparse.Namespace) -> list[Dataset]:
    datasets = [
        *_resolve_directories(args.single_dir, "single"),
        *_resolve_directories(args.batch_dir, "batch"),
    ]
    if not datasets:
        raise ValueError("Provide at least one --single-dir or --batch-dir.")
    labels = [dataset.label for dataset in datasets]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            "Dataset directory names must be unique: " + ", ".join(duplicates)
        )
    return datasets


def sample_from_flux(path: Path) -> str:
    name = path.name
    if not name.endswith(CTX_SUFFIX):
        raise ValueError(f"Not a ctx flux file: {path}")
    prefix = name[: -len(CTX_SUFFIX)]
    match = re.match(r"^([^-]+)", prefix)
    if match is None:
        raise ValueError(f"Cannot infer sample ID from {path.name}")
    return match.group(1)


def _one_flux(directory: Path, sample: str, simulation_type: str) -> Path | None:
    suffix = f"-{simulation_type}-flux.csv"
    exact = directory / f"{sample}{suffix}"
    if exact.is_file():
        return exact.resolve()
    matches = sorted(
        path.resolve()
        for path in directory.glob(f"{sample}*{suffix}")
        if path.is_file()
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple {simulation_type} flux files for {sample} in {directory}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def _reaction_shared_bsl(dataset_path: Path, sample: str) -> Path | None:
    """Resolve the shared formal bsl used by reaction sensitivity members.

    Prefer the exact formal name ``<sample>-bsl-flux.csv``. Search order matches
    current public layout then legacy flat simulation roots:

    1. ``<simulation-root>/context/`` (GE formal)
    2. ``<simulation-root>/ga/`` (GA formal, if present)
    3. ``<simulation-root>/`` itself (legacy)
    """

    parent = dataset_path.parent
    search_dirs = (
        parent / "context",
        parent / "ga",
        parent,
    )
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        exact = directory / f"{sample}{BSL_SUFFIX}"
        if exact.is_file():
            return exact.resolve()
    for directory in search_dirs:
        if directory.is_dir():
            found = _one_flux(directory, sample, "bsl")
            if found is not None:
                return found
    return None


def discover_jobs(
    datasets: list[Dataset],
    requested_samples: list[str] | None,
    *,
    require_bsl: bool,
) -> tuple[list[FluxJob], list[str]]:
    allowed = set(requested_samples or [])
    jobs: list[FluxJob] = []
    discovered_samples: list[str] = []
    for dataset in datasets:
        members = (
            [(dataset.label, dataset.path)]
            if dataset.kind == "single"
            else [
                (path.name, path)
                for path in sorted(dataset.path.iterdir())
                if path.is_dir()
            ]
        )
        if dataset.kind == "batch" and not members:
            raise FileNotFoundError(
                f"Batch dataset has no subdirectories: {dataset.path}"
            )
        dataset_jobs = 0
        for member, directory in members:
            if (
                dataset.sensitivity_type == "reaction"
                and member.upper() == "S000"
            ):
                continue
            ctx_files = sorted(
                path.resolve()
                for path in directory.glob(f"*{CTX_SUFFIX}")
                if path.is_file()
            )
            for ctx_flux in ctx_files:
                sample = sample_from_flux(ctx_flux)
                if allowed and sample not in allowed:
                    continue
                bsl_flux = _one_flux(directory, sample, "bsl")
                # Reaction batches share the formal GE/GA bsl (no per-Sxxx bsl).
                if (
                    bsl_flux is None
                    and dataset.kind == "batch"
                    and dataset.sensitivity_type == "reaction"
                ):
                    bsl_flux = _reaction_shared_bsl(dataset.path, sample)
                if require_bsl and bsl_flux is None:
                    raise FileNotFoundError(
                        f"No bsl flux found for {sample}/{dataset.label}/{member}. "
                        "Place it beside the ctx flux, or provide formal "
                        f"{sample}-bsl-flux.csv under context/ or ga/ next to "
                        "the reaction batch."
                    )
                jobs.append(
                    FluxJob(
                        dataset=dataset.label,
                        input_type=dataset.kind,
                        member=member,
                        sample=sample,
                        ctx_flux=ctx_flux,
                        bsl_flux=bsl_flux,
                        sensitivity_type=dataset.sensitivity_type,
                    )
                )
                dataset_jobs += 1
                if sample not in discovered_samples:
                    discovered_samples.append(sample)
        if dataset_jobs == 0:
            raise FileNotFoundError(
                f"No matching ctx flux files found in {dataset.path}"
            )

    sample_order = list(requested_samples or discovered_samples)
    missing = [
        sample
        for sample in sample_order
        if not any(job.sample == sample for job in jobs)
    ]
    if missing:
        raise FileNotFoundError(
            "No ctx flux data found for requested sample(s): "
            + ", ".join(missing)
        )
    return jobs, sample_order


def partition_metric_rows(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically separate single rows from batch rows."""

    if "input_type" not in table.columns:
        raise ValueError("Metric table lacks the required input_type column.")
    unexpected = sorted(
        set(table["input_type"].dropna().astype(str)) - {"single", "batch"}
    )
    if unexpected:
        raise ValueError(
            "Unexpected metric input_type value(s): " + ", ".join(unexpected)
        )
    return (
        table[table["input_type"].eq("single")].copy(),
        table[table["input_type"].eq("batch")].copy(),
    )


def single_value_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return single-dataset values without statistical aggregation."""

    singles, _ = partition_metric_rows(table)
    if singles.empty:
        return singles
    keys = ["dataset", "sample", "metric"]
    duplicates = singles.duplicated(keys, keep=False)
    if duplicates.any():
        labels = (
            singles.loc[duplicates, keys]
            .drop_duplicates()
            .astype(str)
            .agg("/".join, axis=1)
            .tolist()
        )
        raise ValueError(
            "A single dataset must contain one value per sample/metric: "
            + ", ".join(labels)
        )
    return singles.sort_values(keys).reset_index(drop=True)


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    repetitions: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a reproducible percentile-bootstrap CI for the median."""

    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        value = float(clean[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(repetitions, len(clean)))
    medians = np.median(clean[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(medians, alpha)),
        float(np.quantile(medians, 1.0 - alpha)),
    )


def summarize_batch_values(table: pd.DataFrame) -> pd.DataFrame:
    """Calculate IQR or CI from batch members only, never from singles."""

    columns = [
        "dataset",
        "input_type",
        "sensitivity_type",
        "sample",
        "metric",
        "n",
        "center_statistic",
        "center",
        "interval_type",
        "lower",
        "upper",
    ]
    rows: list[dict[str, object]] = []
    _, batches = partition_metric_rows(table)
    if batches.empty:
        return pd.DataFrame(columns=columns)
    if "sensitivity_type" not in batches.columns:
        raise ValueError("Batch metric table lacks sensitivity_type.")
    unknown = batches[batches["sensitivity_type"].isna()]
    if not unknown.empty:
        labels = sorted(unknown["dataset"].astype(str).unique())
        raise ValueError(
            "Cannot infer sensitivity type for batch dataset(s): "
            + ", ".join(labels)
            + ". Include filter, medium, tradeoff, or reaction in the "
            "batch path."
        )

    keys = ["dataset", "sensitivity_type", "sample", "metric"]
    for key, group in batches.groupby(keys, sort=False, dropna=False):
        values = (
            pd.to_numeric(group["value"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        sensitivity_type = str(key[1])
        center = float(np.median(values)) if len(values) else np.nan
        if sensitivity_type in {"filter", "medium", "tradeoff"}:
            interval_type = "IQR"
            lower = float(np.quantile(values, 0.25)) if len(values) else np.nan
            upper = float(np.quantile(values, 0.75)) if len(values) else np.nan
        elif sensitivity_type == "reaction":
            interval_type = "bootstrap_CI95"
            lower, upper = _bootstrap_median_ci(values)
        else:
            raise ValueError(
                f"Unsupported sensitivity type: {sensitivity_type!r}"
            )
        rows.append(
            {
                "dataset": key[0],
                "input_type": "batch",
                "sensitivity_type": sensitivity_type,
                "sample": key[2],
                "metric": key[3],
                "n": len(values),
                "center_statistic": "median",
                "center": center,
                "interval_type": interval_type,
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(rows, columns=columns)
