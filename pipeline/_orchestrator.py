#!/usr/bin/env python3
"""Internal MetaContext-CMM orchestrator used by the published pipeline entry points."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from presets import METRIC_CHOICES, SENSITIVITY_CHOICES


PIPELINE_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = PIPELINE_DIR.parent
FORMAL_CONTEXT_NAME = "context"
LEGACY_GA_DATASET_NAME = "ga"

BUILD_SCRIPT = Path("workflow/simulation/build_community_model.py")
INFER_SCRIPT = Path("workflow/simulation/infer_context_bounds.py")
SIMULATE_SCRIPT = Path("workflow/simulation/simulate_community_model.py")
MEDIUM_SCRIPT = Path("workflow/simulation/sensitivity_medium.py")
REACTION_SCRIPT = Path("workflow/simulation/sensitivity_reaction.py")
TRADEOFF_SCRIPT = Path("workflow/simulation/sensitivity_tradeoff.py")
EAI_SCRIPT = Path("workflow/analysis/metrics_eai.py")
ARB_SCRIPT = Path("workflow/analysis/metrics_arb.py")
AC_SCRIPT = Path("workflow/analysis/metrics_ac.py")
METRIC_SCRIPTS = {
    "eai": EAI_SCRIPT,
    "arb": ARB_SCRIPT,
    "ac": AC_SCRIPT,
}


def build_parser(preset: dict[str, Any]) -> argparse.ArgumentParser:
    samples = preset["sample"]
    medium_bounds = preset["medium_bounds"]
    tradeoffs = preset["tradeoffs"]
    reaction_realizations = preset["reaction_realizations"]
    ac_heatmap_guilds = preset["ac_heatmap_guilds"]
    output_root = preset["output_root"]
    analysis_dir = preset["analysis_dir"]

    parser = argparse.ArgumentParser(description=preset["description"])
    parser.add_argument(
        "--sample",
        nargs="+",
        default=list(samples),
        help=f"Sample IDs to run (default: {' '.join(samples)}).",
    )
    parser.add_argument(
        "--gem-dir",
        type=Path,
        default=preset["gem_dir"],
        help="Directory containing one <mag_id>.xml GEM per taxon.",
    )
    parser.add_argument(
        "--ge-file",
        type=Path,
        default=preset["ge_file"],
        help=(
            "Taxon GE abundances. Used for discordance QC with --ga-file, "
            "and as baseline abundance unless --abundance-file is set."
        ),
    )
    parser.add_argument(
        "--ga-file",
        type=Path,
        default=preset["ga_file"],
        help="Taxon GA abundances for discordance QC (with --ge-file).",
    )
    parser.add_argument(
        "--abundance-file",
        type=Path,
        default=None,
        help=(
            "Optional taxon abundance CSV for baseline community construction. "
            "Defaults to --ge-file. Use GA abundances here for a GA formal run "
            "while keeping --ge-file/--ga-file for QC."
        ),
    )
    parser.add_argument(
        "--gene-expr-file",
        type=Path,
        default=preset["gene_expr_file"],
        help=(
            "Complete long-format gene expression for RIPTiDe. With default QC, "
            "filter writes context_expression.csv under 02_context/."
        ),
    )
    parser.add_argument(
        "--medium-file",
        type=Path,
        default=preset["medium_file"],
        help="Integrated medium CSV with reaction, flux, and sample_id.",
    )
    parser.add_argument(
        "--medium-reaction-list",
        type=Path,
        default=preset["medium_reaction_list"],
        help="Exchange list for sensitivity_medium.",
    )
    parser.add_argument(
        "--reaction-list",
        type=Path,
        default=preset["reaction_list"],
        help="Reaction-family list for sensitivity_reaction.",
    )
    parser.add_argument(
        "--guild-file",
        type=Path,
        default=preset["guild_file"],
        help="MAG-to-guild map required by metrics_ac; unused by EAI/ARB.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=output_root,
        help=(
            "Output root containing 01_baseline/, 02_context/, 03_simulation/ "
            f"(default: {output_root})."
        ),
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=analysis_dir,
        help=(
            "Metric outputs go to <analysis-dir>/<metric>/ "
            f"(default: {analysis_dir})."
        ),
    )
    parser.add_argument(
        "--sensitivity",
        nargs="*",
        default=list(SENSITIVITY_CHOICES),
        choices=SENSITIVITY_CHOICES,
        metavar="STAGE",
        help=(
            "Sensitivity stages under 03_simulation/. Choices: "
            f"{', '.join(SENSITIVITY_CHOICES)}. Default: all three. "
            "Example: --sensitivity medium tradeoff"
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["eai", "arb", "ac"],
        choices=METRIC_CHOICES,
        metavar="METRIC",
        help=(
            "Metrics after simulation. Choices: "
            f"{', '.join(METRIC_CHOICES)}. Default: eai arb ac. "
            "Pass `--metrics` with no names to skip analysis."
        ),
    )
    parser.add_argument(
        "--medium-bounds",
        type=float,
        nargs="+",
        default=list(medium_bounds),
        help=(
            "Uptake bounds forwarded to sensitivity_medium "
            f"(default: {' '.join(f'{v:g}' for v in medium_bounds)})."
        ),
    )
    parser.add_argument(
        "--tradeoffs",
        type=float,
        nargs="+",
        default=list(tradeoffs),
        help=(
            "Tradeoff fractions forwarded to sensitivity_tradeoff "
            f"(default: {' '.join(f'{v:g}' for v in tradeoffs)})."
        ),
    )
    parser.add_argument(
        "--solve-tradeoff",
        type=float,
        default=1.0,
        help=(
            "Cooperative-tradeoff fraction used by medium and reaction "
            "solves (default: 1). Does not run the tradeoff-sensitivity stage."
        ),
    )
    parser.add_argument(
        "--prior-tradeoff-dir",
        type=Path,
        default=None,
        help=(
            "Existing sensitivity_tradeoff directory to include in metric "
            "plots without solving that stage."
        ),
    )
    parser.add_argument(
        "--prior-medium-dir",
        type=Path,
        default=None,
        help=(
            "Existing sensitivity_medium directory to include in metric "
            "plots without solving that stage."
        ),
    )
    parser.add_argument(
        "--reaction-fraction",
        type=float,
        default=5.0,
        help="Percent of reaction families perturbed per LHS realization (default: 5).",
    )
    parser.add_argument(
        "--reaction-realizations",
        type=int,
        default=reaction_realizations,
        help=(
            "Number of LHS realizations for sensitivity_reaction "
            f"(default: {reaction_realizations})."
        ),
    )
    parser.add_argument(
        "--reaction-seed",
        type=int,
        default=42,
        help="RNG seed for sensitivity_reaction (default: 42).",
    )
    parser.add_argument(
        "--reaction-jobs",
        type=int,
        default=1,
        help="Parallel sample workers for sensitivity_reaction (default: 1).",
    )
    parser.add_argument(
        "--solve-timeout",
        type=float,
        default=1800.0,
        help=(
            "Wall-clock seconds for one sensitivity_reaction sample/realization "
            "worker (default: 1800). On timeout the worker is killed, the job is "
            "recorded as failed, and the schedule continues. Use <=0 to disable."
        ),
    )
    parser.add_argument(
        "--ac-jobs",
        type=int,
        default=1,
        help=(
            "Parallel metrics_ac --jobs across flux scenarios (default: 1). "
            "With --ac-method micom_interaction and jobs>1, --ac-threads is "
            "forced to 1."
        ),
    )
    parser.add_argument(
        "--ac-threads",
        type=int,
        default=1,
        help=(
            "metrics_ac --threads for micom.interaction workers (default: 1; "
            "only used when --ac-method micom_interaction)."
        ),
    )
    parser.add_argument(
        "--ac-method",
        choices=("metacontext_interaction", "micom_interaction"),
        default="metacontext_interaction",
        help=(
            "metrics_ac --method (default: metacontext_interaction). "
            "Caches: *-ctx-interactions.metacontext.csv or "
            "*-ctx-interactions.micom.csv."
        ),
    )
    parser.add_argument(
        "--ac-heatmap-guilds",
        nargs="+",
        default=list(ac_heatmap_guilds),
        help=(
            "Guild subgroups passed to metrics_ac --heatmap-guilds "
            f"(default: {' '.join(ac_heatmap_guilds)})."
        ),
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used to launch child workflow scripts.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild baseline CMMs, re-infer RIPTiDe bounds, rerun formal/"
            "sensitivity simulations, and recalculate metrics. For AC this "
            "forces full recomputation instead of plots-only reuse."
        ),
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and print the plan without running it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Forward --verbose to child workflow scripts.",
    )
    return parser


def unique_preserve(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def ensure_package_cwd() -> None:
    """Run relative paths against the package root."""

    os.chdir(PACKAGE_DIR)


def as_relative(path: Path) -> Path:
    """Expand ``~`` and return a path relative to the package cwd when possible."""

    candidate = path.expanduser()
    absolute = (
        candidate.resolve()
        if candidate.is_absolute()
        else (Path.cwd() / candidate).resolve()
    )
    try:
        return absolute.relative_to(Path.cwd().resolve())
    except ValueError:
        return absolute


def require_path(path: Path, label: str, *, directory: bool) -> Path:
    target = as_relative(path)
    check = target if target.is_absolute() else Path.cwd() / target
    valid = check.is_dir() if directory else check.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} is missing: {target}")
    return target


def nonempty(path: Path) -> bool:
    check = path if path.is_absolute() else Path.cwd() / path
    return check.is_file() and check.stat().st_size > 0


def prepare_directory(path: Path, check_only: bool) -> Path:
    target = as_relative(path)
    check = target if target.is_absolute() else Path.cwd() / target
    if not check_only:
        check.mkdir(parents=True, exist_ok=True)
    return target


def run_command(
    command: Sequence[str | Path],
    *,
    check_only: bool,
    label: str,
    allow_failure: bool = False,
    failures: list[str] | None = None,
) -> bool:
    normalized: list[str] = []
    for index, value in enumerate(command):
        if isinstance(value, Path):
            # Keep the interpreter absolute when it lives outside the package.
            if index == 0:
                normalized.append(str(value.expanduser()))
            else:
                normalized.append(str(as_relative(value)))
        else:
            normalized.append(str(value))
    print(f"[RUN] {label}")
    if check_only:
        return True
    completed = subprocess.run(normalized, check=False)
    if completed.returncode == 0:
        return True
    code = completed.returncode
    if code < 0:
        message = f"{label} failed with signal {-code} (exit code {code})"
    else:
        message = f"{label} failed with exit code {code}"
    if allow_failure:
        print(f"[SKIP] {message}; continuing pipeline")
        if failures is not None:
            failures.append(message)
        return False
    raise subprocess.CalledProcessError(completed.returncode, normalized)


def play_sound(filename: str) -> None:
    sound_file = Path("/System/Library/Sounds") / filename
    if sys.platform != "darwin" or not sound_file.is_file():
        return
    subprocess.run(
        ["afplay", str(sound_file)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def play_completion_sound() -> None:
    play_sound("Glass.aiff")


def play_failure_sound() -> None:
    play_sound("Basso.aiff")


def baseline_complete(directory: Path, sample: str) -> bool:
    return nonempty(directory / f"{sample}-bsl.pickle")


def formal_simulation_complete(directory: Path, sample: str) -> bool:
    return all(
        nonempty(directory / f"{sample}-{mode}-flux.csv")
        for mode in ("bsl", "ctx")
    )


def formal_sample_ready(
    context_dir: Path,
    formal_dir: Path,
    sample: str,
) -> bool:
    """True when pre-solve ctx pickle and formal bsl/ctx fluxes exist."""

    return (
        nonempty(context_dir / f"{sample}-ctx.pickle")
        and formal_simulation_complete(formal_dir, sample)
    )


def warn_if_legacy_output_root(output_root: Path) -> None:
    """Warn when writing into the published case_study/outputs tree."""

    resolved = output_root.expanduser().resolve()
    text = str(resolved).replace("\\", "/")
    if text.rstrip("/").endswith("case_study/outputs") or "/case_study/outputs/" in text + "/":
        print(
            "[WARN] --output-root points at case_study/outputs (or under it). "
            "Pre-solve validation/regeneration should use an isolated run root "
            "such as case_study/runs/presolve_unified_YYYYMMDD/ so old evidence "
            "is not overwritten. See pipeline/bootstrap_presolve_run_root.py."
        )


def context_bounds_mags(context_file: Path, sample: str) -> set[str]:
    """Return mag_id values present for ``sample`` in context_bounds.csv."""

    if not nonempty(context_file):
        return set()
    mags: set[str] = set()
    with context_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            return set()
        if "mag_id" not in reader.fieldnames:
            return set()
        for row in reader:
            if str(row.get("sample_id", "")).strip() != sample:
                continue
            mag_id = str(row.get("mag_id", "")).strip()
            if mag_id:
                mags.add(mag_id)
    return mags


def species_from_gene_expr_file(expression_file: Path, sample: str) -> set[str]:
    """Return MAG ids for ``sample`` listed in --gene-expr-file."""

    if not nonempty(expression_file):
        return set()
    mags: set[str] = set()
    with expression_file.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "sample_id" not in fields or "mag_id" not in fields:
            return set()
        for row in reader:
            if str(row.get("sample_id", "")).strip() != sample:
                continue
            mag_id = str(row.get("mag_id", "")).strip()
            if mag_id:
                mags.add(mag_id)
    return mags


def species_from_context_species(species_file: Path, sample: str) -> set[str]:
    """Return MAG ids marked ``1`` for ``sample`` in context_species.csv."""

    if not nonempty(species_file):
        return set()
    mags: set[str] = set()
    with species_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "Bin Id" not in fields or sample not in fields:
            return set()
        for row in reader:
            if str(row.get(sample, "")).strip() != "1":
                continue
            mag_id = str(row.get("Bin Id", "")).strip()
            if mag_id:
                mags.add(mag_id)
    return mags


def riptide_bounds_complete(
    context_dir: Path,
    sample: str,
    *,
    gene_expr_file: Path,
    enable_qc: bool,
) -> bool:
    """True when context_bounds.csv already matches the species gate.

    With QC (default): required taxa come from filter output
    ``context_species.csv``. Without QC: from ``--gene-expr-file``.
    """

    context_file = context_dir / "context_bounds.csv"
    if enable_qc:
        required = species_from_context_species(
            context_dir / "context_species.csv", sample
        )
    else:
        required = species_from_gene_expr_file(gene_expr_file, sample)
    if not required:
        return False
    existing = context_bounds_mags(context_file, sample)
    return existing == required


def scenario_flux(
    directory: Path,
    sample: str,
    scenario: str,
    mode: str,
) -> bool:
    """True when this job already has a flux file or a recorded failure.

    A nonempty ``*-failed.txt`` marker is treated like an existing output so
    the stage is not entered again just to retry a known solver failure.
    """

    patterns = (
        f"{sample}*{scenario}*{mode}-flux.csv",
        f"{sample}*{scenario}*{mode}-failed.txt",
    )
    return any(
        nonempty(path)
        for pattern in patterns
        for path in directory.glob(pattern)
    )


def medium_complete(
    directory: Path,
    samples: Sequence[str],
    bounds: Sequence[float],
) -> bool:
    return all(
        scenario_flux(
            directory / f"bound{bound:g}",
            sample,
            f"bound{bound:g}",
            mode,
        )
        for sample in samples
        for bound in bounds
        for mode in ("bsl", "ctx")
    )


def tradeoff_label(value: float) -> str:
    """Folder/file tag for a tradeoff fraction (1 → tradeoff1, 0.1 → tradeoff0.1)."""

    return f"tradeoff{float(value):g}"


def tradeoff_complete(
    directory: Path,
    samples: Sequence[str],
    tradeoffs: Sequence[float],
) -> bool:
    return all(
        scenario_flux(directory / label, sample, label, mode)
        for sample in samples
        for value in tradeoffs
        for label in (tradeoff_label(value),)
        for mode in ("bsl", "ctx")
    )


def reaction_complete(
    directory: Path,
    samples: Sequence[str],
    realizations: int,
) -> bool:
    """True when every realization has a flux, a failure marker, or bak LHS.

    Quarantined LHS under ``01_lhs/<sample>/bak/`` is a recorded failed
    attempt and is not solved again.
    """

    lhs_root = directory / "01_lhs"

    def settled(sample: str, scenario: str) -> bool:
        if scenario_flux(directory / scenario, sample, scenario, "ctx"):
            return True
        bak = lhs_root / sample / "bak" / f"LHS_sample_{scenario}.csv"
        return nonempty(bak)

    return all(
        settled(sample, scenario)
        for sample in samples
        for realization in range(realizations + 1)
        for scenario in (f"S{realization:03d}",)
    )


def reaction_lhs_complete(
    directory: Path,
    samples: Sequence[str],
    realizations: int,
) -> bool:
    lhs_root = directory / "01_lhs"

    def _lhs_present(sample: str, realization: int) -> bool:
        name = f"LHS_sample_S{realization:03d}.csv"
        live = lhs_root / sample / name
        bak = lhs_root / sample / "bak" / name
        return nonempty(live) or nonempty(bak)

    return all(
        _lhs_present(sample, realization)
        for sample in samples
        for realization in range(realizations + 1)
    )


def reaction_data_available(directory: Path) -> bool:
    return any(
        path.parent.name.upper() != "S000" and nonempty(path)
        for path in directory.glob("S[0-9][0-9][0-9]/*-ctx-flux.csv")
    )


def ac_metric_tables_complete(directory: Path) -> bool:
    """True when AC analysis CSVs exist (figures may still need refresh)."""

    tables = (
        directory / "ac_all.csv",
        directory / "ac_single_values.csv",
        directory / "ac_batch_statistics.csv",
        directory / "ac_interactions.csv",
    )
    return all(nonempty(path) for path in tables)


def metric_complete(
    directory: Path,
    metric: str,
    *,
    has_batches: bool,
) -> bool:
    tables = [
        directory / f"{metric}_all.csv",
        directory / f"{metric}_single_values.csv",
        directory / f"{metric}_batch_statistics.csv",
    ]
    if metric == "eai":
        figures = (
            (
                directory / "eai_radar_median.pdf",
                directory / "eai_radar_bands.pdf",
            )
            if has_batches
            else (directory / "eai_radar_single.pdf",)
        )
    elif metric == "arb":
        figures = (directory / "arb.pdf",)
    elif metric == "ac":
        # AC reuse is table-based; figures are refreshed via --plots-only.
        return ac_metric_tables_complete(directory)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return all(nonempty(path) for path in (*tables, *figures))


def include_tradeoff_half_single(args: argparse.Namespace) -> bool:
    """Case-study EAI/ARB singles also include tradeoff 0.5.

    The example tutorial pipeline keeps its original single set (formal only).
    """

    if getattr(args, "pipeline_label", "") == "example":
        return False
    if "tradeoff" not in args.sensitivity:
        return False
    return any(abs(float(value) - 0.5) <= 1e-12 for value in args.tradeoffs)


def tradeoff_half_single_dir(simulation_root: Path) -> Path:
    return simulation_root / "sensitivity_tradeoff" / "tradeoff0.5"


def metric_table_has_dataset(
    directory: Path, metric: str, dataset: str
) -> bool:
    """True when ``{metric}_all.csv`` contains rows for ``dataset``."""

    table = directory / f"{metric}_all.csv"
    if not nonempty(table):
        return False
    with table.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "dataset" not in reader.fieldnames:
            return False
        for row in reader:
            if str(row.get("dataset", "")).strip() == dataset:
                return True
    return False


def run_root_paths(run_root: Path) -> tuple[Path, Path, Path]:
    root = as_relative(run_root)
    return (
        root / "01_baseline",
        root / "02_context",
        root / "03_simulation",
    )


def validate_configuration(args: argparse.Namespace) -> None:
    if len(set(args.sample)) != len(args.sample):
        raise ValueError("--sample values must be unique.")
    args.sensitivity = unique_preserve(list(args.sensitivity))
    args.metrics = unique_preserve(list(args.metrics))
    args.enable_qc = True
    if "medium" in args.sensitivity:
        if any(not 0 < value <= 1000 for value in args.medium_bounds):
            raise ValueError("--medium-bounds values must be in (0, 1000].")
    if "tradeoff" in args.sensitivity:
        if any(not 0 < value <= 1 for value in args.tradeoffs):
            raise ValueError("--tradeoffs values must be in (0, 1].")
    if "medium" in args.sensitivity or "reaction" in args.sensitivity:
        if not 0 < float(args.solve_tradeoff) <= 1:
            raise ValueError("--solve-tradeoff must be in (0, 1].")
    if args.prior_tradeoff_dir is not None and "tradeoff" in args.sensitivity:
        raise ValueError(
            "--prior-tradeoff-dir cannot be combined with "
            "--sensitivity tradeoff (that stage would write its own batch)."
        )
    if args.prior_medium_dir is not None and "medium" in args.sensitivity:
        raise ValueError(
            "--prior-medium-dir cannot be combined with "
            "--sensitivity medium (that stage would write its own batch)."
        )
    if "reaction" in args.sensitivity:
        if not 0 < args.reaction_fraction < 100:
            raise ValueError("--reaction-fraction must be in (0, 100).")
        if args.reaction_realizations < 1:
            raise ValueError("--reaction-realizations must be at least 1.")
        if args.reaction_jobs < 1:
            raise ValueError("--reaction-jobs must be at least 1.")
        if args.solve_timeout is not None and not math.isfinite(args.solve_timeout):
            raise ValueError("--solve-timeout must be finite (use <=0 to disable).")


def run_baselines(
    *,
    python_bin: Path,
    samples: Sequence[str],
    gem_dir: Path,
    abundance_file: Path,
    baseline_dir: Path,
    force: bool,
    check_only: bool,
    verbose: bool,
) -> None:
    prepare_directory(baseline_dir, check_only)
    baseline_is_symlink = baseline_dir.is_symlink()
    for sample in samples:
        if baseline_complete(baseline_dir, sample) and not force:
            print(f"[REUSE] baseline/{sample}")
            continue
        if baseline_is_symlink:
            raise SystemExit(
                f"Refusing to write baseline/{sample} through symlink "
                f"{baseline_dir} (read-only reuse for isolated run roots). "
                "Use a real 01_baseline directory, or omit --force when "
                "pickles already exist under the linked tree."
            )
        command: list[str | Path] = [
            python_bin,
            BUILD_SCRIPT,
            "--sample",
            sample,
            "--gem-dir",
            gem_dir,
            "--ge-file",
            abundance_file,
            "--output-dir",
            baseline_dir,
        ]
        if verbose:
            command.append("--verbose")
        run_command(
            command,
            check_only=check_only,
            label=f"baseline/{sample}",
        )


def run_infer_and_formal(
    *,
    python_bin: Path,
    samples: Sequence[str],
    gem_dir: Path,
    gene_expr_file: Path,
    medium_file: Path,
    baseline_dir: Path,
    context_dir: Path,
    formal_dir: Path,
    enable_qc: bool,
    ga_file: Path | None,
    ge_abundance_file: Path | None,
    force: bool,
    check_only: bool,
    verbose: bool,
    failures: list[str],
) -> Path:
    """Infer RIPTiDe bounds and run formal bsl/ctx simulations.

    Returns the context-bounds CSV path.
    """

    prepare_directory(context_dir, check_only)
    prepare_directory(formal_dir, check_only)
    context_file = context_dir / "context_bounds.csv"

    for sample in samples:
        # context/ → formal bsl|ctx flux present?
        formal_ready = formal_simulation_complete(formal_dir, sample) and not force
        # RIPTiDe/ → bounds match filter species (QC) or gene-expr taxa.
        riptide_ready = (
            riptide_bounds_complete(
                context_dir,
                sample,
                gene_expr_file=gene_expr_file,
                enable_qc=enable_qc,
            )
            and not force
        )

        if formal_ready:
            # Formal fluxes imply usable RIPTiDe bounds already existed.
            print(f"[REUSE] RIPTiDe/{sample}")
            print(f"[REUSE] context/{sample}")
            continue

        if riptide_ready:
            print(f"[REUSE] RIPTiDe/{sample}")
        else:
            infer_command: list[str | Path] = [
                python_bin,
                INFER_SCRIPT,
                "--sample",
                sample,
                "--gem-dir",
                gem_dir,
                "--gene-expr-file",
                gene_expr_file,
                "--output-file",
                context_file,
            ]
            if enable_qc:
                if ga_file is None or ge_abundance_file is None:
                    raise ValueError(
                        "Context QC requires --ga-file and --ge-file."
                    )
                infer_command.extend(
                    [
                        "--enable-qc",
                        "--ga-file",
                        ga_file,
                        "--ge-file",
                        ge_abundance_file,
                    ]
                )
            if force:
                infer_command.append("--force")
            if verbose:
                infer_command.append("--verbose")
            run_command(
                infer_command,
                check_only=check_only,
                label=f"RIPTiDe/{sample}",
            )

        simulate_command: list[str | Path] = [
            python_bin,
            SIMULATE_SCRIPT,
            "--sample",
            sample,
            "--baseline-cmm-dir",
            baseline_dir,
            "--context-file",
            context_file,
            "--medium-file",
            medium_file,
            "--output-dir",
            formal_dir,
        ]
        if verbose:
            simulate_command.append("--verbose")
        run_command(
            simulate_command,
            check_only=check_only,
            label=f"context/{sample}",
            allow_failure=True,
            failures=failures,
        )
    return context_file


def run_sensitivities(
    *,
    args: argparse.Namespace,
    python_bin: Path,
    baseline_dir: Path,
    context_dir: Path,
    formal_dir: Path,
    simulation_root: Path,
    medium_file: Path,
    medium_reaction_list: Path,
    reaction_list: Path,
    failures: list[str],
    samples: Sequence[str],
) -> dict[str, Path]:
    """Run selected sensitivity stages for formal-ready samples only."""

    batch_dirs: dict[str, Path] = {}
    if not samples:
        print(
            "[SKIP] sensitivity: no samples with pre-solve ctx pickle and "
            "formal bsl/ctx fluxes in this output root"
        )
        return batch_dirs

    if "medium" in args.sensitivity:
        medium_dir = prepare_directory(
            simulation_root / "sensitivity_medium",
            args.check_only,
        )
        batch_dirs["sensitivity_medium"] = medium_dir
        if (
            medium_complete(medium_dir, samples, args.medium_bounds)
            and not args.force
        ):
            print("[REUSE] sensitivity_medium")
        else:
            command: list[str | Path] = [
                python_bin,
                MEDIUM_SCRIPT,
                "--sample",
                *samples,
                "--baseline-cmm-dir",
                baseline_dir,
                "--medium-file",
                medium_file,
                "--context-cmm-dir",
                context_dir,
                "--output-dir",
                medium_dir,
                "--reaction-list",
                medium_reaction_list,
                "--baseline-flux-dir",
                formal_dir,
                "--bounds",
                *(f"{value:g}" for value in args.medium_bounds),
                "--tradeoff",
                f"{float(args.solve_tradeoff):g}",
            ]
            if args.force:
                command.append("--force")
            run_command(
                command,
                check_only=args.check_only,
                label="sensitivity_medium",
                allow_failure=True,
                failures=failures,
            )

    if "tradeoff" in args.sensitivity:
        tradeoff_dir = prepare_directory(
            simulation_root / "sensitivity_tradeoff",
            args.check_only,
        )
        batch_dirs["sensitivity_tradeoff"] = tradeoff_dir
        if (
            tradeoff_complete(tradeoff_dir, samples, args.tradeoffs)
            and not args.force
        ):
            print("[REUSE] sensitivity_tradeoff")
        else:
            command = [
                python_bin,
                TRADEOFF_SCRIPT,
                "--sample",
                *samples,
                "--baseline-cmm-dir",
                baseline_dir,
                "--context-cmm-dir",
                context_dir,
                "--output-dir",
                tradeoff_dir,
                "--tradeoffs",
                *(f"{value:g}" for value in args.tradeoffs),
            ]
            if args.force:
                command.append("--force")
            run_command(
                command,
                check_only=args.check_only,
                label="sensitivity_tradeoff",
                allow_failure=True,
                failures=failures,
            )

    if "reaction" in args.sensitivity:
        reaction_dir = prepare_directory(
            simulation_root / "sensitivity_reaction",
            args.check_only,
        )
        batch_dirs["sensitivity_reaction"] = reaction_dir
        if (
            reaction_complete(
                reaction_dir,
                samples,
                args.reaction_realizations,
            )
            and not args.force
        ):
            print("[REUSE] sensitivity_reaction")
        else:
            reaction_stage = "all"
            if (
                not args.force
                and reaction_lhs_complete(
                    reaction_dir,
                    samples,
                    args.reaction_realizations,
                )
            ):
                reaction_stage = "simulate"
                print(
                    "[REUSE] sensitivity_reaction LHS; "
                    "running --stage simulate only"
                )
            command = [
                python_bin,
                REACTION_SCRIPT,
                "--sample",
                *samples,
                "--baseline-cmm-dir",
                baseline_dir,
                "--medium-file",
                medium_file,
                "--context-cmm-dir",
                context_dir,
                "--output-dir",
                reaction_dir,
                "--reaction-list",
                reaction_list,
                "--stage",
                reaction_stage,
                "--fraction",
                f"{args.reaction_fraction:g}",
                "--n-realizations",
                str(args.reaction_realizations),
                "--seed",
                str(args.reaction_seed),
                "--jobs",
                str(args.reaction_jobs),
                "--solve-timeout",
                f"{args.solve_timeout:g}",
                "--tradeoff",
                f"{float(args.solve_tradeoff):g}",
            ]
            if args.force:
                command.append("--force")
            run_command(
                command,
                check_only=args.check_only,
                label="sensitivity_reaction",
                allow_failure=True,
                failures=failures,
            )

    return batch_dirs


def pipeline(args: argparse.Namespace) -> int:
    ensure_package_cwd()
    validate_configuration(args)
    failures: list[str] = []
    python_bin = require_path(
        args.python_bin, "Python interpreter", directory=False
    )
    gem_dir = require_path(args.gem_dir, "GEM", directory=True)
    ge_file = require_path(args.ge_file, "Taxon GE", directory=False)
    abundance_file = (
        require_path(args.abundance_file, "Taxon abundance", directory=False)
        if args.abundance_file is not None
        else ge_file
    )
    gene_expr_file = require_path(
        args.gene_expr_file, "Gene expression", directory=False
    )
    medium_file = require_path(args.medium_file, "Medium", directory=False)
    guild_file = require_path(args.guild_file, "Taxon guild", directory=False)

    ga_file: Path | None = None
    if args.enable_qc:
        ga_file = require_path(args.ga_file, "Taxon GA", directory=False)

    medium_reaction_list: Path | None = None
    reaction_list: Path | None = None
    if "medium" in args.sensitivity:
        medium_reaction_list = require_path(
            args.medium_reaction_list,
            "Medium reaction list",
            directory=False,
        )
    if "reaction" in args.sensitivity:
        reaction_list = require_path(
            args.reaction_list,
            "Reaction perturbation list",
            directory=False,
        )

    required_scripts = [BUILD_SCRIPT, INFER_SCRIPT, SIMULATE_SCRIPT]
    if "medium" in args.sensitivity:
        required_scripts.append(MEDIUM_SCRIPT)
    if "tradeoff" in args.sensitivity:
        required_scripts.append(TRADEOFF_SCRIPT)
    if "reaction" in args.sensitivity:
        required_scripts.append(REACTION_SCRIPT)
    for metric in args.metrics:
        required_scripts.append(METRIC_SCRIPTS[metric])
    for script in required_scripts:
        require_path(script, "Workflow script", directory=False)

    baseline_dir, context_dir, simulation_root = run_root_paths(
        args.output_root
    )
    formal_dir = simulation_root / FORMAL_CONTEXT_NAME
    analysis_dir = prepare_directory(args.analysis_dir, args.check_only)
    warn_if_legacy_output_root(args.output_root)

    print("[CONFIG] MetaContext-CMM pipeline")
    print(
        "[CONFIG] context QC: "
        f"on (ga={as_relative(args.ga_file)}, ge={as_relative(args.ge_file)})"
    )
    print(f"[CONFIG] output root: {as_relative(args.output_root)}")
    print(
        "[CONFIG] baseline abundance: "
        + str(as_relative(abundance_file))
        + (
            " (--abundance-file)"
            if args.abundance_file is not None
            else " (--ge-file)"
        )
    )
    print(
        "[CONFIG] sensitivity: "
        + (", ".join(args.sensitivity) if args.sensitivity else "(none)")
    )
    print(
        "[CONFIG] metrics: "
        + (", ".join(args.metrics) if args.metrics else "(none)")
    )
    print(f"[CONFIG] analysis dir: {analysis_dir}")
    print(
        "[CONFIG] ctx pickle: pre-solve model beside context_bounds.csv; "
        "never overwritten after formal ctx solve"
    )

    print("\n=== 1. Baseline / RIPTiDe / context (formal) ===")
    run_baselines(
        python_bin=python_bin,
        samples=args.sample,
        gem_dir=gem_dir,
        abundance_file=abundance_file,
        baseline_dir=baseline_dir,
        force=args.force,
        check_only=args.check_only,
        verbose=args.verbose,
    )
    run_infer_and_formal(
        python_bin=python_bin,
        samples=args.sample,
        gem_dir=gem_dir,
        gene_expr_file=gene_expr_file,
        medium_file=medium_file,
        baseline_dir=baseline_dir,
        context_dir=context_dir,
        formal_dir=formal_dir,
        enable_qc=args.enable_qc,
        ga_file=ga_file,
        ge_abundance_file=ge_file,
        force=args.force,
        check_only=args.check_only,
        verbose=args.verbose,
        failures=failures,
    )

    ready_samples = [
        sample
        for sample in args.sample
        if args.check_only
        or formal_sample_ready(context_dir, formal_dir, sample)
    ]
    blocked_samples = [
        sample for sample in args.sample if sample not in ready_samples
    ]
    if blocked_samples and not args.check_only:
        print(
            "[GATE] formal not ready (need pre-solve "
            f"<sample>-ctx.pickle + bsl/ctx fluxes) for: "
            f"{', '.join(blocked_samples)}"
        )
        for sample in blocked_samples:
            failures.append(f"formal_gate/{sample}")

    print("\n=== 2. Sensitivity batches ===")
    if args.sensitivity and blocked_samples and not ready_samples:
        print(
            "[SKIP] sensitivity: no formal-ready samples in this output root; "
            "refusing to fall back to other trees"
        )
        batch_dirs: dict[str, Path] = {}
    elif args.sensitivity and blocked_samples:
        print(
            "[GATE] running sensitivity only for formal-ready samples: "
            + ", ".join(ready_samples)
        )
        batch_dirs = run_sensitivities(
            args=args,
            python_bin=python_bin,
            baseline_dir=baseline_dir,
            context_dir=context_dir,
            formal_dir=formal_dir,
            simulation_root=prepare_directory(simulation_root, args.check_only),
            medium_file=medium_file,
            medium_reaction_list=medium_reaction_list or Path(),
            reaction_list=reaction_list or Path(),
            failures=failures,
            samples=ready_samples,
        )
    else:
        batch_dirs = run_sensitivities(
            args=args,
            python_bin=python_bin,
            baseline_dir=baseline_dir,
            context_dir=context_dir,
            formal_dir=formal_dir,
            simulation_root=prepare_directory(simulation_root, args.check_only),
            medium_file=medium_file,
            medium_reaction_list=medium_reaction_list or Path(),
            reaction_list=reaction_list or Path(),
            failures=failures,
            samples=ready_samples if not args.check_only else args.sample,
        )

    analysis_samples = ready_samples if not args.check_only else args.sample
    if args.metrics:
        print("\n=== 3. Analysis ===")
        if not analysis_samples and not args.check_only:
            print(
                "[SKIP] analysis: no formal-ready samples; "
                "refusing to mix legacy outputs"
            )
        else:
            single_dirs: list[Path] = [formal_dir]
            print(
                "[CONFIG] analysis single-dirs: "
                + ", ".join(str(as_relative(path)) for path in single_dirs)
            )

            analysis_batch_dirs: list[Path] = []
            for name, path in batch_dirs.items():
                if name == "sensitivity_reaction" and not reaction_data_available(
                    path
                ):
                    print("[EMPTY] reaction batch omitted from analysis inputs")
                    continue
                if path.is_dir() or args.check_only:
                    analysis_batch_dirs.append(path)
            if args.prior_tradeoff_dir is not None:
                prior_tradeoff = as_relative(args.prior_tradeoff_dir)
                if not prior_tradeoff.is_dir() and not args.check_only:
                    raise SystemExit(
                        "Missing --prior-tradeoff-dir: "
                        f"{prior_tradeoff}"
                    )
                print(
                    "[CONFIG] analysis prior tradeoff batch: "
                    f"{prior_tradeoff}"
                )
                analysis_batch_dirs.append(prior_tradeoff)
            if args.prior_medium_dir is not None:
                prior_medium = as_relative(args.prior_medium_dir)
                if not prior_medium.is_dir() and not args.check_only:
                    raise SystemExit(
                        "Missing --prior-medium-dir: "
                        f"{prior_medium}"
                    )
                print(
                    "[CONFIG] analysis prior medium batch: "
                    f"{prior_medium}"
                )
                analysis_batch_dirs.append(prior_medium)

            has_batches = bool(analysis_batch_dirs)
            for metric in args.metrics:
                metric_dir = prepare_directory(
                    analysis_dir / metric,
                    args.check_only,
                )
                reusable = (
                    metric_complete(
                        metric_dir, metric, has_batches=has_batches
                    )
                    and not args.force
                )
                # Drop stale analysis tables that still include the removed GA single.
                if reusable and metric_table_has_dataset(
                    metric_dir, metric, LEGACY_GA_DATASET_NAME
                ):
                    print(
                        f"[STALE] analysis/{metric} still contains "
                        f"'{LEGACY_GA_DATASET_NAME}' single dataset; rerunning"
                    )
                    reusable = False
                if reusable and metric == "ac":
                    # Reuse AC CSVs; refresh network / heatmap / elemental figures.
                    print("[REUSE] analysis/ac tables; plots-only")
                    command = [
                        python_bin,
                        AC_SCRIPT,
                        "--plots-only",
                        "--plot",
                        "network",
                        "heatmap",
                        "elemental",
                        "--sample",
                        *analysis_samples,
                        "--output-dir",
                        metric_dir,
                        "--ge-file",
                        ge_file,
                        "--guild-file",
                        guild_file,
                        "--heatmap-guilds",
                        *args.ac_heatmap_guilds,
                    ]
                    run_command(
                        command,
                        check_only=args.check_only,
                        label="analysis/ac (plots-only)",
                    )
                    continue
                metric_single_dirs = list(single_dirs)
                if metric in {"eai", "arb"} and include_tradeoff_half_single(args):
                    half_dir = tradeoff_half_single_dir(simulation_root)
                    if half_dir.is_dir() or args.check_only:
                        metric_single_dirs.append(half_dir)
                        print(
                            f"[CONFIG] analysis/{metric} extra single-dir: "
                            f"{as_relative(half_dir)}"
                        )
                    else:
                        print(
                            f"[WARN] analysis/{metric}: tradeoff0.5 single "
                            f"omitted; directory missing: {as_relative(half_dir)}"
                        )
                if (
                    reusable
                    and metric in {"eai", "arb"}
                    and include_tradeoff_half_single(args)
                    and not metric_table_has_dataset(
                        metric_dir, metric, "tradeoff0.5"
                    )
                ):
                    print(
                        f"[STALE] analysis/{metric} single values lack "
                        "tradeoff0.5; rerunning"
                    )
                    reusable = False
                if reusable:
                    print(f"[REUSE] analysis/{metric}")
                    continue
                command = [
                    python_bin,
                    METRIC_SCRIPTS[metric],
                    "--single-dir",
                    *metric_single_dirs,
                    "--sample",
                    *analysis_samples,
                    "--output-dir",
                    metric_dir,
                ]
                if analysis_batch_dirs:
                    command.extend(["--batch-dir", *analysis_batch_dirs])
                if metric == "ac":
                    command.extend(["--context-cmm-dir", context_dir])
                    command.extend(["--ge-file", ge_file])
                    command.extend(["--guild-file", guild_file])
                    command.extend(["--method", args.ac_method])
                    command.extend(["--jobs", str(args.ac_jobs)])
                    command.extend(["--threads", str(args.ac_threads)])
                    command.extend(
                        ["--heatmap-guilds", *args.ac_heatmap_guilds]
                    )
                    if args.force:
                        command.append("--force")
                elif metric == "arb":
                    command.extend(["--ge-file", ge_file])
                run_command(
                    command,
                    check_only=args.check_only,
                    label=f"analysis/{metric}",
                )
    else:
        print("\n=== 3. Analysis skipped (no --metrics) ===")

    print("\nPipeline complete.")
    if failures:
        print(
            f"[SUMMARY] skipped {len(failures)} "
            "infeasible/failed simulation stage(s):"
        )
        for item in failures:
            print(f"  - {item}")
    if args.check_only:
        print("Check-only mode: no model, simulation, or analysis command ran.")
    else:
        play_completion_sound()
    return 0 if not failures else 1


def main(preset: dict[str, Any]) -> int:
    args = build_parser(preset).parse_args()
    args.pipeline_label = preset["label"]
    try:
        return pipeline(args)
    except BaseException:
        play_failure_sound()
        raise
