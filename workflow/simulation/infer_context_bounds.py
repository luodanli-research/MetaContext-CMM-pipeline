#!/usr/bin/env python3
"""Build sample-specific RIPTiDe bounds from a gene-expression file.

``--gene-expr-file`` is the complete long-format expression library (shared by
GE/GA runs). Built-in discordance+1e-6 QC is the default gate for which taxa
enter RIPTiDe; the filter writes beside ``--output-file``::

    context_species.csv      wide 0/1 eligibility (upserted per sample)
    context_expression.csv   filtered expression rows for eligible pairs
    context_summary.csv      filter funnel summary
    context_bounds.csv       RIPTiDe reaction bounds (merged across samples)

Reuse rules
-----------
* With ``--enable-qc`` (typical): re-run filter for the sample, upsert
  species/expression outputs, take required taxa from ``context_species.csv``
  (``1`` and present in expression). Reuse matching bounds rows, drop extras,
  RIPTiDe missing taxa, record failures.
* Without QC (``omit --enable-qc``): required taxa are all MAG ids in
  ``--gene-expr-file`` for the sample; same bounds reuse rules. No species /
  expression filter outputs are written.

Provide ``--ga-file`` + ``--ge-file`` or ``--scalings-file`` when QC is on.
GEM files must be named ``<mag_id>.xml``. ``--force`` ignores existing bounds
for the requested sample. ``--audit-only`` skips RIPTiDe.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Iterator, Sequence

import cobra
import numpy as np
import pandas as pd
import riptide
from rich.progress import track

from _cli_utils import configure_logging


LOGGER = logging.getLogger("cmm.contextualize")
REQUIRED_GE_COLUMNS = {"gene_id", "expression", "sample_id", "mag_id"}
OUTPUT_COLUMNS = [
    "gem_rxn_id",
    "lb",
    "ub",
    "rxn_name",
    "equation",
    "sample_id",
    "mag_id",
]
EXPRESSION_COLUMNS = ["gene_id", "expression", "sample_id", "mag_id"]
RPM_FACTOR = 1e6
DEFAULT_DISCORDANCE_FLOOR = 0.01
CONTEXT_SPECIES_NAME = "context_species.csv"
CONTEXT_EXPRESSION_NAME = "context_expression.csv"
CONTEXT_SUMMARY_NAME = "context_summary.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply RIPTiDe contextualization for expression-eligible "
            "<mag_id>.xml models into one context-bounds CSV. Optionally run "
            "discordance+1e-6 QC first (--enable-qc with --ga-file/--ge-file, "
            "or --scalings-file)."
        )
    )
    parser.add_argument(
        "--sample",
        required=True,
        help="Sample identifier to select from gene_expr_file, for example SW60.",
    )
    parser.add_argument(
        "--gem-dir",
        dest="gem_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing input GEMs named <mag_id>.xml "
            "(required unless --audit-only)."
        ),
    )
    parser.add_argument(
        "--gene-expr-file",
        dest="gene_expr_file",
        required=True,
        type=Path,
        help=(
            "Complete long-format gene-expression CSV (gene_id, expression, "
            "sample_id, mag_id). With --enable-qc, filter writes a subset to "
            "context_expression.csv beside --output-file."
        ),
    )
    parser.add_argument(
        "--enable-qc",
        action="store_true",
        help=(
            "Enable discordance + MICOM 1e-6 abundance QC before reuse/"
            "RIPTiDe. Requires --ga-file and --ge-file (or --scalings-file)."
        ),
    )
    parser.add_argument(
        "--ga-file",
        dest="ga_file",
        type=Path,
        default=None,
        help=(
            "Wide taxon GA CSV (Bin Id + sample columns) for discordance QC. "
            "Used with --ge-file when --enable-qc is set."
        ),
    )
    parser.add_argument(
        "--ge-file",
        dest="ge_abundance_file",
        type=Path,
        default=None,
        help=(
            "Wide taxon GE abundance CSV (Bin Id + sample columns) for "
            "discordance QC. Used with --ga-file when --enable-qc is set. "
            "Distinct from --gene-expr-file (long gene expression)."
        ),
    )
    parser.add_argument(
        "--scalings-file",
        dest="scalings_file",
        type=Path,
        default=None,
        help=(
            "Optional legacy paired GA/GE/CN scalings CSV for discordance QC. "
            "Alternative to --ga-file/--ge-file when --enable-qc is set."
        ),
    )
    parser.add_argument(
        "--output-file",
        dest="output_file",
        required=True,
        type=Path,
        help="Path of the integrated context-bounds CSV to create.",
    )
    parser.add_argument(
        "--qc-summary-file",
        dest="qc_summary_file",
        type=Path,
        default=None,
        help=(
            "Summary CSV when --enable-qc is set (default: "
            "<output-file-dir>/context_qc_summary.csv)."
        ),
    )
    parser.add_argument(
        "--discordance-floor",
        type=float,
        default=DEFAULT_DISCORDANCE_FLOOR,
        help=(
            "Practical-zero TPM floor for DNA/RNA discordance "
            f"(default: {DEFAULT_DISCORDANCE_FLOOR:g}; used with --enable-qc)."
        ),
    )
    parser.add_argument(
        "--skip-abundance-prefilter",
        action="store_true",
        help=(
            "With --enable-qc, disable the MICOM rel_ge > 1e-6 abundance mark "
            "(not recommended)."
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help=(
            "Reuse accounting only (and QC funnel if --enable-qc); "
            "do not run RIPTiDe. context_riptide_infeasible_n is 0."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs from RIPTiDe, COBRApy, and their dependencies.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore an existing --output-file and rerun RIPTiDe for every "
            "eligible taxon present in the expression file."
        ),
    )
    args = parser.parse_args(argv)
    has_taxon_pair = args.ga_file is not None and args.ge_abundance_file is not None
    has_partial_taxon = (args.ga_file is None) ^ (args.ge_abundance_file is None)
    if has_partial_taxon:
        parser.error("--ga-file and --ge-file must be provided together.")
    if args.enable_qc:
        if args.scalings_file is None and not has_taxon_pair:
            parser.error(
                "--enable-qc requires --ga-file and --ge-file "
                "(or --scalings-file)."
            )
        if args.scalings_file is not None and has_taxon_pair:
            parser.error(
                "Use either --scalings-file or --ga-file/--ge-file, not both."
            )
    else:
        if args.scalings_file is not None or has_taxon_pair:
            parser.error(
                "--ga-file/--ge-file and --scalings-file require --enable-qc."
            )
    return args


def load_gates_module() -> ModuleType:
    """Load discordance QC helpers from ``_filter_scalings_gates``."""

    path = Path(__file__).resolve().parent / "_filter_scalings_gates.py"
    if not path.is_file():
        raise FileNotFoundError(f"Discordance QC module missing: {path}")
    spec = importlib.util.spec_from_file_location(
        "_micom310_filter_scalings_gates",
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import discordance QC module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_discordance_qc(
    scalings: Path | pd.DataFrame,
    sample: str,
    *,
    discordance_floor: float,
    skip_abundance_prefilter: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, ModuleType]:
    """Return (decisions, filter_summary, gates_module) for one sample."""

    gates = load_gates_module()
    if not math_isfinite_nonneg(discordance_floor):
        raise ValueError("--discordance-floor must be finite and non-negative.")

    if isinstance(scalings, pd.DataFrame):
        frame = scalings.copy()
    else:
        scalings_file = Path(scalings).expanduser().resolve()
        if not scalings_file.is_file():
            raise FileNotFoundError(f"Scalings file not found: {scalings_file}")
        frame = pd.read_csv(scalings_file, encoding="utf-8-sig")
    id_column, samples, columns = gates.validate_and_select_samples(
        frame, [sample]
    )
    long_frame = gates.build_long_table(frame, id_column, samples, columns)
    long_frame = gates.apply_micom_abundance_prefilter(
        long_frame,
        enabled=not skip_abundance_prefilter,
    )
    # Discordance method: no paired-support cutoff, no CN gate.
    support = long_frame["paired_support"].to_numpy(dtype=float)
    positive_n = int(np.sum(support > 0))
    zero_n = int(len(support) - positive_n)
    thresholds = {
        "pooled": gates.ThresholdResult(
            scope="pooled",
            requested_method="discordance",
            selected_method="discordance-only",
            threshold=0.0,
            positive_n=positive_n,
            zero_n=zero_n,
        )
    }
    decisions = gates.assign_decisions(
        long_frame,
        thresholds=thresholds,
        threshold_scope="pooled",
        min_cn=0,
        discordance_floor=float(discordance_floor),
        enforce_discordance_guard=True,
    )
    filter_summary = gates.make_context_summary(
        decisions,
        thresholds=thresholds,
        threshold_scope="pooled",
        min_cn=0,
        discordance_floor=float(discordance_floor),
    )
    return decisions, filter_summary, gates


def math_isfinite_nonneg(value: float) -> bool:
    return bool(np.isfinite(value) and value >= 0)


def load_sample_ge(gene_expr_file: Path, sample: str) -> pd.DataFrame:
    """Load and validate long-format gene expression for one sample."""
    if not gene_expr_file.is_file():
        raise FileNotFoundError(
            f"Gene-expression file not found: {gene_expr_file}"
        )

    ge_table = pd.read_csv(
        gene_expr_file,
        usecols=lambda column: column in REQUIRED_GE_COLUMNS,
        dtype={"gene_id": "string", "sample_id": "string", "mag_id": "string"},
    )
    missing_columns = REQUIRED_GE_COLUMNS.difference(ge_table.columns)
    if missing_columns:
        raise ValueError(
            "Missing required gene-expression column(s): "
            + ", ".join(sorted(missing_columns))
        )

    sample_ge = ge_table.loc[
        ge_table["sample_id"].eq(sample),
        ["gene_id", "expression", "sample_id", "mag_id"],
    ].copy()
    if sample_ge.empty:
        available_samples = sorted(
            ge_table["sample_id"].dropna().astype(str).unique()
        )
        raise ValueError(
            f"Sample {sample!r} is absent from {gene_expr_file}. "
            f"Available samples: {', '.join(available_samples)}"
        )
    if sample_ge[["gene_id", "mag_id"]].isna().any().any():
        raise ValueError(
            f"Missing gene_id or mag_id values for sample {sample!r}."
        )

    sample_ge["expression"] = pd.to_numeric(
        sample_ge["expression"], errors="coerce"
    )
    invalid_mask = sample_ge["expression"].isna() | ~np.isfinite(
        sample_ge["expression"]
    )
    if invalid_mask.any():
        invalid_rows = sample_ge.loc[invalid_mask, ["mag_id", "gene_id"]]
        preview = ", ".join(
            f"{row.mag_id}/{row.gene_id}"
            for row in invalid_rows.head(10).itertuples(index=False)
        )
        raise ValueError(
            f"Non-numeric or non-finite expression values: {preview}"
        )
    if (sample_ge["expression"] < 0).any():
        negative_rows = sample_ge.loc[
            sample_ge["expression"] < 0, ["mag_id", "gene_id"]
        ]
        preview = ", ".join(
            f"{row.mag_id}/{row.gene_id}"
            for row in negative_rows.head(10).itertuples(index=False)
        )
        raise ValueError(f"Negative gene-expression values: {preview}")

    duplicate_mask = sample_ge.duplicated(["mag_id", "gene_id"], keep=False)
    if duplicate_mask.any():
        duplicate_rows = sample_ge.loc[
            duplicate_mask, ["mag_id", "gene_id"]
        ]
        preview = ", ".join(
            f"{row.mag_id}/{row.gene_id}"
            for row in duplicate_rows.head(10).itertuples(index=False)
        )
        raise ValueError(f"Duplicate MAG/gene records: {preview}")

    return sample_ge


def normalize_ge_for_riptide(mag_ge: pd.DataFrame) -> dict[str, list[float]]:
    """Reproduce RIPTiDe's default reads-per-million normalization."""
    ge_values = [float(value) for value in mag_ge["expression"]]
    total_ge = 0.0
    for value in ge_values:
        total_ge += value
    if total_ge <= 0:
        raise ValueError("Cannot normalize gene expression with a zero total.")

    return {
        str(gene_id): [(value / total_ge) * RPM_FACTOR]
        for gene_id, value in zip(mag_ge["gene_id"], ge_values)
    }


def reaction_rows(
    context_model: cobra.Model,
    sample: str,
    mag_id: str,
) -> Iterator[list[object]]:
    """Yield integrated-output rows from a contextualized model."""
    for reaction in context_model.reactions:
        yield [
            reaction.id,
            reaction.lower_bound,
            reaction.upper_bound,
            reaction.name or "",
            reaction.reaction,
            sample,
            mag_id,
        ]


def load_context_partitions(
    context_file: Path,
    sample: str,
) -> tuple[dict[str, list[list[str]]], list[list[str]]]:
    """Group target-sample rows by taxon and preserve other samples."""

    if not context_file.is_file():
        return {}, []

    rows_by_mag: dict[str, list[list[str]]] = {}
    other_rows: list[list[str]] = []
    try:
        with context_file.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != OUTPUT_COLUMNS:
                raise ValueError(
                    f"unexpected header {reader.fieldnames}; "
                    f"expected {OUTPUT_COLUMNS}"
                )
            for row in reader:
                row_sample = str(row["sample_id"])
                mag_id = str(row["mag_id"]).strip()
                values = [str(row[column]) for column in OUTPUT_COLUMNS]
                if row_sample == sample:
                    rows_by_mag.setdefault(mag_id, []).append(values)
                else:
                    other_rows.append(values)
    except (OSError, csv.Error, ValueError) as error:
        raise ValueError(
            f"Cannot reuse existing context file {context_file}: {error}. "
            "Use --force to discard it and recompute all eligible taxa."
        ) from error
    return rows_by_mag, other_rows


def species_from_context_species(
    species_file: Path, sample: str
) -> list[str]:
    """Return MAG ids marked ``1`` for ``sample`` in context_species.csv."""

    with species_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "Bin Id" not in fields or sample not in fields:
            raise ValueError(
                f"{species_file} must contain columns 'Bin Id' and {sample!r}."
            )
        taxa: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            mag_id = str(row["Bin Id"]).strip()
            if not mag_id:
                raise ValueError(
                    f"{species_file}: empty Bin Id at row {row_number}"
                )
            if mag_id in seen:
                raise ValueError(
                    f"{species_file}: duplicate Bin Id {mag_id!r}"
                )
            seen.add(mag_id)
            value = str(row[sample]).strip()
            if value not in {"0", "1"}:
                raise ValueError(
                    f"{species_file}: {sample} must contain only 0 or 1; "
                    f"found {value!r} at row {row_number}"
                )
            if value == "1":
                taxa.append(mag_id)
    return taxa


def write_context_species_matrix(
    decisions: pd.DataFrame,
    output_file: Path,
    sample: str,
    *,
    mag_order: Sequence[str] | None = None,
) -> None:
    """Upsert the ``sample`` column into wide context_species.csv.

    Filter always re-runs for the requested sample; other sample columns
    already on disk are preserved.
    """

    sample_decisions = decisions.loc[
        decisions["sample"].astype(str).eq(sample),
        ["mag_id", "constraint_eligible"],
    ].copy()
    sample_decisions["mag_id"] = sample_decisions["mag_id"].astype(str)
    sample_series = (
        sample_decisions.drop_duplicates("mag_id")
        .set_index("mag_id")["constraint_eligible"]
        .astype(bool)
        .astype(int)
    )

    existing: pd.DataFrame | None = None
    if output_file.is_file() and output_file.stat().st_size > 0:
        existing = pd.read_csv(output_file, encoding="utf-8-sig")
        if "Bin Id" not in existing.columns:
            raise ValueError(
                f"{output_file} is missing required column 'Bin Id'."
            )
        existing["Bin Id"] = existing["Bin Id"].astype(str)
        existing = existing.set_index("Bin Id")

    if existing is None:
        index = (
            [str(mag) for mag in mag_order]
            if mag_order is not None
            else list(sample_series.index)
        )
        frame = pd.DataFrame(index=pd.Index(index, name="Bin Id"))
    else:
        frame = existing.copy()
        if mag_order is not None:
            ordered = [str(mag) for mag in mag_order]
            extras = [mag for mag in frame.index if mag not in set(ordered)]
            frame = frame.reindex(ordered + extras)

    frame[sample] = sample_series.reindex(frame.index).fillna(0).astype(int)
    # Ensure mags present only in this sample's decisions are added.
    missing_mags = sample_series.index.difference(frame.index)
    if len(missing_mags):
        addition = pd.DataFrame(index=missing_mags)
        for column in frame.columns:
            addition[column] = 0
        addition[sample] = sample_series.loc[missing_mags].astype(int)
        frame = pd.concat([frame, addition])

    out = frame.fillna(0).astype(int).reset_index()
    if out.columns[0] != "Bin Id":
        out = out.rename(columns={out.columns[0]: "Bin Id"})
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)


def write_context_summary_rows(
    filter_summary: pd.DataFrame,
    summary_file: Path,
    sample: str,
) -> None:
    """Upsert filter summary rows for ``sample`` into context_summary.csv."""

    new_rows = filter_summary.loc[
        filter_summary["sample"].astype(str).eq(sample)
    ].copy()
    if summary_file.is_file() and summary_file.stat().st_size > 0:
        existing = pd.read_csv(summary_file, encoding="utf-8-sig")
        kept = existing.loc[~existing["sample"].astype(str).eq(sample)]
        combined = pd.concat([kept, new_rows], ignore_index=True)
    else:
        combined = new_rows
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(summary_file, index=False)


def upsert_context_expression(
    sample_ge: pd.DataFrame,
    local_file: Path,
    sample: str,
    required_mags: set[str],
) -> None:
    """Upsert one sample's filtered expression rows into context_expression.csv.

    Other samples already present in ``local_file`` are preserved. Rows for
    ``sample`` are replaced with the subset of ``sample_ge`` whose mag_id is
    in ``required_mags``.
    """

    required = {str(mag) for mag in required_mags}
    fieldnames = list(EXPRESSION_COLUMNS)
    other_rows: list[dict[str, str]] = []
    if local_file.is_file() and local_file.stat().st_size > 0:
        with local_file.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            missing = set(EXPRESSION_COLUMNS).difference(
                reader.fieldnames or []
            )
            if missing:
                raise ValueError(
                    f"{local_file} is missing columns: "
                    + ", ".join(sorted(missing))
                )
            fieldnames = list(reader.fieldnames or fieldnames)
            for row in reader:
                if str(row["sample_id"]) != sample:
                    other_rows.append(
                        {column: str(row[column]) for column in fieldnames}
                    )

    sample_rows = sample_ge.loc[
        sample_ge["mag_id"].astype(str).isin(required),
        EXPRESSION_COLUMNS,
    ]
    present = set(sample_rows["mag_id"].astype(str).unique())
    unresolved = required - present
    if unresolved:
        raise ValueError(
            "Gene-expression library lacks rows for "
            f"{sample}: {', '.join(sorted(unresolved)[:10])}"
        )

    local_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_file.with_name(f".{local_file.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(other_rows)
            for row in sample_rows.itertuples(index=False):
                writer.writerow(
                    {
                        "gene_id": str(row.gene_id),
                        "expression": str(row.expression),
                        "sample_id": str(row.sample_id),
                        "mag_id": str(row.mag_id),
                    }
                )
        os.replace(temporary, local_file)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_and_remove_context_xml(
    context_model: cobra.Model,
    output_dir: Path,
    sample: str,
    mag_id: str,
) -> None:
    """Write a temporary context GEM and immediately remove it."""
    with tempfile.NamedTemporaryFile(
        prefix=f".{sample}-{mag_id}-",
        suffix="-rna.xml",
        dir=output_dir,
        delete=False,
    ) as temporary_file:
        context_path = Path(temporary_file.name)

    try:
        cobra.io.write_sbml_model(context_model, context_path)
    finally:
        context_path.unlink(missing_ok=True)


def _print_aligned_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    header_line = "  ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    )
    print(header_line, flush=True)
    print("  ".join("-" * width for width in widths), flush=True)
    for row in rows:
        print(
            "  ".join(
                str(cell).ljust(widths[index]) for index, cell in enumerate(row)
            ),
            flush=True,
        )


SUMMARY_COLUMNS = [
    "sample",
    "nonzero_mag_n",
    "qc_eligible_n",
    "context_riptide_existing_n",
    "context_riptide_missing_n",
    "context_riptide_infeasible_n",
    "audit_only",
]


def _as_eligible_fraction(count: int, eligible_n: int) -> str:
    return f"{int(count)}/{int(eligible_n)}"


def emit_qc_summary(
    *,
    sample: str,
    filter_summary: pd.DataFrame,
    context_riptide_existing_n: int,
    context_riptide_missing_n: int,
    context_riptide_infeasible_n: int,
    expression_gems: int,
    qc_eligible_n: int,
    required_n: int,
    audit_only: bool,
    qc_summary_file: Path,
) -> pd.DataFrame:
    """Print compact QC + RIPTiDe reuse summary; write CSV."""

    del expression_gems, required_n  # kept for call-site compatibility
    row = filter_summary.loc[filter_summary["sample"].astype(str).eq(sample)]
    if row.empty:
        raise ValueError(f"QC summary missing sample {sample!r}")

    eligible_n = int(qc_eligible_n)
    out = pd.DataFrame(
        [
            {
                "sample": sample,
                "nonzero_mag_n": int(row.iloc[0]["nonzero_mag_n"]),
                "qc_eligible_n": eligible_n,
                "context_riptide_existing_n": _as_eligible_fraction(
                    context_riptide_existing_n, eligible_n
                ),
                "context_riptide_missing_n": _as_eligible_fraction(
                    context_riptide_missing_n, eligible_n
                ),
                "context_riptide_infeasible_n": _as_eligible_fraction(
                    context_riptide_infeasible_n, eligible_n
                ),
                "audit_only": bool(audit_only),
            }
        ]
    )

    print("\n=== Context QC summary (discordance + 1e-6) ===", flush=True)
    r = out.iloc[0]
    headers = SUMMARY_COLUMNS
    table = [[str(r[col]) for col in headers]]
    _print_aligned_table(headers, table)
    if audit_only:
        print(
            "[AUDIT] RIPTiDe skipped; context_riptide_infeasible_n=0/"
            f"{eligible_n}",
            flush=True,
        )

    qc_summary_file = qc_summary_file.expanduser().resolve()
    qc_summary_file.parent.mkdir(parents=True, exist_ok=True)
    # Merge into multi-sample summary if present.
    if qc_summary_file.is_file() and qc_summary_file.stat().st_size > 0:
        existing = pd.read_csv(qc_summary_file)
        existing = existing.loc[~existing["sample"].astype(str).eq(sample)]
        # Drop legacy wide columns from older summary files.
        keep = [c for c in SUMMARY_COLUMNS if c in existing.columns]
        existing = existing.loc[:, keep] if keep else existing.iloc[0:0]
        combined = pd.concat([existing, out], ignore_index=True)
    else:
        combined = out
    combined = combined.reindex(columns=SUMMARY_COLUMNS)
    combined.to_csv(qc_summary_file, index=False)
    print(f"QC summary file: {qc_summary_file}", flush=True)
    return out


def build_scalings_frame_from_taxon_tables(
    ga_file: Path,
    ge_file: Path,
) -> pd.DataFrame:
    """Merge wide GA/GE taxon tables into a gates-compatible scalings frame.

    Discordance QC in this module does not apply a CN gate (``min_cn=0``), so
    synthetic ``{sample}-CN`` columns are filled with 0 for schema compatibility.
    """

    ga_file = ga_file.expanduser().resolve()
    ge_file = ge_file.expanduser().resolve()
    if not ga_file.is_file():
        raise FileNotFoundError(f"Taxon GA file not found: {ga_file}")
    if not ge_file.is_file():
        raise FileNotFoundError(f"Taxon GE file not found: {ge_file}")

    ga = pd.read_csv(ga_file, encoding="utf-8-sig")
    ge = pd.read_csv(ge_file, encoding="utf-8-sig")
    id_column = "Bin Id"
    if id_column not in ga.columns or id_column not in ge.columns:
        raise ValueError(
            f"Both taxon tables must contain a {id_column!r} column."
        )
    ga_samples = [c for c in ga.columns if c != id_column]
    ge_samples = [c for c in ge.columns if c != id_column]
    if set(ga_samples) != set(ge_samples):
        raise ValueError(
            "GA and GE taxon tables must share the same sample columns: "
            f"GA={sorted(ga_samples)}, GE={sorted(ge_samples)}"
        )
    if not ga_samples:
        raise ValueError("Taxon GA/GE tables have no sample columns.")

    ga_renamed = ga.rename(columns={sample: f"{sample}-GA" for sample in ga_samples})
    ge_renamed = ge.rename(columns={sample: f"{sample}-GE" for sample in ge_samples})
    merged = ga_renamed.merge(ge_renamed, on=id_column, how="outer")
    if merged[id_column].isna().any() or merged[id_column].duplicated().any():
        raise ValueError("Taxon GA/GE merge produced missing or duplicate Bin Id.")
    for sample in ga_samples:
        merged[f"{sample}-GA"] = pd.to_numeric(
            merged[f"{sample}-GA"], errors="coerce"
        )
        merged[f"{sample}-GE"] = pd.to_numeric(
            merged[f"{sample}-GE"], errors="coerce"
        )
        merged[f"{sample}-CN"] = 0
    if merged.filter(regex=r"-(GA|GE)$").isna().any().any():
        raise ValueError("Non-numeric or missing values in taxon GA/GE tables.")
    return merged


def contextualize_all(
    sample: str,
    gem_dir: Path | None,
    gene_expr_file: Path,
    out_file: Path,
    *,
    enable_qc: bool = False,
    ga_file: Path | None = None,
    ge_abundance_file: Path | None = None,
    scalings_file: Path | None = None,
    qc_summary_file: Path | None = None,
    discordance_floor: float = DEFAULT_DISCORDANCE_FLOOR,
    skip_abundance_prefilter: bool = False,
    audit_only: bool = False,
    verbose: bool = False,
    force: bool = False,
) -> dict[str, int]:
    """Filter (optional) → reuse bounds → RIPTiDe missing taxa."""

    del verbose  # logging configured by caller
    gene_expr_file = gene_expr_file.resolve()
    out_file = out_file.resolve()
    context_dir = out_file.parent
    species_file = context_dir / CONTEXT_SPECIES_NAME
    expression_file = context_dir / CONTEXT_EXPRESSION_NAME
    summary_file = context_dir / CONTEXT_SUMMARY_NAME

    sample_ge = load_sample_ge(gene_expr_file, sample)
    ge_by_mag = {
        str(mag_id): mag_ge.reset_index(drop=True)
        for mag_id, mag_ge in sample_ge.groupby("mag_id", sort=False)
    }
    expression_order = list(ge_by_mag)

    filter_summary: pd.DataFrame | None = None
    qc_eligible: set[str] | None = None
    if enable_qc:
        if scalings_file is not None:
            scalings_input: Path | pd.DataFrame = scalings_file
        elif ga_file is not None and ge_abundance_file is not None:
            scalings_input = build_scalings_frame_from_taxon_tables(
                ga_file, ge_abundance_file
            )
        else:
            raise ValueError(
                "enable_qc=True requires ga_file+ge_abundance_file "
                "or scalings_file."
            )
        if qc_summary_file is None:
            qc_summary_file = context_dir / "context_qc_summary.csv"
        else:
            qc_summary_file = qc_summary_file.resolve()

        # Always re-run filter; upsert species / summary / expression outputs.
        decisions, filter_summary, _gates = run_discordance_qc(
            scalings_input,
            sample,
            discordance_floor=discordance_floor,
            skip_abundance_prefilter=skip_abundance_prefilter,
        )
        mag_order: Sequence[str] | None = None
        if isinstance(scalings_input, pd.DataFrame):
            mag_order = list(scalings_input.iloc[:, 0].astype(str))
        write_context_species_matrix(
            decisions,
            species_file,
            sample,
            mag_order=mag_order,
        )
        write_context_summary_rows(filter_summary, summary_file, sample)

        species_eligible = species_from_context_species(species_file, sample)
        qc_eligible = set(species_eligible)
        # Eligible for RIPTiDe = species marked 1 that also have expression.
        required_order = [
            mag for mag in expression_order if mag in qc_eligible
        ]
        dropped_by_qc = sorted(set(expression_order) - set(required_order))
        if dropped_by_qc:
            LOGGER.info(
                "[%s] Discordance QC dropped %d expression taxa (e.g. %s)",
                sample,
                len(dropped_by_qc),
                ", ".join(dropped_by_qc[:8]),
            )
        missing_expression = sorted(qc_eligible.difference(expression_order))
        if missing_expression:
            LOGGER.warning(
                "[%s] %d QC-eligible taxa lack expression rows and will "
                "not be RIPTiDe'd (e.g. %s).",
                sample,
                len(missing_expression),
                ", ".join(missing_expression[:8]),
            )

        upsert_context_expression(
            sample_ge, expression_file, sample, set(required_order)
        )
        LOGGER.info(
            "[%s] QC filter updated %s, %s, and %s.",
            sample,
            species_file.name,
            expression_file.name,
            summary_file.name,
        )
    else:
        # No QC: every taxon in the complete expression library for the sample.
        required_order = list(expression_order)
        LOGGER.info(
            "[%s] QC disabled; using all %d taxa from --gene-expr-file.",
            sample,
            len(required_order),
        )

    required = set(required_order)
    ge_by_mag = {mag_id: ge_by_mag[mag_id] for mag_id in required_order}

    context_dir.mkdir(parents=True, exist_ok=True)
    existing_by_mag, preserved_rows = load_context_partitions(out_file, sample)
    reusable_by_mag = (
        {}
        if force
        else {
            mag_id: existing_by_mag[mag_id]
            for mag_id in required_order
            if existing_by_mag.get(mag_id)
        }
    )
    missing_order = [
        mag_id for mag_id in required_order if mag_id not in reusable_by_mag
    ]
    extra_mags = set(existing_by_mag).difference(required)

    summary = {
        "input_gems": 0,
        "expression_gems": len(expression_order),
        "qc_eligible_gems": (
            len(qc_eligible) if qc_eligible is not None else len(required_order)
        ),
        "eligible_gems": len(required_order),
        "reused_gems": len(reusable_by_mag),
        "contextualized_gems": 0,
        "infeasible_gems": 0,
        "low_expression_gems": 0,
        "reused_reaction_rows": sum(
            len(rows) for rows in reusable_by_mag.values()
        ),
        "new_reaction_rows": 0,
        "reaction_rows": sum(
            len(rows) for rows in reusable_by_mag.values()
        ),
    }

    if enable_qc and filter_summary is not None and qc_summary_file is not None:
        emit_qc_summary(
            sample=sample,
            filter_summary=filter_summary,
            context_riptide_existing_n=len(reusable_by_mag),
            context_riptide_missing_n=len(missing_order),
            context_riptide_infeasible_n=0,
            expression_gems=len(expression_order),
            qc_eligible_n=len(qc_eligible or ()),
            required_n=len(required_order),
            audit_only=audit_only,
            qc_summary_file=qc_summary_file,
        )

    if audit_only:
        LOGGER.info(
            "[%s] audit-only: enable_qc=%s; expression=%d; required=%d; "
            "existing context=%d; missing=%d; RIPTiDe skipped.",
            sample,
            enable_qc,
            len(expression_order),
            len(required_order),
            len(reusable_by_mag),
            len(missing_order),
        )
        return summary

    if gem_dir is None:
        raise ValueError("--gem-dir is required unless --audit-only.")
    gem_dir = gem_dir.resolve()
    if not gem_dir.is_dir():
        raise NotADirectoryError(f"GEM directory not found: {gem_dir}")
    gem_files = sorted(gem_dir.glob("*.xml"))
    if not gem_files:
        raise FileNotFoundError(f"No <mag_id>.xml files found in {gem_dir}")
    summary["input_gems"] = len(gem_files)

    LOGGER.info(
        (
            "[%s] RIPTiDe presence check: %d expression taxa; "
            "%d required (qc=%s); %d already in context; "
            "%d missing; %d extra."
        ),
        sample,
        len(expression_order),
        len(required_order),
        enable_qc,
        len(reusable_by_mag),
        len(missing_order),
        len(extra_mags),
    )
    if not missing_order and not extra_mags and not force:
        LOGGER.info(
            "[%s] Context already contains every required species; "
            "skipping RIPTiDe and leaving %s unchanged.",
            sample,
            out_file,
        )
        return summary
    if extra_mags and not force:
        LOGGER.info(
            "[%s] Removing %d context taxa absent from required species list.",
            sample,
            len(extra_mags),
        )

    gem_by_mag = {gem_path.stem: gem_path for gem_path in gem_files}
    missing_gems = [
        mag_id for mag_id in missing_order if mag_id not in gem_by_mag
    ]
    if missing_gems:
        raise FileNotFoundError(
            "Expression taxa requiring RIPTiDe have no GEM XML: "
            + ", ".join(missing_gems[:10])
        )

    temporary_output = out_file.with_name(f".{out_file.name}.tmp")
    try:
        with temporary_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            writer.writerows(preserved_rows)

            for mag_id in required_order:
                reusable_rows = reusable_by_mag.get(mag_id)
                if reusable_rows:
                    writer.writerows(reusable_rows)
            handle.flush()

            mag_iterator = track(
                missing_order,
                description=f"[{sample}] RIPTiDe",
            )
            for mag_id in mag_iterator:
                mag_ge = ge_by_mag.get(mag_id)
                expressed_gene_count = int(
                    mag_ge["expression"].ne(0).sum()
                )
                LOGGER.debug(
                    "Contextualizing %s with %d non-zero genes.",
                    mag_id,
                    expressed_gene_count,
                )
                try:
                    input_model = cobra.io.read_sbml_model(gem_by_mag[mag_id])
                    transcriptome = normalize_ge_for_riptide(mag_ge)
                    riptide_result = riptide.contextualize(
                        model=input_model,
                        transcriptome=transcriptome,
                        silent=True,
                    )
                except Exception as error:
                    summary["infeasible_gems"] += 1
                    LOGGER.warning(
                        "[%s] RIPTiDe failed for %s; context bounds omitted: %s",
                        sample,
                        mag_id,
                        error,
                    )
                    continue
                context_model = riptide_result.model

                rows = list(reaction_rows(context_model, sample, mag_id))
                if not rows:
                    summary["infeasible_gems"] += 1
                    LOGGER.warning(
                        "[%s] RIPTiDe returned no reactions for %s; "
                        "context bounds omitted.",
                        sample,
                        mag_id,
                    )
                    continue
                writer.writerows(rows)
                handle.flush()
                summary["new_reaction_rows"] += len(rows)
                summary["reaction_rows"] += len(rows)
                summary["contextualized_gems"] += 1

                write_and_remove_context_xml(
                    context_model=context_model,
                    output_dir=out_file.parent,
                    sample=sample,
                    mag_id=mag_id,
                )

        os.replace(temporary_output, out_file)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    # Refresh summary file with real infeasible counts after RIPTiDe.
    if enable_qc and filter_summary is not None and qc_summary_file is not None:
        emit_qc_summary(
            sample=sample,
            filter_summary=filter_summary,
            context_riptide_existing_n=len(reusable_by_mag),
            context_riptide_missing_n=len(missing_order),
            context_riptide_infeasible_n=summary["infeasible_gems"],
            expression_gems=len(expression_order),
            qc_eligible_n=len(qc_eligible or ()),
            required_n=len(required_order),
            audit_only=False,
            qc_summary_file=qc_summary_file,
        )

    LOGGER.info(
        (
            "[%s] RIPTiDe complete: %d input GEMs; %d expression taxa; "
            "%d required (qc=%s); "
            "%d reused; %d newly contextualized; %d RIPTiDe-infeasible; "
            "%d reaction rows (%d reused, %d new); output=%s"
        ),
        sample,
        summary["input_gems"],
        summary["expression_gems"],
        summary["eligible_gems"],
        enable_qc,
        summary["reused_gems"],
        summary["contextualized_gems"],
        summary["infeasible_gems"],
        summary["reaction_rows"],
        summary["reused_reaction_rows"],
        summary["new_reaction_rows"],
        out_file,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    if not args.audit_only and args.gem_dir is None:
        raise SystemExit("--gem-dir is required unless --audit-only.")
    contextualize_all(
        sample=args.sample,
        gem_dir=args.gem_dir,
        gene_expr_file=args.gene_expr_file,
        out_file=args.output_file,
        enable_qc=args.enable_qc,
        ga_file=args.ga_file,
        ge_abundance_file=args.ge_abundance_file,
        scalings_file=args.scalings_file,
        qc_summary_file=args.qc_summary_file,
        discordance_floor=args.discordance_floor,
        skip_abundance_prefilter=args.skip_abundance_prefilter,
        audit_only=args.audit_only,
        verbose=args.verbose,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
