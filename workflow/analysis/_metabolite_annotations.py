"""Build and cache the metabolite annotation table used by analysis modules.

This is an internal module. Analysis metrics such as AC, EAI, and ARB should
call :func:`ensure_metabolite_table` instead of implementing their own SBML
extraction or maintaining separate metabolite annotation files.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from cobra.core.formula import elements_and_molecular_weights
from cobra.io import read_sbml_model


LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_FILE = (
    Path(__file__).resolve().parents[2] / "resource" / "metabolites.csv"
)

OUTPUT_COLUMNS = [
    "metabolite",
    "name",
    "Formula",
    "molecular_weight",
    "C_number",
    "N_number",
    "S_number",
    "P_number",
]

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)([0-9.]+)?")
_COMPARTMENT_SUFFIX = re.compile(r"_[a-z]$")


@dataclass(frozen=True)
class _MetaboliteRecord:
    """A deterministic metabolite record consolidated across GEMs."""

    metabolite: str
    name: str
    formula: str
    compartment: str
    charge: float | int | None
    source_model: str


def ensure_metabolite_table(
    gem_dir: str | Path | None = None,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    """Return the shared metabolite table, building it only when absent.

    Parameters
    ----------
    gem_dir
        Directory containing one ``<mag_id>.xml`` model per GEM. It may be
        ``None`` only when ``output_file`` already exists.
    output_file
        Cache location shared by the analysis modules.

    Returns
    -------
    pandas.DataFrame
        The existing or newly generated metabolite annotation table.
    """

    output_path = Path(output_file).expanduser().resolve()

    if output_path.is_file():
        LOGGER.info("Using existing metabolite table: %s", output_path)
        return load_metabolite_table(output_path)

    if gem_dir is None:
        raise ValueError(
            "gem_dir is required because the metabolite table does not exist: "
            f"{output_path}"
        )

    return build_metabolite_table(gem_dir, output_path)


def load_metabolite_table(
    table_file: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    """Load and normalize an existing table without rewriting the cache."""

    table_path = Path(table_file).expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"Metabolite table does not exist: {table_path}")

    table = pd.read_csv(table_path)
    missing = [column for column in OUTPUT_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"Metabolite table {table_path} is missing columns: "
            + ", ".join(missing)
        )

    original_rows = len(table)
    duplicate_count = int(table["metabolite"].duplicated().sum())
    table = _normalize_cached_table(table)

    if duplicate_count or len(table) != original_rows:
        LOGGER.warning(
            "Normalized legacy metabolite cache in memory: %d input rows, "
            "%d duplicate rows, %d analysis-ready rows. The cache file was "
            "not modified.",
            original_rows,
            duplicate_count,
            len(table),
        )

    return table


def build_metabolite_table(
    gem_dir: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    """Extract a consistent metabolite table from a batch of finalized GEMs.

    Full compartment-specific metabolite identifiers are consolidated across
    models. When GEMs contain alternative annotations for the same identifier,
    the first occurrence in model-filename order is selected deterministically,
    reproducing the legacy community-level lookup rule. Community-medium
    ``_m`` records are generated specifically from extracellular ``_e``
    records.
    """

    gem_path = Path(gem_dir).expanduser().resolve()
    output_path = Path(output_file).expanduser().resolve()

    if not gem_path.is_dir():
        raise NotADirectoryError(f"GEM directory does not exist: {gem_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing metabolite table: {output_path}"
        )

    model_files = sorted(gem_path.glob("*.xml"))
    if not model_files:
        raise FileNotFoundError(f"No XML GEM files found in: {gem_path}")

    LOGGER.info(
        "Building metabolite table from %d GEMs in %s",
        len(model_files),
        gem_path,
    )

    (
        records,
        name_conflicts,
        formula_conflicts,
        charge_conflicts,
        missing_names,
        missing_formulas,
    ) = _extract_consistent_records(model_files)
    _add_medium_records(records)

    rows = [_annotation_row(record) for record in records.values()]
    table = (
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        .sort_values("metabolite", kind="stable")
        .reset_index(drop=True)
    )

    if table["metabolite"].duplicated().any():
        raise RuntimeError("Internal error: generated metabolite IDs are not unique.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_atomically(table, output_path)

    if name_conflicts:
        LOGGER.warning(
            "Used the first model-sorted name for %d metabolites with "
            "cross-model name differences.",
            len(name_conflicts),
        )
    if formula_conflicts:
        LOGGER.warning(
            "Used the first model-sorted formula for %d metabolites with "
            "cross-model formula differences.",
            len(formula_conflicts),
        )
    if charge_conflicts:
        LOGGER.warning(
            "Used the first model-sorted charge for %d metabolites with "
            "cross-model charge differences.",
            len(charge_conflicts),
        )
    if missing_names:
        LOGGER.warning(
            "%d metabolite identifiers have an empty name in at least one GEM.",
            len(missing_names),
        )
    if missing_formulas:
        LOGGER.warning(
            "%d metabolite identifiers have an empty formula in at least one "
            "GEM; their elemental counts and molecular weight are recorded as "
            "zero when the selected formula is empty.",
            len(missing_formulas),
        )

    missing_weight = int(table["molecular_weight"].isna().sum())
    if missing_weight:
        LOGGER.warning(
            "Molecular weight is blank for %d metabolites whose formulas "
            "contain unsupported placeholder elements.",
            missing_weight,
        )

    LOGGER.info(
        "Saved %d unique metabolite annotations to %s",
        len(table),
        output_path,
    )
    return table


def _extract_consistent_records(
    model_files: Iterable[Path],
) -> tuple[
    dict[str, _MetaboliteRecord],
    set[str],
    set[str],
    set[str],
    set[str],
    set[str],
]:
    records: dict[str, _MetaboliteRecord] = {}
    name_conflicts: set[str] = set()
    formula_conflicts: set[str] = set()
    charge_conflicts: set[str] = set()
    missing_names: set[str] = set()
    missing_formulas: set[str] = set()
    incomplete_records: list[str] = []

    for model_file in model_files:
        model = read_sbml_model(str(model_file))
        source_model = model_file.stem

        for metabolite in model.metabolites:
            met_id = str(metabolite.id).strip()
            name = str(metabolite.name or "").strip()
            formula = str(metabolite.formula or "").strip()
            compartment = str(metabolite.compartment or "").strip()
            charge = metabolite.charge

            if not met_id or not compartment:
                incomplete_records.append(
                    f"{source_model}:{met_id or '<missing-id>'} has an empty "
                    "metabolite ID or compartment"
                )
                continue
            if not name:
                missing_names.add(met_id)
            if not formula:
                missing_formulas.add(met_id)

            candidate = _MetaboliteRecord(
                metabolite=met_id,
                name=name,
                formula=formula,
                compartment=compartment,
                charge=charge,
                source_model=source_model,
            )
            existing = records.get(met_id)

            if existing is None:
                records[met_id] = candidate
                continue

            if candidate.formula != existing.formula:
                formula_conflicts.add(met_id)
            if candidate.compartment != existing.compartment:
                raise ValueError(
                    f"Metabolite {met_id!r} has inconsistent compartments in "
                    f"{existing.source_model} and {source_model}: "
                    f"{existing.compartment!r} vs {candidate.compartment!r}"
                )
            if candidate.charge != existing.charge:
                charge_conflicts.add(met_id)
            if candidate.name != existing.name:
                name_conflicts.add(met_id)

    if incomplete_records:
        preview = "\n  - ".join(incomplete_records[:20])
        remainder = len(incomplete_records) - min(20, len(incomplete_records))
        suffix = (
            f"\n  ... and {remainder} more incomplete records" if remainder else ""
        )
        raise ValueError(
            "Metabolite metadata are incomplete in the GEM collection:\n  - "
            + preview
            + suffix
        )

    return (
        records,
        name_conflicts,
        formula_conflicts,
        charge_conflicts,
        missing_names,
        missing_formulas,
    )


def _add_medium_records(records: dict[str, _MetaboliteRecord]) -> None:
    """Add MICOM community-medium records from extracellular metabolites."""

    additions: dict[str, _MetaboliteRecord] = {}

    for met_id, record in list(records.items()):
        if not met_id.endswith("_e"):
            continue

        medium_id = _COMPARTMENT_SUFFIX.sub("_m", met_id)
        existing = records.get(medium_id)

        if existing is not None:
            if existing.formula != record.formula:
                raise ValueError(
                    f"Cannot derive {medium_id} from {met_id}: existing medium "
                    f"formula {existing.formula!r} differs from extracellular "
                    f"formula {record.formula!r}."
                )
            continue

        additions[medium_id] = _MetaboliteRecord(
            metabolite=medium_id,
            name=record.name,
            formula=record.formula,
            compartment="m",
            charge=record.charge,
            source_model=record.source_model,
        )

    records.update(additions)


def _normalize_cached_table(table: pd.DataFrame) -> pd.DataFrame:
    """Apply the same ID and formula rules used by a fresh GEM extraction."""

    unique = table.drop_duplicates(subset=["metabolite"], keep="first")
    non_medium = unique[
        ~unique["metabolite"].astype(str).str.endswith("_m")
    ]

    records: dict[str, _MetaboliteRecord] = {}
    for row in non_medium.itertuples(index=False):
        met_id = str(row.metabolite).strip()
        formula = "" if pd.isna(row.Formula) else str(row.Formula).strip()
        name = "" if pd.isna(row.name) else str(row.name).strip()
        suffix_match = _COMPARTMENT_SUFFIX.search(met_id)
        compartment = suffix_match.group(0)[1:] if suffix_match else ""

        records[met_id] = _MetaboliteRecord(
            metabolite=met_id,
            name=name,
            formula=formula,
            compartment=compartment,
            charge=None,
            source_model="<cached-table>",
        )

    _add_medium_records(records)
    rows = [_annotation_row(record) for record in records.values()]
    return (
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        .sort_values("metabolite", kind="stable")
        .reset_index(drop=True)
    )


def _annotation_row(record: _MetaboliteRecord) -> dict[str, object]:
    element_counts, unknown_elements = _parse_formula(record.formula)

    if unknown_elements:
        molecular_weight: float | None = None
    else:
        molecular_weight = round(
            sum(
                elements_and_molecular_weights[element] * count
                for element, count in element_counts.items()
            ),
            5,
        )

    return {
        "metabolite": record.metabolite,
        "name": record.name,
        "Formula": record.formula,
        "molecular_weight": molecular_weight,
        "C_number": _clean_count(element_counts.get("C", 0)),
        "N_number": _clean_count(element_counts.get("N", 0)),
        "S_number": _clean_count(element_counts.get("S", 0)),
        "P_number": _clean_count(element_counts.get("P", 0)),
    }


def _parse_formula(formula: str) -> tuple[dict[str, float], set[str]]:
    if not formula:
        return {}, set()

    if "," in formula:
        raise ValueError(
            f"Multiple comma-separated formulas are not supported: {formula!r}"
        )

    matches = list(_FORMULA_TOKEN.finditer(formula))
    if not matches or "".join(match.group(0) for match in matches) != formula:
        raise ValueError(f"Unsupported metabolite formula syntax: {formula!r}")

    counts: dict[str, float] = {}
    for match in matches:
        element = match.group(1)
        count = float(match.group(2)) if match.group(2) else 1.0
        counts[element] = counts.get(element, 0.0) + count

    unknown = set(counts).difference(elements_and_molecular_weights)
    return counts, unknown


def _clean_count(value: float) -> int | float:
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _write_csv_atomically(table: pd.DataFrame, output_path: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        table.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
