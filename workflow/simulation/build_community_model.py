#!/usr/bin/env python3
"""Build a baseline MICOM community model from taxon abundances.

The abundance table must be a CSV with one ``Bin Id`` column and one numeric
column per sample. Inclusion matches MICOM ``Community`` construction: sample
abundances are converted to relative amounts, then taxa with
``rel_abundance > 1e-6`` (strict ``>``) are kept. Taxa with
``rel_abundance <= 1e-6`` are omitted.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import micom as mc
import numpy as np
import pandas as pd

from _cli_utils import configure_logging


LOGGER = logging.getLogger("cmm.build_baseline")
GE_INDEX_COLUMN = "Bin Id"
# MICOM Community(rel_threshold=1e-6) keeps taxonomy.abundance > rel_threshold
# after sum-normalization (see micom.community.Community.__init__).
MICOM_REL_ABUNDANCE_TOLERANCE = 1e-6
GEM_SUFFIX = ".xml"
SOLVER = "cplex"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build one baseline MICOM community model from a sample-specific "
            "abundance column (MICOM rel_abundance > 1e-6 gate)."
        )
    )
    parser.add_argument(
        "--sample",
        required=True,
        help="Sample column to use from the abundance CSV, for example SW60.",
    )
    parser.add_argument(
        "--gem-dir",
        dest="gem_dir",
        required=True,
        type=Path,
        help="Directory containing one <mag_id>.xml GEM per taxon.",
    )
    parser.add_argument(
        "--ge-file",
        dest="ge_file",
        required=True,
        type=Path,
        help=(
            "Taxon abundance CSV containing 'Bin Id' and sample columns "
            "(GE or GA table)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        type=Path,
        help="Directory in which to write the community model and metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logs from MICOM and its dependencies.",
    )
    return parser.parse_args(argv)


def load_sample_ge(ge_file: Path, sample: str) -> pd.Series:
    """Load abundances and keep taxa with MICOM relative abundance > 1e-6."""
    if not ge_file.is_file():
        raise FileNotFoundError(f"Gene-expression file not found: {ge_file}")

    ge_table = pd.read_csv(ge_file)
    if GE_INDEX_COLUMN not in ge_table.columns:
        raise ValueError(
            f"Missing required column {GE_INDEX_COLUMN!r} in {ge_file}."
        )
    if sample not in ge_table.columns:
        available = ", ".join(
            column for column in ge_table.columns if column != GE_INDEX_COLUMN
        )
        raise ValueError(
            f"Sample {sample!r} is not present in {ge_file}. "
            f"Available sample columns: {available}"
        )
    if ge_table[GE_INDEX_COLUMN].duplicated().any():
        duplicates = sorted(
            ge_table.loc[
                ge_table[GE_INDEX_COLUMN].duplicated(keep=False), GE_INDEX_COLUMN
            ].astype(str).unique()
        )
        raise ValueError(
            "Duplicate taxon identifiers in gene-expression file: "
            + ", ".join(duplicates[:10])
        )

    ge_values = pd.to_numeric(ge_table[sample], errors="coerce")
    invalid_mask = ge_values.isna() | ~np.isfinite(ge_values)
    if invalid_mask.any():
        invalid_taxa = ge_table.loc[invalid_mask, GE_INDEX_COLUMN].astype(str)
        raise ValueError(
            f"Non-numeric or non-finite expression values for sample {sample}: "
            + ", ".join(invalid_taxa.head(10))
        )
    if (ge_values < 0).any():
        negative_taxa = ge_table.loc[ge_values < 0, GE_INDEX_COLUMN].astype(str)
        raise ValueError(
            f"Negative expression values for sample {sample}: "
            + ", ".join(negative_taxa.head(10))
        )

    ge_series = pd.Series(
        ge_values.to_numpy(dtype=float),
        index=ge_table[GE_INDEX_COLUMN].astype(str),
        name="ge",
    )
    # MICOM: abundance /= sum, then keep abundance > rel_threshold.
    total = float(ge_series.sum())
    if total <= 0:
        raise ValueError(
            f"No positive abundances for sample {sample} in {ge_file}."
        )
    relative = ge_series / total
    selected_ge = ge_series[relative > MICOM_REL_ABUNDANCE_TOLERANCE]
    if selected_ge.empty:
        raise ValueError(
            f"No taxa have relative abundance > {MICOM_REL_ABUNDANCE_TOLERANCE:g} "
            f"for sample {sample}."
        )
    dropped = int((relative > 0).sum() - len(selected_ge))
    if dropped:
        LOGGER.info(
            "[%s] MICOM 1e-6 gate dropped %d taxa with 0 < rel_abundance <= %g.",
            sample,
            dropped,
            MICOM_REL_ABUNDANCE_TOLERANCE,
        )
    return selected_ge


def make_species_table(
    selected_ge: pd.Series,
    sample: str,
    gem_dir: Path,
) -> pd.DataFrame:
    """Create and validate the MICOM taxon input table."""
    if not gem_dir.is_dir():
        raise NotADirectoryError(f"GEM directory not found: {gem_dir}")

    gem_dir = gem_dir.resolve()
    gem_paths = [gem_dir / f"{mag_id}{GEM_SUFFIX}" for mag_id in selected_ge.index]
    missing_gems = [
        str(gem_path) for gem_path in gem_paths if not gem_path.is_file()
    ]
    if missing_gems:
        preview = "\n  ".join(missing_gems[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_gems)} GEM file(s):\n  {preview}"
        )

    return pd.DataFrame(
        {
            "id": selected_ge.index,
            "file": [str(gem_path) for gem_path in gem_paths],
            # MICOM requires this exact input-column name.
            "abundance": selected_ge.to_numpy(dtype=float),
            "sample_id": sample,
        }
    )


def write_outputs(
    community: mc.Community,
    sample: str,
    out_dir: Path,
    selected_ge: pd.Series,
) -> tuple[Path, Path]:
    """Write the bsl model and build manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{sample}-bsl.pickle"
    manifest_path = out_dir / f"{sample}-manifest.tsv"

    community.to_pickle(model_path)

    requested_taxa = len(selected_ge)
    found_taxa = len(community.taxa)
    retained_taxa = selected_ge.index.intersection(community.taxa)
    found_fraction = found_taxa / requested_taxa
    found_ge_fraction = (
        float(selected_ge.loc[retained_taxa].sum()) / float(selected_ge.sum())
    )
    manifest = pd.DataFrame(
        [
            {
                "reactions": len(community.reactions),
                "metabolites": len(community.metabolites),
                "file": model_path.name,
                "sample_id": sample,
                "requested_taxa": requested_taxa,
                "found_taxa": found_taxa,
                "found_fraction": found_fraction,
                "found_ge_fraction": found_ge_fraction,
            }
        ]
    )
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return model_path, manifest_path


def build_community(
    sample: str,
    gem_dir: Path,
    ge_file: Path,
    out_dir: Path,
    verbose: bool = False,
) -> mc.Community:
    """Build and save a baseline community for one sample."""
    selected_ge = load_sample_ge(ge_file.resolve(), sample)
    species_table = make_species_table(selected_ge, sample, gem_dir)

    LOGGER.info(
        "[%s] Building bsl CMM from %d taxa (rel_abundance > %g).",
        sample,
        len(species_table),
        MICOM_REL_ABUNDANCE_TOLERANCE,
    )
    community = mc.Community(
        species_table,
        solver=SOLVER,
        progress=True,
    )
    community.id = sample

    model_path, manifest_path = write_outputs(
        community,
        sample,
        out_dir.resolve(),
        selected_ge,
    )
    LOGGER.info(
        (
            "[%s] Bsl CMM complete: retained %d/%d taxa; "
            "%d reactions; %d metabolites; model=%s"
        ),
        sample,
        len(community.taxa),
        len(species_table),
        len(community.reactions),
        len(community.metabolites),
        model_path,
    )
    LOGGER.debug("Manifest: %s", manifest_path)
    return community


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    build_community(
        sample=args.sample,
        gem_dir=args.gem_dir,
        ge_file=args.ge_file,
        out_dir=args.output_dir,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
