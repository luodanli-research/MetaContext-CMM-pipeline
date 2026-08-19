"""Shared resources, table I/O, and plotting utilities for metric modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(
    os.environ.get("MICOM310_PROJECT_DIR", Path(__file__).resolve().parents[2])
).expanduser().resolve()
WORKFLOW_DIR = PROJECT_DIR / "workflow"
RESOURCE_DIR = PROJECT_DIR / "resource"
# GitHub defaults: example/. Override via CLI or set_taxon_input_files().
DEFAULT_INPUT_DIR = PROJECT_DIR / "example" / "inputs"

METABOLITE_FILE = RESOURCE_DIR / "metabolites.csv"
ARB_REACTION_FILE = RESOURCE_DIR / "arb.csv"
TAXON_GE_FILE = DEFAULT_INPUT_DIR / "01_taxon_ge.csv"
TAXON_GUILD_FILE = DEFAULT_INPUT_DIR / "04_taxon_guild.csv"

_taxon_ge_override: Path | None = None
_taxon_guild_override: Path | None = None


def _resolve_data_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def set_taxon_input_files(
    ge_file: str | Path | None = None,
    guild_file: str | Path | None = None,
) -> None:
    """Override default taxon GE / guild CSVs for the current process."""

    global _taxon_ge_override, _taxon_guild_override
    if ge_file is not None:
        _taxon_ge_override = _resolve_data_path(ge_file)
    if guild_file is not None:
        _taxon_guild_override = _resolve_data_path(guild_file)


def load_taxon_ge(
    sample: str,
    ge_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load taxon GE as ``mag_id, ge`` for one sample."""

    path = _resolve_data_path(
        ge_file if ge_file is not None else (_taxon_ge_override or TAXON_GE_FILE)
    )
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Bin Id", sample}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Taxon GE file {path} is missing columns: "
            + ", ".join(sorted(missing))
        )

    out = table[["Bin Id", sample]].copy()
    out.columns = ["mag_id", "ge"]
    out["mag_id"] = out["mag_id"].astype(str)
    out["ge"] = pd.to_numeric(out["ge"], errors="coerce")
    return out


def load_taxon_guild(
    guild_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load the project taxon-to-guild mapping."""

    path = _resolve_data_path(
        guild_file
        if guild_file is not None
        else (_taxon_guild_override or TAXON_GUILD_FILE)
    )
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Bin Id", "Guild"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Taxon guild file {path} is missing columns: "
            + ", ".join(sorted(missing))
        )

    out = table[["Bin Id", "Guild"]].copy()
    out.columns = ["mag_id", "guild"]
    out["mag_id"] = out["mag_id"].astype(str)
    out["guild"] = out["guild"].astype(str)
    return out


def compute_guild_ge(
    sample: str,
    guild_table: pd.DataFrame | None = None,
    ge_file: str | Path | None = None,
) -> pd.DataFrame:
    """Sum taxon GE by guild for one sample."""

    guilds = load_taxon_guild() if guild_table is None else guild_table
    merged = load_taxon_ge(sample, ge_file=ge_file).merge(
        guilds,
        on="mag_id",
        how="inner",
        validate="one_to_one",
    )
    return (
        merged.groupby("guild", as_index=False)["ge"]
        .sum()
        .rename(columns={"ge": "gene_expression"})
        .sort_values("gene_expression", ascending=False)
        .reset_index(drop=True)
    )


def write_csv_atomic(table: pd.DataFrame, output_file: str | Path) -> Path:
    """Write a CSV atomically in its destination directory."""

    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    return output_path


def configure_publication_style() -> None:
    """Apply the shared Python publication-figure style."""

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


# Okabe–Ito without black (black is reserved for formal singles).
# Order favors early pairwise contrast; yellow last (weak on white).
OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
)
FORMAL_SINGLE_NAMES = frozenset({"context", "formal", "ctx"})
FORMAL_SINGLE_COLOR = "#000000"


def format_dataset_label(name: object) -> str:
    """Display dataset labels: underscores → spaces; capitalize first word only."""

    text = str(name).replace("_", " ").strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]


def is_formal_single_name(name: object) -> bool:
    """True when a dataset label is context / formal / ctx (case-insensitive)."""

    return str(name).strip().lower() in FORMAL_SINGLE_NAMES


def _spaced_palette_indices(n: int, length: int) -> list[int]:
    """Return ``n`` unique indices spread across ``[0, length)``."""

    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n >= length:
        return [i % length for i in range(n)]

    raw = [i * (length - 1) / (n - 1) for i in range(n)]
    indices = [int(round(value)) for value in raw]
    for i in range(1, n):
        if indices[i] <= indices[i - 1]:
            indices[i] = indices[i - 1] + 1
    overflow = indices[-1] - (length - 1)
    if overflow > 0:
        indices = [max(0, index - overflow) for index in indices]
        for i in range(1, n):
            if indices[i] <= indices[i - 1]:
                indices[i] = min(length - 1, indices[i - 1] + 1)
    return indices


def okabe_ito_colors(n: int) -> list[str]:
    """Return ``n`` Okabe–Ito colors spaced across the full palette.

    Taking the first ``n`` sequential entries clusters nearby hues. Spacing
    spreads selections over the whole set so small ``n`` still spans the
    chromatic range (e.g. 3 colors → blue / purple / yellow).
    """

    if n <= 0:
        return []
    return [OKABE_ITO[i] for i in _spaced_palette_indices(n, len(OKABE_ITO))]


def assign_category_colors(labels: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Map categorical labels to spaced Okabe–Ito colors (sorted for stability)."""

    ordered = sorted({str(label) for label in labels})
    return dict(zip(ordered, okabe_ito_colors(len(ordered))))


def assign_dataset_colors(
    datasets: list[str] | tuple[str, ...],
    *,
    single_datasets: list[str] | tuple[str, ...] | None = None,
    formal_as_black: bool = True,
) -> dict[str, str]:
    """Map dataset labels to colors.

    Formal single labels (``context`` / ``formal`` / ``ctx``) use black when
    ``formal_as_black`` is True and the label is among ``single_datasets``
    (or among all ``datasets`` if ``single_datasets`` is omitted). Remaining
    labels use Okabe–Ito colors spaced across the full palette.
    """

    singles = (
        {str(name) for name in single_datasets}
        if single_datasets is not None
        else {str(name) for name in datasets}
    )
    auto_labels = [
        str(dataset)
        for dataset in datasets
        if not (
            formal_as_black
            and str(dataset) in singles
            and is_formal_single_name(dataset)
        )
    ]
    auto_colors = okabe_ito_colors(len(auto_labels))
    color_map: dict[str, str] = {}
    auto_index = 0
    for dataset in datasets:
        label = str(dataset)
        if (
            formal_as_black
            and label in singles
            and is_formal_single_name(label)
        ):
            color_map[label] = FORMAL_SINGLE_COLOR
            continue
        color_map[label] = auto_colors[auto_index]
        auto_index += 1
    return color_map


def save_figure(fig, output_stem: str | Path) -> list[Path]:
    """Export one editable PDF and remove obsolete sibling image formats."""

    stem = Path(output_stem).expanduser().resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    output = stem.with_suffix(".pdf")
    for suffix in (".svg", ".png", ".tif", ".tiff"):
        stem.with_suffix(suffix).unlink(missing_ok=True)
    fig.savefig(output, bbox_inches="tight")
    return [output]


def remove_non_pdf_figures(directory: str | Path) -> None:
    """Remove obsolete non-PDF figure exports from one analysis directory."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return
    for suffix in ("*.svg", "*.png", "*.tif", "*.tiff"):
        for path in root.glob(suffix):
            path.unlink()
