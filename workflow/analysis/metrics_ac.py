#!/usr/bin/env python3
"""Calculate and visualize AC for single and batch simulation datasets.

Interaction methods (``--method``):

* ``metacontext_interaction`` (default): fast ANA6-3 vectorized path.
* ``micom_interaction``: official ``micom.interaction`` fallback.

Both try tolerance ``1e-6`` then ``1e-9``. Per-flux caches are written beside
ctx flux tables as ``*-ctx-interactions.metacontext.csv`` or
``*-ctx-interactions.micom.csv``.

Figures: guild AC networks plus ANA2-3-style co-consumption / cross-feeding
heatmaps and elemental networks (one PDF per single or batch dataset; batch
uses member medians).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "micom310-matplotlib"),
)

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
import pandas as pd
from micom import load_pickle
from micom.interaction import interactions, summarize_interactions
from micom.workflows import GrowthResults

from _analysis_utils import (
    TAXON_GE_FILE,
    TAXON_GUILD_FILE,
    assign_category_colors,
    compute_guild_ge,
    configure_publication_style,
    format_dataset_label,
    load_taxon_ge,
    load_taxon_guild,
    remove_non_pdf_figures,
    save_figure,
    set_taxon_input_files,
    write_csv_atomic,
)
from _metabolite_annotations import load_metabolite_table
from _metric_datasets import (
    CTX_SUFFIX,
    FluxJob,
    add_dataset_arguments,
    discover_jobs,
    analysis_output_dir,
    parse_datasets,
    partition_metric_rows,
    single_value_table,
    summarize_batch_values,
)


ELEMENTS = ("mass", "C", "N", "P", "S")
HEATMAP_ELEMENTS = ("C", "N", "P", "S")
HEATMAP_TOP_K = 3
HEATMAP_CLUSTER_METHOD = "weighted"
DEFAULT_HEATMAP_GUILDS = ("GI", "GII")
CF_CMAP = LinearSegmentedColormap.from_list(
    "ac_cf",
    ["#FF6699", "#203764", "cyan"],
)
CC_CMAP = LinearSegmentedColormap.from_list(
    "ac_cc",
    ["grey", "#09FF3E"],
)
EPSILON = 1e-12
# Within-panel edge highlight for mass / elemental AC networks.
EDGE_COLOR_MAX = "#111111"
EDGE_COLOR_OTHER = "#C8C8C8"
TOLERANCES = (1e-6, 1e-9)
DEFAULT_JOBS = 1
DEFAULT_THREADS = 1
METHOD_METACONTEXT = "metacontext_interaction"
METHOD_MICOM = "micom_interaction"
INTERACTION_METHODS = (METHOD_METACONTEXT, METHOD_MICOM)
DEFAULT_METHOD = METHOD_METACONTEXT
# Short tags used only in on-disk cache filenames.
INTERACTION_CACHE_TAGS = {
    METHOD_METACONTEXT: "metacontext",
    METHOD_MICOM: "micom",
}
INTERACTIONS_SUFFIX = "-ctx-interactions.csv"
ABUNDANCE_SUFFIX = "-abundances.csv"
CACHED_INTERACTION_COLUMNS = {
    "sample_id",
    "focal",
    "partner",
    "mass_flux",
    "C_flux",
    "N_flux",
    "P_flux",
    "S_flux",
    "class",
}


def interaction_path_for_flux(
    flux_file: str | Path,
    method: str = DEFAULT_METHOD,
) -> Path:
    """Place the per-simulation interaction cache beside the ctx flux table.

    Cache files use short tags (``metacontext`` / ``micom``) so
    ``metacontext_interaction`` and ``micom_interaction`` results are never
    reused interchangeably.
    """
    path = Path(flux_file)
    name = path.name
    if method not in INTERACTION_METHODS:
        raise ValueError(
            f"Unknown interaction method {method!r}; "
            f"expected one of {INTERACTION_METHODS}."
        )
    if name.endswith(CTX_SUFFIX):
        stem = name[: -len(CTX_SUFFIX)]
    else:
        stem = path.stem
    tag = INTERACTION_CACHE_TAGS[method]
    return path.with_name(f"{stem}-ctx-interactions.{tag}.csv")


def abundance_path_for_sample(context_cmm_dir: str | Path, sample: str) -> Path:
    """Sidecar abundance table next to ``<sample>-ctx.pickle``."""
    return Path(context_cmm_dir).expanduser().resolve() / f"{sample}{ABUNDANCE_SUFFIX}"


def load_cached_interactions(path: Path) -> pd.DataFrame | None:
    """Load a previously written interaction summary, or None if unusable."""
    if not path.is_file():
        return None
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, ValueError):
        return None
    if table.empty:
        # Empty caches are valid (no detectable interactions).
        return _empty_summary()
    missing = CACHED_INTERACTION_COLUMNS.difference(table.columns)
    if missing:
        return None
    return table


def find_context_cmm(context_cmm_dir: str | Path, sample: str) -> Path:
    """Resolve the constrained community paired with the ctx flux table."""
    directory = Path(context_cmm_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Context directory does not exist: {directory}")
    model_path = directory / f"{sample}-ctx.pickle"
    if not model_path.is_file():
        raise FileNotFoundError(f"Context community not found: {model_path}")
    return model_path


def write_abundance_sidecar(
    context_cmm_dir: str | Path,
    sample: str,
    abundances: Mapping[str, float] | pd.Series,
) -> Path:
    """Write ``<sample>-abundances.csv`` beside the ctx community pickle."""
    path = abundance_path_for_sample(context_cmm_dir, sample)
    if isinstance(abundances, pd.Series):
        table = abundances.rename("abundance").rename_axis("taxon").reset_index()
    else:
        table = pd.DataFrame(
            {
                "taxon": list(abundances.keys()),
                "abundance": list(abundances.values()),
            }
        )
    table["taxon"] = table["taxon"].astype(str)
    table["abundance"] = pd.to_numeric(table["abundance"], errors="coerce")
    write_csv_atomic(table, path)
    return path


def ensure_sample_abundances(
    context_cmm_dir: str | Path,
    sample: str,
) -> dict[str, float]:
    """Return sample abundances, building a sidecar from the ctx pickle if needed."""
    directory = Path(context_cmm_dir).expanduser().resolve()
    path = abundance_path_for_sample(directory, sample)
    if path.is_file():
        table = pd.read_csv(path)
        if {"taxon", "abundance"}.issubset(table.columns):
            table = table.dropna(subset=["taxon"])
            return {
                str(taxon): float(value)
                for taxon, value in zip(table["taxon"], table["abundance"])
                if pd.notna(value)
            }
    model_path = find_context_cmm(directory, sample)
    print(f"[AC] building abundance sidecar from {model_path.name}", flush=True)
    community = load_pickle(model_path)
    abundances = {
        str(taxon): float(value) for taxon, value in community.abundances.items()
    }
    write_abundance_sidecar(directory, sample, abundances)
    return abundances


def ensure_abundances_for_samples(
    context_cmm_dir: str | Path,
    samples: list[str],
) -> dict[str, dict[str, float]]:
    """Load or create abundance maps once per sample before job fan-out."""
    return {
        sample: ensure_sample_abundances(context_cmm_dir, sample) for sample in samples
    }


def _empty_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["metabolite", "focal", "partner", "class", "flux", "sample_id"]
    )


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "focal",
            "partner",
            "metabolite",
            "mass_flux",
            "C_flux",
            "N_flux",
            "P_flux",
            "S_flux",
        ]
    )


def exchanges_from_flux(flux: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Vectorized melt of community exchange fluxes into MICOM long format."""
    frames: list[pd.DataFrame] = []
    taxon_labels = flux.index.astype(str)
    non_medium = flux.loc[taxon_labels != "medium"]
    ex_e = [
        column
        for column in flux.columns
        if str(column).startswith("EX_") and str(column).endswith("_e")
    ]
    if ex_e and not non_medium.empty:
        long = (
            non_medium.loc[:, ex_e]
            .stack(future_stack=True)
            .rename("flux")
            .reset_index()
        )
        long.columns = ["taxon", "reaction", "flux"]
        frames.append(long)

    medium_mask = taxon_labels == "medium"
    if medium_mask.any():
        medium = flux.loc[medium_mask]
        ex_m = [
            column
            for column in flux.columns
            if str(column).startswith("EX_") and str(column).endswith("_m")
        ]
        if ex_m:
            long = (
                medium.loc[:, ex_m]
                .stack(future_stack=True)
                .rename("flux")
                .reset_index()
            )
            long.columns = ["taxon", "reaction", "flux"]
            frames.append(long)

    if not frames:
        return pd.DataFrame(
            columns=[
                "taxon",
                "sample",
                "sample_id",
                "reaction",
                "flux",
                "metabolite",
                "direction",
            ]
        )

    exchanges = pd.concat(frames, ignore_index=True)
    exchanges = exchanges[pd.notna(exchanges["flux"])].copy()
    exchanges["taxon"] = exchanges["taxon"].astype(str)
    exchanges["reaction"] = exchanges["reaction"].astype(str)
    exchanges["flux"] = pd.to_numeric(exchanges["flux"], errors="coerce")
    exchanges = exchanges[pd.notna(exchanges["flux"])].copy()
    exchanges["sample"] = sample_id
    exchanges["sample_id"] = sample_id
    exchanges["metabolite"] = exchanges["reaction"].map(
        lambda value: str(value).removeprefix("EX_")
    )
    exchanges["direction"] = np.where(exchanges["flux"] < 0, "import", "export")
    return exchanges[
        [
            "taxon",
            "sample",
            "sample_id",
            "reaction",
            "flux",
            "metabolite",
            "direction",
        ]
    ]


def prepare_exchange_table(
    flux_file: Path,
    abundance_map: Mapping[str, float],
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Load flux CSV into abundance-annotated exchange and growth tables."""
    flux = pd.read_csv(flux_file, index_col=0)
    if "Growth" not in flux.columns:
        raise ValueError(f"Flux table lacks the required Growth column: {flux_file}")

    sample_id = flux_file.name.split("-", 1)[0]
    exchanges = exchanges_from_flux(flux, sample_id)
    if exchanges.empty:
        raise RuntimeError(f"No exchange fluxes found in {flux_file}")

    growth_table = flux[["Growth"]].rename(columns={"Growth": "growth_rate"})
    growth_table.index.name = "taxon"
    growth_table = growth_table.reset_index()
    growth_table["taxon"] = growth_table["taxon"].astype(str)
    growth_table = growth_table[growth_table["taxon"] != "medium"].reset_index(
        drop=True
    )
    exchanges["abundance"] = exchanges["taxon"].map(abundance_map)
    base = exchanges[
        [
            "taxon",
            "sample",
            "sample_id",
            "reaction",
            "flux",
            "abundance",
            "metabolite",
            "direction",
        ]
    ].copy()
    return sample_id, base, growth_table


def summarize_interactions_metacontext(
    exchanges_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    *,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized interaction summary (ANA6-3 / metacontext_interaction path).

    Returns ``(detail_rows, pair_summary)``. Detail rows are one metabolite-
    level interaction per focal/partner pair; the summary matches the bak
    ``summarize_interactions_vectorized`` aggregation used for AC.
    """
    empty_summary = pd.DataFrame(
        columns=[
            "sample_id",
            "focal",
            "partner",
            "class",
            "flux",
            "mass_flux",
            "C_flux",
            "N_flux",
            "n_ints",
            "P_flux",
            "S_flux",
        ]
    )
    active = exchanges_df[
        (exchanges_df["taxon"] != "medium")
        & ((exchanges_df["flux"].abs() * exchanges_df["abundance"]) > tolerance)
    ].copy()
    if active.empty:
        return pd.DataFrame(), empty_summary

    active["weighted_flux"] = active["flux"].abs() * active["abundance"]
    left_cols = [
        "sample_id",
        "metabolite",
        "taxon",
        "direction",
        "weighted_flux",
    ]
    merged = active[left_cols].merge(
        active[left_cols],
        on=["sample_id", "metabolite"],
        suffixes=("_focal", "_partner"),
    )
    merged = merged[merged["taxon_focal"] != merged["taxon_partner"]].copy()
    if merged.empty:
        return pd.DataFrame(), empty_summary

    both_import = (
        (merged["direction_focal"] == "import")
        & (merged["direction_partner"] == "import")
    )
    focal_export_partner_import = (
        (merged["direction_focal"] == "export")
        & (merged["direction_partner"] == "import")
    )
    focal_import_partner_export = (
        (merged["direction_focal"] == "import")
        & (merged["direction_partner"] == "export")
    )
    merged = merged[
        both_import | focal_export_partner_import | focal_import_partner_export
    ].copy()
    if merged.empty:
        return pd.DataFrame(), empty_summary

    merged["class"] = np.select(
        [
            both_import.loc[merged.index],
            focal_export_partner_import.loc[merged.index],
            focal_import_partner_export.loc[merged.index],
        ],
        ["co-consumed", "provided", "received"],
        default="unknown",
    )
    merged = merged[merged["class"] != "unknown"].copy()
    merged["flux"] = merged[["weighted_flux_focal", "weighted_flux_partner"]].min(
        axis=1
    )
    merged = merged.rename(
        columns={
            "taxon_focal": "focal",
            "taxon_partner": "partner",
        }
    )

    annot_cols = [
        "metabolite",
        "molecular_weight",
        "C_number",
        "N_number",
        "P_number",
        "S_number",
    ]
    missing_annot = [column for column in annot_cols if column not in annot_df.columns]
    if missing_annot:
        raise ValueError(
            "Metabolite annotation table lacks columns: "
            + ", ".join(missing_annot)
        )
    merged = merged.merge(annot_df[annot_cols], on="metabolite", how="left")
    for column in annot_cols[1:]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    merged["mass_flux_component"] = merged["flux"] * merged["molecular_weight"] * 1e-3
    merged["C_flux_component"] = merged["flux"] * merged["C_number"]
    merged["N_flux_component"] = merged["flux"] * merged["N_number"]
    merged["P_flux_component"] = merged["flux"] * merged["P_number"]
    merged["S_flux_component"] = merged["flux"] * merged["S_number"]

    detail = merged[
        [
            "sample_id",
            "focal",
            "partner",
            "class",
            "metabolite",
            "flux",
            "molecular_weight",
            "C_number",
            "N_number",
            "P_number",
            "S_number",
        ]
    ].copy()
    summary = (
        merged.groupby(["sample_id", "focal", "partner", "class"], as_index=False)
        .agg(
            flux=("flux", "sum"),
            mass_flux=("mass_flux_component", "sum"),
            C_flux=("C_flux_component", "sum"),
            N_flux=("N_flux_component", "sum"),
            n_ints=("metabolite", "count"),
            P_flux=("P_flux_component", "sum"),
            S_flux=("S_flux_component", "sum"),
        )
    )
    return detail, summary


def build_interaction_tables_metacontext(
    flux_file: Path,
    abundance_map: Mapping[str, float],
    annotation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Default fast path: ANA6-3 vectorized MetaContext interaction algorithm."""
    _sample_id, base, _growth_table = prepare_exchange_table(flux_file, abundance_map)
    detail = pd.DataFrame()
    summary = pd.DataFrame()
    for tolerance in TOLERANCES:
        # Same schedule as micom_interaction: absolute flux gate inside each round.
        filtered = base[base["flux"].abs() > tolerance].copy()
        filtered["tolerance"] = tolerance
        detail, summary = summarize_interactions_metacontext(
            filtered,
            annotation,
            tolerance=tolerance,
        )
        if not summary.empty:
            break
    if summary.empty:
        return pd.DataFrame(), _empty_summary()
    for column in ("C_flux", "N_flux", "P_flux", "S_flux"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
    return detail, summary


def build_interaction_tables_micom(
    flux_file: Path,
    abundance_map: Mapping[str, float],
    annotation: pd.DataFrame,
    *,
    threads: int = DEFAULT_THREADS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback path: official ``micom.interaction.interactions``."""
    _sample_id, base, growth_table = prepare_exchange_table(flux_file, abundance_map)
    annotation_ready = annotation.copy()
    if "metabolite" not in annotation_ready.columns:
        raise ValueError("Metabolite annotation table lacks a metabolite column.")

    full_cn = pd.DataFrame()
    worker_threads = max(1, int(threads))
    for tolerance in TOLERANCES:
        attempt = base[base["flux"].abs() > tolerance].copy()
        attempt["tolerance"] = tolerance
        active = (
            (attempt["taxon"] != "medium")
            & ((attempt["flux"].abs() * attempt["abundance"]) > tolerance)
        )
        active_taxa = set(attempt.loc[active, "taxon"])
        growth_attempt = growth_table[
            growth_table["taxon"].isin(active_taxa)
        ].reset_index(drop=True)
        if attempt.empty or growth_attempt.empty:
            continue
        result = GrowthResults(
            growth_rates=growth_attempt,
            exchanges=attempt,
            annotations=annotation_ready,
        )
        full_cn = interactions(
            result,
            taxa=None,
            threads=worker_threads,
            progress=False,
        )
        if not full_cn.empty and "metabolite" in full_cn.columns:
            break

    if full_cn.empty:
        return pd.DataFrame(), _empty_summary()
    summary_cn = summarize_interactions(full_cn)

    full_ps = full_cn.copy()
    atom_map = annotation_ready.set_index("metabolite")
    full_ps["C_number"] = full_ps["metabolite"].map(atom_map["P_number"]).fillna(
        full_ps["C_number"]
    )
    full_ps["N_number"] = full_ps["metabolite"].map(atom_map["S_number"]).fillna(
        full_ps["N_number"]
    )
    summary_ps = summarize_interactions(full_ps).rename(
        columns={"C_flux": "P_flux", "N_flux": "S_flux"}
    )
    merge_keys = [
        key
        for key in ("sample_id", "focal", "partner", "class")
        if key in summary_cn.columns and key in summary_ps.columns
    ]
    summary = summary_cn.merge(
        summary_ps[merge_keys + ["P_flux", "S_flux"]],
        on=merge_keys,
        how="left",
    )

    for column in ("C_flux", "N_flux", "P_flux", "S_flux"):
        if column not in summary.columns:
            summary[column] = 0.0
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0.0)
    return full_cn, summary


def build_interaction_tables(
    flux_file: Path,
    abundance_map: Mapping[str, float],
    annotation: pd.DataFrame,
    *,
    method: str = DEFAULT_METHOD,
    threads: int = DEFAULT_THREADS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch interaction construction by ``--method``."""
    if method == METHOD_METACONTEXT:
        return build_interaction_tables_metacontext(
            flux_file, abundance_map, annotation
        )
    if method == METHOD_MICOM:
        return build_interaction_tables_micom(
            flux_file,
            abundance_map,
            annotation,
            threads=threads,
        )
    raise ValueError(
        f"Unknown interaction method {method!r}; "
        f"expected one of {INTERACTION_METHODS}."
    )


def aggregate_guild_pairs(
    interactions_table: pd.DataFrame,
    guild_table: pd.DataFrame,
    sample: str,
    *,
    epsilon: float = EPSILON,
) -> pd.DataFrame:
    """Aggregate metabolite exchange into guild-pair AC values."""
    mag_to_guild = guild_table.set_index("mag_id")["guild"].to_dict()
    table = interactions_table.copy()
    table["focal"] = table["focal"].astype(str)
    table["partner"] = table["partner"].astype(str)
    table["taxon_guild"] = table["focal"].map(mag_to_guild)
    table["partner_guild"] = table["partner"].map(mag_to_guild)
    table = table.dropna(subset=["taxon_guild", "partner_guild"])
    table = table[table["taxon_guild"] != table["partner_guild"]]

    guilds = sorted(set(guild_table["guild"].dropna().astype(str)))
    records: list[dict[str, object]] = []
    for guild1, guild2 in combinations(guilds, 2):
        pair = table[
            ((table["taxon_guild"] == guild1) & (table["partner_guild"] == guild2))
            | ((table["taxon_guild"] == guild2) & (table["partner_guild"] == guild1))
        ]
        for element in ELEMENTS:
            value_column = "mass_flux" if element == "mass" else f"{element}_flux"
            values = pd.to_numeric(pair.get(value_column, 0.0), errors="coerce").fillna(0.0)
            types = pair.get("class", pd.Series("", index=pair.index)).astype(str)
            provided = float(
                values[
                    (types == "provided")
                    & (pair["taxon_guild"] == guild1)
                    & (pair["partner_guild"] == guild2)
                ].sum()
            )
            received = float(
                values[
                    (types == "provided")
                    & (pair["taxon_guild"] == guild2)
                    & (pair["partner_guild"] == guild1)
                ].sum()
            )
            direction_1 = float(
                values[
                    (types == "co-consumed")
                    & (pair["taxon_guild"] == guild1)
                    & (pair["partner_guild"] == guild2)
                ].sum()
            )
            direction_2 = float(
                values[
                    (types == "co-consumed")
                    & (pair["taxon_guild"] == guild2)
                    & (pair["partner_guild"] == guild1)
                ].sum()
            )
            cross_feeding = (
                float(np.sqrt(provided * received))
                if provided > 0 and received > 0
                else 0.0
            )
            co_consumption = max(direction_1, direction_2)
            records.append(
                {
                    "sample_id": sample,
                    "guild_pair": f"{guild1}__{guild2}",
                    "guild_1": guild1,
                    "guild_2": guild2,
                    "element": element,
                    "provided": provided,
                    "received": received,
                    "cross_feeding": cross_feeding,
                    "co_consumption": co_consumption,
                    "AC": cross_feeding / (co_consumption + epsilon),
                }
            )
    return pd.DataFrame.from_records(records)


def extract_mass_metrics(pair_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_id",
        "guild_pair",
        "guild_1",
        "guild_2",
        "provided",
        "received",
        "cross_feeding",
        "co_consumption",
        "AC",
    ]
    output = (
        pair_table.loc[pair_table["element"] == "mass", columns]
        .rename(columns={"AC": "AC_mass"})
        .reset_index(drop=True)
    )
    return output[
        output["guild_pair"].isin(("GI__GII", "GI__GIII", "GII__GIII"))
    ].reset_index(drop=True)


def _rescale(values: pd.Series, low: float, high: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(float)
    if len(numeric) == 0:
        return numeric
    if np.allclose(numeric.max(), numeric.min()):
        return np.full(len(numeric), (low + high) / 2)
    return low + (numeric - numeric.min()) * (high - low) / (
        numeric.max() - numeric.min()
    )


def _layout_by_ac(graph: nx.Graph, seed: int = 42) -> dict:
    """Place nodes so stronger AC edges are shorter (closer guild pairs).

    NetworkX spring forces use edge weight as attraction strength, so higher
    AC pulls guilds together. Soft-rescale AC into a moderate range to keep
    layouts stable when AC values differ by orders of magnitude.
    """
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return nx.circular_layout(graph)

    raw_weights = pd.Series(
        [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    )
    # Stronger AC → stronger spring → shorter drawn edge.
    # Keep the attraction ratio mild so length contrast stays subtle.
    layout_weights = _rescale(raw_weights, 1.5, 10.0)
    for (u, v), layout_w in zip(graph.edges(), layout_weights):
        graph.edges[u, v]["layout_weight"] = float(layout_w)

    return nx.spring_layout(
        graph,
        weight="layout_weight",
        seed=seed,
        k=1.5 / max(np.sqrt(graph.number_of_nodes()), 1.0),
        iterations=250,
    )


def plot_network(
    metrics: pd.DataFrame,
    guild_ge: pd.DataFrame,
    sample: str,
    output_base: Path,
) -> pd.DataFrame:
    """Draw one single-dataset AC network; node size represents guild-level GE."""
    selected = guild_ge.nlargest(3, "gene_expression").copy()
    selected_guilds = selected["guild"].tolist()
    edges = metrics[
        metrics["guild_1"].isin(selected_guilds)
        & metrics["guild_2"].isin(selected_guilds)
        & (metrics["AC_mass"] > 0)
    ].copy()

    graph = nx.Graph()
    for row in selected.itertuples(index=False):
        graph.add_node(row.guild, gene_expression=float(row.gene_expression))
    for row in edges.itertuples(index=False):
        graph.add_edge(row.guild_1, row.guild_2, weight=float(row.AC_mass))

    configure_publication_style()
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    positions = _layout_by_ac(graph)
    node_sizes = _rescale(selected.set_index("guild").loc[list(graph.nodes)]["gene_expression"], 900, 7500)
    guild_palette = assign_category_colors(list(graph.nodes))
    node_colors = [guild_palette[str(node)] for node in graph.nodes]
    edge_list = list(graph.edges)
    edge_weights = [float(graph.edges[edge]["weight"]) for edge in edge_list]
    weight_min = float(min(edge_weights)) if edge_weights else 0.0
    weight_max = float(max(edge_weights)) if edge_weights else 0.0
    edge_widths = [
        _scale_range(weight, 0.4, 16.0, weight_min, weight_max)
        for weight in edge_weights
    ]
    edge_colors = _edge_colors_by_panel_max(edge_weights)

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=1.4,
        ax=ax,
    )
    _draw_weighted_edges_with_labels(
        graph,
        positions,
        ax=ax,
        edge_list=edge_list,
        edge_widths=edge_widths,
        edge_colors=edge_colors,
        edge_labels={
            edge: f"{weight:.2f}"
            for edge, weight in zip(edge_list, edge_weights)
        },
        font_size=11,
        alpha=1.0,
        node_size=node_sizes,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=13,
        font_weight="bold",
        font_color="white",
        ax=ax,
    )
    ax.set_title(f"{sample} anabolic–catabolic coupling", fontsize=14)
    ax.text(
        0.5,
        -0.04,
        "Node area: GE · Edge width/label: AC · Edge length: inverse AC (closer = stronger)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        color="#555555",
    )
    ax.axis("off")
    fig.tight_layout()
    save_figure(fig, output_base)
    plt.close(fig)

    return edges[
        ["sample_id", "guild_pair", "guild_1", "guild_2", "AC_mass"]
    ].reset_index(drop=True)


def calculate_job(
    job: FluxJob,
    guild_table: pd.DataFrame,
    annotation: pd.DataFrame,
    abundance_map: Mapping[str, float],
    *,
    job_index: int | None = None,
    job_total: int | None = None,
    force: bool = False,
    epsilon: float = EPSILON,
    threads: int = DEFAULT_THREADS,
    method: str = DEFAULT_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate member-level AC values and return its interaction summary."""

    progress_label = f"{job.dataset}/{job.member}/{job.sample}"
    job_tag = (
        f"job {job_index}/{job_total}: {progress_label}"
        if job_index is not None and job_total is not None
        else progress_label
    )
    cache_path = interaction_path_for_flux(job.ctx_flux, method=method)
    interactions_table = None if force else load_cached_interactions(cache_path)
    if interactions_table is not None:
        print(f"[AC] reuse ({method}) {job_tag}", flush=True)
    else:
        print(f"[AC] starting ({method}) {job_tag}", flush=True)
        _, interactions_table = build_interaction_tables(
            job.ctx_flux,
            abundance_map,
            annotation,
            method=method,
            threads=threads,
        )
        write_csv_atomic(interactions_table, cache_path)
    pairs = aggregate_guild_pairs(
        interactions_table, guild_table, job.sample, epsilon=epsilon
    )
    metrics = extract_mass_metrics(pairs).copy()
    metrics.insert(0, "member", job.member)
    metrics.insert(0, "sensitivity_type", job.sensitivity_type)
    metrics.insert(0, "input_type", job.input_type)
    metrics.insert(0, "dataset", job.dataset)
    metrics["metric"] = metrics["guild_pair"].map(lambda value: f"AC_{value}")
    metrics["value"] = metrics["AC_mass"]
    metrics["ctx_flux"] = str(job.ctx_flux)
    metrics["interaction_method"] = method
    print(f"[AC] complete ({method}) {job_tag}", flush=True)
    return metrics, interactions_table


def _calculate_job_payload(
    payload: dict[str, object],
) -> tuple[int, pd.DataFrame, pd.DataFrame, FluxJob]:
    """Process-pool entry: unpack a picklable job payload."""
    job = payload["job"]
    metrics, interactions_table = calculate_job(
        job,
        payload["guild_table"],
        payload["annotation"],
        payload["abundance_map"],
        job_index=payload["job_index"],
        job_total=payload["job_total"],
        force=payload["force"],
        epsilon=payload["epsilon"],
        threads=payload["threads"],
        method=payload["method"],
    )
    return payload["job_index"], metrics, interactions_table, job


def resolve_ac_concurrency(
    jobs: int,
    threads: int,
    method: str = DEFAULT_METHOD,
) -> tuple[int, int]:
    """Resolve job/thread counts; MICOM nested pools only apply to micom path."""
    job_workers = max(1, int(jobs))
    micom_threads = max(1, int(threads))
    if method != METHOD_MICOM:
        return job_workers, 1
    if job_workers > 1 and micom_threads > 1:
        print(
            "[AC] --jobs > 1 forces per-job --threads=1 "
            "(avoids nested MICOM process pools)",
            flush=True,
        )
        micom_threads = 1
    return job_workers, micom_threads


def plot_datasets(
    table: pd.DataFrame,
    batch_statistics: pd.DataFrame,
    samples: list[str],
    output: Path,
) -> None:
    """Create one four-sample network figure per single or batch dataset."""

    configure_publication_style()
    remove_non_pdf_figures(output)
    normalized = table.rename(columns={"sample_id": "sample"})
    singles, batches = partition_metric_rows(normalized)
    datasets = list(table["dataset"].drop_duplicates())

    for dataset in datasets:
        is_batch = bool(
            (batches["dataset"].eq(dataset)).any()
        )
        edge_rows: list[dict[str, object]] = []
        guild_ge_by_sample: dict[str, pd.DataFrame] = {}
        for sample in samples:
            guild_ge = compute_guild_ge(sample)
            guild_ge_by_sample[sample] = guild_ge
            if is_batch:
                source = batch_statistics[
                    batch_statistics["dataset"].eq(dataset)
                    & batch_statistics["sample"].eq(sample)
                ]
                for row in source.itertuples(index=False):
                    pair = str(row.metric).removeprefix("AC_")
                    parts = pair.split("__", 1)
                    if len(parts) != 2:
                        continue
                    edge_rows.append(
                        {
                            "sample": sample,
                            "guild_1": parts[0],
                            "guild_2": parts[1],
                            "value": float(row.center),
                        }
                    )
            else:
                source = singles[
                    singles["dataset"].eq(dataset)
                    & singles["sample"].eq(sample)
                ]
                for row in source.itertuples(index=False):
                    edge_rows.append(
                        {
                            "sample": sample,
                            "guild_1": row.guild_1,
                            "guild_2": row.guild_2,
                            "value": float(row.value),
                        }
                    )
        edges = pd.DataFrame(edge_rows)
        positive_edges = (
            pd.to_numeric(edges.get("value"), errors="coerce")
            if not edges.empty
            else pd.Series(dtype=float)
        )
        positive_edges = positive_edges[
            np.isfinite(positive_edges) & (positive_edges > 0)
        ]
        edge_min = float(positive_edges.min()) if len(positive_edges) else 0.0
        edge_max = float(positive_edges.max()) if len(positive_edges) else 0.0

        all_ge = pd.concat(
            [
                frame.assign(sample=sample)
                for sample, frame in guild_ge_by_sample.items()
            ],
            ignore_index=True,
        )
        ge_values = pd.to_numeric(
            all_ge["gene_expression"], errors="coerce"
        ).fillna(0.0)
        ge_min = float(ge_values.min()) if len(ge_values) else 0.0
        ge_max = float(ge_values.max()) if len(ge_values) else 0.0

        def scale(value: float, low: float, high: float, vmin: float, vmax: float) -> float:
            if not np.isfinite(value) or np.isclose(vmin, vmax):
                return (low + high) / 2
            return low + (value - vmin) * (high - low) / (vmax - vmin)

        # Stable Okabe–Ito colors for every guild that appears in this figure.
        figure_guilds: list[str] = []
        for sample in samples:
            figure_guilds.extend(
                guild_ge_by_sample[sample]
                .nlargest(3, "gene_expression")["guild"]
                .astype(str)
                .tolist()
            )
        guild_palette = assign_category_colors(figure_guilds)

        ncols = 2
        nrows = int(np.ceil(len(samples) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8.4, 4.2 * nrows),
            squeeze=False,
        )
        for ax, sample in zip(axes.flat, samples):
            selected = guild_ge_by_sample[sample].nlargest(
                3, "gene_expression"
            )
            selected_guilds = selected["guild"].astype(str).tolist()
            graph = nx.Graph()
            for row in selected.itertuples(index=False):
                graph.add_node(
                    str(row.guild),
                    gene_expression=float(row.gene_expression),
                )
            sample_edges = edges[
                edges["sample"].eq(sample)
                & edges["guild_1"].isin(selected_guilds)
                & edges["guild_2"].isin(selected_guilds)
                & (pd.to_numeric(edges["value"], errors="coerce") > 0)
            ] if not edges.empty else edges
            for row in sample_edges.itertuples(index=False):
                graph.add_edge(
                    str(row.guild_1),
                    str(row.guild_2),
                    weight=float(row.value),
                )

            positions = _layout_by_ac(graph)
            node_sizes = [
                scale(
                    float(graph.nodes[node]["gene_expression"]),
                    800,
                    7200,
                    ge_min,
                    ge_max,
                )
                for node in graph.nodes
            ]
            edge_list = list(graph.edges)
            edge_weights = [
                float(graph.edges[edge]["weight"]) for edge in edge_list
            ]
            edge_widths = [
                scale(weight, 0.4, 16.0, edge_min, edge_max)
                for weight in edge_weights
            ]
            edge_colors = _edge_colors_by_panel_max(edge_weights)
            nx.draw_networkx_nodes(
                graph,
                positions,
                node_size=node_sizes,
                node_color=[
                    guild_palette[str(node)]
                    for node in graph.nodes
                ],
                edgecolors="white",
                linewidths=1.3,
                ax=ax,
            )
            _draw_weighted_edges_with_labels(
                graph,
                positions,
                ax=ax,
                edge_list=edge_list,
                edge_widths=edge_widths,
                edge_colors=edge_colors,
                edge_labels={
                    edge: f"{weight:.2f}"
                    for edge, weight in zip(edge_list, edge_weights)
                },
                font_size=10,
                alpha=1.0,
                node_size=node_sizes,
            )
            nx.draw_networkx_labels(
                graph,
                positions,
                font_size=12,
                font_weight="bold",
                font_color="white",
                ax=ax,
            )
            ax.set_title(sample, fontsize=12)
            ax.margins(0.22)
            ax.axis("off")

        for ax in axes.flat[len(samples) :]:
            ax.set_visible(False)
        summary = "batch median" if is_batch else "single"
        fig.suptitle(
            f"{format_dataset_label(dataset)} AC network ({summary})",
            fontsize=13,
            y=0.99,
        )
        fig.text(
            0.5,
            0.015,
            "Node area: GE · Edge width/label: AC · Edge length: inverse AC (closer = stronger)",
            ha="center",
            fontsize=10,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        save_figure(fig, output / f"ac_network_{dataset}")
        plt.close(fig)
        for sample in samples:
            (output / f"ac_network_{dataset}_{sample}.pdf").unlink(
                missing_ok=True
            )
    for sample in samples:
        (output / f"ac_{sample}.pdf").unlink(missing_ok=True)


def _flux_column(element: str) -> str:
    return "mass_flux" if element == "mass" else f"{element}_flux"


def _empty_guild_matrix(guilds: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=guilds, columns=guilds, dtype=float)


def _directed_guild_matrix(
    frame: pd.DataFrame,
    *,
    value_column: str,
    guilds: list[str],
) -> pd.DataFrame:
    """Sum a directed focal_guild→partner_guild value into a square matrix."""

    matrix = _empty_guild_matrix(guilds)
    if frame.empty or value_column not in frame.columns:
        return matrix
    grouped = (
        frame.groupby(["focal_guild", "partner_guild"], sort=False)[value_column]
        .sum()
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        g1 = str(row.focal_guild)
        g2 = str(row.partner_guild)
        if g1 in matrix.index and g2 in matrix.columns:
            matrix.at[g1, g2] = float(getattr(row, value_column))
    return matrix


def guild_cf_cc_matrices(
    interactions_table: pd.DataFrame,
    guild_table: pd.DataFrame,
    sample: str,
    element: str,
    *,
    epsilon: float = EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Guild-level CF/CC matrices for one element (ANA2-3 matrix logic).

    Cross-feeding heatmap values are **directed net fluxes**, not the AC
    geometric-mean score:

    1. At MAG pairs with both ``provided`` and ``received``, net =
       provide − receive (same orientation as ANA2-3).
    2. Sum nets by focal_guild → partner_guild into directed matrix ``D``.
    3. Antisymmetrize for display: ``CF = D − D.T``, so if GI→GII = 3 and
       GII→GI = 2 then CF[GI,GII]=1 and CF[GII,GI]=−1.

    If ``received`` rows are absent, step 1 falls back to summing ``provided``
    only, then still applies ``D − D.T``.

    Co-consumption follows ANA2-3: directed sum of ``co-consumed`` fluxes by
    focal_guild → partner_guild (not antisymmetrized).
    """

    del sample, epsilon  # sample is carried by the interaction rows themselves
    guilds = sorted(
        {str(guild) for guild in guild_table["guild"].dropna().astype(str)}
    )
    empty = _empty_guild_matrix(guilds)
    if interactions_table.empty:
        return empty.copy(), empty.copy()

    flux_col = _flux_column(element)
    mag_to_guild = guild_table.set_index("mag_id")["guild"].astype(str).to_dict()
    table = interactions_table.copy()
    table["focal"] = table["focal"].astype(str)
    table["partner"] = table["partner"].astype(str)
    table["focal_guild"] = table["focal"].map(mag_to_guild)
    table["partner_guild"] = table["partner"].map(mag_to_guild)
    table = table.dropna(subset=["focal_guild", "partner_guild"])
    table = table[table["focal_guild"].astype(str) != table["partner_guild"].astype(str)]
    if table.empty:
        return empty.copy(), empty.copy()

    table[flux_col] = pd.to_numeric(
        table.get(flux_col, 0.0), errors="coerce"
    ).fillna(0.0)
    classes = table.get("class", pd.Series("", index=table.index)).astype(str)
    provided = table[classes.eq("provided")].copy()
    received = table[classes.eq("received")].copy()
    co_consumed = table[classes.eq("co-consumed")].copy()

    if not provided.empty and not received.empty:
        # ANA2-3: net export of focal→partner = provide − receive.
        merged = provided.merge(
            received[["focal", "partner", flux_col]],
            on=["focal", "partner"],
            how="inner",
            suffixes=("", "_received"),
        )
        if not merged.empty:
            received_col = f"{flux_col}_received"
            merged["net_flux"] = (
                pd.to_numeric(merged[flux_col], errors="coerce").fillna(0.0)
                - pd.to_numeric(merged[received_col], errors="coerce").fillna(0.0)
            )
            directed_cf = _directed_guild_matrix(
                merged,
                value_column="net_flux",
                guilds=guilds,
            )
        else:
            directed_cf = _directed_guild_matrix(
                provided,
                value_column=flux_col,
                guilds=guilds,
            )
    else:
        directed_cf = _directed_guild_matrix(
            provided,
            value_column=flux_col,
            guilds=guilds,
        )

    # Antisymmetric net flow between guilds (user-facing CF heatmap).
    cf = directed_cf - directed_cf.T
    np.fill_diagonal(cf.values, 0.0)

    cc = _directed_guild_matrix(
        co_consumed,
        value_column=flux_col,
        guilds=guilds,
    )
    np.fill_diagonal(cc.values, 0.0)
    return cf, cc


def _median_matrix(matrices: list[pd.DataFrame]) -> pd.DataFrame:
    if not matrices:
        return pd.DataFrame(dtype=float)
    if len(matrices) == 1:
        return matrices[0].copy()
    labels = sorted(
        {str(label) for matrix in matrices for label in matrix.index.astype(str)}
    )
    stack = np.stack(
        [
            matrix.reindex(index=labels, columns=labels, fill_value=0.0)
            .to_numpy(dtype=float)
            for matrix in matrices
        ],
        axis=0,
    )
    return pd.DataFrame(np.median(stack, axis=0), index=labels, columns=labels)


def _cluster_reorder_matrix(
    matrix: pd.DataFrame,
    *,
    method: str = HEATMAP_CLUSTER_METHOD,
) -> pd.DataFrame:
    """Reorder rows and columns by hierarchical clustering (ANA2-3 clustermap).

    Matches ``sns.clustermap``: cluster rows from the matrix and columns from
    its transpose, then apply both leaf orders.
    """

    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return matrix
    values = np.nan_to_num(matrix.to_numpy(dtype=float), nan=0.0)
    if not np.any(values) or (
        matrix.shape[0] == matrix.shape[1] and np.allclose(values, 0.0)
    ):
        return matrix
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        def _leaf_order(block: np.ndarray) -> np.ndarray | None:
            if block.shape[0] < 2:
                return np.arange(block.shape[0])
            distances = pdist(block, metric="euclidean")
            if not np.any(distances):
                return np.arange(block.shape[0])
            return leaves_list(linkage(distances, method=method))

        row_order = _leaf_order(values)
        col_order = _leaf_order(values.T)
        if row_order is None or col_order is None:
            return matrix
        row_labels = matrix.index.to_numpy()[row_order]
        col_labels = matrix.columns.to_numpy()[col_order]
        return matrix.loc[row_labels, col_labels]
    except Exception:
        return matrix


def _filter_heatmap_matrix(
    matrix: pd.DataFrame,
    *,
    kind: str,
    top_k: int = HEATMAP_TOP_K,
) -> pd.DataFrame:
    """Select guilds like ANA2-3: CF uses top+bottom halves; CC uses top-k."""

    if matrix.empty:
        return matrix
    row_sums = matrix.sum(axis=1)
    if kind == "cf":
        half = int(math.ceil(top_k / 2))
        selected = pd.Index(
            row_sums.nlargest(half).index.tolist()
            + row_sums.nsmallest(half).index.tolist()
        ).unique()
    elif kind == "cc":
        keep = min(int(top_k), int(len(row_sums)))
        selected = row_sums.nlargest(keep).index
    else:
        raise ValueError(f"Unsupported heatmap kind: {kind}")
    filtered = matrix.loc[selected, selected]
    return _cluster_reorder_matrix(filtered)


def _prepare_heatmap_matrix(
    matrix: pd.DataFrame,
    *,
    kind: str,
    guilds: list[str] | tuple[str, ...] | None = None,
    top_k: int = HEATMAP_TOP_K,
) -> pd.DataFrame:
    """Prepare a heatmap block.

    When ``guilds`` is provided, keep exactly those subgroups and cluster both
    axes (ANA2-3 clustermap style). Otherwise cluster the full matrix, apply
    top-k selection, then cluster again.
    """

    if matrix.empty:
        return matrix
    if guilds:
        labels = [str(guild) for guild in guilds]
        subset = matrix.reindex(index=labels, columns=labels, fill_value=0.0)
        return _cluster_reorder_matrix(subset)
    clustered = _cluster_reorder_matrix(matrix)
    return _filter_heatmap_matrix(clustered, kind=kind, top_k=top_k)


def _draw_guild_heatmap(
    ax,
    matrix: pd.DataFrame,
    *,
    cmap,
    title: str,
) -> None:
    if matrix.empty or matrix.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=8)
        return
    values = matrix.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values))) if np.any(np.isfinite(values)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    # CF net-flux matrices are antisymmetric; always center the color scale.
    if cmap is CF_CMAP or np.any(values < 0):
        vmin = -vmax
    else:
        vmin = 0.0

    try:
        import seaborn as sns

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.8},
            xticklabels=True,
            yticklabels=True,
        )
        ax.tick_params(axis="both", labelsize=6, length=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
    except Exception:
        im = ax.imshow(
            values,
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        n_row, n_col = values.shape
        ax.set_xticks(np.arange(n_col))
        ax.set_yticks(np.arange(n_row))
        ax.set_xticklabels(matrix.columns.astype(str), fontsize=6, rotation=90)
        ax.set_yticklabels(matrix.index.astype(str), fontsize=6)
        ax.set_xticks(np.arange(-0.5, n_col, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_row, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False, length=0)
        ax.tick_params(axis="both", length=0)
        # Keep cell edges inside the image bounds.
        ax.set_xlim(-0.5, n_col - 0.5)
        ax.set_ylim(n_row - 0.5, -0.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        ax.set_title(title, fontsize=8, pad=4)


def _sample_element_cf_cc(
    dataset_interactions: pd.DataFrame,
    guild_table: pd.DataFrame,
    sample: str,
    element: str,
    *,
    use_median: bool,
    epsilon: float = EPSILON,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return CF/CC matrices for one sample×element (median across members)."""

    sample_col = (
        "sample"
        if "sample" in dataset_interactions.columns
        else "sample_id"
    )
    sample_rows = dataset_interactions[
        dataset_interactions[sample_col].astype(str).eq(str(sample))
    ]
    if sample_rows.empty:
        guilds = sorted(
            {str(guild) for guild in guild_table["guild"].dropna().astype(str)}
        )
        empty = pd.DataFrame(0.0, index=guilds, columns=guilds, dtype=float)
        return empty, empty

    member_col = "member" if "member" in sample_rows.columns else None
    members = (
        list(sample_rows[member_col].drop_duplicates())
        if member_col is not None
        else [None]
    )
    cf_mats: list[pd.DataFrame] = []
    cc_mats: list[pd.DataFrame] = []
    for member in members:
        member_rows = (
            sample_rows
            if member is None
            else sample_rows[sample_rows[member_col].eq(member)]
        )
        cf, cc = guild_cf_cc_matrices(
            member_rows,
            guild_table,
            sample,
            element,
            epsilon=epsilon,
        )
        cf_mats.append(cf)
        cc_mats.append(cc)
    if use_median and len(cf_mats) > 1:
        return _median_matrix(cf_mats), _median_matrix(cc_mats)
    return cf_mats[0], cc_mats[0]


def plot_interaction_heatmaps(
    interactions_table: pd.DataFrame,
    guild_table: pd.DataFrame,
    samples: list[str],
    output: Path,
    *,
    heatmap_guilds: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Plot ANA2-3-style CF/CC heatmaps for every single and batch dataset.

    One figure per dataset. Layout: rows = samples, columns = elements
    (C/N/P/S; mass omitted). Each sample×element panel stacks
    co-consumption (top) over cross-feeding (bottom). Batch datasets use the
    member-wise median matrix before clustering. ``heatmap_guilds`` selects
    which subgroups appear; default is GI/GII/GIII.
    """

    if interactions_table is None or interactions_table.empty:
        return

    configure_publication_style()
    table = interactions_table.copy()
    if "sample" not in table.columns and "sample_id" in table.columns:
        table = table.rename(columns={"sample_id": "sample"})
    datasets = list(table["dataset"].drop_duplicates())
    n_samples = max(1, len(samples))
    n_elements = len(HEATMAP_ELEMENTS)
    selected_guilds = (
        [str(guild) for guild in heatmap_guilds]
        if heatmap_guilds
        else list(DEFAULT_HEATMAP_GUILDS)
    )

    for dataset in datasets:
        dataset_rows = table[table["dataset"].eq(dataset)]
        if dataset_rows.empty:
            continue
        use_median = (
            "input_type" in dataset_rows.columns
            and bool(dataset_rows["input_type"].astype(str).eq("batch").any())
        )
        summary = "batch median" if use_median else "single"
        fig = plt.figure(
            figsize=(max(10.0, 2.8 * n_elements), max(4.2, 2.8 * n_samples))
        )
        # Two heatmap rows (CC above CF) per sample.
        # Extra left margin for sample id + vertical metric row names.
        outer = fig.add_gridspec(
            n_samples * 2,
            n_elements,
            hspace=0.28,
            wspace=0.22,
            left=0.16,
            right=0.98,
            top=0.92,
            bottom=0.05,
        )
        sample_row_axes: list[tuple[str, object, object]] = []
        for row_index, sample in enumerate(samples):
            for col_index, element in enumerate(HEATMAP_ELEMENTS):
                cf_raw, cc_raw = _sample_element_cf_cc(
                    dataset_rows,
                    guild_table,
                    sample,
                    element,
                    use_median=use_median,
                )
                cc = _prepare_heatmap_matrix(
                    cc_raw, kind="cc", guilds=selected_guilds
                )
                cf = _prepare_heatmap_matrix(
                    cf_raw, kind="cf", guilds=selected_guilds
                )
                ax_cc = fig.add_subplot(outer[row_index * 2, col_index])
                ax_cf = fig.add_subplot(outer[row_index * 2 + 1, col_index])
                _draw_guild_heatmap(ax_cc, cc, cmap=CC_CMAP, title="")
                _draw_guild_heatmap(ax_cf, cf, cmap=CF_CMAP, title="")
                if row_index == 0:
                    ax_cc.set_title(
                        str(element), fontsize=11, fontweight="bold", pad=10
                    )
                if col_index == 0:
                    # Secondary row names: vertical, immediately left of heatmaps.
                    ax_cc.set_ylabel(
                        "Co-consumption",
                        fontsize=8,
                        rotation=90,
                        labelpad=6,
                    )
                    ax_cf.set_ylabel(
                        "Cross-feeding",
                        fontsize=8,
                        rotation=90,
                        labelpad=6,
                    )
                    sample_row_axes.append((str(sample), ax_cc, ax_cf))

        # Primary row names: sample id, vertical, to the left of metric names.
        for sample, ax_cc, ax_cf in sample_row_axes:
            bbox_cc = ax_cc.get_position()
            bbox_cf = ax_cf.get_position()
            x = min(bbox_cc.x0, bbox_cf.x0) - 0.055
            y = (bbox_cc.y1 + bbox_cf.y0) / 2.0
            fig.text(
                x,
                y,
                sample,
                rotation=90,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )
        fig.suptitle(
            f"{format_dataset_label(dataset)} interaction heatmaps ({summary})",
            fontsize=11,
            y=0.98,
        )
        save_figure(fig, output / f"ac_heatmap_{dataset}")
        plt.close(fig)


def _scale_range(
    value: float,
    low: float,
    high: float,
    vmin: float,
    vmax: float,
) -> float:
    if not np.isfinite(value) or np.isclose(vmin, vmax):
        return (low + high) / 2.0
    return low + (value - vmin) * (high - low) / (vmax - vmin)


def _edge_colors_by_panel_max(weights: list[float]) -> list[str]:
    """Black for the thickest weight(s) in a panel; light gray otherwise."""

    if not weights:
        return []
    max_weight = max(weights)
    return [
        EDGE_COLOR_MAX
        if np.isfinite(weight) and np.isclose(weight, max_weight)
        else EDGE_COLOR_OTHER
        for weight in weights
    ]


def _draw_weighted_edges_with_labels(
    graph: nx.Graph | nx.DiGraph,
    positions: dict,
    *,
    ax,
    edge_list: list,
    edge_widths: list[float],
    edge_colors: list[str],
    edge_labels: dict,
    font_size: int = 7,
    alpha: float = 1.0,
    directed: bool = False,
    node_size: float | list[float] = 300,
    clip_on: bool = True,
) -> None:
    """Draw edges and weight labels; each label matches its edge color."""

    if not edge_list:
        return
    draw_kwargs: dict[str, object] = {
        "edgelist": edge_list,
        "width": edge_widths if edge_widths else 1.0,
        "edge_color": edge_colors if edge_colors else EDGE_COLOR_OTHER,
        "alpha": alpha,
        "ax": ax,
        "node_size": node_size,
    }
    if directed:
        draw_kwargs.update(
            {
                "arrows": True,
                "arrowstyle": "-|>",
                "arrowsize": 12,
                "connectionstyle": "arc3,rad=0.0",
            }
        )
    else:
        draw_kwargs["arrows"] = False
    edge_artists = nx.draw_networkx_edges(graph, positions, **draw_kwargs)
    if not clip_on:
        if isinstance(edge_artists, list):
            for artist in edge_artists:
                artist.set_clip_on(False)
        elif edge_artists is not None:
            edge_artists.set_clip_on(False)

    label_bbox = {
        "boxstyle": "round,pad=0.12",
        "fc": "white",
        "ec": "none",
        "alpha": 1.0,
    }
    for color in (EDGE_COLOR_MAX, EDGE_COLOR_OTHER):
        color_labels = {
            edge: edge_labels[edge]
            for edge, edge_color in zip(edge_list, edge_colors)
            if edge_color == color and edge in edge_labels
        }
        if not color_labels:
            continue
        label_artists = nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=color_labels,
            font_size=font_size,
            font_color=color,
            label_pos=0.5,
            bbox=label_bbox,
            ax=ax,
        )
        if not clip_on:
            for artist in label_artists.values():
                artist.set_clip_on(False)


def _circular_positions(
    guilds: list[str],
    *,
    scale: float = 1.35,
) -> dict[str, np.ndarray]:
    """Equal edge lengths via a shared circular layout order.

    ``scale`` expands the circle so edges read longer and value labels fit,
    while staying inside the panel data window.
    """

    graph = nx.Graph()
    graph.add_nodes_from(guilds)
    positions = nx.circular_layout(graph)
    return {
        node: np.asarray(coord, dtype=float) * scale
        for node, coord in positions.items()
    }


def _undirected_cc_edges(
    matrix: pd.DataFrame,
    guilds: list[str],
    *,
    tol: float = 0.0,
) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    for index, guild_1 in enumerate(guilds):
        for guild_2 in guilds[index + 1 :]:
            forward = float(matrix.at[guild_1, guild_2]) if guild_1 in matrix.index else 0.0
            reverse = float(matrix.at[guild_2, guild_1]) if guild_2 in matrix.index else 0.0
            weight = max(
                forward if np.isfinite(forward) else 0.0,
                reverse if np.isfinite(reverse) else 0.0,
            )
            if weight > tol:
                edges.append((guild_1, guild_2, weight))
    return edges


def _directed_cf_edges(
    matrix: pd.DataFrame,
    guilds: list[str],
    *,
    tol: float = 0.0,
) -> list[tuple[str, str, float]]:
    """Positive net flux only: donor → acceptor."""

    edges: list[tuple[str, str, float]] = []
    for donor in guilds:
        for acceptor in guilds:
            if donor == acceptor:
                continue
            weight = float(matrix.at[donor, acceptor]) if donor in matrix.index else 0.0
            if np.isfinite(weight) and weight > tol:
                edges.append((donor, acceptor, weight))
    return edges


def _draw_elemental_network_panel(
    ax,
    *,
    guilds: list[str],
    guild_ge: pd.DataFrame,
    edges: list[tuple[str, str, float]],
    directed: bool,
    positions: dict[str, np.ndarray],
    guild_palette: dict[str, str],
    edge_min: float,
    edge_max: float,
) -> None:
    """Draw one elemental network panel with fixed circular spacing.

    Node sizes scale by guild GE relative to the min/max among nodes in
    this panel only (not across the whole figure). Within the panel the
    thickest edge(s) are black; remaining edges are light gray. Edge-weight
    labels use the same color as their edge.
    """

    ge_map = {
        str(row.guild): float(row.gene_expression)
        for row in guild_ge.itertuples(index=False)
    }
    graph: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
    for guild in guilds:
        graph.add_node(guild, gene_expression=ge_map.get(guild, 0.0))
    for source, target, weight in edges:
        graph.add_edge(source, target, weight=float(weight))

    panel_ge = [
        float(graph.nodes[node]["gene_expression"]) for node in graph.nodes
    ]
    panel_ge_min = float(min(panel_ge)) if panel_ge else 0.0
    panel_ge_max = float(max(panel_ge)) if panel_ge else 0.0
    # Compact range so relative GE contrast fits circular elemental panels.
    node_sizes = [
        _scale_range(
            float(graph.nodes[node]["gene_expression"]),
            300,
            980,
            panel_ge_min,
            panel_ge_max,
        )
        for node in graph.nodes
    ]
    edge_list = list(graph.edges)
    edge_weights = [
        float(graph.edges[edge]["weight"]) for edge in edge_list
    ]
    edge_widths = [
        _scale_range(weight, 0.7, 11.0, edge_min, edge_max)
        for weight in edge_weights
    ]
    edge_colors = _edge_colors_by_panel_max(edge_weights)
    node_collection = nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color=[guild_palette.get(str(node), "#7A7A7A") for node in graph.nodes],
        edgecolors="white",
        linewidths=1.1,
        ax=ax,
    )
    if node_collection is not None:
        node_collection.set_clip_on(False)
    _draw_weighted_edges_with_labels(
        graph,
        positions,
        ax=ax,
        edge_list=edge_list,
        edge_widths=edge_widths,
        edge_colors=edge_colors,
        edge_labels={
            edge: f"{weight:.2g}"
            for edge, weight in zip(edge_list, edge_weights)
        },
        font_size=7,
        alpha=1.0,
        directed=directed,
        node_size=node_sizes,
        clip_on=False,
    )
    label_artists = nx.draw_networkx_labels(
        graph,
        positions,
        font_size=8,
        font_weight="bold",
        font_color="white",
        ax=ax,
    )
    for artist in label_artists.values():
        artist.set_clip_on(False)
    # Fit circle + node radius + labels inside the axes; spill into gutters
    # is allowed via clip_on=False so columns can stay tight.
    ax.set_aspect("equal")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.05, 2.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)
    ax.set_clip_on(False)


def _edge_width_legend_handles(
    values: list[float],
    *,
    edge_min: float,
    edge_max: float,
    label: str,
) -> list[object]:
    """Build proxy artists for a shared edge-width legend."""

    from matplotlib.lines import Line2D

    finite = [float(value) for value in values if np.isfinite(value) and value > 0]
    if not finite:
        return [
            Line2D(
                [0],
                [0],
                color=EDGE_COLOR_OTHER,
                lw=2.0,
                label=f"{label}: no edges",
            )
        ]
    exemplars = sorted({min(finite), float(np.median(finite)), max(finite)})
    handles: list[object] = []
    max_value = max(finite)
    for value in exemplars:
        width = _scale_range(value, 0.7, 11.0, edge_min, edge_max)
        handles.append(
            Line2D(
                [0],
                [0],
                color=(
                    EDGE_COLOR_MAX
                    if np.isclose(value, max_value)
                    else EDGE_COLOR_OTHER
                ),
                lw=max(width / 2.0, 0.6),
                label=f"{label} {value:.2g}",
            )
        )
    return handles


def plot_elemental_networks(
    interactions_table: pd.DataFrame,
    guild_table: pd.DataFrame,
    samples: list[str],
    output: Path,
    *,
    heatmap_guilds: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Plot elemental CF/CC networks with the same panel grid as heatmaps.

    Layout: rows = samples, columns = C/N/P/S; within each sample×element,
    Co-consumption (undirected) is above Cross-feeding (directed net arrows).
    Edge lengths are equal (circular layout). Edge widths share one scale per
    interaction type across all samples/elements in the figure. Node sizes
    scale by guild GE within each panel only. Within each panel the thickest
    edge is black; other edges use light gray. Edge-weight labels match the
    edge color.
    """

    if interactions_table is None or interactions_table.empty:
        return

    configure_publication_style()
    table = interactions_table.copy()
    if "sample" not in table.columns and "sample_id" in table.columns:
        table = table.rename(columns={"sample_id": "sample"})
    datasets = list(table["dataset"].drop_duplicates())
    n_samples = max(1, len(samples))
    n_elements = len(HEATMAP_ELEMENTS)
    selected_guilds = (
        [str(guild) for guild in heatmap_guilds]
        if heatmap_guilds
        else list(DEFAULT_HEATMAP_GUILDS)
    )
    positions = _circular_positions(selected_guilds)
    guild_palette = assign_category_colors(selected_guilds)

    for dataset in datasets:
        dataset_rows = table[table["dataset"].eq(dataset)]
        if dataset_rows.empty:
            continue
        use_median = (
            "input_type" in dataset_rows.columns
            and bool(dataset_rows["input_type"].astype(str).eq("batch").any())
        )
        summary = "batch median" if use_median else "single"

        guild_ge_by_sample = {
            sample: compute_guild_ge(sample) for sample in samples
        }

        panel_data: dict[tuple[str, str], tuple[list, list]] = {}
        cc_weights: list[float] = []
        cf_weights: list[float] = []
        for sample in samples:
            for element in HEATMAP_ELEMENTS:
                cf_raw, cc_raw = _sample_element_cf_cc(
                    dataset_rows,
                    guild_table,
                    sample,
                    element,
                    use_median=use_median,
                )
                cf_mat = cf_raw.reindex(
                    index=selected_guilds, columns=selected_guilds, fill_value=0.0
                )
                cc_mat = cc_raw.reindex(
                    index=selected_guilds, columns=selected_guilds, fill_value=0.0
                )
                cc_edges = _undirected_cc_edges(cc_mat, selected_guilds)
                cf_edges = _directed_cf_edges(cf_mat, selected_guilds)
                panel_data[(sample, element)] = (cc_edges, cf_edges)
                cc_weights.extend(weight for _, _, weight in cc_edges)
                cf_weights.extend(weight for _, _, weight in cf_edges)

        cc_positive = [w for w in cc_weights if np.isfinite(w) and w > 0]
        cf_positive = [w for w in cf_weights if np.isfinite(w) and w > 0]
        cc_min = float(min(cc_positive)) if cc_positive else 0.0
        cc_max = float(max(cc_positive)) if cc_positive else 0.0
        cf_min = float(min(cf_positive)) if cf_positive else 0.0
        cf_max = float(max(cf_positive)) if cf_positive else 0.0

        fig = plt.figure(
            figsize=(max(12.5, 3.9 * n_elements), max(6.2, 3.6 * n_samples))
        )
        # Narrow column gutters: each panel gets more width; clip_on=False
        # lets nodes/labels spill slightly without looking cut off.
        outer = fig.add_gridspec(
            n_samples * 2,
            n_elements,
            hspace=0.08,
            wspace=0.03,
            left=0.12,
            right=0.995,
            top=0.93,
            bottom=0.08,
        )
        sample_row_axes: list[tuple[str, object, object]] = []
        for row_index, sample in enumerate(samples):
            guild_ge = guild_ge_by_sample[sample]
            for col_index, element in enumerate(HEATMAP_ELEMENTS):
                cc_edges, cf_edges = panel_data[(sample, element)]
                ax_cc = fig.add_subplot(outer[row_index * 2, col_index])
                ax_cf = fig.add_subplot(outer[row_index * 2 + 1, col_index])
                _draw_elemental_network_panel(
                    ax_cc,
                    guilds=selected_guilds,
                    guild_ge=guild_ge,
                    edges=cc_edges,
                    directed=False,
                    positions=positions,
                    guild_palette=guild_palette,
                    edge_min=cc_min,
                    edge_max=cc_max,
                )
                _draw_elemental_network_panel(
                    ax_cf,
                    guilds=selected_guilds,
                    guild_ge=guild_ge,
                    edges=cf_edges,
                    directed=True,
                    positions=positions,
                    guild_palette=guild_palette,
                    edge_min=cf_min,
                    edge_max=cf_max,
                )
                if row_index == 0:
                    ax_cc.set_title(
                        str(element), fontsize=11, fontweight="bold", pad=3
                    )
                if col_index == 0:
                    ax_cc.set_ylabel(
                        "Co-consumption",
                        fontsize=8,
                        rotation=90,
                        labelpad=4,
                    )
                    ax_cf.set_ylabel(
                        "Cross-feeding",
                        fontsize=8,
                        rotation=90,
                        labelpad=4,
                    )
                    sample_row_axes.append((str(sample), ax_cc, ax_cf))

        for sample, ax_cc, ax_cf in sample_row_axes:
            bbox_cc = ax_cc.get_position()
            bbox_cf = ax_cf.get_position()
            x = min(bbox_cc.x0, bbox_cf.x0) - 0.048
            y = (bbox_cc.y1 + bbox_cf.y0) / 2.0
            fig.text(
                x,
                y,
                sample,
                rotation=90,
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        legend_handles = [
            *_edge_width_legend_handles(
                cc_positive,
                edge_min=cc_min,
                edge_max=cc_max,
                label="Co-consumption",
            ),
            *_edge_width_legend_handles(
                cf_positive,
                edge_min=cf_min,
                edge_max=cf_max,
                label="Cross-feeding",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(6, max(1, len(legend_handles))),
            fontsize=8,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.suptitle(
            f"{format_dataset_label(dataset)} elemental networks ({summary})",
            fontsize=11,
            y=0.98,
        )
        save_figure(fig, output / f"ac_elemental_network_{dataset}")
        plt.close(fig)


def run_plots_only(args) -> dict[str, Path]:
    """Rebuild AC figures from existing analysis CSVs (no recomputation)."""

    set_taxon_input_files(
        ge_file=getattr(args, "ge_file", None),
        guild_file=getattr(args, "guild_file", None),
    )
    output = analysis_output_dir(args)
    interactions_path = output / "ac_interactions.csv"
    table_path = output / "ac_all.csv"
    if not interactions_path.is_file():
        raise FileNotFoundError(
            f"--plots-only requires existing interactions table: {interactions_path}"
        )
    if not table_path.is_file():
        raise FileNotFoundError(
            f"--plots-only requires existing AC table: {table_path}"
        )

    interactions_out = pd.read_csv(interactions_path)
    table = pd.read_csv(table_path)
    normalized = table.rename(columns={"sample_id": "sample"})
    if "sample" not in normalized.columns:
        raise ValueError(f"{table_path} lacks sample / sample_id column")
    samples = (
        list(args.sample)
        if getattr(args, "sample", None)
        else sorted(normalized["sample"].astype(str).drop_duplicates())
    )
    guild_table = load_taxon_guild()
    heatmap_guilds = getattr(args, "heatmap_guilds", None) or list(
        DEFAULT_HEATMAP_GUILDS
    )
    selected_plots = {
        str(name) for name in (getattr(args, "plot", None) or ["elemental"])
    }
    print(
        f"[AC] plots-only from {output}; plots={sorted(selected_plots)}; "
        f"samples={samples}",
        flush=True,
    )

    outputs: dict[str, Path] = {
        "all": table_path,
        "interactions": interactions_path,
    }
    if "network" in selected_plots:
        singles = single_value_table(normalized)
        batch_statistics = summarize_batch_values(normalized)
        plot_datasets(table, batch_statistics, samples, output)
    if "heatmap" in selected_plots:
        plot_interaction_heatmaps(
            interactions_out,
            guild_table,
            samples,
            output,
            heatmap_guilds=heatmap_guilds,
        )
    if "elemental" in selected_plots:
        plot_elemental_networks(
            interactions_out,
            guild_table,
            samples,
            output,
            heatmap_guilds=heatmap_guilds,
        )
    return outputs


def run(args) -> dict[str, Path]:
    if bool(getattr(args, "plots_only", False)):
        return run_plots_only(args)

    set_taxon_input_files(
        ge_file=getattr(args, "ge_file", None),
        guild_file=getattr(args, "guild_file", None),
    )
    datasets = parse_datasets(args)
    jobs, samples = discover_jobs(
        datasets,
        args.sample,
        require_bsl=False,
    )
    context_cmm_dir = args.context_cmm_dir.expanduser().resolve()
    guild_table = load_taxon_guild()
    annotation = load_metabolite_table()
    output = analysis_output_dir(args)

    method = str(getattr(args, "method", DEFAULT_METHOD))
    if method not in INTERACTION_METHODS:
        raise ValueError(
            f"Unknown --method {method!r}; expected one of {INTERACTION_METHODS}."
        )
    job_workers, micom_threads = resolve_ac_concurrency(
        getattr(args, "jobs", DEFAULT_JOBS),
        getattr(args, "threads", DEFAULT_THREADS),
        method=method,
    )
    force = bool(getattr(args, "force", False))
    abundance_by_sample = ensure_abundances_for_samples(context_cmm_dir, samples)

    frames: list[pd.DataFrame] = []
    interaction_frames: list[pd.DataFrame] = []
    job_total = len(jobs)
    print(
        f"[AC] discovered {job_total} job(s); method={method}; "
        f"jobs={job_workers}, micom_threads={micom_threads}",
        flush=True,
    )

    payloads = [
        {
            "job": job,
            "guild_table": guild_table,
            "annotation": annotation,
            "abundance_map": abundance_by_sample[job.sample],
            "job_index": job_index,
            "job_total": job_total,
            "force": force,
            "epsilon": EPSILON,
            "threads": micom_threads,
            "method": method,
        }
        for job_index, job in enumerate(jobs, start=1)
    ]

    results_by_index: dict[int, tuple[pd.DataFrame, pd.DataFrame, FluxJob]] = {}
    if job_workers == 1 or job_total <= 1:
        for payload in payloads:
            job_index, metrics, interactions_table, job = _calculate_job_payload(
                payload
            )
            results_by_index[job_index] = (metrics, interactions_table, job)
    else:
        try:
            pool_cm = ProcessPoolExecutor(max_workers=job_workers)
            pool_cm.__enter__()
        except (PermissionError, OSError) as error:
            print(
                f"[AC] process pool unavailable ({error}); "
                "falling back to in-process sequential jobs",
                flush=True,
            )
            for payload in payloads:
                job_index, metrics, interactions_table, job = _calculate_job_payload(
                    payload
                )
                results_by_index[job_index] = (metrics, interactions_table, job)
        else:
            try:
                futures = [
                    pool_cm.submit(_calculate_job_payload, payload)
                    for payload in payloads
                ]
                for future in as_completed(futures):
                    job_index, metrics, interactions_table, job = future.result()
                    results_by_index[job_index] = (metrics, interactions_table, job)
            finally:
                pool_cm.__exit__(None, None, None)

    for job_index in sorted(results_by_index):
        metrics, interactions_table, job = results_by_index[job_index]
        frames.append(metrics)
        if not interactions_table.empty:
            one = interactions_table.copy()
            one.insert(0, "member", job.member)
            one.insert(0, "input_type", job.input_type)
            one.insert(0, "dataset", job.dataset)
            interaction_frames.append(one)

    print("[AC] writing tables and figures", flush=True)
    table = pd.concat(frames, ignore_index=True)
    interactions_out = (
        pd.concat(interaction_frames, ignore_index=True)
        if interaction_frames
        else _empty_summary()
    )
    normalized = table.rename(columns={"sample_id": "sample"})
    singles = single_value_table(normalized)
    batch_statistics = summarize_batch_values(normalized)
    outputs = {
        "all": write_csv_atomic(table, output / "ac_all.csv"),
        "single": write_csv_atomic(
            singles,
            output / "ac_single_values.csv",
        ),
        "batch_statistics": write_csv_atomic(
            batch_statistics,
            output / "ac_batch_statistics.csv",
        ),
        "interactions": write_csv_atomic(
            interactions_out,
            output / "ac_interactions.csv",
        ),
    }
    plot_datasets(table, batch_statistics, samples, output)
    heatmap_guilds = getattr(args, "heatmap_guilds", None) or list(
        DEFAULT_HEATMAP_GUILDS
    )
    plot_interaction_heatmaps(
        interactions_out,
        guild_table,
        samples,
        output,
        heatmap_guilds=heatmap_guilds,
    )
    plot_elemental_networks(
        interactions_out,
        guild_table,
        samples,
        output,
        heatmap_guilds=heatmap_guilds,
    )
    return outputs


def main() -> None:
    import argparse

    parser = add_dataset_arguments(
        argparse.ArgumentParser(
            description=(
                "Calculate AC (guild-pair interaction scores) from single "
                "and/or batch flux directories. Default --method "
                "metacontext_interaction is the fast ANA6-3 vectorized path; "
                "micom_interaction falls back to micom.interaction. Both try "
                "tolerance 1e-6 then 1e-9. Requires --context-cmm-dir for "
                "<sample>-ctx.pickle and optional <sample>-abundances.csv "
                "sidecars (unless --plots-only). Caches per flux as "
                "*-ctx-interactions.metacontext.csv or "
                "*-ctx-interactions.micom.csv."
            )
        ),
        require_context=False,
    )
    parser.add_argument(
        "--context-cmm-dir",
        type=Path,
        help=(
            "Directory with <sample>-ctx.pickle (and optional "
            "<sample>-abundances.csv sidecars). Required unless --plots-only."
        ),
    )
    parser.add_argument(
        "--ge-file",
        type=Path,
        default=TAXON_GE_FILE,
        help=(
            "Wide taxon GE abundances for guild GE node sizes "
            "(default: example/inputs/01_taxon_ge.csv)."
        ),
    )
    parser.add_argument(
        "--guild-file",
        type=Path,
        default=TAXON_GUILD_FILE,
        help=(
            "MAG-to-guild map CSV with Bin Id,Guild "
            "(default: example/inputs/04_taxon_guild.csv)."
        ),
    )
    parser.add_argument(
        "--method",
        choices=INTERACTION_METHODS,
        default=DEFAULT_METHOD,
        help=(
            "Interaction algorithm (default: metacontext_interaction). "
            "metacontext_interaction: fast vectorized ANA6-3 path. "
            "micom_interaction: official micom.interaction fallback. "
            "Both try |flux| and |flux|*abundance gates at 1e-6, then 1e-9 "
            "if the result is empty. Cache tag: .metacontext.csv / .micom.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore method-specific *-ctx-interactions.{metacontext,micom}.csv "
            "caches beside each ctx flux and recompute them."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "Parallel AC jobs across flux scenarios (default: 1). "
            "When --method micom_interaction and --jobs > 1, --threads is "
            "forced to 1 to avoid nested MICOM process pools."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=(
            "micom.interaction worker processes per job (default: 1). "
            "Used only with --method micom_interaction; ignored for "
            "metacontext_interaction. Prefer raising --jobs for many "
            "scenarios."
        ),
    )
    parser.add_argument(
        "--heatmap-guilds",
        nargs="+",
        default=list(DEFAULT_HEATMAP_GUILDS),
        help=(
            "Guild subgroups to include in interaction heatmaps / elemental "
            f"networks (default: {' '.join(DEFAULT_HEATMAP_GUILDS)})."
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help=(
            "Skip all AC recomputation. Rebuild selected figures from "
            "existing ac_all.csv and ac_interactions.csv under --output-dir."
        ),
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        choices=("network", "heatmap", "elemental"),
        default=["elemental"],
        help=(
            "With --plots-only, which figures to rebuild "
            "(default: elemental). Choices: network heatmap elemental."
        ),
    )
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1")
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if not args.heatmap_guilds:
        raise SystemExit("--heatmap-guilds must list at least one guild")
    if args.plots_only:
        if args.context_cmm_dir is None:
            # unused in plots-only mode
            pass
    elif args.context_cmm_dir is None:
        raise SystemExit("--context-cmm-dir is required unless --plots-only")
    outputs = run(args)
    print(f"AC complete: {outputs['all'].parent}")


if __name__ == "__main__":
    main()
