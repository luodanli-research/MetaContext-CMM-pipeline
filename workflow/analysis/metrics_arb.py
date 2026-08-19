#!/usr/bin/env python3
"""Calculate and visualize ARB for single and batch simulation datasets."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "micom310-matplotlib"),
)

import matplotlib.pyplot as plt

from _analysis_utils import (
    ARB_REACTION_FILE,
    TAXON_GE_FILE,
    assign_dataset_colors,
    configure_publication_style,
    format_dataset_label,
    load_taxon_ge,
    remove_non_pdf_figures,
    save_figure,
    set_taxon_input_files,
    write_csv_atomic,
)
from _metric_datasets import (
    FluxJob,
    add_dataset_arguments,
    discover_jobs,
    analysis_output_dir,
    parse_datasets,
    partition_metric_rows,
    single_value_table,
    summarize_batch_values,
)


ARB_TYPES = ("Ana2Cata", "Ana2Detox", "Cata2Ana", "Detox2Ana")
EPSILON_ARB = 1e-7


def load_arb_reactions(
    reaction_file: str | Path = ARB_REACTION_FILE,
) -> pd.DataFrame:
    """Load the fixed ARB reaction classification."""

    table = pd.read_csv(reaction_file, encoding="utf-8-sig")
    required = {"reaction_id", "lb", "ub", "type"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"ARB reaction table is missing columns: "
            + ", ".join(sorted(missing))
        )

    table = table[table["type"].isin(ARB_TYPES)].copy()
    if table.empty:
        raise RuntimeError("No supported ARB reactions were found.")
    table["reaction_id"] = table["reaction_id"].astype(str).str.strip()
    return table


def directional_flux(bsl_flux: float, lower_bound: float, upper_bound: float) -> float:
    """Reproduce the legacy forward-direction ARB flux transformation."""

    flux = float(bsl_flux)
    if pd.isna(flux):
        return 0.0
    if float(upper_bound) == 0:
        flux = -flux
    return max(flux, 0.0)


def calculate_arb(
    sample: str,
    flux_file: str | Path,
    reaction_table: pd.DataFrame,
    taxon_ge: pd.DataFrame,
    *,
    epsilon_arb: float = EPSILON_ARB,
) -> pd.DataFrame:
    """Calculate ctx ARB using GE-weighted directional reaction flux."""

    flux_path = Path(flux_file).resolve()
    flux = pd.read_csv(flux_path)
    if "compartment" not in flux.columns:
        raise ValueError(f"{flux_path} lacks the 'compartment' column.")
    flux = flux.set_index("compartment")

    ge_map = taxon_ge.set_index("mag_id")["ge"].to_dict()
    taxon_rows = [taxon for taxon in flux.index if str(taxon) != "medium"]
    totals = {arb_type: 0.0 for arb_type in ARB_TYPES}
    matched = 0
    missing = 0
    positive_entries = 0

    for reaction in reaction_table.itertuples(index=False):
        if reaction.reaction_id not in flux.columns:
            missing += 1
            continue
        matched += 1

        for taxon in taxon_rows:
            ge = ge_map.get(str(taxon))
            if ge is None or pd.isna(ge):
                continue
            forward = directional_flux(
                flux.at[taxon, reaction.reaction_id],
                reaction.lb,
                reaction.ub,
            )
            weighted = forward * float(ge)
            if weighted > 0:
                positive_entries += 1
            totals[reaction.type] += weighted

    numerator = totals["Ana2Cata"] + totals["Ana2Detox"] + epsilon_arb
    denominator = totals["Cata2Ana"] + totals["Detox2Ana"] + epsilon_arb
    arb = float(np.log2(numerator / denominator))

    row: dict[str, object] = {
        "sample": sample,
        "simulation_type": "ctx",
        "flux_file": flux_path.name,
        "n_arb_reactions_total": len(reaction_table),
        "n_arb_reactions_matched": matched,
        "n_arb_reactions_missing": missing,
        "n_positive_weighted_mag_reaction_entries": positive_entries,
    }
    row.update(totals)
    row["ARB"] = arb
    return pd.DataFrame([row])


def calculate_job(
    job: FluxJob,
    reaction_table: pd.DataFrame,
    *,
    ge_file: str | Path | None = None,
    epsilon_arb: float = EPSILON_ARB,
) -> dict[str, object]:
    metrics = calculate_arb(
        job.sample,
        job.ctx_flux,
        reaction_table,
        load_taxon_ge(job.sample, ge_file=ge_file),
        epsilon_arb=epsilon_arb,
    )
    row = metrics.iloc[0]
    return {
        "dataset": job.dataset,
        "input_type": job.input_type,
        "sensitivity_type": job.sensitivity_type,
        "member": job.member,
        "sample": job.sample,
        "metric": "ARB",
        "value": float(row["ARB"]),
        **{arb_type: float(row[arb_type]) for arb_type in ARB_TYPES},
        "ctx_flux": str(job.ctx_flux),
    }


def plot_datasets(
    table: pd.DataFrame,
    batch_statistics: pd.DataFrame,
    samples: list[str],
    output: Path,
) -> None:
    """Plot all samples together as single bars or batch boxes + single points."""

    configure_publication_style()
    remove_non_pdf_figures(output)
    singles, batches = partition_metric_rows(table)
    single_datasets = list(singles["dataset"].drop_duplicates())
    batch_datasets = list(batches["dataset"].drop_duplicates())
    all_datasets = list(table["dataset"].drop_duplicates())
    # Color only the series that are actually drawn as bars/boxes so that
    # Okabe–Ito spacing matches the visible count (do not reserve a slot for
    # formal singles that are drawn as points instead).
    if batches.empty:
        bar_color_map = assign_dataset_colors(
            single_datasets,
            single_datasets=single_datasets,
            formal_as_black=True,
        )
    else:
        bar_color_map = assign_dataset_colors(
            batch_datasets,
            formal_as_black=False,
        )
    # Points: formal singles (context/formal/ctx) stay black.
    point_color_map = assign_dataset_colors(
        single_datasets,
        single_datasets=single_datasets,
        formal_as_black=True,
    )
    sample_centers = np.arange(len(samples), dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(5.6, 1.35 * len(samples)), 3.8)
    )

    if batches.empty:
        width = min(0.72 / max(1, len(single_datasets)), 0.24)
        offsets = (
            np.arange(len(single_datasets))
            - (len(single_datasets) - 1) / 2
        ) * width
        for offset, dataset in zip(offsets, single_datasets):
            values = []
            for sample in samples:
                one = pd.to_numeric(
                    singles.loc[
                        singles["dataset"].eq(dataset)
                        & singles["sample"].eq(sample),
                        "value",
                    ],
                    errors="coerce",
                ).dropna()
                values.append(float(one.iloc[0]) if len(one) else np.nan)
            ax.bar(
                sample_centers + offset,
                values,
                width=width * 0.88,
                color=bar_color_map[dataset],
                edgecolor="#333333",
                linewidth=0.6,
                label=format_dataset_label(dataset),
            )
    else:
        width = min(0.66 / max(1, len(batch_datasets)), 0.22)
        offsets = (
            np.arange(len(batch_datasets))
            - (len(batch_datasets) - 1) / 2
        ) * width
        for offset, dataset in zip(offsets, batch_datasets):
            box_values = []
            positions = []
            for center, sample in zip(sample_centers, samples):
                values = pd.to_numeric(
                    batches.loc[
                        batches["dataset"].eq(dataset)
                        & batches["sample"].eq(sample),
                        "value",
                    ],
                    errors="coerce",
                ).dropna().to_numpy(float)
                if len(values):
                    box_values.append(values)
                    positions.append(center + offset)
            if not box_values:
                continue
            color = bar_color_map[dataset]
            boxes = ax.boxplot(
                box_values,
                positions=positions,
                widths=width * 0.82,
                patch_artist=True,
                showfliers=True,
                boxprops={"edgecolor": color, "linewidth": 1.0},
                medianprops={"color": "#202020", "linewidth": 1.3},
                whiskerprops={"color": color, "linewidth": 0.9},
                capprops={"color": color, "linewidth": 0.9},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "markersize": 2.5,
                    "alpha": 0.45,
                },
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.28)
            ax.plot([], [], color=color, linewidth=7, alpha=0.28, label=format_dataset_label(dataset))

        point_offsets = (
            np.arange(len(single_datasets))
            - (len(single_datasets) - 1) / 2
        ) * 0.045
        for offset, dataset in zip(point_offsets, single_datasets):
            values = []
            positions = []
            for center, sample in zip(sample_centers, samples):
                one = pd.to_numeric(
                    singles.loc[
                        singles["dataset"].eq(dataset)
                        & singles["sample"].eq(sample),
                        "value",
                    ],
                    errors="coerce",
                ).dropna()
                if len(one):
                    values.append(float(one.iloc[0]))
                    positions.append(center + offset)
            ax.scatter(
                positions,
                values,
                s=30,
                color=point_color_map[dataset],
                edgecolor="#222222",
                linewidth=0.5,
                zorder=5,
                label=format_dataset_label(dataset),
            )

    ax.axhline(0, color="#999999", linewidth=0.7)
    ax.set_xticks(sample_centers)
    ax.set_xticklabels(samples)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Anabolic Reallocation Bias (ARB)")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=min(4, max(1, len(all_datasets))),
    )
    fig.tight_layout()
    save_figure(fig, output / "arb")
    plt.close(fig)
    for sample in samples:
        (output / f"arb_{sample}.pdf").unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    set_taxon_input_files(ge_file=args.ge_file)
    datasets = parse_datasets(args)
    jobs, samples = discover_jobs(
        datasets,
        args.sample,
        require_bsl=False,
    )
    reactions = load_arb_reactions()
    table = pd.DataFrame(
        calculate_job(job, reactions, ge_file=args.ge_file) for job in jobs
    )
    singles = single_value_table(table)
    batch_statistics = summarize_batch_values(table)
    output = analysis_output_dir(args)
    outputs = {
        "all": write_csv_atomic(table, output / "arb_all.csv"),
        "single": write_csv_atomic(
            singles,
            output / "arb_single_values.csv",
        ),
        "batch_statistics": write_csv_atomic(
            batch_statistics,
            output / "arb_batch_statistics.csv",
        ),
    }
    plot_datasets(table, batch_statistics, samples, output)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = add_dataset_arguments(
        argparse.ArgumentParser(
            description=(
                "Calculate ARB from single datasets and/or batches of "
                "simulation-member directories."
            )
        )
    )
    parser.add_argument(
        "--ge-file",
        type=Path,
        default=TAXON_GE_FILE,
        help=(
            "Wide taxon GE abundances (Bin Id + sample columns). "
            "Default: example/inputs/01_taxon_ge.csv."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    print(f"ARB complete: {outputs['all'].parent}")


if __name__ == "__main__":
    main()
