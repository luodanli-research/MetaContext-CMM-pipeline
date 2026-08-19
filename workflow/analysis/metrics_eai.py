#!/usr/bin/env python3
"""Calculate and visualize EAI for single and batch simulation datasets."""

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
    FORMAL_SINGLE_COLOR,
    METABOLITE_FILE,
    assign_dataset_colors,
    configure_publication_style,
    format_dataset_label,
    remove_non_pdf_figures,
    save_figure,
    write_csv_atomic,
)
from _metabolite_annotations import load_metabolite_table
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


ELEMENTS = ("C", "N", "P", "S")
BASELINE_TYPE = "bsl"
CONTEXT_TYPE = "ctx"
EPSILON = 1e-7
BUDGET_ALPHA = 1e-7


def build_element_lookup(annotation: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build exact exchange-metabolite and legacy base-ID element lookups."""

    columns = [f"{element}_number" for element in ELEMENTS]
    exact = annotation.set_index("metabolite")[columns].to_dict("index")
    lookup = dict(exact)

    medium = annotation[
        annotation["metabolite"].astype(str).str.endswith("_m")
    ].copy()
    medium["base_metabolite"] = medium["metabolite"].str[:-2]
    for base_id, values in (
        medium.drop_duplicates("base_metabolite", keep="first")
        .set_index("base_metabolite")[columns]
        .to_dict("index")
        .items()
    ):
        lookup.setdefault(base_id, values)

    return lookup


def parse_exchange_metabolite(reaction_id: str) -> tuple[str, str]:
    """Return exact and compartment-free IDs from an exchange reaction."""

    exact = str(reaction_id).removeprefix("EX_")
    base = exact[:-2] if exact.endswith(("_m", "_e")) else exact
    return exact, base


def compute_element_flux(
    medium_flux: pd.Series,
    annotation_lookup: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Reproduce the legacy medium elemental input/output calculation."""

    result = {
        element: {"in": 0.0, "out": 0.0, "net": 0.0}
        for element in ELEMENTS
    }

    for reaction, bsl_flux in medium_flux.items():
        if not str(reaction).startswith("EX_"):
            continue
        flux = float(bsl_flux)
        exact_id, base_id = parse_exchange_metabolite(str(reaction))
        counts = annotation_lookup.get(exact_id)
        if counts is None:
            counts = annotation_lookup.get(base_id)
        if counts is None:
            continue

        for element in ELEMENTS:
            count = pd.to_numeric(
                counts.get(f"{element}_number"),
                errors="coerce",
            )
            if pd.isna(count) or count == 0:
                continue

            amount = abs(flux) * float(count)
            if flux < 0:
                result[element]["in"] += amount
            elif flux > 0:
                result[element]["out"] += amount

    for element in ELEMENTS:
        result[element]["net"] = (
            result[element]["in"] - result[element]["out"]
        )
    return result


def calculate_flux_record(
    sample: str,
    simulation_type: str,
    flux_file: str | Path,
    annotation_lookup: dict[str, dict[str, float]],
) -> dict[str, object]:
    """Calculate elemental flux totals for one MICOM solution."""

    flux_path = Path(flux_file).resolve()
    flux = pd.read_csv(flux_path)
    if "compartment" not in flux.columns:
        raise ValueError(f"{flux_path} lacks the 'compartment' column.")

    medium = flux[flux["compartment"] == "medium"]
    if len(medium) != 1:
        raise ValueError(
            f"{flux_path} must contain exactly one medium row; found {len(medium)}."
        )

    medium_flux = pd.to_numeric(
        medium.iloc[0].drop(labels="compartment"),
        errors="coerce",
    ).fillna(0.0)
    element_flux = compute_element_flux(medium_flux, annotation_lookup)

    record: dict[str, object] = {
        "sample": sample,
        "simulation_type": simulation_type,
        "solution_id": flux_path.name.removesuffix("-flux.csv"),
        "flux_file": flux_path.name,
    }
    for element in ELEMENTS:
        for direction in ("in", "out", "net"):
            record[f"{element}_{direction}"] = element_flux[element][direction]

    c_net = element_flux["C"]["net"]
    n_net = element_flux["N"]["net"]
    p_net = element_flux["P"]["net"]
    s_net = element_flux["S"]["net"]
    record.update(
        {
            "C_N": _safe_ratio(c_net, n_net),
            "C_P": _safe_ratio(c_net, p_net),
            "C_S": _safe_ratio(c_net, s_net),
            "N_P": _safe_ratio(n_net, p_net),
        }
    )
    return record


def add_eai_metrics(
    table: pd.DataFrame,
    *,
    epsilon: float = EPSILON,
    budget_alpha: float = BUDGET_ALPHA,
) -> pd.DataFrame:
    """Add legacy EAI and EAIshape values using bsl as the fixed baseline."""

    output = table.copy()
    baseline_rows = output[output["simulation_type"] == BASELINE_TYPE]
    if len(baseline_rows) != 1:
        raise ValueError(
            "EAI requires exactly one bsl baseline row per flux pair; "
            f"found {len(baseline_rows)}."
        )

    baseline = baseline_rows.iloc[0]
    baseline_positive = {
        element: max(float(baseline[f"{element}_net"]), 0.0)
        for element in ELEMENTS
    }
    baseline_budget = sum(baseline_positive.values())
    baseline_proportions = {
        element: (
            (baseline_positive[element] + budget_alpha)
            / (baseline_budget + len(ELEMENTS) * budget_alpha)
        )
        for element in ELEMENTS
    }

    for element in ELEMENTS:
        output[f"{element}_net_pos"] = output[f"{element}_net"].clip(lower=0)

    output["budget_EAI"] = output[
        [f"{element}_net_pos" for element in ELEMENTS]
    ].sum(axis=1)

    for element in ELEMENTS:
        output[f"EAI_{element}"] = (
            (output[f"{element}_net_pos"] + epsilon)
            / (baseline_positive[element] + epsilon)
        )
        output[f"p_{element}"] = (
            (output[f"{element}_net_pos"] + budget_alpha)
            / (output["budget_EAI"] + len(ELEMENTS) * budget_alpha)
        )
        output[f"EAIshape_{element}"] = (
            output[f"p_{element}"] / baseline_proportions[element]
        )

    output["EAI"] = output.apply(
        lambda row: ";".join(
            f"{element}={row[f'EAI_{element}']:.6g}"
            for element in ELEMENTS
        ),
        axis=1,
    )
    output["EAIshape"] = output.apply(
        lambda row: ";".join(
            f"{element}={row[f'EAIshape_{element}']:.6g}"
            for element in ELEMENTS
        ),
        axis=1,
    )

    mode_order = pd.Categorical(
        output["simulation_type"],
        categories=[BASELINE_TYPE, CONTEXT_TYPE],
        ordered=True,
    )
    return (
        output.assign(_mode_order=mode_order)
        .sort_values("_mode_order")
        .drop(columns="_mode_order")
        .reset_index(drop=True)
    )


def radar_source(table: pd.DataFrame) -> pd.DataFrame:
    """Return one source-data row per elemental radar spoke."""

    rows = []
    for element in ELEMENTS:
        row = {"element": element}
        for simulation_type in (BASELINE_TYPE, CONTEXT_TYPE):
            values = table.loc[
                table["simulation_type"] == simulation_type,
                f"EAIshape_{element}",
            ]
            if len(values) != 1:
                raise ValueError(
                    f"Expected one {simulation_type} EAIshape_{element} value."
                )
            row[simulation_type] = float(values.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_radar(source: pd.DataFrame, sample: str, output_stem: Path) -> None:
    """Plot ctx elemental-demand shape relative to the bsl unit baseline."""

    configure_publication_style()
    angles = np.linspace(0, 2 * np.pi, len(ELEMENTS), endpoint=False)
    closed_angles = np.append(angles, angles[0])

    bsl = source["bsl"].to_numpy(dtype=float)
    ctx = source["ctx"].to_numpy(dtype=float)
    bsl_closed = np.append(bsl, bsl[0])
    ctx_closed = np.append(ctx, ctx[0])

    finite = ctx[np.isfinite(ctx)]
    radial_max = max(1.1, (float(finite.max()) * 1.12) if finite.size else 1.1)

    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(
        closed_angles,
        bsl_closed,
        color="#767676",
        linewidth=1.1,
        linestyle="--",
        label="Baseline",
    )
    ax.plot(
        closed_angles,
        ctx_closed,
        color=FORMAL_SINGLE_COLOR,
        linewidth=1.6,
        marker="o",
        markersize=3.5,
        label="ctx",
    )
    ax.fill(closed_angles, ctx_closed, color=FORMAL_SINGLE_COLOR, alpha=0.12)
    ax.set_xticks(angles)
    ax.set_xticklabels(ELEMENTS, fontsize=11)
    ax.set_ylim(0, radial_max)
    ax.grid(color="#D8D8D8", linewidth=0.6)
    ax.spines["polar"].set_color("#A8A8A8")
    ax.set_title(
        f"{sample} elemental adaptation",
        pad=14,
        fontsize=12,
    )
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, output_stem)
    plt.close(fig)


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    if pd.isna(denominator) or abs(denominator) < epsilon:
        return np.nan
    return numerator / denominator


def calculate_job(
    job: FluxJob,
    annotation_lookup: dict[str, dict[str, float]],
    *,
    epsilon: float = EPSILON,
    budget_alpha: float = BUDGET_ALPHA,
) -> list[dict[str, object]]:
    """Calculate the four EAIshape spokes for one flux pair."""

    if job.bsl_flux is None:
        raise RuntimeError(f"EAI job lacks bsl flux: {job}")
    pair = add_eai_metrics(
        pd.DataFrame(
            [
                calculate_flux_record(
                    job.sample, BASELINE_TYPE, job.bsl_flux, annotation_lookup
                ),
                calculate_flux_record(
                    job.sample, CONTEXT_TYPE, job.ctx_flux, annotation_lookup
                ),
            ]
        ),
        epsilon=epsilon,
        budget_alpha=budget_alpha,
    )
    ctx = pair.loc[pair["simulation_type"].eq(CONTEXT_TYPE)].iloc[0]
    return [
        {
            "dataset": job.dataset,
            "input_type": job.input_type,
            "sensitivity_type": job.sensitivity_type,
            "member": job.member,
            "sample": job.sample,
            "metric": f"EAIshape_{element}",
            "element": element,
            "value": float(ctx[f"EAIshape_{element}"]),
            "bsl_flux": str(job.bsl_flux),
            "ctx_flux": str(job.ctx_flux),
        }
        for element in ELEMENTS
    ]


def plot_datasets(
    table: pd.DataFrame,
    batch_statistics: pd.DataFrame,
    samples: list[str],
    output: Path,
) -> None:
    """Plot all samples in shared-scale radar grids."""

    configure_publication_style()
    remove_non_pdf_figures(output)
    singles, batches = partition_metric_rows(table)
    angles = np.linspace(0, 2 * np.pi, len(ELEMENTS), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    datasets = list(table["dataset"].drop_duplicates())
    single_datasets = list(singles["dataset"].drop_duplicates())
    color_map = assign_dataset_colors(
        datasets,
        single_datasets=single_datasets,
        formal_as_black=True,
    )

    iqr_rows: list[dict[str, object]] = []
    for key, group in batches.groupby(
        ["dataset", "sample", "metric"], sort=False
    ):
        values = pd.to_numeric(
            group["value"], errors="coerce"
        ).dropna().to_numpy(float)
        iqr_rows.append(
            {
                "dataset": key[0],
                "sample": key[1],
                "metric": key[2],
                "lower": (
                    float(np.quantile(values, 0.25))
                    if len(values)
                    else np.nan
                ),
                "upper": (
                    float(np.quantile(values, 0.75))
                    if len(values)
                    else np.nan
                ),
            }
        )
    batch_iqr = pd.DataFrame(
        iqr_rows,
        columns=["dataset", "sample", "metric", "lower", "upper"],
    )

    def radial_limit(*series: pd.Series) -> float:
        values = pd.concat(
            [pd.to_numeric(item, errors="coerce") for item in series],
            ignore_index=True,
        )
        finite = values[np.isfinite(values)]
        return max(
            1.1,
            float(finite.max()) * 1.08 if len(finite) else 1.1,
        )

    sample_mask = (
        batch_statistics["sample"].isin(samples)
        if not batch_statistics.empty
        else pd.Series(dtype=bool)
    )
    iqr_mask = (
        batch_iqr["sample"].isin(samples)
        if not batch_iqr.empty
        else pd.Series(dtype=bool)
    )
    # Median plot: only batch medians across the requested samples.
    # Bands plot: only IQR edges. Do not use raw realization extremes —
    # those inflate the axis far above the plotted summaries.
    if batches.empty:
        median_radial_max = radial_limit(singles["value"])
        bands_radial_max = median_radial_max
    else:
        singles_values = singles.loc[
            singles["sample"].isin(samples), "value"
        ]
        # Median plot draws both formal singles and batch medians.
        median_radial_max = radial_limit(
            singles_values,
            batch_statistics.loc[sample_mask, "center"],
        )
        # Bands plot also overlays formal singles, so include them in the
        # radial limit along with IQR edges.
        bands_radial_max = radial_limit(
            singles_values,
            batch_iqr.loc[iqr_mask, "lower"],
            batch_iqr.loc[iqr_mask, "upper"],
        )

    def ordered_values(group: pd.DataFrame, column: str) -> np.ndarray:
        values = []
        for element in ELEMENTS:
            row = group[group["metric"].eq(f"EAIshape_{element}")]
            values.append(
                float(row[column].iloc[0]) if len(row) else np.nan
            )
        return np.asarray(values, dtype=float)

    def draw_grid(mode: str, output_stem: Path, radial_max: float) -> None:
        ncols = 2
        nrows = int(np.ceil(len(samples) / ncols))
        # Scale canvas with sample count; constrained layout spaces rows/cols.
        fig, axes = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": "polar"},
            figsize=(7.6, max(4.0, 4.2 * nrows)),
            squeeze=False,
            layout="constrained",
        )
        for ax, sample in zip(axes.flat, samples):
            ax.plot(
                closed_angles,
                np.ones(len(ELEMENTS) + 1),
                color="#777777",
                linestyle="--",
                linewidth=0.8,
                label="Baseline",
                zorder=2,
            )
            single_sample = singles[singles["sample"].eq(sample)]
            for dataset, group in single_sample.groupby(
                "dataset", sort=False
            ):
                values = ordered_values(group, "value")
                ax.plot(
                    closed_angles,
                    np.append(values, values[0]),
                    marker="o",
                    markersize=3.0,
                    linewidth=1.5,
                    color=color_map[dataset],
                    label=format_dataset_label(dataset),
                    zorder=4,
                )

            batch_sample = (
                batch_statistics[batch_statistics["sample"].eq(sample)]
                if mode == "median"
                else batch_iqr[batch_iqr["sample"].eq(sample)]
            )
            for dataset, group in batch_sample.groupby(
                "dataset", sort=False
            ):
                color = color_map[dataset]
                if mode == "median":
                    center = ordered_values(group, "center")
                    ax.plot(
                        closed_angles,
                        np.append(center, center[0]),
                        linewidth=1.5,
                        color=color,
                        label=f"{format_dataset_label(dataset)} median",
                        zorder=3,
                    )
                elif mode == "bands":
                    lower = np.maximum(ordered_values(group, "lower"), 0.0)
                    upper = ordered_values(group, "upper")
                    ax.fill_between(
                        closed_angles,
                        np.append(lower, lower[0]),
                        np.append(upper, upper[0]),
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                        label=f"{format_dataset_label(dataset)} IQR",
                        zorder=1,
                    )
                else:
                    raise ValueError(f"Unsupported radar mode: {mode}")

            ax.set_xticks(angles)
            ax.set_xticklabels(ELEMENTS, fontsize=11)
            ax.set_ylim(0, radial_max)
            ax.set_title(sample, pad=14, fontsize=12)
            ax.grid(color="#D8D8D8", linewidth=0.6)
            ax.spines["polar"].set_color("#A8A8A8")
            ax.tick_params(axis="y", labelsize=9)

        for ax in axes.flat[len(samples) :]:
            ax.set_visible(False)
        handles: list[object] = []
        labels: list[str] = []
        for ax in axes.flat[: len(samples)]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            ncol=min(4, max(1, len(labels))),
            fontsize=10,
        )
        # Extra row/column padding so polar titles do not collide as nrows grows.
        fig.get_layout_engine().set(
            h_pad=0.12,
            w_pad=0.06,
            hspace=0.18,
            wspace=0.10,
        )
        save_figure(fig, output_stem)
        plt.close(fig)

    if batches.empty:
        draw_grid("median", output / "eai_radar_single", median_radial_max)
    else:
        draw_grid("median", output / "eai_radar_median", median_radial_max)
        draw_grid("bands", output / "eai_radar_bands", bands_radial_max)
    for sample in samples:
        (output / f"eai_radar_{sample}.pdf").unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    datasets = parse_datasets(args)
    jobs, samples = discover_jobs(
        datasets,
        args.sample,
        require_bsl=True,
    )
    annotation = load_metabolite_table(METABOLITE_FILE)
    lookup = build_element_lookup(annotation)
    table = pd.DataFrame(
        row for job in jobs for row in calculate_job(job, lookup)
    )
    singles = single_value_table(table)
    batch_statistics = summarize_batch_values(table)
    output = analysis_output_dir(args)
    outputs = {
        "all": write_csv_atomic(table, output / "eai_all.csv"),
        "single": write_csv_atomic(
            singles,
            output / "eai_single_values.csv",
        ),
        "batch_statistics": write_csv_atomic(
            batch_statistics,
            output / "eai_batch_statistics.csv",
        ),
    }
    plot_datasets(table, batch_statistics, samples, output)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    return add_dataset_arguments(
        argparse.ArgumentParser(
        description=(
                "Calculate EAI from single datasets and/or batches of "
                "simulation-member directories."
            )
        )
    )


def main() -> None:
    args = build_parser().parse_args()
    outputs = run(args)
    print(f"EAI complete: {outputs['all'].parent}")


if __name__ == "__main__":
    main()
