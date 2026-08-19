#!/usr/bin/env python3
"""Fixed defaults for the published example and case-study pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SENSITIVITY_CHOICES = ("medium", "tradeoff", "reaction")
METRIC_CHOICES = ("eai", "arb", "ac")


def paths_for(data_root: Path) -> dict[str, Path]:
    return {
        "gem_dir": data_root / "inputs/01_gems",
        "ge_file": data_root / "inputs/01_taxon_ge.csv",
        "ga_file": data_root / "inputs/01_taxon_ga.csv",
        "gene_expr_file": data_root / "inputs/02_gene_expression.csv",
        "medium_file": data_root / "inputs/03_medium.csv",
        "medium_reaction_list": data_root / "inputs/04_medium_sensitivity_list.csv",
        "reaction_list": data_root / "inputs/04_reaction_sensitivity_list.csv",
        "guild_file": data_root / "inputs/04_taxon_guild.csv",
        "output_root": data_root / "outputs",
        "analysis_dir": data_root / "outputs/04_analysis",
    }


def _preset(**kwargs: Any) -> dict[str, Any]:
    data_root = kwargs["data_root"]
    return {**kwargs, **paths_for(data_root)}


EXAMPLE = _preset(
    label="example",
    description=(
        "MetaContext-CMM tutorial pipeline for the shipped example/ community "
        "(sample EX01): baseline + RIPTiDe/context formal, optional "
        "sensitivity batches, then metrics under 04_analysis/<metric>/."
    ),
    data_root=Path("example"),
    sample=["EX01"],
    medium_bounds=[1000.0, 100.0, 10.0],
    tradeoffs=[0.1, 0.5, 1.0],
    reaction_realizations=5,
    ac_heatmap_guilds=["GI", "GII"],
)

CASE_STUDY = _preset(
    label="case_study",
    description=(
        "MetaContext-CMM publication pipeline for the Zenodo case_study/ "
        "community (samples SW46 SW51 SW55 SW61): baseline + RIPTiDe/context "
        "formal, sensitivity batches, then metrics under 04_analysis/<metric>/."
    ),
    data_root=Path("case_study"),
    sample=["SW46", "SW51", "SW55", "SW61"],
    medium_bounds=[1000, 619, 383, 237, 146, 90, 56, 34, 21, 13, 8, 5],
    tradeoffs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    reaction_realizations=100,
    ac_heatmap_guilds=["GI", "GII", "GIII"],
)
