#!/usr/bin/env python3
"""Discordance / abundance gates used by ``infer_context_bounds`` QC.

Internal helper (not a published analysis CLI). Provides:

- ``discordance`` — reject DNA/RNA practical-zero mismatches
- ``gates`` — discordance + optional gene-count gate
- ``min-clear`` — smallest paired-support cutoff that clears discordance

Loaded by ``infer_context_bounds.load_gates_module()``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_RE = re.compile(r"^(?P<sample>.+)-(?P<measure>GA|GE|CN)$", re.IGNORECASE)

DEFAULT_METHOD = "gates"
DEFAULT_FIXED_THRESHOLD = 0.01
DEFAULT_THRESHOLD_SCOPE = "pooled"
DEFAULT_MIN_CN = 10
DEFAULT_DISCORDANCE_FLOOR = 0.01
MICOM_REL_ABUNDANCE_TOLERANCE = 1e-6
LITE_METHODS = ("discordance", "gates", "min-clear")


@dataclass(frozen=True)
class ThresholdResult:
    scope: str
    requested_method: str
    selected_method: str
    threshold: float
    positive_n: int
    zero_n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lightweight MAG/sample constraint gates: discordance, gates, "
            "and min-clear (no near-zero threshold algorithms)."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory containing context_species.csv and "
            "context_summary.csv."
        ),
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Samples to process (default: infer GA/GE/CN columns).",
    )
    parser.add_argument(
        "--method",
        choices=LITE_METHODS,
        default=DEFAULT_METHOD,
        help=(
            "gates: discordance + --min-cn (no near-zero cutoff). "
            "discordance: discordance only. "
            "min-clear: smallest paired-support cutoff that clears "
            f"discordance (default: {DEFAULT_METHOD})."
        ),
    )
    parser.add_argument(
        "--threshold-scope",
        choices=("pooled", "sample"),
        default=DEFAULT_THRESHOLD_SCOPE,
        help=(
            "Estimate one pooled threshold or a threshold per sample "
            f"(default: {DEFAULT_THRESHOLD_SCOPE})."
        ),
    )
    parser.add_argument(
        "--min-cn",
        type=int,
        default=DEFAULT_MIN_CN,
        help=(
            "Minimum expressed model-associated gene count, inclusive "
            f"(default: {DEFAULT_MIN_CN}; ignored by discordance/min-clear)."
        ),
    )
    parser.add_argument(
        "--discordance-floor",
        type=float,
        default=DEFAULT_DISCORDANCE_FLOOR,
        help=(
            "Practical-zero TPM floor for DNA/RNA zero-status mismatches "
            f"(default: {DEFAULT_DISCORDANCE_FLOOR:g})."
        ),
    )
    parser.add_argument(
        "--skip-abundance-prefilter",
        action="store_true",
        help=(
            "Disable the per-sample GE relative-abundance mark "
            f"(default: mark rel_ge <= {MICOM_REL_ABUNDANCE_TOLERANCE:g})."
        ),
    )
    parser.add_argument(
        "--gene-expression-file",
        type=Path,
        default=None,
        help="Optional full long-format gene-expression CSV to filter.",
    )
    parser.add_argument(
        "--filtered-gene-expression-file",
        type=Path,
        default=None,
        help=(
            "Output path for the filtered gene-expression CSV; "
            "requires --gene-expression-file."
        ),
    )
    return parser.parse_args()


def require_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}")


def discover_samples(columns: Iterable[str]) -> dict[str, dict[str, str]]:
    discovered: dict[str, dict[str, str]] = {}
    for column in columns:
        match = COLUMN_RE.match(str(column))
        if match is None:
            continue
        sample = match.group("sample")
        measure = match.group("measure").upper()
        discovered.setdefault(sample, {})[measure] = str(column)
    return discovered


def validate_and_select_samples(
    frame: pd.DataFrame, requested: list[str] | None
) -> tuple[str, list[str], dict[str, dict[str, str]]]:
    if frame.empty:
        raise ValueError("Input table is empty.")
    id_column = str(frame.columns[0])
    if frame[id_column].isna().any():
        raise ValueError(f"ID column {id_column!r} contains missing values.")
    if frame[id_column].astype(str).duplicated().any():
        duplicates = (
            frame.loc[frame[id_column].astype(str).duplicated(), id_column]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Duplicate MAG IDs: {', '.join(duplicates)}")

    discovered = discover_samples(frame.columns)
    complete = sorted(
        sample
        for sample, measures in discovered.items()
        if {"GA", "GE", "CN"}.issubset(measures)
    )
    if requested is None:
        samples = complete
    else:
        samples = requested
        missing = [sample for sample in samples if sample not in complete]
        if missing:
            raise ValueError(
                "Requested samples lack GA, GE or CN columns: "
                + ", ".join(missing)
            )
    if not samples:
        raise ValueError("No samples with GA, GE and CN columns were found.")
    return id_column, samples, discovered


def build_long_table(
    frame: pd.DataFrame,
    id_column: str,
    samples: list[str],
    columns: dict[str, dict[str, str]],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sample in samples:
        measure_columns = columns[sample]
        part = pd.DataFrame(
            {
                "mag_id": frame[id_column].astype(str),
                "sample": sample,
                "dna_tpm": pd.to_numeric(
                    frame[measure_columns["GA"]], errors="raise"
                ),
                "rna_tpm": pd.to_numeric(
                    frame[measure_columns["GE"]], errors="raise"
                ),
                "expressed_gene_count": pd.to_numeric(
                    frame[measure_columns["CN"]], errors="raise"
                ),
            }
        )
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)

    numeric_columns = ("dna_tpm", "rna_tpm", "expressed_gene_count")
    if not np.isfinite(result[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("GA, GE and CN values must all be finite.")
    if (result[list(numeric_columns)] < 0).any().any():
        raise ValueError("GA, GE and CN values must all be non-negative.")
    result["paired_support"] = result[["dna_tpm", "rna_tpm"]].min(axis=1)
    return result


def apply_micom_abundance_prefilter(
    long_frame: pd.DataFrame,
    *,
    enabled: bool = True,
    tolerance: float = MICOM_REL_ABUNDANCE_TOLERANCE,
) -> pd.DataFrame:
    """Per-sample L1-normalize GE and mark MICOM relative-abundance failures."""

    result = long_frame.copy()
    rna = result["rna_tpm"].to_numpy(dtype=float)
    sample_sums = result.groupby("sample", sort=False)["rna_tpm"].transform(
        "sum"
    ).to_numpy(dtype=float)
    rel_ge = np.zeros(len(result), dtype=float)
    positive_sum = sample_sums > 0
    rel_ge[positive_sum] = rna[positive_sum] / sample_sums[positive_sum]
    result["rel_ge"] = rel_ge
    if enabled:
        result["abundance_prefilter_pass"] = result["rel_ge"] > tolerance
    else:
        result["abundance_prefilter_pass"] = True
    return result


def clears_dna_rna_discordance(
    frame: pd.DataFrame,
    threshold: float,
    *,
    min_cn: int,
    discordance_floor: float,
) -> tuple[bool, int]:
    """Return whether a cutoff leaves zero DNA/RNA practical-zero mismatches."""

    dna = frame["dna_tpm"].to_numpy(dtype=float)
    rna = frame["rna_tpm"].to_numpy(dtype=float)
    cn = frame["expressed_gene_count"].to_numpy(dtype=float)
    eligible = (dna >= threshold) & (rna >= threshold) & (cn >= min_cn)
    mismatch = (dna < discordance_floor) ^ (rna < discordance_floor)
    retained_mismatch = int(np.count_nonzero(eligible & mismatch))
    return retained_mismatch == 0, retained_mismatch


def minimum_discordance_clearing_threshold(
    frame: pd.DataFrame,
    *,
    min_cn: int,
    discordance_floor: float,
) -> float:
    """Smallest cutoff that clears DNA/RNA practical-zero mismatches."""

    required = {"dna_tpm", "rna_tpm", "expressed_gene_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            "min-clear requires decision columns: "
            + ", ".join(sorted(missing))
        )
    dna = frame["dna_tpm"].to_numpy(dtype=float)
    rna = frame["rna_tpm"].to_numpy(dtype=float)
    cn = frame["expressed_gene_count"].to_numpy(dtype=float)
    mismatch = (dna < discordance_floor) ^ (rna < discordance_floor)
    relevant = mismatch & (cn >= min_cn)
    if not np.any(relevant):
        return 0.0
    low = np.minimum(dna[relevant], rna[relevant])
    critical = float(np.max(low))
    if not np.isfinite(critical):
        raise ValueError("min-clear could not determine a finite cutoff.")
    return float(np.nextafter(critical, np.inf))


def min_clear_threshold(
    positive_support: np.ndarray,
    scope: str,
    zero_n: int,
    decision_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> ThresholdResult:
    threshold = minimum_discordance_clearing_threshold(
        decision_frame,
        min_cn=0,
        discordance_floor=args.discordance_floor,
    )
    clears, retained = clears_dna_rna_discordance(
        decision_frame,
        threshold,
        min_cn=0,
        discordance_floor=args.discordance_floor,
    )
    if not clears:
        raise RuntimeError(
            f"min-clear cutoff {threshold:g} still retains {retained} "
            f"DNA/RNA discordance pair(s) in scope {scope}."
        )
    return ThresholdResult(
        scope=scope,
        requested_method="min-clear",
        selected_method="min-clear-discordance",
        threshold=threshold,
        positive_n=int(positive_support.size),
        zero_n=zero_n,
    )


def estimate_threshold(
    support: np.ndarray,
    scope: str,
    args: argparse.Namespace,
    decision_frame: pd.DataFrame | None = None,
) -> ThresholdResult:
    positive = support[support > 0]
    zero_n = int((support <= 0).sum())
    if args.method == "gates":
        return ThresholdResult(
            scope=scope,
            requested_method="gates",
            selected_method="gates-discordance-mincn",
            threshold=0.0,
            positive_n=int(positive.size),
            zero_n=zero_n,
        )
    if args.method == "discordance":
        return ThresholdResult(
            scope=scope,
            requested_method="discordance",
            selected_method="discordance-only",
            threshold=0.0,
            positive_n=int(positive.size),
            zero_n=zero_n,
        )
    if args.method == "min-clear":
        if decision_frame is None:
            raise ValueError(
                "min-clear threshold estimation requires the MAG/sample "
                "decision frame."
            )
        return min_clear_threshold(
            positive,
            scope=scope,
            zero_n=zero_n,
            decision_frame=decision_frame,
            args=args,
        )
    raise ValueError(f"Unsupported threshold method: {args.method}")


def assign_decisions(
    long_frame: pd.DataFrame,
    thresholds: dict[str, ThresholdResult],
    threshold_scope: str,
    min_cn: int,
    discordance_floor: float,
    enforce_discordance_guard: bool,
) -> pd.DataFrame:
    result = long_frame.copy()
    if "abundance_prefilter_pass" not in result.columns:
        result["abundance_prefilter_pass"] = True
    if "rel_ge" not in result.columns:
        result["rel_ge"] = np.nan

    if threshold_scope == "pooled":
        threshold = thresholds["pooled"].threshold
        result["paired_support_threshold"] = threshold
        result["threshold_method"] = thresholds["pooled"].selected_method
    else:
        result["paired_support_threshold"] = result["sample"].map(
            {sample: item.threshold for sample, item in thresholds.items()}
        )
        result["threshold_method"] = result["sample"].map(
            {sample: item.selected_method for sample, item in thresholds.items()}
        )

    result["dna_pass"] = result["dna_tpm"] >= result["paired_support_threshold"]
    result["rna_pass"] = result["rna_tpm"] >= result["paired_support_threshold"]
    result["paired_support_pass"] = result["dna_pass"] & result["rna_pass"]
    result["cn_pass"] = result["expressed_gene_count"] >= min_cn
    result["dna_practical_zero"] = result["dna_tpm"] < discordance_floor
    result["rna_practical_zero"] = result["rna_tpm"] < discordance_floor
    result["dna_rna_zero_mismatch"] = (
        result["dna_practical_zero"] ^ result["rna_practical_zero"]
    )
    result["discordance_guard_applied"] = enforce_discordance_guard
    result["discordance_guard_pass"] = (
        ~result["dna_rna_zero_mismatch"]
        if enforce_discordance_guard
        else True
    )
    result["constraint_eligible"] = (
        result["paired_support_pass"]
        & result["cn_pass"]
        & result["discordance_guard_pass"]
        & result["abundance_prefilter_pass"]
    )
    result["fixed_0_01_paired_pass"] = (
        (result["dna_tpm"] >= DEFAULT_FIXED_THRESHOLD)
        & (result["rna_tpm"] >= DEFAULT_FIXED_THRESHOLD)
    )
    result["fixed_0_01_constraint_eligible"] = (
        result["fixed_0_01_paired_pass"] & result["cn_pass"]
    )

    def reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not bool(row["abundance_prefilter_pass"]):
            reasons.append("abundance_below_micom_rtol")
        if enforce_discordance_guard and row["dna_rna_zero_mismatch"]:
            reasons.append("dna_rna_zero_mismatch")
        if not row["dna_pass"] and not row["rna_pass"]:
            reasons.append("both_near_zero")
        elif not row["dna_pass"]:
            reasons.append("dna_near_zero")
        elif not row["rna_pass"]:
            reasons.append("rna_near_zero")
        if not row["cn_pass"]:
            reasons.append("expressed_gene_count_below_min")
        return "eligible" if not reasons else ";".join(reasons)

    result["decision_reason"] = result.apply(reason, axis=1)
    return result


def make_context_summary(
    decisions: pd.DataFrame,
    thresholds: dict[str, ThresholdResult],
    threshold_scope: str,
    min_cn: int,
    discordance_floor: float,
) -> pd.DataFrame:
    """Abundance-first sequential drop / kept counts among DNA>0 or RNA>0."""

    rows: list[dict[str, object]] = []
    for sample, sample_frame in decisions.groupby("sample", sort=False):
        threshold = (
            thresholds["pooled"]
            if threshold_scope == "pooled"
            else thresholds[str(sample)]
        )
        diagnostics = asdict(threshold)
        diagnostics.pop("scope")
        diagnostics["cutoff"] = diagnostics.pop("threshold")

        in_scope = (
            (sample_frame["dna_tpm"].to_numpy(dtype=float) > 0)
            | (sample_frame["rna_tpm"].to_numpy(dtype=float) > 0)
        )
        scoped = sample_frame.loc[in_scope]

        if len(scoped) == 0:
            drop_abundance_prefilter_n = 0
            abundance_prefilter_pass_n = 0
            drop_threshold_n = 0
            drop_discordance_n = 0
            drop_cn_n = 0
            kept_nonzero_n = 0
            paired_pass_n = 0
            cn_pass_n = 0
            mismatch_n = 0
            mismatch_guard_excluded_n = 0
            mismatch_guard_retained_n = 0
            mismatch_final_eligible_n = 0
        else:
            pass_abundance = scoped["abundance_prefilter_pass"]
            drop_abundance_prefilter_n = int((~pass_abundance).sum())
            after_abundance = scoped.loc[pass_abundance]
            abundance_prefilter_pass_n = int(len(after_abundance))

            pass_threshold = after_abundance["paired_support_pass"]
            drop_threshold_n = int((~pass_threshold).sum())
            after_threshold = after_abundance.loc[pass_threshold]

            pass_discordance = after_threshold["discordance_guard_pass"]
            drop_discordance_n = int((~pass_discordance).sum())
            after_discordance = after_threshold.loc[pass_discordance]

            pass_cn = after_discordance["cn_pass"]
            drop_cn_n = int((~pass_cn).sum())
            kept_nonzero_n = int(pass_cn.sum())

            paired_pass_n = int(pass_threshold.sum())
            cn_pass_n = int(after_abundance["cn_pass"].sum())
            mismatch = after_abundance["dna_rna_zero_mismatch"]
            mismatch_n = int(mismatch.sum())
            mismatch_guard_excluded_n = drop_discordance_n
            mismatch_guard_retained_n = int(
                (mismatch & after_abundance["discordance_guard_pass"]).sum()
            )
            mismatch_final_eligible_n = int(
                (mismatch & after_abundance["constraint_eligible"]).sum()
            )

        row: dict[str, object] = {
            "sample": sample,
            "threshold_scope": threshold_scope,
            "min_cn": min_cn,
            "discordance_floor": discordance_floor,
            "fixed_reference_cutoff": DEFAULT_FIXED_THRESHOLD,
            **diagnostics,
            "total_mag_n": len(sample_frame),
            "nonzero_mag_n": int(in_scope.sum()),
            "exact_zero_mag_n": int((~in_scope).sum()),
            "abundance_prefilter_pass_n": abundance_prefilter_pass_n,
            "drop_abundance_prefilter_n": drop_abundance_prefilter_n,
            "paired_support_pass_n": paired_pass_n,
            "paired_support_excluded_n": drop_threshold_n,
            "cn_pass_n": cn_pass_n,
            "cn_excluded_n": drop_cn_n,
            "discordance_guard_applied": bool(
                sample_frame["discordance_guard_applied"].iloc[0]
            ),
            "constraint_eligible_n": int(sample_frame["constraint_eligible"].sum()),
            "constraint_excluded_n": int(
                (~sample_frame["constraint_eligible"]).sum()
            ),
            "drop_threshold_n": drop_threshold_n,
            "drop_discordance_n": drop_discordance_n,
            "drop_cn_n": drop_cn_n,
            "kept_nonzero_n": kept_nonzero_n,
            "dna_rna_zero_mismatch_n": mismatch_n,
            "dna_rna_zero_mismatch_guard_excluded_n": mismatch_guard_excluded_n,
            "dna_rna_zero_mismatch_guard_retained_n": mismatch_guard_retained_n,
            "dna_rna_zero_mismatch_final_eligible_n": mismatch_final_eligible_n,
            "fixed_0_01_constraint_eligible_n": int(
                sample_frame["fixed_0_01_constraint_eligible"].sum()
            ),
        }
        for category in (
            "eligible",
            "abundance_below_micom_rtol",
            "dna_near_zero",
            "rna_near_zero",
            "both_near_zero",
            "dna_rna_zero_mismatch",
            "expressed_gene_count_below_min",
        ):
            row[f"decision_reason_{category}_n"] = (
                int(
                    scoped["decision_reason"].str.contains(
                        rf"(?:^|;){re.escape(category)}(?:$|;)", regex=True
                    ).sum()
                )
                if len(scoped)
                else 0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_filtered_gene_expression(
    source_file: Path,
    output_file: Path,
    decisions: pd.DataFrame,
) -> tuple[int, int]:
    source_path = source_file.expanduser().resolve()
    output_path = output_file.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Gene-expression CSV not found: {source_path}"
        )
    if source_path == output_path:
        raise ValueError(
            "Filtered gene-expression output must differ from its source."
        )

    eligible_pairs = {
        (str(row.mag_id), str(row.sample))
        for row in decisions.loc[
            decisions["constraint_eligible"], ["mag_id", "sample"]
        ].itertuples(index=False)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    written_rows = 0
    seen_pairs: set[tuple[str, str]] = set()

    try:
        with (
            source_path.open("r", encoding="utf-8-sig", newline="") as source,
            os.fdopen(descriptor, "w", encoding="utf-8", newline="") as target,
        ):
            reader = csv.DictReader(source)
            required = {"gene_id", "expression", "mag_id", "sample_id"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    "Gene-expression CSV missing required columns: "
                    + ", ".join(sorted(missing))
                )
            writer = csv.DictWriter(
                target,
                fieldnames=list(reader.fieldnames or []),
            )
            writer.writeheader()
            for row in reader:
                pair = (str(row["mag_id"]), str(row["sample_id"]))
                if pair in eligible_pairs:
                    writer.writerow(row)
                    written_rows += 1
                    seen_pairs.add(pair)

        missing_pairs = eligible_pairs - seen_pairs
        if missing_pairs:
            preview = ", ".join(
                f"{mag}/{sample}" for mag, sample in sorted(missing_pairs)[:10]
            )
            raise ValueError(
                "Eligible MAG/sample pairs are absent from the gene-expression "
                f"source: {preview}"
            )
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return written_rows, len(seen_pairs)


def main() -> None:
    args = parse_args()
    require_nonnegative("--discordance-floor", args.discordance_floor)
    if args.min_cn < 0:
        raise ValueError("--min-cn must be non-negative.")
    if (args.gene_expression_file is None) != (
        args.filtered_gene_expression_file is None
    ):
        raise ValueError(
            "--gene-expression-file and --filtered-gene-expression-file "
            "must be supplied together."
        )

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "context_species.csv"
    summary_file = output_dir / "context_summary.csv"
    if (
        args.filtered_gene_expression_file is not None
        and args.filtered_gene_expression_file.expanduser().resolve()
        in {output_file, summary_file}
    ):
        raise ValueError(
            "--filtered-gene-expression-file conflicts with a standard "
            "filter output filename."
        )

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    id_column, samples, columns = validate_and_select_samples(frame, args.samples)
    long_frame = build_long_table(frame, id_column, samples, columns)
    long_frame = apply_micom_abundance_prefilter(
        long_frame,
        enabled=not args.skip_abundance_prefilter,
    )

    thresholds: dict[str, ThresholdResult] = {}
    if args.threshold_scope == "pooled":
        thresholds["pooled"] = estimate_threshold(
            long_frame["paired_support"].to_numpy(dtype=float),
            "pooled",
            args,
            decision_frame=long_frame,
        )
    else:
        for sample in samples:
            sample_frame = long_frame.loc[
                long_frame["sample"] == sample
            ].copy()
            thresholds[sample] = estimate_threshold(
                sample_frame["paired_support"].to_numpy(dtype=float),
                sample,
                args,
                decision_frame=sample_frame,
            )

    effective_min_cn = (
        0 if args.method in {"discordance", "min-clear"} else args.min_cn
    )
    decisions = assign_decisions(
        long_frame,
        thresholds=thresholds,
        threshold_scope=args.threshold_scope,
        min_cn=effective_min_cn,
        discordance_floor=args.discordance_floor,
        enforce_discordance_guard=args.method in {"discordance", "gates"},
    )
    summary = make_context_summary(
        decisions,
        thresholds=thresholds,
        threshold_scope=args.threshold_scope,
        min_cn=effective_min_cn,
        discordance_floor=args.discordance_floor,
    )

    eligibility = decisions.pivot(
        index="mag_id", columns="sample", values="constraint_eligible"
    )
    eligibility = eligibility.reindex(frame[id_column].astype(str))
    eligibility = eligibility.astype(int).reset_index()
    eligibility.columns.name = None
    eligibility.to_csv(output_file, index=False)
    summary.to_csv(summary_file, index=False)

    filtered_gene_expression_stats: tuple[int, int] | None = None
    if args.gene_expression_file is not None:
        filtered_gene_expression_stats = write_filtered_gene_expression(
            args.gene_expression_file,
            args.filtered_gene_expression_file,
            decisions,
        )

    print(f"Input: {input_path}")
    print(f"Context species: {output_file}")
    print(f"Context summary: {summary_file}")
    for scope, item in thresholds.items():
        print(
            f"Threshold [{scope}]: {item.threshold:.8g} "
            f"({item.selected_method})"
        )
    for row in summary.itertuples(index=False):
        print(
            f"{row.sample}: {row.constraint_eligible_n}/{row.total_mag_n} eligible "
            f"(kept_nonzero={row.kept_nonzero_n} of "
            f"{row.abundance_prefilter_pass_n} after 1e-6; "
            f"drop_1e-6={row.drop_abundance_prefilter_n}; "
            f"fixed 0.01 comparison: "
            f"{row.fixed_0_01_constraint_eligible_n}/{row.total_mag_n}; "
            f"DNA/RNA zero mismatches guard-excluded: "
            f"{row.dna_rna_zero_mismatch_guard_excluded_n}/"
            f"{row.dna_rna_zero_mismatch_n})"
        )
    if filtered_gene_expression_stats is not None:
        print(
            "Filtered gene-expression: "
            f"{filtered_gene_expression_stats[0]} rows across "
            f"{filtered_gene_expression_stats[1]} MAG/sample pairs -> "
            f"{args.filtered_gene_expression_file.expanduser().resolve()}"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
