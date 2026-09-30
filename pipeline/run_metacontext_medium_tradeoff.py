#!/usr/bin/env python3
"""Independent MetaContext medium + tradeoff pipeline (isolated run root).

Does **not** modify published ``pipeline/run_case_study.py`` / ``_orchestrator.py``.
Reuses existing pre-solve simulation scripts under ``workflow/``.

Default MetaContext source: ``case_study/outputs/02_context/`` (copied, not
symlinked). Baseline CMMs: read-only symlink to ``01_baseline``.

All writes go under a new timestamped tree::

    case_study/runs/metacontext_medium_tradeoff_<YYYYMMDD_HHMMSS>/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from presets import CASE_STUDY  # noqa: E402

SIMULATE = Path("workflow/simulation/simulate_community_model.py")
TRADEOFF = Path("workflow/simulation/sensitivity_tradeoff.py")
MEDIUM = Path("workflow/simulation/sensitivity_medium.py")
EAI = Path("workflow/analysis/metrics_eai.py")
ARB = Path("workflow/analysis/metrics_arb.py")
AC = Path("workflow/analysis/metrics_ac.py")

CONTEXT_TABLES = (
    "context_bounds.csv",
    "context_species.csv",
    "context_expression.csv",
    "context_summary.csv",
)

# Acceptance tolerances (ACCEPTANCE_PRESOLVE.md)
GROWTH_ATOL, GROWTH_RTOL, GROWTH_RMIN = 1e-9, 1e-8, 1e-12
MEMBER_ATOL, MEMBER_RTOL, MEMBER_RMIN = 1e-8, 1e-8, 1e-12
FLUX_ATOL, FLUX_RTOL, FLUX_RMIN = 1e-6, 1e-5, 1e-6


def repo_relative(path: str | Path) -> str:
    """Path relative to the repository when recording provenance."""

    candidate = Path(path).expanduser()
    absolute = (
        candidate.resolve()
        if candidate.is_absolute()
        else (REPO_ROOT / candidate).resolve()
    )
    try:
        relative = absolute.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return str(path)
    if not relative.parts:
        return "."
    return relative.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def nonempty_table(path: Path) -> bool:
    """True when CSV path exists and has more than blank/whitespace content."""

    if not path.is_file() or path.stat().st_size == 0:
        return False
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return bool(text)


def read_failure_table(path: Path) -> pd.DataFrame:
    if not nonempty_table(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def pass_tol(e: float, r: float, atol: float, rtol: float, r_min: float) -> bool:
    if e <= atol:
        return True
    if r >= r_min and e <= rtol * r:
        return True
    return False


def compare_flux_tables(
    formal_path: Path,
    other_path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """Compare formal vs tradeoff1 flux CSVs with OR absolute/relative gates."""

    a = pd.read_csv(formal_path, index_col=0)
    b = pd.read_csv(other_path, index_col=0)
    report: dict[str, Any] = {
        "mode": mode,
        "formal": repo_relative(formal_path),
        "other": repo_relative(other_path),
        "pass_overall": True,
        "failures": [],
    }

    # Community growth ≈ abundance-weighted is not stored; use member Growth mean
    # of finite values as community proxy, plus full Growth column check.
    if "Growth" not in a.columns or "Growth" not in b.columns:
        report["pass_overall"] = False
        report["failures"].append({"kind": "missing_Growth_column"})
        return report

    ga = a["Growth"].astype(float)
    gb = b["Growth"].reindex(ga.index).astype(float)
    # Align indices
    idx = ga.index.union(gb.index)
    ga = ga.reindex(idx)
    gb = gb.reindex(idx)

    # Community growth_rate proxy: nansum of member Growth (MICOM stores per-taxon)
    # Prefer documented: solution.growth_rate ≈ sum(abundance * growth); without
    # abundances use nanmean of finite member growth as secondary scalar.
    finite = ga.notna() & gb.notna() & np.isfinite(ga) & np.isfinite(gb)
    if finite.any():
        # Use sum of member Growth when both finite (stable scalar for gate)
        c_a = float(np.nansum(ga[finite]))
        c_b = float(np.nansum(gb[finite]))
        e = abs(c_a - c_b)
        r = abs(c_a)
        ok = pass_tol(e, r, GROWTH_ATOL, GROWTH_RTOL, GROWTH_RMIN)
        report["community_growth_proxy"] = {
            "formal": c_a,
            "other": c_b,
            "abs_err": e,
            "pass": ok,
        }
        if not ok:
            report["pass_overall"] = False
            report["failures"].append(
                {"kind": "community_growth_proxy", "abs_err": e, "formal": c_a, "other": c_b}
            )

    # Member Growth
    member_fail = 0
    for taxon in idx:
        va, vb = ga.get(taxon), gb.get(taxon)
        a_ok = va is not None and np.isfinite(va)
        b_ok = vb is not None and np.isfinite(vb)
        if not a_ok and not b_ok:
            continue
        if a_ok != b_ok:
            member_fail += 1
            report["failures"].append(
                {"kind": "member_Growth", "taxon": str(taxon), "detail": "finite_mismatch"}
            )
            continue
        e = abs(float(va) - float(vb))
        r = abs(float(va))
        if not pass_tol(e, r, MEMBER_ATOL, MEMBER_RTOL, MEMBER_RMIN):
            member_fail += 1
            if member_fail <= 20:
                report["failures"].append(
                    {
                        "kind": "member_Growth",
                        "taxon": str(taxon),
                        "formal": float(va),
                        "other": float(vb),
                        "abs_err": e,
                    }
                )
    report["member_Growth_fail_n"] = member_fail
    if member_fail:
        report["pass_overall"] = False

    # EX_ fluxes
    ex_cols = sorted(set(c for c in a.columns if str(c).startswith("EX_")) & set(b.columns))
    flux_fail = 0
    for col in ex_cols:
        sa = a[col].astype(float)
        sb = b[col].reindex(sa.index).astype(float)
        for taxon in sa.index.union(sb.index):
            va, vb = sa.get(taxon), sb.get(taxon)
            a_ok = va is not None and np.isfinite(va)
            b_ok = vb is not None and np.isfinite(vb)
            if not a_ok and not b_ok:
                continue
            if a_ok != b_ok:
                flux_fail += 1
                continue
            e = abs(float(va) - float(vb))
            r = abs(float(va))
            if not pass_tol(e, r, FLUX_ATOL, FLUX_RTOL, FLUX_RMIN):
                flux_fail += 1
                if flux_fail <= 30:
                    report["failures"].append(
                        {
                            "kind": "EX_flux",
                            "taxon": str(taxon),
                            "reaction": str(col),
                            "formal": float(va),
                            "other": float(vb),
                            "abs_err": e,
                        }
                    )
    report["ex_flux_fail_n"] = flux_fail
    report["ex_cols_compared"] = len(ex_cols)
    if flux_fail:
        report["pass_overall"] = False
    return report


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def append_tsv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    import csv

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_cmd(
    cmd: Sequence[str | Path],
    *,
    log_path: Path,
    label: str,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display = " ".join(repo_relative(c) for c in cmd)
    print(f"[RUN] {label}")
    print(f"      {display}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {label} ===\n{display}\n")
        log.flush()
        proc = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(REPO_ROOT),
            env=env or os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(f"\n[exit {proc.returncode}]\n")
    if proc.returncode != 0:
        print(f"[FAIL] {label} exit={proc.returncode} (see {log_path})")
    else:
        print(f"[OK] {label}")
    return proc.returncode


def bootstrap_run_root(
    *,
    case_root: Path,
    metacontext_src: Path,
    stamp: str,
) -> tuple[Path, Path]:
    run_root = case_root / "runs" / f"metacontext_medium_tradeoff_{stamp}"
    analysis_root = case_root / "runs" / f"metacontext_medium_tradeoff_{stamp}_analysis"
    if run_root.exists():
        raise SystemExit(f"Refusing to overwrite existing run root: {run_root}")

    src_baseline = case_root / "outputs" / "01_baseline"
    if not src_baseline.is_dir():
        raise SystemExit(f"Missing baseline: {src_baseline}")
    bounds = metacontext_src / "context_bounds.csv"
    if not bounds.is_file():
        raise SystemExit(f"Missing MetaContext bounds: {bounds}")

    run_root.mkdir(parents=True, exist_ok=False)
    analysis_root.mkdir(parents=True, exist_ok=True)
    (run_root / "03_simulation" / "context").mkdir(parents=True)
    (run_root / "logs").mkdir()
    (run_root / "compare").mkdir()

    baseline_link = run_root / "01_baseline"
    baseline_link.symlink_to(src_baseline.resolve(), target_is_directory=True)

    context_dir = run_root / "02_context"
    context_dir.mkdir()
    copied: list[dict[str, str]] = []
    for name in CONTEXT_TABLES:
        src = metacontext_src / name
        if not src.is_file():
            continue
        dst = context_dir / name
        shutil.copy2(src, dst)
        if dst.is_symlink():
            raise SystemExit(f"Context table must be a regular file: {dst}")
        copied.append(
            {
                "name": name,
                "source": repo_relative(src),
                "sha256": sha256_file(dst),
                "bytes": str(dst.stat().st_size),
            }
        )
    if not any(c["name"] == "context_bounds.csv" for c in copied):
        raise SystemExit("context_bounds.csv was not copied")

    write_json(
        run_root / "00_input_manifest.json",
        {
            "stamp": stamp,
            "metacontext_source": repo_relative(metacontext_src),
            "baseline_symlink": repo_relative(src_baseline),
            "copied_context_tables": copied,
            "case_study_preset_samples": CASE_STUDY["sample"],
            "medium_bounds": CASE_STUDY["medium_bounds"],
            "tradeoffs": CASE_STUDY["tradeoffs"],
            "medium_reaction_list": repo_relative(CASE_STUDY["medium_reaction_list"]),
            "medium_file": repo_relative(CASE_STUDY["medium_file"]),
        },
    )
    return run_root, analysis_root


def formal_ready(context_dir: Path, formal_dir: Path, sample: str) -> bool:
    return (
        nonempty(context_dir / f"{sample}-ctx.pickle")
        and nonempty(formal_dir / f"{sample}-bsl-flux.csv")
        and nonempty(formal_dir / f"{sample}-ctx-flux.csv")
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--case-study-root",
        type=Path,
        default=Path("case_study"),
    )
    p.add_argument(
        "--metacontext-dir",
        type=Path,
        default=None,
        help="Source MetaContext dir (default: <case>/outputs/02_context).",
    )
    p.add_argument(
        "--stamp",
        default=None,
        help="Run stamp YYYYMMDD_HHMMSS (default: now).",
    )
    p.add_argument(
        "--python-bin",
        type=Path,
        default=Path(sys.executable),
    )
    p.add_argument(
        "--resume-root",
        type=Path,
        default=None,
        help="Resume an existing metacontext_medium_tradeoff_* run root.",
    )
    p.add_argument(
        "--skip-metrics",
        action="store_true",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    case_root = args.case_study_root.expanduser().resolve()
    metacontext = (
        args.metacontext_dir.expanduser().resolve()
        if args.metacontext_dir
        else (case_root / "outputs" / "02_context").resolve()
    )
    python_bin = args.python_bin.expanduser().resolve()
    stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    medium_bounds = [float(x) for x in CASE_STUDY["medium_bounds"]]
    tradeoffs_full = [float(x) for x in CASE_STUDY["tradeoffs"]]
    samples_all = list(CASE_STUDY["sample"])
    medium_file = Path(CASE_STUDY["medium_file"]).resolve()
    medium_rxn = Path(CASE_STUDY["medium_reaction_list"]).resolve()
    ge_file = Path(CASE_STUDY["ge_file"]).resolve()
    guild_file = Path(CASE_STUDY["guild_file"]).resolve()

    if args.resume_root:
        run_root = args.resume_root.expanduser().resolve()
        analysis_root = Path(str(run_root) + "_analysis")
        if not analysis_root.is_dir():
            analysis_root.mkdir(parents=True, exist_ok=True)
        print(f"[RESUME] {run_root}")
    else:
        run_root, analysis_root = bootstrap_run_root(
            case_root=case_root,
            metacontext_src=metacontext,
            stamp=stamp,
        )

    context_dir = run_root / "02_context"
    baseline_dir = run_root / "01_baseline"
    formal_dir = run_root / "03_simulation" / "context"
    tradeoff_dir = run_root / "03_simulation" / "sensitivity_tradeoff"
    medium_dir = run_root / "03_simulation" / "sensitivity_medium"
    log_path = run_root / "logs" / "commands.log"
    status_path = run_root / "logs" / "task_status.tsv"

    # Environment versions
    env_info: dict[str, Any] = {
        "python": str(python_bin),
        "cwd": repo_relative(REPO_ROOT),
        "run_root": repo_relative(run_root),
        "analysis_root": repo_relative(analysis_root),
        "metacontext_source": repo_relative(metacontext),
        "samples": samples_all,
        "medium_bounds": medium_bounds,
        "tradeoffs": tradeoffs_full,
        "medium_reactions": [
            line.strip().lstrip("\ufeff")
            for line in medium_rxn.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ],
    }
    try:
        import micom  # type: ignore

        env_info["micom"] = getattr(micom, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover
        env_info["micom_import_error"] = str(exc)
    write_json(run_root / "00_env.json", env_info)

    print("\n=== Independent MetaContext medium/tradeoff pipeline ===")
    print(f"MetaContext source: {metacontext}")
    print(f"Run root:           {run_root}")
    print(f"Analysis root:      {analysis_root}")
    print(f"Samples:            {samples_all}")
    print(f"Tradeoffs:          {tradeoffs_full}")
    print(f"Medium bounds:      {medium_bounds}")
    print(f"Medium reactions:   {env_info['medium_reactions']}")
    print(
        "Est. top-level solves: formal 8 + tradeoff 80 + medium ctx ~44 "
        "(bound1000/bsl copy) ≈ 132"
    )

    def log_status(task: str, sample: str, status: str, detail: str = "") -> None:
        append_tsv(
            status_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "task": task,
                "sample": sample,
                "status": status,
                "detail": detail,
            },
        )

    # --- Stage A: SW46 formal ---
    gate_sample = "SW46"
    if not formal_ready(context_dir, formal_dir, gate_sample):
        rc = run_cmd(
            [
                python_bin,
                SIMULATE,
                "--sample",
                gate_sample,
                "--baseline-cmm-dir",
                baseline_dir,
                "--context-file",
                context_dir / "context_bounds.csv",
                "--medium-file",
                medium_file,
                "--output-dir",
                formal_dir,
            ],
            log_path=log_path,
            label=f"formal/{gate_sample}",
        )
        log_status("formal", gate_sample, "ok" if rc == 0 else "fail", f"exit={rc}")
        if rc != 0 or not formal_ready(context_dir, formal_dir, gate_sample):
            print("[STOP] SW46 formal incomplete")
            return 1
    else:
        print(f"[REUSE] formal/{gate_sample}")
        log_status("formal", gate_sample, "reuse")

    # Ensure pickle was not written under published tree
    pub_pickle = case_root / "outputs" / "02_context" / f"{gate_sample}-ctx.pickle"
    new_pickle = context_dir / f"{gate_sample}-ctx.pickle"
    if new_pickle.resolve() == pub_pickle.resolve():
        print("[STOP] ctx pickle path resolved to published outputs; abort")
        return 1

    # --- Stage A: SW46 tradeoff=1 ---
    tradeoff_dir.mkdir(parents=True, exist_ok=True)
    t1_bsl = tradeoff_dir / "tradeoff1" / f"{gate_sample}-tradeoff1-bsl-flux.csv"
    t1_ctx = tradeoff_dir / "tradeoff1" / f"{gate_sample}-tradeoff1-ctx-flux.csv"
    if not (nonempty(t1_bsl) and nonempty(t1_ctx)):
        rc = run_cmd(
            [
                python_bin,
                TRADEOFF,
                "--sample",
                gate_sample,
                "--baseline-cmm-dir",
                baseline_dir,
                "--context-cmm-dir",
                context_dir,
                "--output-dir",
                tradeoff_dir,
                "--tradeoffs",
                "1.0",
            ],
            log_path=log_path,
            label=f"tradeoff1/{gate_sample}",
        )
        log_status("tradeoff1", gate_sample, "ok" if rc == 0 else "fail", f"exit={rc}")
        if rc != 0:
            return 1
    else:
        print(f"[REUSE] tradeoff1/{gate_sample}")
        log_status("tradeoff1", gate_sample, "reuse")

    # Consistency gate
    reports = {}
    overall_ctx_ok = True
    overall_bsl_ok = True
    for mode in ("bsl", "ctx"):
        formal_flux = formal_dir / f"{gate_sample}-{mode}-flux.csv"
        other_flux = (
            tradeoff_dir / "tradeoff1" / f"{gate_sample}-tradeoff1-{mode}-flux.csv"
        )
        rep = compare_flux_tables(formal_flux, other_flux, mode=mode)
        reports[mode] = rep
        if mode == "ctx" and not rep["pass_overall"]:
            overall_ctx_ok = False
        if mode == "bsl" and not rep["pass_overall"]:
            overall_bsl_ok = False
    gate = {
        "sample": gate_sample,
        "ctx_aligned": overall_ctx_ok,
        "bsl_aligned": overall_bsl_ok,
        "overall_consistency_pass": overall_ctx_ok and overall_bsl_ok,
        "note": (
            "bsl mismatch ⇒ overall must not be marked passed"
            if overall_ctx_ok and not overall_bsl_ok
            else ""
        ),
        "reports": reports,
    }
    write_json(run_root / "compare" / f"{gate_sample}_formal_vs_tradeoff1.json", gate)
    print(
        f"[GATE] {gate_sample}: ctx_aligned={overall_ctx_ok} "
        f"bsl_aligned={overall_bsl_ok} overall={gate['overall_consistency_pass']}"
    )
    if not overall_ctx_ok:
        print("[STOP] SW46 formal ctx vs tradeoff1 ctx failed hard gate")
        return 1
    if not overall_bsl_ok:
        print(
            "[STOP] SW46 formal bsl vs tradeoff1 bsl failed; "
            "overall consistency not passed — not expanding"
        )
        return 1

    # --- Stage B: remaining samples formal + tradeoff1 ---
    other_samples = [s for s in samples_all if s != gate_sample]
    for sample in other_samples:
        if not formal_ready(context_dir, formal_dir, sample):
            rc = run_cmd(
                [
                    python_bin,
                    SIMULATE,
                    "--sample",
                    sample,
                    "--baseline-cmm-dir",
                    baseline_dir,
                    "--context-file",
                    context_dir / "context_bounds.csv",
                    "--medium-file",
                    medium_file,
                    "--output-dir",
                    formal_dir,
                ],
                log_path=log_path,
                label=f"formal/{sample}",
            )
            log_status("formal", sample, "ok" if rc == 0 else "fail", f"exit={rc}")
            if rc != 0 or not formal_ready(context_dir, formal_dir, sample):
                print(f"[STOP] formal incomplete for {sample}")
                return 1
        else:
            print(f"[REUSE] formal/{sample}")
            log_status("formal", sample, "reuse")

        t1b = tradeoff_dir / "tradeoff1" / f"{sample}-tradeoff1-bsl-flux.csv"
        t1c = tradeoff_dir / "tradeoff1" / f"{sample}-tradeoff1-ctx-flux.csv"
        if not (nonempty(t1b) and nonempty(t1c)):
            rc = run_cmd(
                [
                    python_bin,
                    TRADEOFF,
                    "--sample",
                    sample,
                    "--baseline-cmm-dir",
                    baseline_dir,
                    "--context-cmm-dir",
                    context_dir,
                    "--output-dir",
                    tradeoff_dir,
                    "--tradeoffs",
                    "1.0",
                ],
                log_path=log_path,
                label=f"tradeoff1/{sample}",
            )
            log_status("tradeoff1", sample, "ok" if rc == 0 else "fail", f"exit={rc}")
            if rc != 0:
                return 1
        else:
            print(f"[REUSE] tradeoff1/{sample}")
            log_status("tradeoff1", sample, "reuse")

        sample_gate = {}
        ctx_ok = bsl_ok = True
        for mode in ("bsl", "ctx"):
            rep = compare_flux_tables(
                formal_dir / f"{sample}-{mode}-flux.csv",
                tradeoff_dir / "tradeoff1" / f"{sample}-tradeoff1-{mode}-flux.csv",
                mode=mode,
            )
            sample_gate[mode] = rep
            if mode == "ctx" and not rep["pass_overall"]:
                ctx_ok = False
            if mode == "bsl" and not rep["pass_overall"]:
                bsl_ok = False
        payload = {
            "sample": sample,
            "ctx_aligned": ctx_ok,
            "bsl_aligned": bsl_ok,
            "overall_consistency_pass": ctx_ok and bsl_ok,
            "reports": sample_gate,
        }
        write_json(run_root / "compare" / f"{sample}_formal_vs_tradeoff1.json", payload)
        print(
            f"[GATE] {sample}: ctx={ctx_ok} bsl={bsl_ok} overall={ctx_ok and bsl_ok}"
        )
        if not ctx_ok or not bsl_ok:
            print(f"[STOP] consistency failure for {sample}; not expanding grids")
            return 1

    # --- Stage C: full tradeoff grid (0.1..1.0) all samples ---
    print("\n=== Full tradeoff grid ===")
    rc = run_cmd(
        [
            python_bin,
            TRADEOFF,
            "--sample",
            *samples_all,
            "--baseline-cmm-dir",
            baseline_dir,
            "--context-cmm-dir",
            context_dir,
            "--output-dir",
            tradeoff_dir,
            "--tradeoffs",
            *[f"{v:g}" for v in tradeoffs_full],
        ],
        log_path=log_path,
        label="tradeoff_full",
    )
    log_status("tradeoff_full", ",".join(samples_all), "ok" if rc == 0 else "fail")
    if rc != 0:
        return 1
    # Check tradeoff failures file (empty / whitespace-only = no failures)
    fail_csv = tradeoff_dir / "00_sensitivity_tradeoff_failures.csv"
    fails = read_failure_table(fail_csv)
    if len(fails) > 0:
        print(f"[STOP] tradeoff failures: {len(fails)} rows in {fail_csv}")
        return 1

    # --- Stage D: medium sensitivity ---
    print("\n=== Medium sensitivity ===")
    print(f"[CONFIG] medium bounds: {medium_bounds}")
    print(f"[CONFIG] medium reaction list ({medium_rxn}):")
    for rxn in env_info["medium_reactions"]:
        print(f"  - {rxn}")
    medium_dir.mkdir(parents=True, exist_ok=True)
    rc = run_cmd(
        [
            python_bin,
            MEDIUM,
            "--sample",
            *samples_all,
            "--baseline-cmm-dir",
            baseline_dir,
            "--medium-file",
            medium_file,
            "--context-cmm-dir",
            context_dir,
            "--output-dir",
            medium_dir,
            "--reaction-list",
            medium_rxn,
            "--baseline-flux-dir",
            formal_dir,
            "--bounds",
            *[f"{v:g}" for v in medium_bounds],
        ],
        log_path=log_path,
        label="medium_full",
    )
    log_status("medium_full", ",".join(samples_all), "ok" if rc == 0 else "fail")
    if rc != 0:
        return 1
    med_fail = medium_dir / "00_sensitivity_medium_failures.csv"
    med_fails = read_failure_table(med_fail)
    if len(med_fails) > 0:
        print(f"[STOP] medium failures: {len(med_fails)} rows in {med_fail}")
        return 1

    # Verify bound1000 copied from this-batch formal
    bound1000_check: dict[str, Any] = {}
    for sample in samples_all:
        src = formal_dir / f"{sample}-ctx-flux.csv"
        dst = medium_dir / "bound1000" / f"{sample}-bound1000-ctx-flux.csv"
        ok = nonempty(src) and nonempty(dst) and sha256_file(src) == sha256_file(dst)
        bound1000_check[sample] = {
            "match": ok,
            "formal_sha256": sha256_file(src) if nonempty(src) else None,
            "bound1000_sha256": sha256_file(dst) if nonempty(dst) else None,
        }
    write_json(run_root / "compare" / "bound1000_vs_formal.json", bound1000_check)
    if not all(v["match"] for v in bound1000_check.values()):
        print("[STOP] bound1000 does not match same-batch formal ctx flux")
        return 1

    # --- Stage E: metrics ---
    if not args.skip_metrics:
        print("\n=== Metrics (EAI / ARB / AC) ===")
        batch_dirs = [medium_dir, tradeoff_dir]
        for metric, script in (("eai", EAI), ("arb", ARB), ("ac", AC)):
            metric_dir = analysis_root / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            single_dirs: list[Path] = [formal_dir]
            if metric in {"eai", "arb"}:
                half_dir = tradeoff_dir / "tradeoff0.5"
                single_dirs.append(half_dir)
            cmd: list[str | Path] = [
                python_bin,
                script,
                "--single-dir",
                *single_dirs,
                "--batch-dir",
                *batch_dirs,
                "--sample",
                *samples_all,
                "--output-dir",
                metric_dir,
            ]
            if metric == "arb":
                cmd.extend(["--ge-file", ge_file])
            if metric == "ac":
                cmd.extend(
                    [
                        "--context-cmm-dir",
                        context_dir,
                        "--ge-file",
                        ge_file,
                        "--guild-file",
                        guild_file,
                        "--heatmap-guilds",
                        *CASE_STUDY["ac_heatmap_guilds"],
                    ]
                )
            rc = run_cmd(cmd, log_path=log_path, label=f"metrics/{metric}")
            log_status("metrics", metric, "ok" if rc == 0 else "fail")
            if rc != 0:
                return 1

    summary = {
        "status": "success",
        "run_root": repo_relative(run_root),
        "analysis_root": repo_relative(analysis_root),
        "samples": samples_all,
        "consistency_gates": "see compare/*_formal_vs_tradeoff1.json",
        "message": "Batch complete; incomplete batches must not be reported as success.",
    }
    write_json(run_root / "00_SUMMARY.json", summary)
    print("\nPipeline complete successfully.")
    print(f"Results: {run_root}")
    print(f"Analysis: {analysis_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
