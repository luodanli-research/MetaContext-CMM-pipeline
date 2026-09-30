# Pre-solve ctx pickle — acceptance checklist

Status after code change: **代码修改完成，数值验证待执行**.

Do not claim formal/tradeoff consistency is solved until the stages below pass.
Do not overwrite `case_study/outputs/`. Bootstrap an isolated run root first:

```bash
python pipeline/bootstrap_presolve_run_root.py --case-study-root case_study
```

## Tolerance formula

For error \(e = |x - x_{\mathrm{ref}}|\) and \(r = |x_{\mathrm{ref}}|\):

`pass = (e ≤ atol) OR (r ≥ r_min AND e ≤ rtol · r)`

Near-zero references (`r < r_min`) use the absolute branch only. Never auto-loosen
thresholds on failure.

| Quantity | atol | rtol | r_min |
|---|---|---|---|
| community `growth_rate` | 1e-9 | 1e-8 | 1e-12 |
| member Growth | 1e-8 | 1e-8 | 1e-12 |
| EX_ fluxes | 1e-6 | 1e-5 | 1e-6 |
| EAIshape_* | 1e-6 | (absolute only) | — |

EAI must use the **same-batch** formal bsl. Record ε / budget_alpha (`1e-7`).

**Consistency gate:** formal ctx vs tradeoff1 ctx (and S000 vs formal ctx) are hard
gates. Formal bsl vs tradeoff1 bsl is reported separately; **bsl mismatch ⇒ overall
consistency must not be marked passed**.

## Stages (top-level solves; internal retries logged separately)

| Stage | Content | Top-level solves |
|---|---|---|
| 0 | Control-flow / cache: pre-solve save, no post-solve overwrite, per-fraction load, copied bounds, no legacy REUSE | 0 |
| 1 | SW46 formal bsl+ctx; tradeoff1 bsl+ctx | 4 |
| 2 | SW51/SW55/SW61 same as stage 1 (**stop if SW46 fails**) | 12 |
| 3 | SW46 S000 vs formal ctx; bound1000 copy-source check (0 solves) | 1 |
| **Total** | | **17** |

Stage 1 example (after bootstrap; adjust stamp):

```bash
RUN=case_study/runs/presolve_unified_YYYYMMDD
ANA=case_study/runs/presolve_unified_YYYYMMDD_analysis
python pipeline/run_case_study.py \
  --output-root "$RUN" --analysis-dir "$ANA" \
  --sample SW46 --metrics \
  --sensitivity tradeoff --tradeoffs 1.0
```

Then compare `$RUN/03_simulation/context/SW46-*-flux.csv` to
`$RUN/03_simulation/sensitivity_tradeoff/tradeoff1/SW46-tradeoff1-*-flux.csv`.
