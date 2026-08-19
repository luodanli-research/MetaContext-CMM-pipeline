# MetaContext-CMM

**MetaContext-CMM** is a meta-omics contextualised community metabolic modelling workflow that builds context-specific community-scale metabolic models (CMMs) by integrating MAG-derived network topology, metatranscriptome-informed taxon scaling and reaction constraints, and geochemistry-based environmental constraints.

This repository contains the reproducible MetaContext-CMM implementation for hot-spring microbial communities. Reusable code lives under `workflow/`. The shipped `example/` tree is an example community for learning and testing.

The workflow builds paired baseline and context-specific CMMs for each sample, computes flux-derived metrics—the Elemental Allocation Index (EAI), Anabolic Reallocation Bias (ARB), and Association Coefficient (AC)—to characterise community metabolic states, and runs sensitivity analyses over medium bounds, cooperative trade-off settings, and reaction constraints to test the robustness of model-derived conclusions.

## Installation

From the project root:

```bash
conda env create -f workflow/config/environment.yml
conda activate metacontext
```

Verify the installation:

```bash
python -c "import cobra, cplex, micom, riptide; print('OK')"
```

CPLEX is distributed by IBM and may require a separate license if the PyPI
package cannot be used on the target system.

## Workflow

![MetaContext-CMM workflow](https://github.com/user-attachments/assets/777e4e98-b2ac-406d-a7af-98530b7dbabb)

## Inputs

Each input type below has a corresponding format example under `example/inputs/`.

**Genome-scale metabolic model (GEM) of MAGs:**

```text
01_gems/
├── MAG001.xml
├── MAG002.xml
└── ...
```
Each file should contain one GEM for one MAG in SBML format.

**Taxon-level scaling:**

`01_taxon_ge.csv`: housekeeping-gene-based RNA abundance (TPM), used for taxon scaling in baseline and context CMMs.

`01_taxon_ga.csv`: housekeeping-gene-based DNA abundance (TPM), used only for DNA–RNA concordance quality control.

**Gene expression:**

`02_gene_expression.csv`: gene-level expression profiles for each MAG.

**Medium:**

`03_medium.csv`: medium composition for exchange-reaction constraints, based on measured geochemistry or user-defined settings.

**Sensitivity analysis:**

`04_medium_sensitivity_list.csv`: medium exchange reactions included in medium-bound sensitivity analysis.

`04_reaction_sensitivity_list.csv`: reaction families included in reaction-constraint sensitivity analysis.

**Community guild:**

`04_taxon_guild.csv`: community guild assignment used for Association Coefficient (AC) calculation.

## Run Pipeline

**Example pipeline:**

Three-MAG community (`example/inputs`) for a quick workstation run:

```bash
python pipeline/run_example.py
```
Output layout (under `{example|case_study}/outputs/`; shared by both launchers):

| Stage | Item | Notes |
|---|---|---|
| `01_baseline/` | `<sample>-bsl.pickle` | baseline community model |
| `01_baseline/` | `<sample>-manifest.tsv` | taxon manifest |
| `02_context/` | `context_bounds.csv` | RIPTiDe reaction bounds |
| `02_context/` | `context_species.csv` | QC eligibility table |
| `02_context/` | `context_expression.csv` | QC-filtered expression |
| `02_context/` | `context_summary.csv` | QC funnel summary |
| `02_context/` | `<sample>-ctx.pickle` | context community for sensitivity reuse |
| `03_simulation/context/` | `<sample>-{bsl,ctx}-flux.csv` | formal (single) flux dataset |
| `03_simulation/sensitivity_medium/bound<value>/` | `<sample>-bound<value>-{bsl,ctx}-flux.csv` | one folder per medium bound |
| `03_simulation/sensitivity_tradeoff/tradeoff<value>/` | `<sample>-tradeoff<value>-{bsl,ctx}-flux.csv` | one folder per tradeoff fraction |
| `03_simulation/sensitivity_reaction/01_lhs/<sample>/` | `LHS_sample_S*.csv` | generated Latin Hypercube Sampling (LHS) intervals |
| `03_simulation/sensitivity_reaction/S*/` | `<sample>-reaction-S*-ctx-flux.csv` | one folder per realization |
| `04_analysis/eai/<dataset>/` | EAI CSVs and radar PDFs | one subfolder per analyzed flux dataset (`context` or a sensitivity batch member) |
| `04_analysis/arb/<dataset>/` | ARB CSVs and bar PDFs | anabolic/catabolic/detox routing summaries per flux dataset |
| `04_analysis/ac/<dataset>/` | AC CSVs, networks, and heatmap PDFs | guild interaction scores, networks, and co-consumption heatmaps per flux dataset |

**Case-study pipeline:**

Full case-study data and outputs for the hot-spring system are distributed separately via [Zenodo DOI 10.5281/zenodo.20132606](https://doi.org/10.5281/zenodo.20132606).

After download, run `python pipeline/run_case_study.py` from the repo root. Output paths and file formats match the example layout above.

```bash
python pipeline/run_case_study.py
```

**Key options:**

The flags below apply to both `run_example.py` and `run_case_study.py`; only preset defaults differ (samples, sensitivity grids, and AC heatmap guilds). For the full flag set and defaults, run `python pipeline/run_example.py --help` or `python pipeline/run_case_study.py --help`.

| Option | Role |
|---|---|
| `--sample` | Subset of preset samples to run |
| `--sensitivity` | `medium` / `tradeoff` / `reaction` (default: all three) |
| `--metrics` | `eai` / `arb` / `ac` (default: `eai arb ac`) |
| `--medium-bounds` | Medium sensitivity uptake bounds (preset differs by launcher) |
| `--tradeoffs` | Tradeoff fractions (preset differs by launcher) |
| `--reaction-realizations` | Latin Hypercube Sampling (LHS) realization count for reaction sensitivity (preset differs by launcher) |
| `--force` | Rebuild baselines, re-infer bounds, rerun simulations and metrics |
| `--check-only` | Validate inputs and print the plan without solving |

Completed stages are reused unless `--force` is set. Infeasible formal or
sensitivity simulations may be skipped after recording a failure; baseline,
RIPTiDe, and metric stages stop the pipeline on error.

## Run Individual Modules

Every module is also a standalone CLI under `workflow/`; you can run modules individually or assemble your own pipeline. Run all commands from the repo root. In the **In** / **Out** tables below, bare filenames and stage folders are shorthands for `{example|case_study}/inputs/…` and `{example|case_study}/outputs/…`, respectively. For flags, defaults, and command-line examples, run `python <script> --help` on any module or pipeline launcher.

**1. `workflow/simulation/build_community_model.py`**

Build a baseline MICOM community from taxon abundances.

| | Item | Notes |
|---|---|---|
| In | `01_gems/<mag_id>.xml` | one SBML GEM per MAG |
| In | `01_taxon_ge.csv` | default abundance source; or `--abundance-file` |
| In | `--sample` | sample ID |
| Out | `01_baseline/<sample>-bsl.pickle` | baseline community model |
| Out | `01_baseline/<sample>-manifest.tsv` | taxon manifest |

**2. `workflow/simulation/infer_context_bounds.py`**

Infer sample-specific RIPTiDe reaction bounds from gene expression after discordance and MICOM abundance QC.

| | Item | Notes |
|---|---|---|
| In | `01_gems/<mag_id>.xml` | one SBML GEM per MAG |
| In | `02_gene_expression.csv` | long-format expression |
| In | `01_taxon_ge.csv`, `01_taxon_ga.csv` | discordance QC inputs |
| In | `--sample` | sample ID |
| Out | `02_context/context_bounds.csv` | RIPTiDe reaction bounds |
| Out | `02_context/context_species.csv` | QC eligibility table |
| Out | `02_context/context_expression.csv` | QC-filtered expression |
| Out | `02_context/context_summary.csv` | QC funnel summary |

**3. `workflow/simulation/simulate_community_model.py`**

Run baseline (`bsl`) and context (`ctx`) cooperative-tradeoff simulations.

| | Item | Notes |
|---|---|---|
| In | `01_baseline/<sample>-bsl.pickle` | baseline community |
| In | `02_context/context_bounds.csv` | RIPTiDe bounds |
| In | `03_medium.csv` | exchange constraints |
| In | `--sample` | sample ID |
| Out | `03_simulation/context/<sample>-{bsl,ctx}-flux.csv` | formal flux tables |
| Out | `02_context/<sample>-ctx.pickle` | ctx community for sensitivity reuse |

**4. `workflow/simulation/sensitivity_medium.py`**

Medium-bound exchange sensitivity: compress listed uptakes on the ctx community per scenario bound.

| | Item | Notes |
|---|---|---|
| In | `02_context/<sample>-ctx.pickle` | context community |
| In | `03_medium.csv` | exchange constraints |
| In | `04_medium_sensitivity_list.csv` | exchanges to perturb |
| In | `<sample>-ctx-flux.csv` | formal flux; reused for `bound1000` |
| In | `--medium-bounds`, `--sample` | scenario bounds and sample ID |
| Out | `03_simulation/sensitivity_medium/bound<value>/<sample>-bound<value>-{bsl,ctx}-flux.csv` | one folder per bound |

**5. `workflow/simulation/sensitivity_tradeoff.py`**

Cooperative-tradeoff fraction sensitivity (solver parameter sweep per sample).

| | Item | Notes |
|---|---|---|
| In | `01_baseline/<sample>-bsl.pickle` | baseline community |
| In | `02_context/context_bounds.csv` | RIPTiDe bounds |
| In | `03_medium.csv` | exchange constraints |
| In | `--tradeoffs`, `--sample` | fraction grid and sample ID |
| Out | `03_simulation/sensitivity_tradeoff/tradeoff<value>/<sample>-tradeoff<value>-{bsl,ctx}-flux.csv` | one folder per fraction |

**6. `workflow/simulation/sensitivity_reaction.py`**

RIPTiDe-bound Latin Hypercube Sampling (LHS) reaction-constraint sensitivity (one solve per realization).

| | Item | Notes |
|---|---|---|
| In | `02_context/<sample>-ctx.pickle` | context community |
| In | `02_context/context_bounds.csv` | interval source |
| In | `04_reaction_sensitivity_list.csv` | reaction families to perturb |
| In | `--reaction-realizations`, `--sample` | LHS realization count and sample ID |
| Out | `03_simulation/sensitivity_reaction/01_lhs/<sample>/LHS_sample_S*.csv` | generated LHS interval tables |
| Out | `03_simulation/sensitivity_reaction/S*/<sample>-reaction-S*-ctx-flux.csv` | one folder per realization |

**7. `workflow/analysis/metrics_eai.py`**

Elemental Allocation Index (EAI) and radar plots.

| | Item | Notes |
|---|---|---|
| In | `03_simulation/context/` | `--single-dir`; formal fluxes |
| In | `03_simulation/sensitivity_*/` | `--batch-dir`; optional sensitivity batches |
| In | `--sample` | sample ID |
| Out | `04_analysis/eai/<dataset>/…` | CSVs and PDFs |

**8. `workflow/analysis/metrics_arb.py`**

Anabolic Reallocation Bias (ARB) and bar plots.

| | Item | Notes |
|---|---|---|
| In | flux directories | `--single-dir` / `--batch-dir` |
| In | `01_taxon_ge.csv` | `--ge-file` |
| In | `--sample` | sample ID |
| Out | `04_analysis/arb/<dataset>/…` | CSVs and PDFs |

**9. `workflow/analysis/metrics_ac.py`**

Association Coefficient (AC), guild networks, and co-consumption / cross-feeding heatmaps.

| | Item | Notes |
|---|---|---|
| In | flux directories | `--single-dir` / `--batch-dir` |
| In | `02_context/<sample>-ctx.pickle` | `--context-cmm-dir` |
| In | `01_taxon_ge.csv`, `04_taxon_guild.csv` | scaling and guild map |
| In | `--sample` | sample ID |
| Out | `04_analysis/ac/<dataset>/…` | CSVs and PDFs |
| Out | `*-ctx-interactions.{metacontext,micom}.csv` | interaction cache beside flux tables |

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
