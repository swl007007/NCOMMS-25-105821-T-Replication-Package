# Food-Crisis Monitoring and Prediction Replication Package

This repository contains the source data, notebooks, controlled Python
generators, configuration files, and audited artifacts used to reproduce the
food-crisis monitoring and prediction analyses prepared for a Nature
Communications submission.

The repository is currently a private replication release. Citation metadata,
a project license, and confirmation of third-party data redistribution rights
remain author actions before any public release.

## Quick start

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/swl007007/NCOMMS-25-105821-T-Replication-Package.git
cd NCOMMS-25-105821-T-Replication-Package
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the non-model readiness checks:

```bash
python run_replication.py --check-only
```

Run the controlled core workflow from the repository root:

```bash
python run_replication.py
```

The runner executes, in dependency order:

1. canonical 1,170-row temporal-test prediction generation;
2. the shared Forecasting, Nowcasting, and random-five-fold Contemporaneous
   evaluation artifacts;
3. the temporal-test-prefixed evaluation family; and
4. the cumulative-phase Forecasting/Nowcasting/Contemporaneous scatter family.

The model steps can take substantially longer than the readiness check. Exact
predictions may drift across XGBoost and numerical-library versions; each
controlled generator records environment and source lineage in its audit
artifacts.

## Notebook workflow

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Open notebooks under `2.Source Code/` and use that directory as the notebook
working directory. The notebooks use paths relative to that directory and have
been cleared of saved execution output. `Figure3_Flowchart.ipynb` additionally
requires the system Graphviz executable (`dot -V`).

There are 11 active analysis notebooks. See `INSTALL.md` for their recommended order
and `REPLICATION_PACKAGE.md` for the detailed evaluation and artifact contracts.

## Evaluation boundary

### Current manuscript and Response Table R1

The authoritative Response letter is `writing/Response_2_v4_090826.docx`.
Edit that DOCX first, then regenerate its PDF and `writing/response_numbered.txt`.
The PDF and numbered text are derived reading aids, not independent authorities;
they must not overwrite newer DOCX content.

The manuscript's Forecasting/Nowcasting main results use the frozen
`main-result-figure1-v1` predictions, also used by the base panels of
Supplementary Fig. 4. Table R1 is generated without model fitting by
`2.Source Code/generate_response_table_r1_metrics.py`; its three-row output is
`2.Source Code/produced_graph/response_table_r1_macro_metrics.csv`, with explicit
prediction paths, hashes, label columns and averaging rules. Contemporaneous
in the existing R1 generator still uses the earlier saved seed-0 row-level
full-OOF predictions. The selected contemporaneous result was updated on
15 September 2026 as documented below; that generator and existing figures
have not yet been migrated to the newly selected source.

### Selected contemporaneous main result — random area CV (15 September 2026)

The `Nowcasting` row in
[`leave_area_out_20pct_random_cv_metrics.csv`](2.Source%20Code/produced_graph/leave_area_out_20pct_random_cv_metrics.csv)
is selected as **Contemporaneous (using nowcasting dataset; random area
five-fold CV)**. Its complete six-CSV family has been restored to
`2.Source Code/produced_graph/`. The original generator is
[`generate_leave_area_out_20pct_random_cv.py`](2.Source%20Code/generate_leave_area_out_20pct_random_cv.py).
See the [result provenance and metric definitions](2.Source%20Code/produced_graph/leave_area_out_20pct_random_cv_README.md).

This uses all dates, 5,575 observations and 1,198 areas, with areas randomly
assigned to five folds (seed 0). Results are fold means and sample SDs,
not pooled metrics or a temporal holdout. The cascading model uses Forecasting
inputs in layer 1 and the Nowcasting dataset in layer 2. Original CSV model
labels and audit hashes are preserved; `Nowcasting` is the selected row's
source label, while `Contemporaneous` is its manuscript reporting label.

`1.Source Data/All_prediction.csv` is an alternative Forecasting/Nowcasting
prediction lineage, **not the source for the current manuscript main results
or Table R1**. It remains in place for map/population and core-workflow inputs.
The two old `all_prediction*_macro_metrics` presentation families and uncited
calibration outputs are now archived. Running the core workflow can recreate
those alternative outputs; it does not reproduce the frozen manuscript
lineage, and its metrics must not be substituted.

### Archived presentation outputs (8 September 2026)

The current TeX manuscript and Response letter were inventoried before moving
obsolete or unselected figures/tables to
[`0.Archived/2026-09-08_obsolete_presentation/`](0.Archived/2026-09-08_obsolete_presentation/README.md).
The archive contains original relative paths, reasons, SHA-256 hashes and
restoration instructions. Historical output paths elsewhere in the package
may now resolve there; retained generators can recreate unselected outputs.
`Conflit_Simulation.ipynb` is archived, not an active replication step.
The corrected `generate_conflict_perturbation.py` and
`produced_graph/conflict_perturbation_10pct/` remain exploratory, at the
author's request, and are not manuscript evidence. Current writing files,
table sources and complete hash-bound dependency families remain in place.

Forecasting and cascading two-layer Nowcasting use the complete 1,170-row 2022
temporal holdout. The earlier contemporaneous artifacts use reproducible seed-0 random
five-fold row-level cross-validation over 5,575 observations. These protocols
and populations differ from the selected random-area-CV result above and
must not be silently interchanged.

## Release asset

The following formal lineage artifact is distributed with GitHub release
`v1.2.0` rather than stored in ordinary Git history:

`spatial_feature_interpolation_audit.csv.gz`

Restore it to `2.Source Code/produced_graph/` with GitHub CLI:

```bash
gh release download v1.2.0 \
  --pattern spatial_feature_interpolation_audit.csv.gz \
  --dir "2.Source Code/produced_graph"
sha256sum --check RELEASE_ASSET_SHA256.txt
```

The checksum manifest records the required restoration path. The other formal
spatial-comparison outputs remain in the repository.

## Repository layout

Result-to-generator coverage, including notebook SHAP export locations and
the author-confirmed retirement of historical outputs, is documented in
[RESULT_SCRIPT_AUDIT.md](RESULT_SCRIPT_AUDIT.md), with the per-file inventory
in [result_script_inventory.csv](result_script_inventory.csv). Thirteen obsolete
files without active readers were archived on 15 September 2026; the two
legacy R-squared anchor tables still consumed by current scripts remain in place.

- `1.Source Data/`: released model-ready inputs and supporting tables.
- `2.Source Code/`: notebooks, generators, shared functions, parameters, and
  `produced_graph/` artifacts.
- `tests/`: focused generator and artifact-contract checks.
- `run_replication.py`: dependency-ordered core workflow entry point.
- `INSTALL.md`: environment and notebook instructions.
- `REPLICATION_PACKAGE.md`: detailed methods, provenance, and validation
  contracts.

## Citation and license

Citation information and a license have not yet been assigned. Until those are
added, access to this private repository does not itself grant redistribution or
reuse rights.
