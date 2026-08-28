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

There are 12 analysis notebooks. See `INSTALL.md` for their recommended order
and `REPLICATION_PACKAGE.md` for the detailed evaluation and artifact contracts.

## Evaluation boundary

Forecasting and cascading two-layer Nowcasting use the complete 1,170-row 2022
temporal holdout. Contemporaneous results use reproducible seed-0 random
five-fold row-level cross-validation over 5,575 observations. These protocols
and populations are intentionally labeled and are not directly comparable.

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
