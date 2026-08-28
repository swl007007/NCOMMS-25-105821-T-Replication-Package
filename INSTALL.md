# Installation and Execution Guide

This guide describes the supported fresh-clone workflow for the food-crisis
monitoring and prediction replication package.

## System requirements

- Python 3.9 or newer
- Windows, macOS, or Linux
- At least 8 GB RAM; 16 GB is recommended for cross-validation and SHAP
- At least 5 GB free disk space
- Graphviz system executables only for `Figure3_Flowchart.ipynb`

## 1. Clone the private repository

GitHub authentication is required while the repository remains private.

```bash
git clone https://github.com/swl007007/NCOMMS-25-105821-T-Replication-Package.git
cd NCOMMS-25-105821-T-Replication-Package
```

Alternatively, download and extract the GitHub release source archive, then
open a terminal in the extracted package root. The root is the directory that
contains `requirements.txt`, `1.Source Data/`, and `2.Source Code/`.

## 2. Create and activate an environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the compatibility-constrained dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the main imports:

```bash
python -c "import pandas, numpy, xgboost, sklearn, shap, matplotlib, seaborn, geopandas, statsmodels.api as sm; from statsmodels.miscmodels.ordinal_model import OrderedModel; assert sm.OLS is not None and OrderedModel is not None; print('Python environment ready')"
```

`requirements.txt` provides a supported compatibility envelope. Exact package
versions used by controlled generators are retained in their corresponding
source-audit CSV files because XGBoost and numerical-library changes can alter
predictions.

## 3. Optional Graphviz setup

Only `Figure3_Flowchart.ipynb` requires the Graphviz `dot` executable.

Windows: install Graphviz from <https://graphviz.org/download/> and add it to
`PATH`.

macOS:

```bash
brew install graphviz
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install graphviz
```

Verify the installation:

```bash
dot -V
```

## 4. Restore the release-only lineage asset

The 75.8 MiB spatial-interpolation lineage artifact is attached to GitHub
release `v1.2.0` instead of being stored in ordinary Git history. It is not
required for the core model rerun, but it is required to inspect or independently
verify the complete spatial-interpolation lineage.

With GitHub CLI:

```bash
gh release download v1.2.0 \
  --pattern spatial_feature_interpolation_audit.csv.gz \
  --dir "2.Source Code/produced_graph"
sha256sum --check RELEASE_ASSET_SHA256.txt
```

Expected SHA-256:

```text
c6769b453538c19be0745e66772ad7c794960eb2c77e96e914d35a422f8e39b4
```

Windows users without `sha256sum` can run:

```powershell
certutil -hashfile "2.Source Code\produced_graph\spatial_feature_interpolation_audit.csv.gz" SHA256
```

## 5. Check the package without fitting models

From the repository root:

```bash
python run_replication.py --check-only
```

The check validates required inputs and generators, parses all 12 notebooks,
rejects executable absolute paths, confirms that notebook outputs are cleared,
and verifies the release asset checksum when the asset is present.

## 6. Run the controlled core workflow

```bash
python run_replication.py
```

This runs the following generators in dependency order with one formal outer
worker where applicable:

1. `generate_all_prediction_temporal_test.py`
2. `generate_all_prediction_evaluation.py`
3. `generate_all_prediction_temporal_test_evaluation.py`
4. `generate_phase_cumulative_scatter_comparison.py`

The first generator rebuilds the canonical 1,170-row temporal-test prediction
artifact. The second creates the shared random-five-fold Contemporaneous OOF
sidecar before the scatter generator consumes it. Forecasting and Nowcasting
use the 2022 temporal holdout; Contemporaneous uses seed-0 random five-fold
row-level cross-validation. Their metric levels are not directly comparable.

To run a generator individually from the package root, use the same Python
environment, for example:

```bash
python "2.Source Code/generate_spatial_feature_comparison.py" --workers 1
python "2.Source Code/generate_simple_baseline_comparison.py"
```

Long robustness workflows and their restart options are documented in
`REPLICATION_PACKAGE.md`.

## 7. Run the notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Open `2.Source Code/` in JupyterLab and run notebooks with that directory as the
working directory. Inputs use `../1.Source Data/...`, parameter files use paths
relative to `2.Source Code/`, and generated tabular artifacts are directed to
`2.Source Code/produced_graph/`. No machine-specific path editing is required.

Recommended order:

1. Main results:
   - `Table1_Forecasting_main.ipynb`
   - `Table1_Contemporaneous_main.ipynb`
   - `Table1_Nowcasting_two_layer.ipynb`
2. Phase-change analyses:
   - `Table1_Forecast_phasechange.ipynb`
   - `Table1_Nowcast_phasechange.ipynb`
3. Feature importance:
   - `Figure2_Feature_Importance_Forecasting.ipynb`
   - `Figure2_Nowcasting_two_layer_feature_importance.ipynb`
4. Standalone figures and descriptives:
   - `Figure1_multiple_figures.ipynb`
   - `Figure3_Flowchart.ipynb`
   - `Figure4_Descriptives.ipynb`
   - `figuresS1_hyperparameter.ipynb`
5. Conflict simulation:
   - `Conflit_Simulation.ipynb`

`Figure4_Descriptives.ipynb` uses only released files:
`Nowcasting_Analysis_010825.csv` and `area_country_lookup.csv`.

## Data files

Required model-ready inputs in `1.Source Data/`:

- `Forecasting_Analysis_010825.csv`
- `Nowcasting_Analysis_010825.csv`
- `area_country_lookup.csv`

The package also retains the canonical `All_prediction.csv`, reference metric
tables, and four optional preassigned-fold CSV files. The fold files support
custom cross-validation diagnostics but are not required by every notebook.

## Validation commands

Fast structural checks:

```bash
python run_replication.py --check-only
python -m py_compile run_replication.py "2.Source Code"/*.py tests/*.py
python -m unittest discover -s tests -p "test_run_replication.py"
```

The repository also retains model- and artifact-specific regression tests. Some
of those tests intentionally enforce exact historical Windows environments or
older frozen artifact contracts, so they are not the universal fresh-clone
release gate. Run the complete historical suite only in the matching audited
environment:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The complete model workflow is computationally heavier than these checks and
should be run in the intended audited environment before claiming numerical
reproduction.

## Troubleshooting

`ModuleNotFoundError`: activate the intended virtual environment and rerun
`python -m pip install -r requirements.txt`.

`Graphviz ExecutableNotFound`: verify `dot -V`, then restart the terminal or
Jupyter process after updating `PATH`.

`FileNotFoundError`: confirm the notebook is running from `2.Source Code/`, or
run the root-level generators from the package root.

Slow SHAP or cross-validation: this is expected. Close other applications and
retain the documented worker settings for formal outputs.

## Citation and license

Citation information and a license are pending author confirmation. The GitHub
repository must remain private until those metadata and third-party data
redistribution rights are resolved.

**Last updated:** August 2026
