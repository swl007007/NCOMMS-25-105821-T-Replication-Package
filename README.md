# Food-Crisis Monitoring and Prediction Replication Package

Source data, analysis notebooks, Python generators, and saved results for
the food-crisis monitoring and prediction study.

- `1.Source Data/`: model-ready inputs.
- `2.Source Code/`: notebooks, generators, and model parameters.
- `2.Source Code/produced_graph/`: saved figures, tables, and predictions.
- `0.Archived/`: historical outputs.

## Setup

For the main analyses, use Windows Python 3.11.3 and the package versions in
`2.Source Code/main_result_figure1_v1_environment_lock.txt`.

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -r requirements.txt -c "2.Source Code/main_result_figure1_v1_environment_lock.txt"
python run_replication.py --check-only
```

## Reproduce results

Start `jupyter lab`, select the installed environment, and run notebooks
from top to bottom with `2.Source Code/` as the working directory.
The main Forecasting and two-layer Nowcasting sources are
`Table1_Forecasting_main.ipynb` and
`Figure2_Nowcasting_two_layer_feature_importance.ipynb`.

For Figure 1 contemporaneous row-level CV, run from the repository root:

```bash
python "2.Source Code/generate_contemporaneous_main.py"
```

For the selected contemporaneous result (using the Nowcasting dataset,
random area five-fold CV), run from the repository root:

```bash
python "2.Source Code/generate_leave_area_out_20pct_random_cv.py"
```

Use the `Nowcasting` row in
`produced_graph/leave_area_out_20pct_random_cv_metrics.csv`.
For LOCO, use the same environment and run:

```bash
python "2.Source Code/generate_strict_temporal_leave_one_country_out_robustness.py" --workers 4
python "2.Source Code/generate_leave_one_country_out_robustness.py" --workers 4
```

The `Nowcasting` row in `leave_one_country_out_micro_metrics.csv` is the
contemporaneous LOCO result using the Nowcasting dataset.

Figure notebooks and additional result generators are in `2.Source Code/`;
Graphviz (`dot -V`) is required for the flowchart notebook.

See [INSTALL.md](INSTALL.md) for execution order and the core pipeline,
and [REPLICATION_PACKAGE.md](REPLICATION_PACKAGE.md) for individual analyses.

## Spatial audit asset

If absent, restore the release asset before auditing spatial interpolation:

```bash
gh release download v1.2.0 --pattern spatial_feature_interpolation_audit.csv.gz --dir "2.Source Code/produced_graph"
sha256sum --check RELEASE_ASSET_SHA256.txt
```
