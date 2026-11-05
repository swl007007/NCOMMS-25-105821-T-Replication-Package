# Reproduction guide

## Saved-result exports

The README commands regenerate all ten complete figures, seven CSV tables and
the 17-file Source Data archive. Figure rendering uses `export_final_artwork.py`;
table rendering uses `export_final_tables.py`, which reuses `build_source_data.py`.
All commands below run from the package root; script names are in `2.Source Code/`.

| Output | Numeric source / generating script |
| --- | --- |
| Figure 1; Table 1, main rows | `main_result_figure1_v1.py`, `generate_filtered_main_result_metrics.py` (full-population calibration before repeated-area refit), `generate_contemporaneous_main.py` |
| Figure 2 | `export_final_artwork_shap.py`; original Forecasting and two-layer Nowcasting SHAP notebooks |
| Figure 3 | Conceptual diagram drawn by `export_final_artwork.py`; phase conversion in `food_crisis_functions.py` |
| Figure 4 | Released Nowcasting input; counted by `export_final_artwork.py` |
| SI Figure 1 | `generate_sample_imbalance_chart.py` |
| SI Figure 2 | Saved `All_prediction.csv`, bundled geometry; selection in `generate_2022_alert_map_panel.py` |
| SI Figures 3, 5 | `generate_direct_phase3_vs_phase45_rescue.py`, `generate_direct_phase3_vs_phase45_rescue_contemporaneous.py`; five-class checks in `generate_direct_phase3_vs_phase45_rescue_five_class_confusion.py` |
| SI Figure 4 | `generate_phase_cumulative_scatter_comparison.py` |
| SI Figure 6 | Conceptual temporal validation design drawn by `export_final_artwork.py` |
| SI Table 1 | `generate_persistence_baseline_comparison.py`, `generate_missingness_sensitivity.py`, `generate_filtered_main_result_metrics.py` |
| SI Table 2 | `generate_spatial_feature_comparison.py` |
| SI Table 3 | `generate_strict_temporal_leave_one_country_out_robustness.py`, `generate_leave_one_country_out_robustness.py`, `generate_leave_area_out_20pct_fivefold_robustness.py`, `generate_leave_area_out_20pct_random_cv.py` |
| SI Table 4 | `Dataset_Preparation.xlsx`, exported by `build_source_data.py` |
| SI Table 5 | Search grid in `build_source_data.py`; fitted hyperparameters in the JSON files |
| SI Table 6 | `generate_missingness_sensitivity.py` |

The main Figure 1 lineage and the supplementary map's `All_prediction.csv`
lineage remain separate. Temporal evaluation retains all 1,170 observations;
Contemporaneous row CV retains 5,575 out-of-fold observations. SI Table 3 uses
its separate country/area holdout protocols. Do not substitute predictions
between these analyses. Table-specific definitions are in `tables/README.md`.

## Optional model refitting

Use a separate **Windows Python 3.11.3** environment. The model scripts enforce
the frozen core versions and, where applicable, the XGBoost binary hash:

```text
python -m venv .venv-model
.venv-model\Scripts\activate
python -m pip install -r requirements-model.txt
```

The rendering environment is only for exporting saved numbers. Do not refit
models in it. Run optional refits in a second extracted copy of the release so
the reference results remain available for comparison. With the bundled
prerequisite outputs, the following entry points can be run individually:

```bash
python "2.Source Code/generate_contemporaneous_main.py"
python "2.Source Code/generate_filtered_main_result_metrics.py"
python "2.Source Code/export_final_artwork_shap.py"
python "2.Source Code/generate_persistence_baseline_comparison.py" --country-lookup "1.Source Data/area_country_lookup.csv"
python "2.Source Code/generate_simple_baseline_comparison.py"
python "2.Source Code/generate_missingness_sensitivity.py"
python "2.Source Code/generate_spatial_feature_comparison.py" --workers 1
python "2.Source Code/generate_strict_temporal_leave_one_country_out_robustness.py" --workers 4
python "2.Source Code/generate_leave_one_country_out_robustness.py" --workers 4
python "2.Source Code/generate_leave_area_out_20pct_fivefold_robustness.py"
python "2.Source Code/generate_leave_area_out_20pct_random_cv.py"
python "2.Source Code/generate_phase_cumulative_scatter_comparison.py" --workers 1
```

These are independent entry points, not a batch command: several depend on
hash-bound reference inputs that another command can rewrite. The 10% area
holdout generator is retained only as an imported helper for the 20% scripts;
its historical results are not included.

The rescue scripts require an absent or empty output directory:

```bash
python "2.Source Code/generate_direct_phase3_vs_phase45_rescue_contemporaneous.py" --output-dir rerun_rescue_contemporaneous
python "2.Source Code/refit_rescue_from_release.py" --check-only
python "2.Source Code/refit_rescue_from_release.py" --output-dir rerun_rescue_temporal
```

Their frozen predecessor bundles are included in `produced_graph/` because the
scripts verify every member and hash. Some auxiliary figures and model files
inside those bundles are required inputs to these checks. Final publication
figures are collected only in `writing/final_artwork/`.

The temporal rescue entry point accommodates the later addition of Contemporaneous
rows to the spatial results. It checks all three published spatial-file hashes
and proves that the 9,360 temporal prediction records, including their header,
are byte-identical to the original frozen input before rebinding the three
whole-file checks. It then calls the unchanged rescue implementation and its
remaining frozen guards. `--check-only` performs these checks without fitting.

Table 1's changed-phase summaries are retained as reported historical values;
the original routines are included as the two `Table1_*phasechange.ipynb`
notebooks, but their historical keyed predictions are unavailable. The SHAP
exporter executes selected cells of the original notebooks without executing
their unused notebook imports or historical output cells.

This release was checked by rebuilding saved-data figures, tables and Source
Data in an isolated extracted directory, checking numerical assertions and
PDF/SVG properties, and checking frozen-model input/environment gates. Model
training and all robustness experiments were not rerun for this packaging task.
