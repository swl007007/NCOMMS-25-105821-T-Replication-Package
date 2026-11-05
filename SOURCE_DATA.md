# Source Data

Source data for **Predicting Acute Food Insecurity with Machine Learning**.

`Source Data.zip` uses the journal's zipped-folder option: one CSV or text file
per main or supplementary figure/table, using the numbering in the final main
manuscript and separate Supplementary Information downloaded on 4 November 2026.
Files are UTF-8; CSVs include a byte-order mark for Excel. No models are refitted.

## File contents

| File | Contents |
| --- | --- |
| Figure 1.csv | Frozen summary metrics, 1,170 classification predictions and scatter pairs per temporal model, and 5,575 contemporaneous out-of-fold predictions. |
| Figure 2.csv | The two original analysis datasets, signed SHAP sums, plotted top-ten features and normalized group contributions for the final vector artwork. |
| Figure 3.txt | Classification rules for the conceptual workflow; no empirical source measurements. |
| Figure 4.csv | All 5,575 source IPC phase labels and dates; count by phase for panel A and by year/phase for panel B. |
| Table 1.csv | Main-model data from Figure 1 and the separately labelled reported changed-phase summary values. |
| Supplementary Figure 1.csv | Assessment-level phases and the saved observation-frequency histogram counts. |
| Supplementary Figure 2.csv | 646 mapped areas, coordinates, maximum predicted/observed shares, 70th-percentile cutoffs and alert flags. |
| Supplementary Figure 3.csv | Paired baseline/rescue predictions for the selected policies. |
| Supplementary Figure 4.csv | All 23,745 scatter points and nine panel summaries for cumulative IPC shares. |
| Supplementary Figure 5.csv | Nine binary confusion matrices (36 cells) and the corresponding saved predictions. |
| Supplementary Figure 6.csv | Four temporal tuning folds and the final holdout design. |
| Supplementary Table 1.csv | Selected benchmark and main-model summaries, with saved persistence predictions. |
| Supplementary Table 2.csv | Metrics and predictions for the three alternative spatial strategies. |
| Supplementary Table 3.csv | Strict-temporal and ordinary LOCO, temporal area holdout and random-area CV data, with protocols kept separate. |
| Supplementary Table 4.csv | Original variable dictionary, units, descriptions and data providers. |
| Supplementary Table 5.csv | Hyperparameter search grids and saved effective parameter sets. |
| Supplementary Table 6.csv | Missingness-indicator, predictor-removal and country-removal benchmark results. |

## Reading the files

CSV files with more than one data block use `_record_type` to separate
observations, predictions, metrics and design information. `_source_file` and
`_source_row` identify the original artifact and its 1-based CSV/worksheet row,
including the header. `_panel` maps supplementary-table rows to panels.
`_reported_model` gives the manuscript label when it differs from the original
artifact's model name. Original columns and stored numerical precision are
retained. Empty cells mean missing, undefined or not applicable to that block;
they are never imputed to zero. Population shares are fractions, not percentages.
Sensitivity is recall. R-squared columns are dimensionless; continuous-share R²
and binary-indicator R² are separate quantities.

Figure 1 temporal scatter sources have no observation keys and remain separate
blocks; their row order is not asserted to match the keyed classification data.
The frozen main results are used for Figure 1/Table 1. The supplementary alert
map instead follows its original `All_prediction.csv` input and maximum-per-area
selection; ties at the 70th percentile are retained.

Figure 2 contains the model inputs confirmed by the authors and the numerical
SHAP summaries exported by `2.Source Code/export_final_artwork_shap.py` from
the original forecasting and nowcasting notebooks. The export uses the frozen
Windows Python 3.11.3 / XGBoost 2.0.3 environment with SHAP 0.46.0. Individual
importance is the absolute value of the signed sum over 4,405 training rows;
group importance sums signed contributions over the original feature groups,
takes absolute values and normalizes across the included groups. These are not
mean absolute SHAP values. Nowcasting summaries describe layer 2. The original
inputs and derived contributions are distinguished by `_record_type`.
For Table 1, changed-phase values are transcribed reported summaries; historical
per-observation predictions for that subset were not saved. The full input
datasets are available in Figure 2.csv. Some benchmark/refit artifacts likewise
retain summaries rather than per-observation predictions. These exports do not
claim to recreate unretained historical intermediate values.

For Supplementary Figure 3, use `reconstructed_overall_phase` as truth and
`base_overall_phase_pred`/`final_overall_phase_pred` for baseline/rescue. Temporal
records include separate `frozen_base` and `direct_phase45_sqrt_balance` rows;
contemporaneous records use `direct_phase45_full_balance`, with both predictions
in each row. Combine predicted phases 4 and 5 for the rescue panels. Counts
and within-actual-class percentages can be calculated directly from these rows.
For Supplementary Figure 5, confusion cells retain the complete benchmark/OOF
population; gate-only diagnostics are excluded.

Supplementary Table 3 contemporaneous rows use the original `Nowcasting` model
label in ordinary LOCO and random-area CV. Temporal rows each retain 1,170
observations; contemporaneous rows each retain 5,575. Random-area CV summaries
are arithmetic means across five folds (the fold metrics are included); other
primary rows are pooled metrics. Supplementary Table 2 uses the saved pooled
metrics from its own temporal/row-level-CV protocols. Source variable names in
Supplementary Table 4 are the dataset dictionary names, which may differ from
the manuscript's abbreviated display labels.

## Rebuild and provenance

From the repository root, with pandas and openpyxl installed:

```bash
python "2.Source Code/build_source_data.py"
```

The builder checks populations, the main classification metrics, map counts,
the file inventory and the ZIP contents. `manifest.json` records input/output
SHA-256 hashes and block row counts. The CSVs retain existing results without
retraining or changing manuscript values.

The relevant figure legends should contain:
“Source data are provided as a Source Data file.”
