# Replication Package Documentation

**Prepared for Nature Communications Submission**
**Initial release**: January 2026
**Last updated**: August 2026

---

## Overview

This package provides reproducible controlled workflows, source data, configuration, and audited artifacts for the food-crisis monitoring and prediction analyses. Stored aggregate outputs from the original unpinned notebook environment are retained as provenance references and may drift under a modern controlled rerun.

---

## Package Contents

### 1. Environment Setup Files

- **`requirements.txt`**: Minimum Python package dependency constraints; exact versions used for audited generators are recorded in their source-audit files
- **`INSTALL.md`**: Comprehensive installation guide with step-by-step instructions for:
  - Virtual environment creation (`.venv`)
  - Package installation
  - System dependency setup (Graphviz)
  - Troubleshooting common issues

### 2. Documentation Files

- **`README.md`**: Fresh-clone entry point and core workflow summary
- **`REPLICATION_PACKAGE.md`** (this file): Replication package overview
- **`RELEASE_ASSET_SHA256.txt`**: Checksum and restoration path for the
  release-only spatial interpolation lineage artifact

### 3. Source Code (`2.Source Code/`)

All 12 Jupyter notebooks have been prepared for distribution with:
- **Markdown header cells** explaining purpose and methodology
- **Step-by-step workflow documentation** (2-3 high-level steps per notebook)
- **Portable relative input and output paths** anchored to `2.Source Code/`
- **Expected outputs** for verification
- **Execution dependencies** noting prerequisite notebooks
- **Cleared execution counts and saved outputs** so the repository contains
  source notebooks rather than author-machine kernel state

#### Notebook Categories:

**Main Analysis Notebooks** (Table 1):
- `Table1_Forecasting_main.ipynb` - 12-month ahead forecasting
- `Table1_Contemporaneous_main.ipynb` - Current month nowcasting
- `Table1_Nowcasting_two_layer.ipynb` - Two-layer ensemble approach

**Visualization Notebooks**:
- `Figure1_multiple_figures.ipynb` - Geographic and temporal distributions, comparative visualizations
- `Figure2_Feature_Importance_Forecasting.ipynb` - SHAP analysis for forecasting models
- `Figure2_Nowcasting_two_layer_feature_importance.ipynb` - SHAP analysis for two-layer nowcasting
- `Figure3_Flowchart.ipynb` - IPC classification flowchart (requires Graphviz)
- `Figure4_Descriptives.ipynb` - Descriptive statistics and data summaries
- `Conflit_Simulation.ipynb` - Conflict sensitivity analysis
- `figuresS1_hyperparameter.ipynb` - Supplementary hyperparameter tuning results

**Auxiliary Analysis Notebooks**:
- `Table1_Forecast_phasechange.ipynb` - Phase transition analysis for forecasting
- `Table1_Nowcast_phasechange.ipynb` - Phase transition analysis for nowcasting

**Robustness Generators**:
- `generate_leave_one_country_out_robustness.py` - Fixed-hyperparameter, pure-spatial leave-one-country-out evaluation for the forecasting ensemble and cascading two-layer nowcasting model
- `generate_strict_temporal_leave_one_country_out_robustness.py` - Combined country and 2022 temporal holdout with pooled-micro metrics
- `generate_leave_area_out_10pct_robustness.py` - Fixed 10% country-stratified joint area holdout for settings where primary labels are sparse or unavailable for selected areas
- `generate_phase_cumulative_scatter_comparison.py` - Reproducible legacy Phase 2+/4+/5 Forecasting/Nowcasting scatter figures plus the Phase 2+/3+/4+ by Forecasting/Nowcasting/Contemporaneous 3 x 3 grid
- `generate_spatial_feature_comparison.py` - Four-condition comparison of temporal Forecasting/Nowcasting and seed-0 random five-fold Contemporaneous models under latitude/longitude ablation, pure geodesic KNN-5 means, and inclusive 200 km spatial means
- `generate_simple_baseline_comparison.py` - Audited 2022 temporal comparison of Persistence, frozen Multinomial, direct Ordered Probit, and cumulative-architecture Ensemble OLS baselines, with a 2 x 2 Phase 3+ precision/recall figure and stored-original references

### 4. Data Files (`1.Source Data/`)

**Required Input Files**:
- `Forecasting_Analysis_010825.csv` - Pre-processed forecasting dataset with 12-month lagged features
- `Nowcasting_Analysis_010825.csv` - Pre-processed nowcasting dataset with concurrent features
- `area_country_lookup.csv` - Unique `area_id` to ISO3 country mapping used for country and stratified area holdouts; deterministically derived from `0.Archived/new_merge_0108_with_country_code.csv` using only `area_id` and `country_code_3`

**Optional Files** (for custom cross-validation analysis):
- `forecasting_df_with_folds.csv` - 10-fold CV assignments for forecasting
- `forecasting_df_with_folds_5.csv` - 5-fold CV assignments for forecasting
- `nowcasting_df_with_folds.csv` - 10-fold CV assignments for nowcasting
- `nowcasting_df_with_folds_5.csv` - 5-fold CV assignments for nowcasting

**Configuration Files** (`2.Source Code/`):
- `forecasting_hyperparameters.json` - XGBoost parameters used for Phase 2+
- `forecasting_hyperparameters_p3.json` - XGBoost parameters used for Phase 3+, Phase 4+, and Phase 5, matching the notebooks' effective loop carry-over
- `contemporaneous_hyperparameters.json` - Nowcasting XGBoost parameters
- `contemporaneous_hyperparameters_p3.json` - Nowcasting phase 3 parameters

**Custom Modules**:
- `food_crisis_functions.py` - Core utility functions

---

## Reproducibility Instructions

### Step 1: Environment Setup

```bash
# Navigate to the replication-package root containing requirements.txt,
# 1.Source Data/, and 2.Source Code/
cd /path/to/NCOMMS-25-105821-T-Replication-Package

# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For exact comparison with published generator artifacts, verify the package
# versions against the corresponding source-audit CSV before execution.

# Install Graphviz (system dependency)
# See INSTALL.md for platform-specific instructions
```

### Step 2: Working-Directory Contract

No machine-specific path editing is required. Open the notebooks from
`2.Source Code/` and use that directory as the notebook working directory.
Notebook inputs use `../1.Source Data/...`, configuration files are local to
`2.Source Code/`, and generated tabular artifacts use `produced_graph/`.

Root-level generator scripts resolve paths from their own file location and may
be run from the package root. Before fitting models, run:

```bash
python run_replication.py --check-only
```

### Step 3: Execution Order

#### For Main Results:

1. **Main Models** (independent execution):
   ```
   Table1_Forecasting_main.ipynb
   Table1_Contemporaneous_main.ipynb
   Table1_Nowcasting_two_layer.ipynb
   ```

2. **Phase Change Analysis** (independent, uses source data):
   ```
   Table1_Forecast_phasechange.ipynb
   Table1_Nowcast_phasechange.ipynb
   ```

3. **Leave-One-Country-Out Robustness Analysis** (independent, uses both main model tables):
   ```bash
   # Full 29-country run with bounded country-level multiprocessing and restart support
   python "2.Source Code/generate_leave_one_country_out_robustness.py" --workers 4 --resume

   # Bounded smoke test; a non-default output directory is mandatory for partial runs
   python "2.Source Code/generate_leave_one_country_out_robustness.py" \
     --countries TZA \
     --workers 1 \
     --output-dir /tmp/loco_tza_smoke

   # Rebuild only the two-row pooled-micro table from saved predictions
   python "2.Source Code/generate_leave_one_country_out_robustness.py" \
     --aggregate-existing
   ```

4. **10% Leave-Area-Out Robustness Analysis** (independent, uses both main model tables):
   ```bash
   # Fixed seed-0 sample of 120 areas; Forecasting and Nowcasting run in parallel
   python "2.Source Code/generate_leave_area_out_10pct_robustness.py" --workers 2
   ```

5. **Cumulative-Phase Scatter Comparison** (depends on the shared Contemporaneous OOF sidecar):
   ```bash
   # The evaluation generator creates the random-five-fold sidecar first.
   python "2.Source Code/generate_all_prediction_evaluation.py"
   python "2.Source Code/generate_phase_cumulative_scatter_comparison.py" --workers 1
   ```

6. **Spatial Feature Comparison** (independent, uses both main model tables):
   ```bash
   # Formal four-condition, three-model run; outer execution is serial
   python "2.Source Code/generate_spatial_feature_comparison.py" --workers 1

   # Bounded smoke run; use a fresh non-default directory
   SMOKE_DIR=$(mktemp -d /tmp/spatial_feature_smoke.XXXXXX)
   python "2.Source Code/generate_spatial_feature_comparison.py" \
     --conditions no_lat_lon \
     --workers 1 \
     --output-dir "$SMOKE_DIR"
   ```

   On Windows, replace `mktemp` with a new empty directory under the user's
   temporary directory and pass that path to `--output-dir`.

7. **Simple Baseline Comparison** (independent, uses both main model tables and the frozen baseline artifacts):
   ```bash
   python "2.Source Code/generate_simple_baseline_comparison.py"
   ```

   The formal contract uses the default 2022 cutoff, both tasks, all four
   methods, Ordered Probit `bfgs`, and `maxiter=1000`. Alternate task/method or
   optimizer selections are diagnostic only and cannot publish into the formal
   output namespace.

#### For Figures:

1. **Feature Importance** (standalone retraining from released source data):
   ```
   Figure2_Feature_Importance_Forecasting.ipynb
   Figure2_Nowcasting_two_layer_feature_importance.ipynb
   ```

2. **Other Visualizations**:
   ```
   Figure1_multiple_figures.ipynb (uses source data, standalone)
   Figure3_Flowchart.ipynb (standalone, requires Graphviz)
   Figure4_Descriptives.ipynb (uses source data, standalone)
   Conflit_Simulation.ipynb (requires trained models)
   figuresS1_hyperparameter.ipynb (standalone, hyperparameter analysis)
   ```

---

## Expected Computational Requirements

### Runtime Estimates

| Notebook Type | Typical Runtime | Notes |
|--------------|----------------|-------|
| Main models (Table1_*.ipynb) | 2-5 minutes | Depends on CPU cores |
| Feature importance (Figure2_*.ipynb) | 10-30 minutes | SHAP is CPU-intensive |
| Visualizations (Figure1, 3, 4) | 1-5 minutes | Fast execution |
| Conflict simulation (Conflit_Simulation.ipynb) | 5-15 minutes | Multiple model runs |
| Phase change analysis | 2-5 minutes | Similar to main models |
| Leave-one-country-out robustness | Hardware-dependent; approximately 348 XGBoost fits | Uses bounded country-level processes, `n_jobs=1` per XGBoost model, and restart checkpoints |
| 10% leave-area-out robustness | Hardware-dependent; 12 XGBoost fits | At most two model-level processes; each XGBoost uses `n_jobs=1` |
| Spatial feature comparison | Hardware-dependent; 128 XGBoost fits plus spatial interpolation | Four conditions; each has 4 Forecasting fits, 8 cascading-Nowcasting fits, and 20 Contemporaneous OOF fits. Formal outer execution is serial; XGBoost uses its environment-default thread count |
| Simple baseline comparison | Approximately 4-5 minutes | Two full Ordered Probit BFGS fits dominate runtime; Persistence and frozen Multinomial are adapters, and the eight OLS fits are comparatively fast |

### Memory Requirements

- **Minimum**: 8GB RAM
- **Recommended**: 16GB RAM (for large-scale CV and SHAP analysis)

### Storage

- Source data: ~100MB
- Model outputs: ~50MB
- Figures: ~20MB

---

## Verification Procedure

### Expected Performance Metrics

The original Table 1 notebooks store the following reference values:

| Model | Accuracy | Sensitivity | Precision | R² (Phase 3+) |
|-------|----------|-------------|-----------|---------------|
| **Forecasting** (12-mo ahead) | 0.65 | 0.94 | 0.78 | 0.25 |
| **Contemporaneous** (current month) | ~0.70 | ~0.91 | ~0.80 | ~0.64 |
| **Two-Layer Nowcast** | 0.67 | 0.92 | 0.80 | 0.28 |

**Note**: The Forecasting and Two-Layer Nowcast references use the 2022 temporal holdout. The stored Contemporaneous reference comes from the notebook's unseeded random five-fold workflow and is not directly comparable to the temporal results. Sensitivity and precision focus on crisis phases (IPC 3+). Because the original notebooks were executed in an unpinned Windows XGBoost environment, controlled reruns must report their actual metrics and environment. Differences around ±0.02 are useful calibration checks, but larger drift must be documented rather than corrected by filtering rows or altering predictions.

### Canonical 1,170-Row Prediction Artifact

`1.Source Data/All_prediction.csv` now uses the complete temporal test population: 1,170 unique `(area_id, date)` observations across 646 areas, including source-row indices `3374`, `3517`, `3534`, `3553`, and `3567`. The canonical key SHA-256 is `288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2`.

Regenerate it from the two source tables with:

```bash
python "2.Source Code/generate_all_prediction_temporal_test.py"
```

The formal run executes Forecasting and Nowcasting sequentially and preserves the XGBoost estimator's default thread setting, matching the notebook execution contract. In the verified Windows Python 3.12.10 / XGBoost 3.0.0 environment, the regenerated Forecasting accuracy and Phase 3+ R-squared are `0.666667` and `0.267910`; the regenerated Nowcasting values are `0.666667` and `0.275545`. The Nowcasting metrics reproduce the stored Table 1 result, while Forecasting remains within the package's approximately 0.02 calibration tolerance.

The macro evaluation leaves the frozen ten-column `All_prediction.csv` contract unchanged. Forecasting and Nowcasting continue to use its 1,170-row 2022 temporal holdout. Contemporaneous instead follows the effective implementation in `Table1_Contemporaneous_main.ipynb`: the 5,575 current-month rows are sorted, reproducibly shuffled with seed 0, split into five equal row-level folds, and assigned one out-of-fold prediction per row. The notebook's `kfolds` column remains an input predictor, and all four cumulative targets use `contemporaneous_hyperparameters.json`, matching the notebook's effective parameter overwrite. Predictions are rounded to two decimals before applying the 0.20 cumulative-share threshold. The sidecar and audit are `all_prediction_contemporaneous_random_cv_predictions.csv` and `all_prediction_contemporaneous_random_cv_source_audit.csv` under `2.Source Code/produced_graph/`.

Regenerate the shared sidecar and both evaluation artifact families with:

```bash
python "2.Source Code/generate_all_prediction_evaluation.py"
python "2.Source Code/generate_all_prediction_temporal_test_evaluation.py"
```

The reproducible Windows Python 3.11.3 / XGBoost 2.0.3 random-CV rerun has full-OOF macro precision `0.636860`, macro recall `0.328301`, and macro F1 `0.333608`; its Phase 3+ full-OOF R² is `0.641619`. The original notebook used an unrecorded shuffle seed and exported only its last 1,115-row fold, so this rerun is not claimed to reproduce the historical `0.602` last-fold value exactly. The `all_prediction_temporal_test_*` prefix is retained for artifact continuity, but every CSV and figure records task-specific protocol, population, and `n`; all three-model comparisons are explicitly descriptive and not directly comparable. The older `generate_all_prediction_1165_evaluation.py` filename remains only a compatibility entry point.

### Leave-One-Country-Out Robustness Contract

The LOCO analysis addresses a different scenario from the Table 1 temporal holdout: a held-out country has no IPC labels available for model fitting, while its predictor variables remain available. For each fold, all dates from one country are held out and all dates from the other countries are used for training.

Important interpretation boundaries:

- The protocol is `fixed_hyperparameter_loco`: it reuses the manuscript JSON hyperparameters and is not nested hyperparameter selection.
- Training uses all dates from non-held countries, so this is cross-country transfer rather than out-of-time or historical-point-in-time validation.
- The Layer 1 predictor `fews_ipc_ha` is retained. Results therefore assume FEWSNET information is available and do not represent countries where that predictor is also unavailable.
- Phase 2+ uses the general forecasting parameters; Phase 3+, 4+, and 5 use the Phase 3 parameter file, matching the notebooks' effective loop behavior.
- The two-layer nowcast retains Layer 1 in-sample residual learning for Layer 2.
- Rows whose four rounded cumulative predictions sum to a non-positive value are retained, classified as Phase 1 when no 0.20 threshold is met, and explicitly flagged.

The current inputs contain 5,575 unique `(area_id, date)` rows and 29 countries. The generator verifies these live contracts and does not use the counts as filtering rules. A complete run must produce exactly one forecasting and one nowcasting out-of-country prediction for every source row.

Per-country precision or recall is stored as missing when its denominator is zero. R-squared is missing for fewer than two observations or a constant actual Phase 3+ share. The figure includes only countries whose precision and recall are both defined; omitted countries remain in the metrics CSV with support counts and an explicit reason. No pooled, macro, or weighted summary point is added to the country scatter figure.

The primary two-row table pools all 5,575 out-of-country predictions for each
model, without first averaging by area or country. Accuracy is exact five-phase
accuracy; precision and recall use the pooled Phase 3+ confusion counts; R-squared
is calculated once from the pooled Phase 3+ shares after actual and predicted
shares are rounded to two decimals. The canonical primary table is
`leave_one_country_out_micro_metrics.csv`; its name now matches the pooled-micro
estimand.

| model | accuracy | precision | recall | R2(p3) |
|---|---:|---:|---:|---:|
| Nowcasting | 0.565740 | 0.666312 | 0.917864 | 0.118206 |
| Forecasting | 0.558386 | 0.661736 | 0.905544 | 0.102838 |

The canonical table SHA-256 is
`64dbae3ae190b2ef9887614f763e585ef3cd226ac556daf4e590e26503849ff9`.

Regenerate only this table from the saved full-LOCO predictions with:

```bash
python "2.Source Code/generate_leave_one_country_out_robustness.py" \
  --aggregate-existing
```

Generated outputs under `2.Source Code/produced_graph/`:

- `leave_one_country_out_country_metrics.csv`
- `leave_one_country_out_micro_metrics.csv`
- `leave_one_country_out_forecasting_predictions.csv`
- `leave_one_country_out_nowcasting_predictions.csv`
- `leave_one_country_out_source_audit.csv`
- `precision_recall_scatter_leave_one_country_out.jpg`
- `precision_recall_scatter_leave_one_country_out.png`
- `precision_recall_scatter_leave_one_country_out.pdf`

Restart checkpoints are stored under the hidden `.leave_one_country_out_checkpoints/` subdirectory. `--resume` reuses a country only when the input, lookup, parameter, script, seed, and country manifest hashes match exactly.

### Strict Temporal Leave-One-Country-Out Robustness

The additional strict experiment uses, for each held-out country `c`:

```text
train(c) = country != c AND date < 2022-01-01
test(c)  = country == c AND date >= 2022-01-01
```

Run it with:

```bash
python "2.Source Code/generate_strict_temporal_leave_one_country_out_robustness.py" \
  --workers 4 --resume
```

The verified test union contains 1,170 unique `(area_id,date)` rows, 646 areas,
and 27 countries; `LSO` and `ZWE` have no post-cutoff rows and are recorded as
skipped folds. The primary four metrics pool all 1,170 test observations for each
model, without first averaging by area or country. The canonical main-table path
is `strict_temporal_loco_micro_metrics.csv`. Per-area metrics and their
metric-specific defined-area counts remain available only as diagnostics in
`strict_temporal_loco_area_metrics.csv` and
`strict_temporal_loco_area_macro_denominators.csv`; they do not enter the primary
2 x 4 table.

| model | accuracy | precision | recall | R2(p3) |
|---|---:|---:|---:|---:|
| Nowcasting | 0.652991 | 0.777467 | 0.949943 | -0.007144 |
| Forecasting | 0.652137 | 0.774373 | 0.948805 | -0.013764 |

The canonical table SHA-256 is
`5729dcefc77970b1e4e957b80e3949fd90239c2435386dd57764e837927f2ac0`.

For consistency with the reporting precision, both actual and predicted Phase
3+ shares are rounded to two decimals before one pooled R-squared is computed
across the 1,170 observations. Accuracy remains exact five-phase accuracy, while
precision and recall pool the global Phase 3+ confusion counts. The fixed manuscript hyperparameters,
`fews_ipc_ha`, and notebook-faithful in-sample Layer 1 residuals are retained, so
this is fixed-pipeline retrospective robustness rather than a complete crisis-
information cold start or vintage-faithful real-time nowcast.

Regenerate only the replacement table from the verified saved predictions,
without refitting the 27 country folds:

```bash
python "2.Source Code/generate_strict_temporal_leave_one_country_out_robustness.py" \
  --aggregate-existing
```

The generator writes the two-row pooled-micro main table, both prediction files,
per-area diagnostic metrics, diagnostic denominators, fold audit, skipped-fold
audit, source audit, and restart checkpoints under the `strict_temporal_loco_*` namespace in
`2.Source Code/produced_graph/`.

### 10% Leave-Area-Out Robustness Contract

This analysis targets a narrower missing-primary-data scenario than LOCO: IPC-style primary labels are sparse or unavailable for selected areas, but labels from other areas in the same country may still be available. The two main models use one shared joint spatial holdout.

The fixed sampling and split contract is:

- The sampling frame contains 1,198 unique areas in 29 countries. Exactly 120 areas are selected with seed 0.
- Every country receives at least one sampled area. Remaining slots are allocated by the Hamilton largest-remainder method using each country's remaining capacity (`area_count - 1`), with ISO3 ordering for tied remainders.
- One shared `numpy.random.default_rng(0)` stream samples areas in ISO3 order. The saved sample SHA-256 is `b9cd7f069330ed5cdd0991ebe2c4b5a04d8ae418b1bc155010a1cb7d919f06b7`.
- All dates for the 120 sampled areas form one pooled test set. All dates for the remaining 1,078 areas form the training set.
- Training may retain other areas from the same country and date. Labels from held-out areas do not enter the Forecasting model or either Nowcasting layer.
- The protocol reuses the manuscript JSON hyperparameters, retains `fews_ipc_ha`, and keeps training in-sample residual learning for Nowcasting Layer 2.
- Precision and recall use the pooled Phase 3+ binary threshold (`overall_phase >= 3`). Accuracy is pooled exact-match overall-phase accuracy. R-squared compares actual and predicted Phase 3+ population shares. Metrics are calculated over all 508 held-out rows, not averaged by area or country.

The seed-0 run produces:

| Model | Phase 3+ precision | Phase 3+ recall | Overall-phase accuracy | Phase 3+ share R² |
|-------|-------------------:|----------------:|-----------------------:|------------------:|
| Forecasting | 0.830303 | 0.925676 | 0.748031 | 0.727879 |
| Nowcasting | 0.834862 | 0.922297 | 0.753937 | 0.736254 |

Generated outputs under `2.Source Code/produced_graph/`:

- `leave_area_out_10pct_sample.csv`
- `leave_area_out_10pct_forecasting_predictions.csv`
- `leave_area_out_10pct_nowcasting_predictions.csv`
- `leave_area_out_10pct_metrics.csv`
- `leave_area_out_10pct_source_audit.csv`
- `precision_recall_accuracy_p3r2_leave_area_out_10pct.jpg`
- `precision_recall_accuracy_p3r2_leave_area_out_10pct.png`
- `precision_recall_accuracy_p3r2_leave_area_out_10pct.pdf`

### Cumulative-Phase Scatter-Figure Contract

The legacy figures extend the Phase 3+ actual-vs-predicted scatter logic in `Figure1_multiple_figures.ipynb` to Phase 2+, Phase 4+, and Phase 5 for the manuscript Forecasting model and cascading two-layer Nowcasting model. A combined 3 x 3 figure retains those outputs and adds the random five-fold full-OOF Contemporaneous model for Phase 2+, Phase 3+, and Phase 4+.

The reproducible model and evaluation contract is:

- Training rows have `date < 2022-01-01`; test rows have `date >= 2022-01-01`.
- Forecasting and Nowcasting use all 1,170 temporal-test area-date observations across 646 areas, matching the canonical `All_prediction.csv` and original Figure 1 anchors. Contemporaneous uses all 5,575 random-CV OOF rows across 1,198 areas. No join or key match redefines either population.
- Phase 2+ is `phase2_percent + phase3_percent + phase4_percent + phase5_percent`; Phase 3+ is `phase3_percent + phase4_percent + phase5_percent`; Phase 4+ is `phase4_percent + phase5_percent`; Phase 5 is `phase5_percent`.
- Phase 2+ uses `forecasting_hyperparameters.json`. Phase 4+ and Phase 5 use `forecasting_hyperparameters_p3.json`, matching the notebooks' effective Phase 3 parameter carry-over.
- Layer 1 retains `fews_ipc_ha`. Nowcasting Layer 2 learns the Layer 1 in-sample training residual and adds the predicted residual to the Layer 1 test prediction.
- Each figure places predicted share on x and actual share on y, reports `r2_score(actual, predicted)`, limits the linear fit to the observed x range, and includes the `y=x` perfect-prediction reference. The 3 x 3 grid additionally reports the intercept and slope from `actual = intercept + slope * predicted` in every panel.
- The 3 x 3 grid uses rows Phase 2+/3+/4+ and columns Forecasting/Nowcasting/Contemporaneous. All three models share one x/y scale within a row; different cumulative phases retain phase-specific scales. Each panel states R², intercept, slope, and its own `n`.
- Contemporaneous predictions come from `all_prediction_contemporaneous_random_cv_predictions.csv`; the audit records the seed-0 fold assignment, five 1,115-row folds, feature count including `kfolds`, and the full-OOF population key and prediction hashes. They are not joined to the temporal holdout.
- Formal regeneration uses one outer model worker and the frozen Forecasting/Nowcasting environment's default XGBoost threading. The figure and tables explicitly state that the temporal-holdout and random-CV protocols/populations are not directly comparable.

The controlled run recorded in `phase_cumulative_scatter_source_audit.csv` produces:

| Cumulative phase | Forecasting R² | Nowcasting R² |
|------------------|---------------:|--------------:|
| Phase 2+ | 0.284453 | 0.285322 |
| Phase 4+ | 0.199356 | 0.198471 |
| Phase 5 | -0.030956 | -0.030956 |

In this environment, all rounded Phase 5 predictions are `0.00` for both models. The Phase 5 figure therefore shows the constant prediction column, states that a linear fit is not estimable, and does not include a misleading linear-fit legend entry.

The 3 x 3 panel statistics written to `phase_cumulative_three_model_scatter_metrics.csv` are:

| Cumulative phase | Model | R² | Intercept | Slope |
|---|---|---:|---:|---:|
| Phase 2+ | Forecasting | 0.284453 | 0.119160 | 0.891315 |
| Phase 2+ | Nowcasting | 0.285322 | 0.118529 | 0.892355 |
| Phase 2+ | Contemporaneous (random CV; n=5,575) | 0.697891 | -0.079610 | 1.127868 |
| Phase 3+ | Forecasting | 0.248999 | 0.005093 | 1.112952 |
| Phase 3+ | Nowcasting | 0.252785 | 0.004427 | 1.109484 |
| Phase 3+ | Contemporaneous (random CV; n=5,575) | 0.641619 | -0.047227 | 1.162332 |
| Phase 4+ | Forecasting | 0.199356 | -0.044281 | 1.734019 |
| Phase 4+ | Nowcasting | 0.198471 | -0.038883 | 1.676945 |
| Phase 4+ | Contemporaneous (random CV; n=5,575) | 0.442766 | -0.020895 | 1.295013 |

The source audit calibrates the regenerated Forecasting and Nowcasting Phase 3+ predictions against the legacy Figure 1 anchors, records both model environments, and links the random-CV Contemporaneous prediction and audit hashes. The Contemporaneous values above are full five-fold OOF metrics from the reproducible seed-0 rerun; they are not the historical notebook's unrecorded-shuffle last fold.

Generated outputs under `2.Source Code/produced_graph/`:

- `all_prediction_contemporaneous_random_cv_predictions.csv`
- `all_prediction_contemporaneous_random_cv_source_audit.csv`
- `phase2plus_actual_vs_predicted_forecasting_nowcasting.jpg`
- `phase2plus_actual_vs_predicted_forecasting_nowcasting.png`
- `phase2plus_actual_vs_predicted_forecasting_nowcasting.pdf`
- `phase4plus_actual_vs_predicted_forecasting_nowcasting.jpg`
- `phase4plus_actual_vs_predicted_forecasting_nowcasting.png`
- `phase4plus_actual_vs_predicted_forecasting_nowcasting.pdf`
- `phase5_actual_vs_predicted_forecasting_nowcasting.jpg`
- `phase5_actual_vs_predicted_forecasting_nowcasting.png`
- `phase5_actual_vs_predicted_forecasting_nowcasting.pdf`
- `phase_cumulative_actual_vs_predicted_forecasting_nowcasting_contemporaneous.jpg`
- `phase_cumulative_actual_vs_predicted_forecasting_nowcasting_contemporaneous.png`
- `phase_cumulative_actual_vs_predicted_forecasting_nowcasting_contemporaneous.pdf`
- `phase_cumulative_three_model_scatter_predictions.csv`
- `phase_cumulative_three_model_scatter_metrics.csv`
- `phase_cumulative_scatter_predictions.csv`
- `phase_cumulative_scatter_metrics.csv`
- `phase_cumulative_scatter_source_audit.csv`

### Spatial Feature Comparison Contract

This analysis asks whether removing raw coordinates or replacing them with
spatially aggregated predictor information changes model performance. Forecasting
and cascading two-layer Nowcasting retain the fixed 2022 temporal holdout;
Contemporaneous uses the canonical seed-0 random five-fold row-CV full-OOF
population. Feature-condition deltas are interpreted within a model and protocol.
The temporal and random-CV metric levels are explicitly not directly comparable.
This is a predictive robustness analysis, not a causal estimate of spatial
spillovers.

The four controlled conditions are run in this order:

1. `baseline_with_lat_lon`: current-environment rerun with latitude and longitude;
2. `no_lat_lon`: remove latitude and longitude without replacement;
3. `knn5_spatial_means`: remove coordinates and append equal-weight means from the five nearest other areas;
4. `d200_spatial_means`: remove coordinates and append equal-weight means from all other areas no more than 200 km away.

The first condition is a controlled baseline, not a fourth requested experiment.
Within each condition, Forecasting uses four XGBoost fits, cascading Nowcasting
uses eight fits, and Contemporaneous uses 20 fits (four cumulative targets across
five held-out folds). Across four conditions this gives 128 total fits. Formal
outer execution is serial. Forecasting/Nowcasting reproduce the stored main-result
environment without an explicit estimator seed; Contemporaneous fixes estimator
and shuffle seed 0 and uses the environment-default XGBoost thread count.

#### Evaluation population and feature contract

- Forecasting and Nowcasting train on rows with `date < 2022-01-01` and evaluate all 1,170 rows with `date >= 2022-01-01`, covering 646 areas and 27 test countries.
- Contemporaneous stably sorts the 5,575 source rows by area/date, shuffles once with seed 0, and applies five non-shuffled KFold blocks of 1,115 rows. The saved result is full OOF across 1,198 areas and 29 countries.
- The four conditions reuse one identical Contemporaneous fold assignment with SHA-256 `bc2b0a13def3c2979b562236cedbeff0cc030bf67d4b5de9160f8b916937b099`.
- The temporal test-key SHA-256 is `288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2`; the Contemporaneous full-population key SHA-256 is `f540e216e9ee286c7502b3aa465fe222a66a1a87eec014158d0db9ace13f6651`.
- The prediction artifact contains 31,660 rows: four times `(1,170 Forecasting + 1,170 Nowcasting + 5,575 Contemporaneous)`.
- Layer 1 contains 106 features in the controlled baseline, 104 after coordinate removal, and 208 after adding either KNN-5 or D200 means.
- Nowcasting Layer 2 contains 69 original features and 138 features after spatial means are appended.
- Contemporaneous uses the canonical source-predictor contract plus `kfolds`: 174 baseline features, 172 without coordinates, and 343 with either spatial-mean feature family.
- The feature manifest classifies two coordinates, 31 Layer 1 static features, 73 Layer 1 dynamic features, and 69 Layer 2 dynamic features.
- Reported `phase3plus_r2` uses unrounded actual shares and two-decimal Phase 3+ predictions, matching the frozen Figure 1 contract. `phase3plus_r2_raw` is retained as a separate raw-prediction diagnostic. Two-decimal cumulative predictions also drive the 0.20 overall-phase threshold conversion.

#### Spatial and interpolation contract

Distances are Haversine great-circle distances with Earth radius `6371.0088 km`.
KNN-5 is pure directed geodesic nearest-five: it excludes self, permits
cross-country neighbors, uses equal weights, and resolves exact-distance ties by
ascending `area_id`. It is not polygon contiguity. D200 includes every other area
with distance `<= 200.0 km`, permits cross-country neighbors, uses equal weights,
and leaves spatial means missing when an area has no neighbor; it never falls
back to KNN-5 or another support definition.

Spatial means use only original observed values. Missing neighbor slots are
resolved in strict order: the missing neighbor's own history, then an eligible
same-country source, then an eligible global source. Spatial fallback excludes
both the aggregation target A and missing neighbor B. Imputed values are never
used recursively as sources.

For dynamic Layer 1 predictors, the resolved `feature_month` must be no later
than `target_month - 12 calendar months`. This is a feature-time contract, not a
second transformation of already lagged columns. Dynamic Layer 2 predictors may
use the target month but never a future month. Static predictors do not carry a
feature month. The formal lineage contains 15,295,392 imputation events, all with
`temporal_contract_passed=True`; the 692-row summary satisfies all documented
slot identities.

#### Model references and controlled baselines

The primary metrics in each row are recomputed from that condition's saved
predictions. For Forecasting and Nowcasting, the `baseline_*` fields,
signed/absolute deltas, and hollow figure markers use the stored main-result
notebook outputs. For Contemporaneous, they use the same-protocol
`baseline_with_lat_lon` seed-0 full-OOF rerun.

| Model/reference | Precision | Recall | Accuracy | Phase 3+ R-squared |
|---|---:|---:|---:|---:|
| Forecasting | 0.775070 | 0.940842 | 0.649573 | 0.248999 |
| Nowcasting | 0.777465 | 0.941980 | 0.653846 | 0.252785 |
| Contemporaneous controlled baseline | 0.797259 | 0.904371 | 0.693453 | 0.641619 |

The formal environment reproduces the Forecasting and Nowcasting stored
main-result metrics. The Contemporaneous baseline also reproduces the canonical
5,575-row random-CV prediction sidecar after normalizing two-decimal floating
serialization. The source audit marks all temporal-versus-random-CV comparisons
as not directly comparable and retains both rounded-contract and raw R-squared.

#### Formal metrics

The following table was generated directly from
`spatial_feature_comparison_metrics.csv`:

| Condition | Model | Precision | Recall | Accuracy | R-squared | Raw R-squared |
|---|---|---:|---:|---:|---:|---:|
| Baseline with latitude/longitude | Forecasting | 0.775070 | 0.940842 | 0.649573 | 0.248999 | 0.248065 |
| Baseline with latitude/longitude | Nowcasting | 0.777465 | 0.941980 | 0.653846 | 0.252785 | 0.254387 |
| Baseline with latitude/longitude | Contemporaneous | 0.797259 | 0.904371 | 0.693453 | 0.641619 | 0.642211 |
| No latitude/longitude | Forecasting | 0.789276 | 0.954494 | 0.674359 | 0.216680 | 0.215287 |
| No latitude/longitude | Nowcasting | 0.793756 | 0.954494 | 0.676068 | 0.223429 | 0.223539 |
| No latitude/longitude | Contemporaneous | 0.789017 | 0.906131 | 0.686099 | 0.628174 | 0.628498 |
| KNN-5 spatial means | Forecasting | 0.795977 | 0.945392 | 0.670940 | 0.134165 | 0.135130 |
| KNN-5 spatial means | Nowcasting | 0.803485 | 0.944255 | 0.673504 | 0.140797 | 0.140046 |
| KNN-5 spatial means | Contemporaneous | 0.803516 | 0.911704 | 0.699552 | 0.663294 | 0.663416 |
| 200 km spatial means | Forecasting | 0.801951 | 0.935154 | 0.665812 | 0.131397 | 0.132412 |
| 200 km spatial means | Nowcasting | 0.804497 | 0.936291 | 0.672650 | 0.135867 | 0.137216 |
| 200 km spatial means | Contemporaneous | 0.803748 | 0.905837 | 0.697937 | 0.679755 | 0.680244 |

Within the temporal protocol, removing coordinates raises precision, recall, and
accuracy but lowers Phase 3+ share R-squared; KNN-5 and D200 retain the precision
gain while further reducing R-squared. Within random five-fold Contemporaneous,
coordinate removal slightly reduces accuracy and R-squared, whereas both spatial
mean conditions improve the controlled-baseline accuracy and R-squared. These
within-protocol findings do not make the temporal and Contemporaneous metric
levels directly comparable.

#### Formal execution evidence and outputs

The formal run used Python 3.11.3, pandas 2.2.3, NumPy 1.26.4,
scikit-learn 1.5.2, XGBoost 2.0.3, and matplotlib 3.10.1. It completed with exit
status 0 under serial outer execution and environment-default XGBoost threads.
The generator's staged validation and an independent readback confirmed 31,660
predictions, 12 metric rows, 12 source-audit rows, five folds of 1,115 rows for
every Contemporaneous condition, one shared fold hash, zero temporal-contract
violations, and matching artifact SHA-256 values. Raster/PDF visual and format
checks passed. No test suite was run for this bounded update.

Generated outputs under `2.Source Code/produced_graph/`:

- `spatial_feature_comparison_predictions.csv`
- `spatial_feature_comparison_metrics.csv`
- `spatial_feature_comparison_feature_manifest.csv`
- `spatial_feature_comparison_weight_diagnostics.csv`
- `spatial_feature_interpolation_audit.csv.gz`
- `spatial_feature_interpolation_summary.csv`
- `spatial_feature_comparison_source_audit.csv`
- `precision_recall_accuracy_p3r2_spatial_feature_comparison.jpg`
- `precision_recall_accuracy_p3r2_spatial_feature_comparison.png`
- `precision_recall_accuracy_p3r2_spatial_feature_comparison.pdf`

The 75.8 MiB `spatial_feature_interpolation_audit.csv.gz` lineage file is
attached to GitHub release `v1.2.0` rather than stored in ordinary Git history.
After downloading it to the path above, verify it with
`sha256sum --check RELEASE_ASSET_SHA256.txt`. Its expected SHA-256 is
`c6769b453538c19be0745e66772ad7c794960eb2c77e96e914d35a422f8e39b4`.

The figure remains a 3 x 4 grid: rows are coordinate removal, KNN-5 means, and
D200 means; columns are Phase 3+ precision, recall, overall-phase accuracy, and
Phase 3+ R-squared. Each panel contains Forecasting, Nowcasting, and
Contemporaneous. Filled marks are condition results. Hollow marks are the stored
main-result reference for Forecasting/Nowcasting and the same-protocol
latitude/longitude baseline for Contemporaneous. The subtitle gives both sample
sizes and states that the protocols are not directly comparable. Numeric labels
are placed away from their reference connectors. The source audit is the
completion/provenance record for inputs, environment, features, matrices,
interpolation, artifact paths, and SHA-256 values.

### Simple Baseline Comparison Contract

This analysis compares four deliberately simple methods with the stored Figure 1
original-model anchors for detecting IPC Phase 3 or above. It is a fixed 2022
temporal evaluation, not a leave-country-out or leave-area-out experiment.

The formal method order is Persistence, Multinomial, Ordered Probit, and
Ensemble OLS. Training rows have `date < 2022-01-01`; test rows have
`date >= 2022-01-01`. Every generated task-method group contains the same 1,170
unique `(area_id,date)` keys across 646 areas and uses canonical key SHA-256
`288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2`.
The source `overall_phase` differs from the 0.20 cumulative-share reconstruction
on five test rows; those rows remain in the evaluation, and the reconstructed
five-level phase is the common truth and Ordered Probit outcome for every
method.

Precision and recall are pooled row-level Phase 3+ micro metrics. The saved
metrics also retain five-class exact accuracy, binary Phase 3+ accuracy, and
TP/FP/FN/TN. No join, model fit, or missing-value step may reduce the 1,170-row
denominator.

#### Baseline definitions and lineage

- Persistence calls the existing source-selection implementation unchanged.
  Forecasting sources are no later than target month minus 12 months;
  Nowcasting sources are strictly earlier than the target month.
- Multinomial consumes and validates the frozen task-specific prediction
  artifacts. It is not refitted under scikit-learn 1.8 because that refit changes
  three Nowcasting negative-class predictions.
- Ordered Probit fits one direct five-class `OrderedModel(distr="probit")` per
  task without an intercept. Median imputation, scaling, duplicate/rank pruning,
  and all feature choices are learned from training rows only.
- Ensemble OLS replaces the original cumulative-target XGBoost estimators with
  OLS. Forecasting fits four cumulative-share regressions. Nowcasting retains
  keyed in-sample Layer 1 residual learning and four Layer 2 residual
  regressions. The temporary evaluation label is explicitly excluded from Layer
  1 predictors; source feature counts are 106 for Forecasting and 175 for the
  combined Nowcasting architecture.
- Raw OLS predictions remain unconstrained and are neither clipped nor projected.
  Separate two-decimal copies are used only for the existing 0.20 phase
  conversion. The original Layer 1 workflow retains `fews_ipc_ha`, so this is not
  a complete crisis-information cold-start claim.

The selected comparison values labeled `Main result` are stored notebook
results rather than current-environment predictions:

- Forecasting precision `0.7750702905342081`, recall `0.9408418657565415`;
- Nowcasting precision `0.7774647887323943`, recall `0.9419795221843004`.

They are rendered as a neutral-gray dashed line plus hollow diamond in each
panel, not as a fifth bar. The different Table 1 Nowcasting anchor and the
controlled spatial rerun remain provenance-only values in the source audit.

#### Formal metrics

The following table was generated directly from
`simple_baseline_comparison_metrics.csv`:

| Task | Method | Role | Precision | Recall | Five-class accuracy | Phase 3+ accuracy | Phase 3-above R² |
|---|---|---|---:|---:|---:|---:|---:|
| Nowcasting | Persistence | Baseline | 0.822864 | 0.745165 | 0.588034 | 0.688034 | -0.669540 |
| Nowcasting | Multinomial | Baseline | 0.796043 | 0.594994 | 0.481197 | 0.581197 | -1.241300 |
| Nowcasting | Ordered Probit | Baseline | 0.825291 | 0.564278 | 0.489744 | 0.582906 | -1.232152 |
| Nowcasting | Ensemble OLS | Baseline | 0.794271 | 0.693970 | 0.545299 | 0.635043 | -0.953133 |
| Nowcasting | Main result | Original reference | 0.777465 | 0.941980 | 0.653846 | 0.753846 | -0.317336 |
| Forecasting | Persistence | Baseline | 0.852615 | 0.612059 | 0.510256 | 0.629060 | -0.985152 |
| Forecasting | Multinomial | Baseline | 0.791444 | 0.673493 | 0.505128 | 0.621368 | -1.026319 |
| Forecasting | Ordered Probit | Baseline | 0.792647 | 0.613197 | 0.490598 | 0.588889 | -1.200134 |
| Forecasting | Ensemble OLS | Baseline | 0.783959 | 0.689420 | 0.533333 | 0.623932 | -1.012596 |
| Forecasting | Main result | Original reference | 0.775070 | 0.940842 | 0.649573 | 0.750427 | -0.335632 |

`phase3above_r2` applies scikit-learn's `r2_score` to the binary actual and
predicted indicators `1[IPC phase >= 3]`. It is therefore distinct from the
continuous Phase 3+ population-share R² used elsewhere in the package. A
negative value means that the binary predictions have greater squared error
than the constant observed-mean benchmark.

The baseline pattern is higher precision but materially lower recall than the
stored original anchors. This precision-recall tradeoff is visible directly in
the 2 x 2 figure and is not summarized by averaging the two metrics.

Both Ordered Probit fits converged under BFGS. Forecasting retained 103 features
with maximum absolute gradient `5.7858e-06`; Nowcasting retained 163 with
`9.8499e-06`. All parameters and five-class probabilities were finite, every
probability row summed to one, and transformed cutpoints were strictly ordered.
Ensemble OLS retained 103 Forecasting and 165 combined Nowcasting features. Its
unconstrained audit recorded 829/907 out-of-range prediction cells and 168/215
cumulative-order-violation rows for Forecasting/Nowcasting, respectively; those
values are diagnostics and were not post-processed away.

#### Formal execution, verification, and outputs

The current formal artifacts were regenerated in the frozen Windows environment,
including a refreshed live hash binding to the 12-row spatial metrics artifact:
Python 3.11.3, pandas 2.2.3, NumPy 1.26.4, SciPy 1.17.1,
scikit-learn 1.5.2, XGBoost 2.0.3, statsmodels 0.14.6, patsy 1.0.1, and
matplotlib 3.10.1. The generator's staged validation reloaded the saved files,
verified 9,360 predictions and ten metric rows, reconstructed the generated
metrics, and checked every payload hash before writing the source audit last.
No test suite was run for the bounded `phase3above_r2` column update.

| Artifact | SHA-256 |
|---|---|
| `simple_baseline_comparison_predictions.csv` | `05cb67ab4f39a7e75751926c30cb6181301e6232e29b424bb9eaa740eb36ae16` |
| `simple_baseline_comparison_metrics.csv` | `d7f6dc6eae93b1db5a52c7f624c91ea96af1d3f6dcba90a4666d9354b35ddcdb` |
| `simple_baseline_comparison_feature_manifest.csv` | `f44dd4659647a8afa8ddb491effe8f7692d0262d5dbf10f5a8eb03c3e89d4e17` |
| `simple_baseline_comparison_model_audit.csv` | `beeb3249779b459f5043bfb1b9d380969e82d1640e5372c829800e8965c02297` |
| `simple_baseline_comparison_source_audit.csv` | `1ec4b3d7b9a25eab79edf23edf12d05c21ed34130aebec5a67a2e4b199b80647` |
| `phase3plus_precision_recall_simple_baseline_comparison.jpg` | `d39950f82981c1c67d712519955d7cd0fb28630384fc880761b06683b6ff3b9f` |
| `phase3plus_precision_recall_simple_baseline_comparison.png` | `d8deba46f9e4a6a61bd21b4eb988784a0629ae6dfd68ba6eefcbbe855dbe766e` |
| `phase3plus_precision_recall_simple_baseline_comparison.pdf` | `a2eab9c4024b00b923af8b3ea7c02476bd5c4f04fa08e81f2dbcef59da8173c6` |

Generated outputs under `2.Source Code/produced_graph/` are exactly the eight
artifacts listed in the hash table. The source audit is written last; consumers
must require `run_status=complete` and verify every listed payload hash.

### Verification Checklist

- [ ] Virtual environment created and activated
- [ ] All packages installed without errors
- [ ] Graphviz system dependency installed and verified (`dot -V`)
- [x] All 12 notebooks use portable executable paths, contain no saved outputs, and have null execution counts
- [ ] Data files present in `1.Source Data/`
- [ ] Main notebooks execute without errors
- [ ] Performance metrics within expected range
- [ ] Figures generate successfully
- [ ] LOCO forecasting predictions contain one row per source `(area_id, date)`
- [ ] LOCO nowcasting predictions contain one row per source `(area_id, date)`
- [ ] LOCO metrics contain one row per model-country pair
- [x] `leave_one_country_out_micro_metrics.csv` contains exactly two model rows and four metric columns, pools all 5,575 out-of-country predictions per model without area/country averaging, and uses pooled Phase 3+ confusion counts
- [x] `strict_temporal_loco_micro_metrics.csv` contains exactly two model rows and four metric columns, pools all 1,170 strict temporal test predictions per model, and uses the documented two-decimal pooled R-squared contract
- [ ] LOCO audit confirms the held-out country is absent from training and is the only country in its test fold
- [ ] LOCO figure coordinates match the non-missing per-country metrics and contain no aggregate point
- [ ] Leave-area sample contains 120 unique areas, covers all 29 countries, and matches the documented SHA-256
- [ ] Leave-area predictions contain all and only the 508 source rows from sampled areas
- [ ] Leave-area audit confirms zero held-area overlap with training, a shared test set for both models, two model-level workers, and `n_jobs=1` per XGBoost fit
- [ ] Leave-area figure values match the pooled metrics CSV and use the documented metric semantics
- [ ] Legacy cumulative-phase scatter predictions contain 1,170 rows per model and phase, without join- or match-based filtering
- [ ] Three-model grid predictions contain six temporal groups of 1,170 rows and three Contemporaneous groups of 5,575 rows, with protocol-specific key hashes and no population-redefining join
- [ ] Three-model grid metrics contain nine rows whose R², intercept, and slope agree with all panel annotations
- [ ] Cumulative-phase scatter audit records the fixed 2022 split, both model environments, input/parameter/generator/sidecar hashes, one formal outer model worker, and the legacy Phase 3+ calibration
- [ ] Phase 5 panels identify constant predictions, omit a non-existent linear fit, and retain the complete evaluation population
- [x] Spatial comparison predictions contain 31,660 rows and 12 condition-model groups: eight temporal groups of 1,170 rows and four Contemporaneous groups of 5,575 rows
- [x] Forecasting/Nowcasting groups share the canonical temporal key hash; all Contemporaneous groups share the canonical full-OOF key and fold hashes, with no duplicate evaluation keys
- [ ] Spatial feature manifest confirms coordinate 2, Layer 1 static 31, Layer 1 dynamic 73, and Layer 2 dynamic 69
- [ ] Spatial feature counts are Layer 1 106/104/208 and Nowcasting Layer 2 69/138 as applicable
- [ ] Every KNN-5 area has five unique non-self neighbors; every D200 distance is <=200 km and zero-neighbor areas remain explicit
- [ ] Interpolation summaries satisfy all slot identities and lineage confirms observed-only sources, strict tier priority, A/B exclusion, and zero temporal violations
- [x] All 12 saved metric rows recompute from predictions using the documented rounded-contract R-squared, with raw R-squared retained separately
- [x] `baseline_*` values and hollow markers match stored main-result references for Forecasting/Nowcasting and the canonical same-protocol baseline for Contemporaneous
- [x] Source-audit artifact hashes match the published files
- [x] The formal spatial figure has 12 panels, letters a-l, shared within-column scales, three model positions per panel, and annotations agreeing with metrics
- [x] All 10 spatial-comparison output files exist: seven tabular artifacts plus JPG, PNG, and PDF
- [x] Spatial source audit records `production_run=True`, Python 3.11.3/XGBoost 2.0.3, serial outer execution, seed-0 Contemporaneous folds, default XGBoost threads, and non-comparability across protocols
- [x] JPG is 300 DPI; JPG/PNG are readable at 3,515 x 2,165 pixels; PDF is readable, single-page, vector-rendered, and uses embedded TrueType fonts
- [ ] `python -m unittest discover -s tests -p 'test_*.py'` reports 243 passing tests in the verified Windows calibration environment
- [ ] Generator, spatial-comparison, and simple-baseline test files pass `python -m py_compile`
- [x] Simple-baseline predictions contain exactly 9,360 rows: 1,170 canonical area-date rows for each of four methods in both tasks, with no duplicate keys
- [x] Every simple-baseline task-method group uses the identical canonical 1,170-row test population and common test-key SHA-256
- [x] Simple-baseline metrics contain eight generated baseline rows and two stored-original reference rows, and recompute from predictions using pooled Phase 3+ micro precision/recall
- [x] Persistence, frozen Multinomial, direct Ordered Probit, and cumulative-architecture Ensemble OLS satisfy their documented fitting and prediction contracts
- [x] The simple-baseline feature manifest and model audit agree with the saved task-method feature sets, model diagnostics, and training/test split
- [x] The 2 x 2 Phase 3+ precision/recall figure has four panels and its bars, reference markers, and numeric labels agree with the metrics CSV
- [x] All eight simple-baseline output artifacts exist, their SHA-256 values match the completed source audit, and two formal runs were byte-identical
- [x] The 17 focused simple-baseline tests pass, the full repository suite passes 243 tests, and the generator and test module pass `python -m py_compile`

---

## Path Audit Summary

All executable notebook input and tabular-output paths have been normalized for
the released directory layout. Run notebooks from `2.Source Code/`:

- input tables resolve through `../1.Source Data/`;
- hyperparameter JSON files and `food_crisis_functions.py` resolve locally;
- generated tabular artifacts resolve through `produced_graph/`;
- `Figure4_Descriptives.ipynb` consumes the released Nowcasting table and
  `area_country_lookup.csv`, with no archived author-machine input.

The root-level `run_replication.py --check-only` command parses all notebooks,
checks code cells for executable absolute paths, and confirms that stored
outputs and execution counts are absent.

---

## Documentation Standards

Each notebook includes:

1. **Header Markdown Cell**:
   - Title and purpose statement
   - High-level methodology overview (2-3 steps)
   - Expected outputs and metrics
   - Working-directory and dependency notes where needed

2. **Code Organization**:
   - Import statements and hyperparameter loading
   - Data loading with package-relative paths
   - Model training and evaluation
   - Results output

3. **Reproducibility Requirements**:
   - Run notebooks from `2.Source Code/`
   - Verify data files exist in `1.Source Data/`
   - Install Graphviz system dependency (for Figure3_Flowchart.ipynb only)
   - Expect 2-30 minute runtime depending on notebook type

---

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Solution: Activate virtual environment, reinstall requirements

2. **FileNotFoundError for data files**
   - Solution: Update file paths, verify data files exist

3. **Graphviz ExecutableNotFound**
   - Solution: Install system Graphviz (see INSTALL.md)

4. **Out of Memory errors**
   - Solution: Close other applications, reduce CV folds

5. **Slow SHAP computation**
   - Expected behavior: SHAP takes 10-30 minutes on typical hardware

---

## Contact and Support

For questions regarding:
- **Code execution**: Refer to INSTALL.md and notebook documentation
- **Methodology**: See manuscript
- **Data**: Contact corresponding author

---

## Version History

- **v1.2.0** (August 2026): GitHub replication release, pooled-micro LOCO replacement, and verified simple-baseline comparison
  - Added the fresh-clone `README.md`, root-level controlled workflow runner, portable notebook paths, cleared notebook state, and a checksum-bound release asset for the large interpolation lineage
  - Replaced both pure-spatial and strict-temporal LOCO 2 x 4 main tables with pooled-micro metrics under canonical `*_micro_metrics.csv` paths
  - Added Persistence, frozen Multinomial, direct Ordered Probit, and cumulative-architecture Ensemble OLS comparisons for Forecasting and Nowcasting
  - Added the canonical 1,170-row metrics contract, model/feature/source audits, deterministic artifact hashes, focused tests, and the 2 x 2 Phase 3+ precision/recall figure
  - Added the Phase 2+/3+/4+ by Forecasting/Nowcasting/Contemporaneous 3 x 3 scatter grid, keyed predictions, panel statistics, and linked source-audit hashes
  - Extended the four-condition spatial-feature comparison with canonical seed-0 random five-fold Contemporaneous OOF results, 12-row metrics/audit tables, and three-model figure panels with explicit cross-protocol non-comparability
- **v1.1.0** (August 2026): Added the verified spatial-feature comparison
  - Added coordinate ablation, pure KNN-5, and inclusive D200 conditions for Forecasting and cascading two-layer Nowcasting
  - Added leakage-safe interpolation lineage, neighbor/feature diagnostics, source-audit provenance, and the 3 x 4 metric figure
  - Preserved original notebook stored outputs as explicit legacy references while separately auditing the controlled rerun baseline
- **v1.0.0** (January 2026): Initial replication package for Nature Communications submission
  - Added requirements.txt with minimum dependency constraints
  - Created comprehensive INSTALL.md
  - Added markdown documentation to the analysis notebooks
  - Audited machine-specific notebook paths
  - Documented expected outputs and execution order

---

## Citation

When using this replication package, please cite:

[Citation to be added upon publication]

---

## License

[License information to be added by authors]

---

## Acknowledgments

This replication package was prepared following Nature Communications reproducibility guidelines and best practices for computational research transparency.

**Replication Package Status**: Spatial-feature, pooled-micro LOCO, simple-baseline, and cumulative three-model scatter computational artifacts verified; notebook paths normalized and publication files prepared. Final citation, license, and third-party data redistribution confirmation remain author actions, so the GitHub repository is private.
