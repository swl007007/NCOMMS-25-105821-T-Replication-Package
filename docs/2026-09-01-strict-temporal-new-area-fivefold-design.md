# Strict-temporal random new-area five-fold evaluation

## Status

Design approved on 2026-09-01. This document authorizes no implementation or
model run by itself.

## Goal

Extend the existing 10% leave-area-out robustness logic with a reproducible
five-fold, 20%-per-fold evaluation of 2022 observations from previously
untrained areas. Evaluate both the four-model cumulative Forecasting ensemble
and the notebook-faithful cascading two-layer Nowcasting pipeline.

## Population and folds

- Use the canonical `1.Source Data/All_prediction.csv` key universe: 1,170
  unique `(area_id, date)` observations across 646 `area_id` values.
- Verify that these keys equal the `date >= 2022-01-01` keys in both released
  model inputs before fitting.
- Sort the 646 unique area IDs, then assign them with
  `KFold(n_splits=5, shuffle=True, random_state=0)`. Each area belongs to one
  fold only; folds contain 129 or 130 areas.
- Do not reuse the optional `*_df_with_folds_5.csv` files: their universe is
  not the canonical 646-area temporal-test universe.

For held-out fold `k`, construct each model frame from only:

```text
train = date < 2022-01-01  AND area_id not in held_areas[k]
test  = date >= 2022-01-01 AND area_id in held_areas[k]
```

Thus every historical row of a held-out area is excluded from training, and
2022 rows from the other four folds are excluded from both fitting and that
fold's evaluation. Across the five folds, every canonical test key receives
one out-of-fold prediction.

## Fixed model contract

- Inputs: `Forecasting_Analysis_010825.csv`, `Nowcasting_Analysis_010825.csv`,
  `area_country_lookup.csv`, and the existing two Forecasting parameter JSONs.
- Reuse `prepare_model_inputs`, `fit_forecasting_split`,
  `fit_nowcasting_split`, and the existing cumulative-share phase conversion.
- Retain `fews_ipc_ha`; this measures label-history cold start for new areas,
  not predictor cold start.
- Retain four cumulative targets, Phase 2 general parameters, Phase 3--5 P3
  parameters, Layer-1 in-sample residual learning for Layer 2, two-decimal
  prediction rounding, and the `>= 0.20` phase threshold.
- Set XGBoost `random_state=0` and `n_jobs=1`, matching the current 10% area
  robustness generator. This is 20 Forecasting and 40 Nowcasting fits.
- Reconstruct the true five-level `overall_phase` from actual cumulative shares
  with the same threshold; retain source `overall_phase` only for audit.

## Metrics and artifacts

The primary result is pooled OOF: concatenate the five held-out prediction
sets for each model and calculate Phase 3+ precision, recall, exact five-class
accuracy, and continuous Phase 3+ share R-squared once across all 1,170 rows.
Report each metric's five-fold arithmetic mean and sample standard deviation
(`ddof=1`) as stability summaries, but do not substitute those means for the
primary pooled estimate.

Write only new files under `2.Source Code/produced_graph/`:

- `leave_area_out_20pct_fivefold_area_folds.csv`
- `leave_area_out_20pct_fivefold_forecasting_predictions.csv`
- `leave_area_out_20pct_fivefold_nowcasting_predictions.csv`
- `leave_area_out_20pct_fivefold_fold_metrics.csv`
- `leave_area_out_20pct_fivefold_metrics.csv`
- `leave_area_out_20pct_fivefold_source_audit.csv`

Do not modify any `leave_area_out_10pct_*` artifact and do not create a figure,
checkpoint/resume system, dependency, notebook edit, or new model abstraction.

## Acceptance checks

- Both source tables and `All_prediction.csv` have identical canonical test
  key sets; no duplicate keys or missing required values.
- The fold file has exactly 646 unique areas, five folds, and no multi-fold
  area.
- In every fold, all train dates precede the cutoff, all test dates meet it,
  held areas are absent from train, and other 2022 rows are absent from the
  fold frame.
- Each model produces exactly 1,170 unique OOF keys matching
  `All_prediction.csv`; Forecasting and Nowcasting key/fold assignments agree.
- Independently recompute pooled and fold metrics from saved predictions;
  record input, parameter, fold, script, and output hashes plus environment
  versions in the source audit.
- Confirm the original 10% files remain byte-identical.
