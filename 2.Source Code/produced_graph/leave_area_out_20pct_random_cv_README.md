# Contemporaneous (using nowcasting dataset; random area five-fold CV)

Selected for the manuscript main results by the author on 15 September 2026.
Use the `model=Nowcasting` row in `leave_area_out_20pct_random_cv_metrics.csv`.
The source label is preserved to keep the original audit hashes valid.

## Generator and data

- Generator: `2.Source Code/generate_leave_area_out_20pct_random_cv.py`.
- Model: cascading two-layer XGBoost. Layer 1 uses
  `1.Source Data/Forecasting_Analysis_010825.csv`; the residual-correction
  layer uses `1.Source Data/Nowcasting_Analysis_010825.csv`.
- CV: sorted unique areas, `KFold(n_splits=5, shuffle=True, random_state=0)`.
  All dates of held-out areas are tested; all dates of other areas are used
  for training. This is random area CV, not row-level CV or out-of-time testing.
- Population: 5,575 observations across 1,198 areas. Fold test row counts:
  1,187, 1,038, 1,203, 1,066, 1,081.
- Aggregation: unweighted mean and sample SD (`ddof=1`) across five folds.

## Selected results

| Metric | Mean | Sample SD |
| --- | ---: | ---: |
| Phase 3+ precision | 0.8297738142 | 0.0185872205 |
| Phase 3+ recall | 0.9234407342 | 0.0122867312 |
| Overall phase accuracy | 0.7448125425 | 0.0091126926 |
| Phase 3+ population-share R-squared | 0.7183263787 | 0.0178927732 |

Precision and recall concern the binary `overall_phase >= 3` outcome;
accuracy concerns exact overall-phase class agreement. These are not macro
five-class precision/recall or pooled OOF metrics.

## Restoration and provenance

All six original CSVs were copied from
`0.Archived/2026-09-08_obsolete_presentation/2.Source Code/produced_graph/`:
area folds, Forecasting predictions, Nowcasting predictions, fold metrics,
summary metrics and source audit. Archived copies remain intact. No training
was rerun and no numeric values or CSV headers were changed.

The audit's historical worktree paths describe the original generation run.
Its hashes identify the restored files and original generator. The Forecasting
row remains a companion random-CV result, not the temporal Forecasting main
result. Existing Figure 1, R1 and scatter consumers still point to the earlier
`all_prediction_contemporaneous_random_cv_*` row-CV family; selecting this
restored result does not mean those consumers have already been regenerated.
