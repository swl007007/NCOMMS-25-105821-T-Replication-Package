# Presentation archive — 8 September 2026

## Restoration update — 15 September 2026

At the author's request, all six `leave_area_out_20pct_random_cv_*.csv`
files were copied back to `2.Source Code/produced_graph/` with their original
bytes and hashes. The metrics file's `Nowcasting` row is now selected as
**Contemporaneous (using nowcasting dataset; random area five-fold CV)**.
See `2.Source Code/produced_graph/leave_area_out_20pct_random_cv_README.md`
for the model lineage and metric contract. The Forecasting row and companion
files are retained for audit completeness, not selected as temporal main results.

The archived copies and `inventory.csv` remain the historical 8 September
snapshot. Its description of this family as unselected is superseded by this
restoration. The one-off `archive_snapshot.py` retains its historical policy:
its `--verify` absence check will reject these deliberately restored source
paths, and its preview classification is not the current selection policy.

This is a recoverable snapshot, not a deletion or a new experiment. Decisions
use `writing/Predict_food_crisis_NC_RandR.tex` and
`writing/Response_2_v4_090826.docx`, checked against their current PDFs and text
indices. No manuscript, Response, model or numerical result was rewritten by
this cleanup. Not being cited does not by itself invalidate an analysis.

## Inventory and scope

`inventory.csv` records every non-hidden file scanned, its disposition and
reason. Archived files mirror their original package-relative paths beneath
this directory. Each moved file, retained scientific artifact, and retained
writing file has a pre-move SHA-256 hash. Non-artifact support files are listed
but not hashed. Hidden files/directories, environments, `__pycache__`,
`node_modules` and this archive directory are excluded. Existing
`missingness_sensitivity/archive/` and `1.Source Data/old_graph/` are inventoried
as existing archives and were not moved again. Empty former output directories
may remain. This is a point-in-time inventory, not a replacement experiment
registry.

The manuscript contains 16 image imports in 10 figure environments, six inline
tables, and no active external table imports. The Response DOCX contains no
embedded images; its three table objects comprise R1 (macro metrics), one
empty table and R2 (hyperparameter search). Inline table values are not live
CSV links; table-source retention follows the audited experiment lineage,
not a claim that every printed cell already agrees with its source.

## Current figure authority

All names below are in `writing/`; these files and the current TeX/PDF,
Response DOCX/PDF and numbered indices stay in place unchanged.

| Figure | Image files |
| --- | --- |
| 1 | `accuracy.jpg`, `precision_sensitivity.jpg`, `R2.jpg` |
| 2 | `Picture6.jpg`, `Picture8.jpg`, `Picture7.jpg`, `Picture9.jpg` |
| 3 | `Picture3.jpg` |
| 4 | `Picture1.jpg`, `Picture2.jpg` |
| Supplementary 1 | `sample_imbalance_area_frequency.png` |
| Supplementary 2 | `2022_alert_map_panel_2x3.png` |
| Supplementary 3 | `Picture4.jpg` |
| Supplementary 4 | `direct_phase3_vs_phase45_rescue_five_class_confusion_atlas.png` |
| Supplementary 5 | `phase_cumulative_actual_vs_predicted_forecasting_nowcasting_contemporaneous.png` |
| Supplementary 6 | `direct_phase3_vs_phase45_rescue_binary_confusion_atlas.png` |

## Current table sources retained

Paths below refer to `2.Source Code/produced_graph/` unless otherwise noted.

| Table/evidence | Retained source families |
| --- | --- |
| Main Table 1; SI Table 1 main rows | `main_result_figure1_v1.*`, contemporaneous random-CV predictions/audit, original Table1 notebooks and source inputs |
| SI Table 1 baselines and sensitivities | `simple_baseline_comparison_*`, `persistence_baseline_*`, upstream `multinomial_baseline_*`, `selected_figure1_repeated_area_refit_metrics.csv`, `missingness_sensitivity/*.csv` |
| SI Table 2 | `spatial_feature_comparison_*`, spatial interpolation summary and release audit |
| SI Table 3 | `strict_temporal_loco_*`, `leave_area_out_20pct_fivefold_*` including fold/area diagnostics |
| SI Tables 4–5; Response R2 | Inline variable/parameter tables, input columns, JSON hyperparameters, tuning notebook |
| Response R1 | `response_table_r1_macro_metrics.csv`, frozen rescue benchmark predictions, contemporaneous predictions/audit |
| Missingness and rescue evidence cited in text | `variable_missingness_balance.csv`, all current missingness CSVs, complete rescue packages |

Uncited plots inside `phase4_rescue_classifier/`,
`direct_phase3_vs_phase45_rescue/`, and its contemporaneous predecessor remain:
their generators validate whole-directory file sets and hashes. Simple
baseline and spatial comparison payloads also stay intact for dependency
verification. Supporting predictions/audits and companion exports of current
figures are not treated as obsolete simply because not imported into TeX.

## What was archived

- Withdrawn `Conflit_Simulation.ipynb` and its `produced_graph/Picture5.jpg`.
  This legacy truth-versus-perturbed comparison must not be described as
  baseline-versus-perturbed prediction transitions. The different
  `writing/Picture5.jpg` is an unused map, also archived.
- Alternative `all_prediction_*` presentation tables/figures (but not the
  current contemporaneous predictions/audit or `1.Source Data/All_prediction.csv`).
- Uncited pure-spatial LOCO, 10% area and ordinary random-20%-area result
  families, superseded for SI Table 3 by strict-temporal analyses.
- Both uncited calibration output directories, together with the temporal
  evaluation outputs to which their saved audits are hash-bound.
- Unselected two-model scatter panels, standalone baseline comparison plots,
  alternative alert maps and their diagnostics, legacy SHAP/flow images,
  unused writing images and two unreferenced historical summary CSVs.

No active source datasets were removed. Legacy generators other than the
withdrawn conflict notebook remain available; several share utilities with
current analyses. Their output-writing code is not a current manuscript
citation. Rerunning them (including the root core workflow) can recreate
archived alternative outputs at old paths. Do not substitute such outputs for
the current frozen manuscript/Response lineage.

## Explicit exceptions and unresolved provenance

The corrected `generate_conflict_perturbation.py` and entire
`conflict_perturbation_10pct/` stay in place at the author's explicit request,
although the claim was removed from the manuscript. `writing/sn-article.pdf`
also remains because its hash is recorded in that exploratory audit.

Files marked `retain_uncertain` are current Figure 2/3 notebook outputs,
top-level SHAP variants and editable `flowchart_SI/` sources. The renamed
writing images have no exact byte/pixel match to these files; at least one
SHAP variant has different feature ordering and values. They must not be
silently equated or discarded. Resolving this artwork lineage is separate
from the confirmed-obsolete cleanup.

Separate table issues discovered during source mapping were not changed:
SI Table 1 persistence Nowcasting precision appears to copy recall; its
common-sample prose does not match the retained baseline support; and the
contemporaneous baseline precision/recall entries appear swapped. Keeping
their source evidence does not endorse the printed values.

## Recovery and verification

From the package root, run the non-mutating snapshot check:

```bash
python3 0.Archived/2026-09-08_obsolete_presentation/archive_snapshot.py --verify
python3 run_replication.py --check-only
```

For recovery, select rows with `status=archive` in `inventory.csv`, check each
archived file's SHA-256, and copy it from `archive_path` to `original_path`.
Create parent directories as needed; never overwrite a newly regenerated
file without resolving the conflict. Keep the archived copy and manifest.
The snapshot verifier expects moved sources to be absent; restoring files
deliberately changes that invariant.

Restore both calibration families and the complete archived
`all_prediction_temporal_test_*` set together before checking their historical
saved audits at the original paths. Restore LOCO predictions before using its
`--aggregate-existing` option. Old source audits are immutable historical
records, not silently rewritten to pretend their files never moved.
