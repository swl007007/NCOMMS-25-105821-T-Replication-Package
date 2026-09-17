# LOCO Windows XGBoost 2.0.3 rerun comparison

Tolerance: absolute unrounded difference <= 0.001. Display: three decimals.
Both undefined cells match; a one-sided undefined value fails. No rows were dropped.
Historical sources remain unchanged in archives. Both rerun families were promoted to produced_graph on 2026-09-17.

## Primary pooled metrics

| Family | Model | Metric | Historical | Rerun | Absolute difference | Pass |
| --- | --- | --- | ---: | ---: | ---: | --- |
| strict_temporal | Forecasting | accuracy | 0.652 | 0.658 | 0.005983 | FAIL |
| strict_temporal | Forecasting | precision | 0.774 | 0.773 | 0.001479 | FAIL |
| strict_temporal | Forecasting | recall | 0.949 | 0.960 | 0.011377 | FAIL |
| strict_temporal | Forecasting | R2(p3) | -0.014 | -0.004 | 0.009384 | FAIL |
| strict_temporal | Nowcasting | accuracy | 0.653 | 0.665 | 0.011966 | FAIL |
| strict_temporal | Nowcasting | precision | 0.777 | 0.781 | 0.003178 | FAIL |
| strict_temporal | Nowcasting | recall | 0.950 | 0.964 | 0.013652 | FAIL |
| strict_temporal | Nowcasting | R2(p3) | -0.007 | 0.008 | 0.014842 | FAIL |
| ordinary | Forecasting | accuracy | 0.558 | 0.578 | 0.019372 | FAIL |
| ordinary | Forecasting | precision | 0.662 | 0.666 | 0.004101 | FAIL |
| ordinary | Forecasting | recall | 0.906 | 0.942 | 0.036668 | FAIL |
| ordinary | Forecasting | R2(p3) | 0.103 | 0.131 | 0.027674 | FAIL |
| ordinary | Nowcasting | accuracy | 0.566 | 0.583 | 0.017578 | FAIL |
| ordinary | Nowcasting | precision | 0.666 | 0.671 | 0.005170 | FAIL |
| ordinary | Nowcasting | recall | 0.918 | 0.941 | 0.022881 | FAIL |
| ordinary | Nowcasting | R2(p3) | 0.118 | 0.136 | 0.018240 | FAIL |

## Separate primary and detailed verdicts

- strict_temporal / primary: FAIL; 8/8 failures; max delta 0.014842; both undefined 0; definition changes 0.
- strict_temporal / detail: FAIL; 1027/5168 failures; max delta 84.0000031; both undefined 948; definition changes 88.
- ordinary / primary: FAIL; 8/8 failures; max delta 0.036668; both undefined 0; definition changes 0.
- ordinary / detail: FAIL; 156/232 failures; max delta 4.575000; both undefined 10; definition changes 2.

Full precision comparisons: metric_comparison.csv. Machine-readable checks: comparison_summary.json.
Full population and truth keys verified; primary metrics independently recomputed from saved predictions.

## Provenance limitation

Current shared LOCO script hash differs from historical audits; this is current-code reproduction, not an isolated environment ablation.

Training environment and exact commands: run_context.json. Logs: strict_temporal.log and ordinary.log.
Recheck with: python3 -B compare_results.py (requires NumPy; does not retrain).
