# Fresh-run verification — 16 September 2026

Command from repository root:

    python "2.Source Code/generate_contemporaneous_main.py" --output-dir "2.Source Code/produced_graph/contemporaneous_main_verification_20260916"

Executed with Windows Python 3.11.3 / XGBoost 2.0.3, fixed seed 0 and default
XGBoost thread setting. All 5,575 freshly fitted OOF predictions are byte-identical
to the existing Figure 1 sidecar. Accuracy/recall/precision/R-squared:
0.693 / 0.904 / 0.797 / 0.642. Exact values are in contemporaneous_main_metrics.csv.
Original Figure 1 results and shared generator were preserved.

Recheck without model fitting:

    python3 -B "2.Source Code/produced_graph/contemporaneous_main_verification_20260916/verify_saved_results.py"
