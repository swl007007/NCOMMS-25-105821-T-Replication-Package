# Archived contemporaneous notebook — 16 September 2026

Table1_Contemporaneous_main.ipynb is preserved here unchanged as historical source.
It used an unseeded data shuffle and exported only the final fold.

The active replacement is 2.Source Code/generate_contemporaneous_main.py.
It reuses the existing generate_all_prediction_evaluation.py helper with seed 0,
the original default XGBoost thread policy and all 5,575 row-level OOF predictions.
It requires the frozen Windows Python 3.11.3 / XGBoost 2.0.3 environment.

Run from the package root:
    python "2.Source Code/generate_contemporaneous_main.py"

Outputs under 2.Source Code/produced_graph/:
- all_prediction_contemporaneous_random_cv_predictions.csv
- all_prediction_contemporaneous_random_cv_source_audit.csv
- contemporaneous_main_metrics.csv

Figure 1 continues to read the same saved prediction/audit filenames.
The random-area CV analysis remains a separate valid result.
inventory.csv preserves the original notebook path and SHA-256 for recovery.
