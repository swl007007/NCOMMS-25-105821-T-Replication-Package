"""Reproduce Figure 1's contemporaneous row-level five-fold CV results.

Uses seed 0, all 5,575 rows, the saved general hyperparameters, and XGBoost's
default thread setting. Run with the frozen Windows Python 3.11.3 / XGBoost
2.0.3 environment. The separate random-area CV analysis has its own generator.
"""
from pathlib import Path
import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, r2_score

from generate_all_prediction_evaluation import (
    DEFAULT_OUTPUT_DIR,
    write_contemporaneous_random_cv_artifacts,
)
from main_result_figure1_v1 import assert_frozen_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    assert_frozen_environment()
    paths = write_contemporaneous_random_cv_artifacts(
        output_dir=args.output_dir, random_state=0, estimator_n_jobs=None
    )
    predictions = pd.read_csv(
        paths["contemporaneous_predictions_csv"], float_precision="round_trip"
    )
    actual = predictions["overall_phase"]
    predicted = predictions["contemporaneous_predict"]
    metrics = pd.DataFrame([{
        "model": "Contemporaneous",
        "evaluation_protocol": "random_5fold_row_cv",
        "aggregation": "pooled_oof",
        "n": len(predictions),
        "accuracy": accuracy_score(actual, predicted),
        "recall": recall_score(actual.ge(3), predicted.ge(3)),
        "precision": precision_score(actual.ge(3), predicted.ge(3)),
        "phase3plus_r2": r2_score(
            predictions["phase3_actual"], predictions["phase3_contemporaneous"]
        ),
    }])
    paths["metrics_csv"] = args.output_dir / "contemporaneous_main_metrics.csv"
    metrics.to_csv(paths["metrics_csv"], index=False, float_format="%.17g")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
