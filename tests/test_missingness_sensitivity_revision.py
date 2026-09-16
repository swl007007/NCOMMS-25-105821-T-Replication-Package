from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

try:
    import xgboost  # noqa: F401
except ImportError:
    xgboost_stub = types.ModuleType("xgboost")
    xgboost_stub.XGBRegressor = object
    xgboost_stub.core = types.SimpleNamespace(XGBoostError=RuntimeError)
    sys.modules["xgboost"] = xgboost_stub


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE = REPO_ROOT / "2.Source Code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))

import generate_missingness_sensitivity as sensitivity


def complete_internal_metrics() -> pd.DataFrame:
    rows = []
    for experiment in ("feature_removal", "country_removal"):
        for threshold in (0, 5, 10):
            for task in ("Forecasting", "Nowcasting", "Contemporaneous"):
                for model in ("Main model", "Ensemble OLS", "Ordered Probit"):
                    row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
                    row.update(
                        {
                            "experiment": experiment,
                            "threshold_percent": threshold,
                            "task": task,
                            "model": model,
                            "removed_feature_count": (
                                {
                                    "Forecasting": {0: 0, 5: 6, 10: 11},
                                    "Nowcasting": {0: 0, 5: 9, 10: 18},
                                    "Contemporaneous": {0: 0, 5: 9, 10: 18},
                                }[task][threshold]
                                if experiment == "feature_removal"
                                else 0
                            ),
                            "removed_country_count": (
                                {0: 0, 5: 2, 10: 3}[threshold]
                                if experiment == "country_removal"
                                else 0
                            ),
                            "removed_country_iso3": (
                                ";".join(["AAA", "BBB", "CCC"][: {0: 0, 5: 2, 10: 3}[threshold]])
                                if experiment == "country_removal"
                                else ""
                            ),
                            "overall_accuracy": 0.6,
                            "phase3plus_precision": 0.7,
                            "phase3plus_recall": 0.8,
                            "phase3plus_population_share_r2": (
                                np.nan if model == "Ordered Probit" else 0.2
                            ),
                            "n_train": 4460 if task == "Contemporaneous" else 4405,
                            "n_test": 5575 if task == "Contemporaneous" else 1170,
                            "status": "generated",
                            "reason": "",
                        }
                    )
                    rows.append(row)
    for task in ("Forecasting", "Nowcasting", "Contemporaneous"):
        for model in ("Main model", "Ensemble OLS", "Ordered Probit"):
            row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
            row.update(
                {
                    "experiment": "missing_indicators",
                    "threshold_percent": np.nan,
                    "task": task,
                    "model": model,
                    "removed_feature_count": 0,
                    "removed_country_count": 0,
                    "removed_country_iso3": "",
                    "overall_accuracy": 0.6,
                    "phase3plus_precision": 0.7,
                    "phase3plus_recall": 0.8,
                    "phase3plus_population_share_r2": (
                        np.nan if model == "Ordered Probit" else 0.2
                    ),
                    "n_train": 4460 if task == "Contemporaneous" else 4405,
                    "n_test": 5575 if task == "Contemporaneous" else 1170,
                    "status": (
                        "frozen_reference" if model == "Main model" else "generated"
                    ),
                    "reason": "",
                }
            )
            rows.append(row)
    return sensitivity.add_main_model_deltas(
        pd.DataFrame(rows, columns=sensitivity.METRIC_COLUMNS)
    )


class RevisedMetricContractTests(unittest.TestCase):
    def test_generated_metrics_use_continuous_phase3plus_population_share_r2(self):
        predictions = sensitivity.prediction_frame([1, 3, 4], [1, 2, 4])
        predictions["phase3_actual_cumulative"] = [0.0, 0.5, 1.0]
        predictions["phase3_pred_rounded"] = [0.0, 0.4, 0.8]
        record = {
            "experiment": "feature_removal",
            "threshold_percent": 0,
            "task": "Forecasting",
            "model": "Main model",
            "removed_feature_count": 0,
            "removed_country_count": 0,
            "removed_country_iso3": "",
            "n_train": 4405,
            "n_test": 1170,
            "status": "generated",
            "reason": "",
            "predictions": predictions,
        }
        row = sensitivity.record_metrics(record)
        self.assertAlmostEqual(row["phase3plus_population_share_r2"], 0.9)
        self.assertNotIn("phase3above_r2", row)

        record["model"] = "Ordered Probit"
        ordered = sensitivity.record_metrics(record)
        self.assertTrue(math.isnan(ordered["phase3plus_population_share_r2"]))

    def test_zero_percent_condition_cache_reuses_one_fit(self):
        calls = 0

        def fit():
            nonlocal calls
            calls += 1
            return sensitivity.prediction_frame([1], [1]), 10, 2

        cache = {}
        kwargs = {
            "threshold_percent": 0,
            "task": "Forecasting",
            "model": "Main model",
            "removed_feature_count": 0,
            "removed_country_count": 0,
            "removed_country_iso3": "",
            "n_train": 10,
            "n_test": 2,
            "fit": fit,
            "fit_cache": cache,
            "fit_cache_key": ("Forecasting", "Main model"),
        }
        sensitivity._condition_record(experiment="feature_removal", **kwargs)
        sensitivity._condition_record(experiment="country_removal", **kwargs)
        self.assertEqual(calls, 1)

    def test_internal_grid_writes_only_54_row_removal_and_9_row_indicator_csvs(self):
        metrics = complete_internal_metrics()
        sensitivity.validate_metrics(metrics)
        with tempfile.TemporaryDirectory() as directory:
            paths = sensitivity.write_outputs(metrics, Path(directory))
            self.assertEqual(
                {path.name for path in Path(directory).iterdir()},
                {
                    "missingness_sensitivity_metrics.csv",
                    "missing_indicator_baseline_comparison_metrics.csv",
                },
            )
            removal = pd.read_csv(paths["metrics_csv"])
            indicator = pd.read_csv(paths["indicator_metrics_csv"])
        self.assertEqual(len(removal), 54)
        self.assertEqual(len(indicator), 9)
        self.assertEqual(
            set(removal["threshold_percent"].astype(int)), {0, 5, 10}
        )
        self.assertEqual(
            set(removal["task"]), {"Forecasting", "Nowcasting", "Contemporaneous"}
        )
        self.assertNotIn("phase3above_r2", removal.columns)
        self.assertIn("phase3plus_population_share_r2", indicator.columns)

    def test_validator_rejects_wrong_feature_removal_count(self):
        metrics = complete_internal_metrics()
        selector = (
            metrics["experiment"].eq("feature_removal")
            & metrics["task"].eq("Nowcasting")
            & metrics["threshold_percent"].eq(5)
        )
        metrics.loc[selector, "removed_feature_count"] = 8
        with self.assertRaisesRegex(ValueError, "Feature-removal count"):
            sensitivity.validate_metrics(metrics)


class ContemporaneousFoldTests(unittest.TestCase):
    def test_feature_ranking_uses_only_each_folds_training_rows(self):
        design = pd.DataFrame(
            {
                "area_id": np.arange(10),
                "date": pd.date_range("2020-01-01", periods=10, freq="MS"),
                "country_code_3": ["AAA"] * 10,
                "overall_phase": [1, 2, 3, 4, 5] * 2,
                "source_row_index": np.arange(10),
                "fold": np.repeat(np.arange(5), 2),
                "shuffle_position": np.arange(10),
                "a": [np.nan, np.nan] + [1.0] * 8,
                "b": [1.0, 1.0, np.nan] + [1.0] * 7,
                "kfolds": np.repeat(np.arange(5), 2),
            }
        )
        for phase in range(2, 6):
            design[f"phase{phase}_worse"] = np.linspace(0.0, 0.9, 10)

        all_missing_columns = []

        class FakeRegressor:
            def __init__(self, **params):
                pass

            def fit(self, x, y):
                all_missing_columns.append(tuple(x.columns[x.isna().all()].tolist()))
                return self

            def predict(self, x):
                return np.zeros(len(x))

        with mock.patch.object(sensitivity.xgb, "XGBRegressor", FakeRegressor):
            sensitivity.fit_contemporaneous_task(
                model="Main model",
                design=design,
                feature_columns=("a", "b", "kfolds"),
                params={},
                feature_removal_percent=5,
            )

        self.assertEqual(all_missing_columns[0], ("b",))


if __name__ == "__main__":
    unittest.main()
