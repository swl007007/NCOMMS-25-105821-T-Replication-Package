from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
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


def synthetic_complete_metrics() -> pd.DataFrame:
    rows = []
    country_counts = {0: 0, 5: 2, 10: 3}
    for experiment in ("feature_removal", "country_removal"):
        for threshold in sensitivity.THRESHOLDS:
            for task in sensitivity.TASKS:
                for model in sensitivity.MODELS:
                    removed_count = (
                        country_counts[threshold]
                        if experiment == "country_removal"
                        else 0
                    )
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
                            "removed_country_count": removed_count,
                            "removed_country_iso3": ";".join(
                                f"C{index:02d}" for index in range(removed_count)
                            ),
                            "overall_accuracy": 0.6,
                            "phase3plus_precision": 0.7,
                            "phase3plus_recall": 0.8,
                            "phase3plus_population_share_r2": (
                                np.nan if model == "Ordered Probit" else 0.1
                            ),
                            "n_train": 4460 if task == "Contemporaneous" else 4405,
                            "n_test": 5575 if task == "Contemporaneous" else 1170,
                            "status": "generated",
                            "reason": "",
                        }
                    )
                    rows.append(row)
    for task in sensitivity.TASKS:
        for model in sensitivity.MODELS:
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
                        np.nan if model == "Ordered Probit" else 0.1
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


def synthetic_removal_metrics() -> pd.DataFrame:
    rows = []
    removed_countries = {0: 0, 5: 2, 10: 3}
    for experiment in ("feature_removal", "country_removal"):
        for task in sensitivity.TEMPORAL_TASKS:
            for threshold in sensitivity.THRESHOLDS:
                for model_index, model in enumerate(sensitivity.MODELS):
                    value = 0.80 - threshold / 500 - model_index * 0.03
                    row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
                    row.update(
                        {
                            "experiment": experiment,
                            "threshold_percent": threshold,
                            "task": task,
                            "model": model,
                            "removed_feature_count": int(
                                math.ceil(threshold / 100 * 106)
                            ),
                            "removed_country_count": (
                                removed_countries[threshold]
                                if experiment == "country_removal"
                                else 0
                            ),
                            "removed_country_iso3": (
                                ";".join(
                                    f"C{index:02d}"
                                    for index in range(removed_countries[threshold])
                                )
                                if experiment == "country_removal"
                                else ""
                            ),
                            "overall_accuracy": value,
                            "phase3plus_precision": value - 0.02,
                            "phase3plus_recall": value + 0.02,
                            "phase3plus_population_share_r2": (
                                np.nan
                                if model == "Ordered Probit"
                                else value - 0.30
                            ),
                            "n_train": 4405 - removed_countries[threshold] * 10,
                            "n_test": 1170 - removed_countries[threshold] * 5,
                            "status": "generated",
                            "reason": "",
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows, columns=sensitivity.METRIC_COLUMNS)


class SelectionTests(unittest.TestCase):
    def test_feature_ranking_uses_training_only_and_source_order_for_ties(self):
        data = pd.DataFrame(
            {
                "a": [np.nan, 1.0, np.nan, np.nan],
                "b": [np.inf, 2.0, 3.0, np.nan],
                "c": [1.0, np.nan, 4.0, np.nan],
            }
        )
        train_mask = pd.Series([True, True, False, False])
        self.assertEqual(
            sensitivity.rank_features(data, ["a", "b", "c"], train_mask),
            ("a", "b", "c"),
        )

    def test_selection_count_uses_ceiling(self):
        self.assertEqual(sensitivity.selection_count(5, 29), 2)
        self.assertEqual(sensitivity.selection_count(10, 29), 3)
        self.assertEqual(sensitivity.selection_count(0, 29), 0)


class TransformationTests(unittest.TestCase):
    def test_country_ranking_combines_layer_feature_cells(self):
        layer1 = pd.DataFrame(
            {
                "country_code_3": ["AAA", "AAA", "BBB", "BBB"],
                "x": [np.nan, 1.0, np.nan, np.nan],
                "y": [1.0, 1.0, np.nan, 1.0],
            }
        )
        layer2 = pd.DataFrame(
            {
                "country_code_3": ["AAA", "AAA", "BBB", "BBB"],
                "z": [1.0, np.inf, np.nan, np.nan],
            }
        )
        train = pd.Series([True, True, True, True])
        ranked = sensitivity.rank_countries(
            [(layer1, ["x", "y"], train), (layer2, ["z"], train)]
        )
        self.assertEqual(ranked, ("BBB", "AAA"))

    def test_country_ranking_keeps_countries_without_training_rows_last(self):
        data = pd.DataFrame(
            {
                "country_code_3": ["AAA", "CCC", "BBB"],
                "x": [1.0, np.nan, np.nan],
            }
        )
        train = pd.Series([True, False, False])
        self.assertEqual(
            sensitivity.rank_countries([(data, ["x"], train)]),
            ("AAA", "BBB", "CCC"),
        )

    def test_suppress_features_does_not_mutate_source(self):
        source = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = sensitivity.suppress_features(source, ["x"])
        self.assertTrue(result["x"].isna().all())
        self.assertEqual(source["x"].tolist(), [1.0, 2.0])
        self.assertEqual(result["y"].tolist(), [3.0, 4.0])

    def test_indicators_capture_nan_and_infinity_before_imputation(self):
        source = pd.DataFrame({"x": [1.0, np.nan, np.inf, -np.inf]})
        result, indicators = sensitivity.add_missing_indicators(source, ["x"])
        self.assertEqual(indicators, ("x__missing",))
        self.assertEqual(result["x__missing"].tolist(), [0, 1, 1, 1])
        self.assertTrue(np.isinf(result.loc[[2, 3], "x"]).all())


class FitRoutingTests(unittest.TestCase):
    def test_prediction_frame_requires_equal_valid_phase_vectors(self):
        result = sensitivity.prediction_frame([1, 3, 5], [2, 3, 4])
        self.assertEqual(result.columns.tolist(), ["actual_phase", "predicted_phase"])
        with self.assertRaisesRegex(ValueError, "same length"):
            sensitivity.prediction_frame([1, 2], [1])
        with self.assertRaisesRegex(ValueError, "between one and five"):
            sensitivity.prediction_frame([1], [6])

    def test_safe_fit_records_known_non_estimable_error(self):
        def fail():
            raise ValueError("missing Ordered Probit class")

        predictions, n_train, n_test, status, reason = sensitivity.run_fit_safely(
            fail,
            n_train=20,
            n_test=5,
        )
        self.assertIsNone(predictions)
        self.assertEqual((n_train, n_test), (20, 5))
        self.assertEqual(status, "not_estimable")
        self.assertEqual(reason, "missing Ordered Probit class")

    def test_xgboost_forecasting_routes_through_authoritative_split_helper(self):
        forecasting = pd.DataFrame(
            {
                "area_id": [1, 1, 2],
                "date": ["2021-01-01", "2022-01-01", "2022-01-01"],
                "country_code_3": ["AAA", "AAA", "BBB"],
                "evaluation_phase": [1, 2, 3],
            }
        )
        train = pd.Series([True, False, False])
        test = ~train
        returned = pd.DataFrame(
            {
                "overall_phase": [2, 4],
                "overall_phase_pred": [3, 4],
                "phase3_test": [0.1, 0.6],
                "phase3_pred": [0.2, 0.5],
            }
        )
        with mock.patch.object(
            sensitivity.loco, "fit_forecasting_split", return_value=returned
        ) as fitted:
            result, n_train, n_test = sensitivity.fit_xgboost_task(
                task="Forecasting",
                forecasting=forecasting,
                nowcasting=forecasting.copy(),
                forecasting_train_mask=train,
                forecasting_test_mask=test,
                nowcasting_train_mask=train,
                nowcasting_test_mask=test,
                general_params={"max_depth": 1},
                phase3_params={"max_depth": 1},
            )
        fitted.assert_called_once()
        self.assertNotIn("evaluation_phase", fitted.call_args.args[0].columns)
        self.assertEqual((n_train, n_test), (1, 2))
        self.assertEqual(result["predicted_phase"].tolist(), [3, 4])
        self.assertEqual(result["phase3_pred_rounded"].tolist(), [0.2, 0.5])

    def test_ordered_probit_uses_supplied_direct_features(self):
        phases = np.array([1, 2, 3, 4, 5] * 2 + [2, 4])
        data = pd.DataFrame(
            {
                "date": ["2021-01-01"] * 10 + ["2022-01-01"] * 2,
                "x": np.arange(12, dtype=float),
                "x__missing": [0, 1] * 6,
            }
        )
        for phase in range(1, 6):
            data[f"phase{phase}_percent"] = (phases == phase).astype(float)
        train = pd.Series([True] * 10 + [False] * 2)
        test = ~train
        returned = (
            np.array([2, 4]),
            np.full((2, 5), 0.2),
            {"converged": True},
        )
        with mock.patch.object(
            sensitivity.simple, "fit_ordered_probit_arrays", return_value=returned
        ) as fitted:
            result, n_train, n_test = sensitivity.fit_ordered_probit_task(
                data=data,
                features=["x", "x__missing"],
                train_mask=train,
                test_mask=test,
                task="Forecasting",
            )
        self.assertEqual(fitted.call_args.args[0].shape[0], 10)
        self.assertEqual((n_train, n_test), (10, 2))
        self.assertEqual(result["predicted_phase"].tolist(), [2, 4])

    def test_ensemble_ols_nowcasting_adds_layer1_and_residual(self):
        phases = np.array([1, 2, 3, 4, 5, 3, 1])
        forecasting = pd.DataFrame(
            {
                "area_id": np.arange(1, 8),
                "date": ["2021-01-01"] * 5 + ["2022-01-01"] * 2,
                "country_code_3": ["AAA"] * 7,
                "x": np.arange(7, dtype=float),
                "evaluation_phase": phases,
            }
        )
        for phase in range(1, 6):
            forecasting[f"phase{phase}_percent"] = (phases == phase).astype(float)
        forecasting = sensitivity.loco.add_cumulative_targets(forecasting)
        nowcasting = forecasting[["area_id", "date", "country_code_3"]].copy()
        nowcasting["z"] = np.arange(10, 17, dtype=float)
        train = pd.Series([True] * 5 + [False] * 2)
        test = ~train
        effects = []
        for phase in range(2, 6):
            layer1_test = np.zeros(2)
            layer1_train = np.zeros(5)
            residual_test = np.array([0.25, 0.0]) if phase == 3 else np.zeros(2)
            effects.extend(
                [
                    (layer1_test, {}),
                    (layer1_train, {}),
                    (residual_test, {}),
                ]
            )
        with mock.patch.object(
            sensitivity.simple, "fit_ols_arrays", side_effect=effects
        ) as fitted:
            result, n_train, n_test = sensitivity.fit_ensemble_ols_task(
                task="Nowcasting",
                forecasting=forecasting,
                nowcasting=nowcasting,
                layer1_features=["x"],
                layer2_features=["z"],
                forecasting_train_mask=train,
                forecasting_test_mask=test,
                nowcasting_train_mask=train,
                nowcasting_test_mask=test,
            )
        self.assertEqual(fitted.call_count, 12)
        self.assertEqual((n_train, n_test), (5, 2))
        self.assertEqual(result["predicted_phase"].tolist(), [3, 1])


class LiveSourceContractTests(unittest.TestCase):
    def test_live_feature_and_country_contract(self):
        bundle = sensitivity.simple.load_prepared_inputs()
        contract = sensitivity.source_feature_contract(bundle)
        self.assertEqual(len(contract["forecast_direct"]), 106)
        self.assertEqual(len(contract["nowcast_direct"]), 173)
        self.assertEqual(len(contract["layer1"]), 106)
        self.assertEqual(len(contract["layer2"]), 69)
        self.assertEqual(len(contract["countries"]), 29)
        self.assertEqual(
            set(contract["forecast_source_to_layer1"]),
            set(contract["forecast_direct"]),
        )
        forecast_ranking = sensitivity.rank_countries(
            [
                (
                    bundle.forecasting,
                    contract["layer1"],
                    bundle.forecasting_train_mask,
                )
            ]
        )
        nowcast_ranking = sensitivity.rank_countries(
            [
                (
                    bundle.forecasting,
                    contract["layer1"],
                    bundle.forecasting_train_mask,
                ),
                (
                    bundle.nowcasting,
                    contract["layer2"],
                    bundle.nowcasting_train_mask,
                ),
            ]
        )
        self.assertEqual(len(forecast_ranking), 29)
        self.assertEqual(len(nowcast_ranking), 29)
        self.assertEqual(set(forecast_ranking), set(contract["countries"]))
        self.assertEqual(set(nowcast_ranking), set(contract["countries"]))

    def test_forecast_source_to_layer1_mapping_preserves_values(self):
        bundle = sensitivity.simple.load_prepared_inputs()
        contract = sensitivity.source_feature_contract(bundle)
        raw = bundle.raw_forecasting.sort_values(
            sensitivity.loco.KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        prepared = bundle.forecasting.sort_values(
            sensitivity.loco.KEY_COLUMNS, kind="mergesort"
        ).reset_index(drop=True)
        for source, layer1 in contract["forecast_source_to_layer1"].items():
            left = raw[source].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            right = (
                prepared[layer1]
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=float)
            )
            np.testing.assert_allclose(
                left, right, rtol=0.0, atol=0.0, equal_nan=True
            )


class ExperimentGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = sensitivity.simple.load_prepared_inputs()

    @staticmethod
    def successful_fit():
        return sensitivity.prediction_frame([1, 3], [1, 3]), 4405, 1170

    def test_temporal_feature_and_country_grids_have_eighteen_conditions_each(self):
        success = self.successful_fit()
        with mock.patch.object(
            sensitivity, "fit_xgboost_task", return_value=success
        ), mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", return_value=success
        ), mock.patch.object(
            sensitivity, "fit_ordered_probit_task", return_value=success
        ):
            feature = sensitivity.run_feature_removal(self.bundle, {}, {})
            country = sensitivity.run_country_removal(self.bundle, {}, {})
        self.assertEqual(len(feature), 18)
        self.assertEqual(len(country), 18)
        zero_feature = [row for row in feature if row["threshold_percent"] == 0]
        self.assertTrue(
            all(row["removed_feature_count"] == 0 for row in zero_feature)
        )
        expected_counts = {0: 0, 5: 2, 10: 3}
        for threshold, expected in expected_counts.items():
            rows = [
                row for row in country if row["threshold_percent"] == threshold
            ]
            self.assertTrue(
                all(row["removed_country_count"] == expected for row in rows)
            )
        grouped = pd.DataFrame(country).groupby(
            ["task", "threshold_percent"], observed=True
        )["removed_country_iso3"].nunique()
        self.assertTrue(grouped.eq(1).all())

    def test_indicator_grid_has_four_fits_and_never_calls_xgboost(self):
        success = self.successful_fit()
        with mock.patch.object(
            sensitivity, "fit_xgboost_task"
        ) as xgboost_fit, mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", return_value=success
        ), mock.patch.object(
            sensitivity, "fit_ordered_probit_task", return_value=success
        ):
            rows = sensitivity.run_missing_indicators(self.bundle)
        self.assertEqual(len(rows), 4)
        xgboost_fit.assert_not_called()
        self.assertEqual(
            {(row["task"], row["model"]) for row in rows},
            {
                ("Forecasting", "Ensemble OLS"),
                ("Nowcasting", "Ensemble OLS"),
                ("Forecasting", "Ordered Probit"),
                ("Nowcasting", "Ordered Probit"),
            },
        )

    def test_indicator_grid_uses_each_models_source_roles(self):
        success = self.successful_fit()
        with mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", return_value=success
        ) as ols_fit, mock.patch.object(
            sensitivity, "fit_ordered_probit_task", return_value=success
        ) as ordered_fit:
            sensitivity.run_missing_indicators(self.bundle)
        ols_calls = {
            call.kwargs["task"]: call.kwargs for call in ols_fit.call_args_list
        }
        ordered_calls = {
            call.kwargs["task"]: call.kwargs for call in ordered_fit.call_args_list
        }
        self.assertEqual(len(ols_calls["Forecasting"]["layer1_features"]), 212)
        self.assertEqual(len(ols_calls["Nowcasting"]["layer1_features"]), 212)
        self.assertEqual(len(ols_calls["Nowcasting"]["layer2_features"]), 138)
        self.assertEqual(len(ordered_calls["Forecasting"]["features"]), 212)
        self.assertEqual(len(ordered_calls["Nowcasting"]["features"]), 346)


class LiveRegressionTests(unittest.TestCase):
    def test_live_zero_percent_regression_matches_frozen_baselines(self):
        try:
            sensitivity.frozen_main_result.assert_frozen_environment(
                ("matplotlib", "statsmodels", "patsy")
            )
        except RuntimeError as error:
            self.skipTest(str(error))

        bundle = sensitivity.simple.load_prepared_inputs()
        contract = sensitivity.source_feature_contract(bundle)
        general_params, phase3_params = sensitivity.loco.load_hyperparameters(
            sensitivity.DEFAULT_GENERAL_PARAMS,
            sensitivity.DEFAULT_PHASE3_PARAMS,
            random_state=None,
            estimator_n_jobs=None,
        )
        frozen = pd.read_csv(
            SOURCE_CODE
            / "produced_graph"
            / "selected_figure1_repeated_area_refit_metrics.csv",
            float_precision="round_trip",
        )

        with mock.patch.object(
            sensitivity, "THRESHOLDS", (0,)
        ), mock.patch.object(
            sensitivity,
            "fit_xgboost_task",
            wraps=sensitivity.fit_xgboost_task,
        ) as xgboost_fit, mock.patch.object(
            sensitivity,
            "fit_ensemble_ols_task",
            wraps=sensitivity.fit_ensemble_ols_task,
        ) as ols_fit, mock.patch.object(
            sensitivity,
            "fit_ordered_probit_task",
            wraps=sensitivity.fit_ordered_probit_task,
        ) as ordered_fit:
            feature_raw = sensitivity.run_feature_removal(
                bundle, general_params, phase3_params
            )
        self.assertEqual(
            (xgboost_fit.call_count, ols_fit.call_count, ordered_fit.call_count),
            (2, 2, 2),
        )
        self.assertTrue(all(row["status"] == "generated" for row in feature_raw))
        cache = {
            (row["task"], row["model"]): (
                row["predictions"],
                row["n_train"],
                row["n_test"],
            )
            for row in feature_raw
        }

        def assert_prepared_inputs(kwargs):
            pd.testing.assert_frame_equal(kwargs["forecasting"], bundle.forecasting)
            pd.testing.assert_frame_equal(kwargs["nowcasting"], bundle.nowcasting)
            for name in (
                "forecasting_train_mask",
                "forecasting_test_mask",
                "nowcasting_train_mask",
                "nowcasting_test_mask",
            ):
                pd.testing.assert_series_equal(
                    kwargs[name], getattr(bundle, name), check_names=False
                )

        def cached_xgboost(**kwargs):
            assert_prepared_inputs(kwargs)
            self.assertEqual(kwargs["general_params"], general_params)
            self.assertEqual(kwargs["phase3_params"], phase3_params)
            return cache[(kwargs["task"], "Main model")]

        def cached_ols(**kwargs):
            assert_prepared_inputs(kwargs)
            self.assertEqual(tuple(kwargs["layer1_features"]), contract["layer1"])
            self.assertEqual(tuple(kwargs["layer2_features"]), contract["layer2"])
            return cache[(kwargs["task"], "Ensemble OLS")]

        raw_masks = {
            "Forecasting": sensitivity.simple.temporal_masks(
                bundle.raw_forecasting["date"]
            ),
            "Nowcasting": sensitivity.simple.temporal_masks(bundle.raw_nowcasting["date"]),
        }

        def cached_ordered(**kwargs):
            task = kwargs["task"]
            expected_data = (
                bundle.raw_forecasting if task == "Forecasting" else bundle.raw_nowcasting
            )
            expected_features = (
                contract["forecast_direct"]
                if task == "Forecasting"
                else contract["nowcast_direct"]
            )
            pd.testing.assert_frame_equal(kwargs["data"], expected_data)
            self.assertEqual(tuple(kwargs["features"]), expected_features)
            pd.testing.assert_series_equal(
                kwargs["train_mask"], raw_masks[task][0], check_names=False
            )
            pd.testing.assert_series_equal(
                kwargs["test_mask"], raw_masks[task][1], check_names=False
            )
            return cache[(task, "Ordered Probit")]

        with mock.patch.object(
            sensitivity, "THRESHOLDS", (0,)
        ), mock.patch.object(
            sensitivity, "fit_xgboost_task", side_effect=cached_xgboost
        ), mock.patch.object(
            sensitivity, "fit_ensemble_ols_task", side_effect=cached_ols
        ), mock.patch.object(
            sensitivity, "fit_ordered_probit_task", side_effect=cached_ordered
        ):
            country_raw = sensitivity.run_country_removal(
                bundle, general_params, phase3_params
            )

        for row in [*feature_raw, *country_raw]:
            self.assertEqual(row["removed_feature_count"], 0)
            self.assertEqual(row["removed_country_count"], 0)
            self.assertEqual(row["removed_country_iso3"], "")

        keys = ["task", "model"]
        comparison_columns = [
            *sensitivity.METRIC_NAMES,
            "n_train",
            "n_test",
            "status",
            "reason",
        ]
        feature = pd.DataFrame(map(sensitivity.record_metrics, feature_raw)).set_index(
            keys
        ).sort_index()
        country = pd.DataFrame(map(sensitivity.record_metrics, country_raw)).set_index(
            keys
        ).sort_index()
        pd.testing.assert_frame_equal(
            feature[comparison_columns],
            country[comparison_columns],
            check_exact=True,
        )

        baseline_method = {
            "Main model": "Main result",
            "Ensemble OLS": "Ensemble OLS",
            "Ordered Probit": "Ordered Probit",
        }
        for (task, model), row in feature.iterrows():
            expected = frozen.loc[
                frozen["task"].eq(task)
                & frozen["method"].eq(baseline_method[model])
            ]
            self.assertEqual(len(expected), 1)
            expected_row = expected.iloc[0]
            self.assertEqual((int(row["n_train"]), int(row["n_test"])), (4405, 1170))
            self.assertEqual(
                (int(expected_row["n_train"]), int(expected_row["n_test"])),
                (4405, 1170),
            )
            expected_values = expected_row[
                ["overall_accuracy", "phase3plus_precision", "phase3plus_recall"]
            ].tolist()
            expected_values.append(expected_row["phase3plus_continuous_r2"])
            np.testing.assert_allclose(
                row[list(sensitivity.METRIC_NAMES)].to_numpy(dtype=float),
                np.asarray(expected_values, dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )


class MetricTableTests(unittest.TestCase):
    def test_generated_record_uses_existing_pooled_metric_definitions(self):
        predictions = sensitivity.prediction_frame([1, 3, 4], [1, 2, 4])
        predictions["phase3_actual_cumulative"] = [0.0, 0.5, 1.0]
        predictions["phase3_pred_rounded"] = [0.0, 0.4, 0.8]
        raw = {
            "experiment": "feature_removal",
            "threshold_percent": 10,
            "task": "Forecasting",
            "model": "Ensemble OLS",
            "removed_feature_count": 11,
            "removed_country_count": 0,
            "removed_country_iso3": "",
            "n_train": 4405,
            "n_test": 1170,
            "status": "generated",
            "reason": "",
            "predictions": predictions,
        }
        row = sensitivity.record_metrics(raw)
        self.assertAlmostEqual(row["overall_accuracy"], 2 / 3)
        self.assertAlmostEqual(row["phase3plus_precision"], 1.0)
        self.assertAlmostEqual(row["phase3plus_recall"], 0.5)
        self.assertAlmostEqual(row["phase3plus_population_share_r2"], 0.9)

    def test_not_estimable_record_has_nan_metrics_and_keeps_counts(self):
        raw = {
            "experiment": "country_removal",
            "threshold_percent": 50,
            "task": "Nowcasting",
            "model": "Ordered Probit",
            "removed_feature_count": 0,
            "removed_country_count": 15,
            "removed_country_iso3": "AAA;BBB",
            "n_train": 100,
            "n_test": 20,
            "status": "not_estimable",
            "reason": "missing Ordered Probit class",
            "predictions": None,
        }
        row = sensitivity.record_metrics(raw)
        self.assertTrue(math.isnan(row["overall_accuracy"]))
        self.assertEqual(row["n_test"], 20)
        self.assertEqual(row["reason"], "missing Ordered Probit class")

    def test_frozen_main_references_cover_all_three_tasks(self):
        source = pd.DataFrame(
            {
                "overall_phase": [1, 2, 3, 4, 5],
                "contemporaneous_predict": [1, 2, 3, 4, 5],
                "phase3_actual": [0.0, 0.0, 0.3, 0.5, 0.8],
                "phase3_contemporaneous": [0.0, 0.0, 0.3, 0.5, 0.8],
                "fold": [0, 1, 2, 3, 4],
            }
        )
        references = sensitivity.load_frozen_main_references(source)
        self.assertEqual(len(references), 3)
        self.assertEqual({row["task"] for row in references}, set(sensitivity.TASKS))
        self.assertTrue(all(row["model"] == "Main model" for row in references))
        self.assertTrue(all(row["status"] == "frozen_reference" for row in references))

    def test_deltas_use_same_experiment_task_and_threshold_xgboost(self):
        rows = []
        for experiment, threshold, xgb_value, baseline_value in (
            ("feature_removal", 10, 0.8, 0.7),
            ("country_removal", 10, 0.6, 0.55),
            ("missing_indicators", np.nan, 0.75, 0.65),
        ):
            for model, value in (
                ("Main model", xgb_value),
                ("Ensemble OLS", baseline_value),
            ):
                row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
                row.update(
                    {
                        "experiment": experiment,
                        "threshold_percent": threshold,
                        "task": "Forecasting",
                        "model": model,
                        "removed_feature_count": 0,
                        "removed_country_count": 0,
                        "removed_country_iso3": "",
                        "n_train": 4405,
                        "n_test": 1170,
                        "status": (
                            "frozen_reference"
                            if experiment == "missing_indicators"
                            and model == "Main model"
                            else "generated"
                        ),
                        "reason": "",
                    }
                )
                for metric in sensitivity.METRIC_NAMES:
                    row[metric] = value
                rows.append(row)
        result = sensitivity.add_main_model_deltas(pd.DataFrame(rows))
        observed = result.loc[result["model"].eq("Ensemble OLS")].set_index(
            "experiment"
        )
        self.assertAlmostEqual(
            observed.loc[
                "feature_removal", "delta_overall_accuracy_vs_main_model"
            ],
            -0.1,
        )
        self.assertAlmostEqual(
            observed.loc[
                "country_removal", "delta_overall_accuracy_vs_main_model"
            ],
            -0.05,
        )
        self.assertAlmostEqual(
            observed.loc[
                "missing_indicators", "delta_overall_accuracy_vs_main_model"
            ],
            -0.1,
        )
        xgboost = result.loc[result["model"].eq("Main model")]
        self.assertTrue(
            xgboost["delta_overall_accuracy_vs_main_model"].eq(0.0).all()
        )

    def test_delta_is_nan_when_reference_metric_is_undefined(self):
        rows = []
        for model, value in (("Main model", np.nan), ("Ordered Probit", 0.4)):
            row = {column: np.nan for column in sensitivity.METRIC_COLUMNS}
            row.update(
                {
                    "experiment": "feature_removal",
                    "threshold_percent": 50,
                    "task": "Nowcasting",
                    "model": model,
                    "removed_feature_count": 88,
                    "removed_country_count": 0,
                    "removed_country_iso3": "",
                    "n_train": 4405,
                    "n_test": 1170,
                    "status": "generated",
                    "reason": "",
                    "overall_accuracy": value,
                }
            )
            rows.append(row)
        result = sensitivity.add_main_model_deltas(pd.DataFrame(rows))
        delta = result.loc[
            result["model"].eq("Ordered Probit"),
            "delta_overall_accuracy_vs_main_model",
        ].iloc[0]
        self.assertTrue(math.isnan(delta))

    def test_validate_metrics_accepts_the_exact_sixty_three_row_grid(self):
        sensitivity.validate_metrics(synthetic_complete_metrics())

    def test_validate_metrics_rejects_misplaced_frozen_reference(self):
        metrics = synthetic_complete_metrics()
        metrics.loc[0, "status"] = "frozen_reference"
        with self.assertRaisesRegex(ValueError, "Frozen reference"):
            sensitivity.validate_metrics(metrics)

    def test_validate_metrics_rejects_infinite_metric(self):
        metrics = synthetic_complete_metrics()
        metrics.loc[0, "phase3plus_population_share_r2"] = np.inf
        with self.assertRaisesRegex(ValueError, "infinite"):
            sensitivity.validate_metrics(metrics)

    def test_validate_metrics_rejects_reason_on_generated_row(self):
        metrics = synthetic_complete_metrics()
        metrics.loc[0, "reason"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "no reason"):
            sensitivity.validate_metrics(metrics)


class FigureAndOutputTests(unittest.TestCase):
    def test_figure_is_the_fixed_four_by_four_grid(self):
        metrics = synthetic_removal_metrics()
        figure = sensitivity.create_sensitivity_figure(metrics)
        self.assertEqual(len(figure.axes), 16)
        for row in range(4):
            for axis in figure.axes[row * 4 : row * 4 + 3]:
                self.assertEqual(axis.get_ylim(), (0.0, 1.0))
                self.assertEqual(len(axis.lines), 3)
        r2_axes = figure.axes[3::4]
        self.assertEqual(len({axis.get_ylim() for axis in r2_axes}), 1)
        for axis in r2_axes:
            low, high = axis.get_ylim()
            self.assertLess(low, 0.0)
            self.assertGreater(high, 0.0)
            self.assertEqual(len(axis.lines), 4)
            np.testing.assert_array_equal(axis.lines[3].get_ydata(), [0.0, 0.0])
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            ["Main model", "Ensemble OLS", "Ordered Probit"],
        )
        self.assertEqual(
            figure.axes[3].get_title(loc="left"),
            "Phase 3+ population-share R²",
        )
        country_axis = figure.axes[8]
        self.assertEqual(
            [label.get_text() for label in country_axis.get_xticklabels()],
            ["0", "2", "3"],
        )
        sensitivity.plt.close(figure)

    def test_not_estimable_metric_is_not_replaced_with_a_point(self):
        metrics = synthetic_removal_metrics()
        selector = (
            metrics["experiment"].eq("feature_removal")
            & metrics["task"].eq("Forecasting")
            & metrics["model"].eq("Ordered Probit")
            & metrics["threshold_percent"].eq(10)
        )
        metrics.loc[selector, "overall_accuracy"] = np.nan
        figure = sensitivity.create_sensitivity_figure(metrics)
        line = figure.axes[0].lines[2]
        self.assertTrue(np.isnan(line.get_ydata()[2]))
        sensitivity.plt.close(figure)

    def test_tick_labels_and_bottom_legend_do_not_overlap(self):
        figure = sensitivity.create_sensitivity_figure(synthetic_removal_metrics())
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            boxes = [
                label.get_window_extent(renderer)
                for label in axis.get_xticklabels()
                if label.get_visible() and label.get_text()
            ]
            for index, first in enumerate(boxes):
                for second in boxes[index + 1 :]:
                    self.assertFalse(first.overlaps(second))
        legend_box = figure.legends[0].get_window_extent(renderer)
        for axis in figure.axes[-4:]:
            self.assertFalse(
                legend_box.overlaps(axis.xaxis.label.get_window_extent(renderer))
            )
        sensitivity.plt.close(figure)

    def test_write_outputs_creates_only_the_two_metric_csvs(self):
        complete = synthetic_complete_metrics()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "sentinel.txt"
            sentinel.write_bytes(b"frozen")
            output_dir = root / "sensitivity"
            paths = sensitivity.write_outputs(complete, output_dir)
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    sensitivity.METRICS_FILENAME,
                    sensitivity.INDICATOR_METRICS_FILENAME,
                },
            )
            self.assertEqual(sentinel.read_bytes(), b"frozen")
            self.assertEqual(
                pd.read_csv(paths["metrics_csv"]).columns.tolist(),
                list(sensitivity.METRIC_COLUMNS),
            )
            self.assertEqual(
                pd.read_csv(paths["indicator_metrics_csv"]).columns.tolist(),
                list(sensitivity.INDICATOR_COLUMNS),
            )

    def test_cli_has_paths_but_no_model_or_threshold_controls(self):
        parser_result = sensitivity.parse_args([])
        self.assertEqual(parser_result.output_dir, sensitivity.DEFAULT_OUTPUT_DIR)
        self.assertFalse(hasattr(parser_result, "thresholds"))
        self.assertFalse(hasattr(parser_result, "models"))
        self.assertFalse(hasattr(parser_result, "optimizer"))
        self.assertFalse(hasattr(parser_result, "random_state"))
