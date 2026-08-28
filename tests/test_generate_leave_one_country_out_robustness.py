import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_leave_one_country_out_robustness as loco


def synthetic_model_data(nowcasting=False):
    rows = [
        (1, "2020-01-01", 3, [0.80, 0.00, 0.20, 0.00, 0.00], 10.0),
        (1, "2020-02-01", 4, [0.80, 0.00, 0.00, 0.20, 0.00], 11.0),
        (2, "2020-01-01", 1, [1.00, 0.00, 0.00, 0.00, 0.00], 20.0),
        (2, "2020-02-01", 2, [0.80, 0.20, 0.00, 0.00, 0.00], 21.0),
    ]
    records = []
    for area_id, date, phase, shares, signal in rows:
        record = {
            "area_id": area_id,
            "date": date,
            "overall_phase": phase,
            "signal": signal,
            "fews_ipc_ha": phase,
            "infra_index_m12": signal / 10,
            "infra_index_s12": signal / 20,
        }
        record.update({f"phase{i}_percent": shares[i - 1] for i in range(1, 6)})
        if nowcasting:
            record["current_signal"] = signal + 100
        records.append(record)
    return pd.DataFrame(records)


def synthetic_lookup():
    return pd.DataFrame(
        {"area_id": [1, 2], "country_code_3": ["AAA", "BBB"]}
    )


class RecordingMeanRegressor:
    created = []

    def __init__(self, **params):
        self.params = params
        self.fit_index = None
        self.fit_columns = None
        self.mean_ = 0.0
        type(self).created.append(self)

    def fit(self, X, y):
        self.fit_index = list(X.index)
        self.fit_columns = list(X.columns)
        self.mean_ = float(pd.Series(y).mean())
        return self

    def predict(self, X):
        return pd.Series(self.mean_, index=X.index).to_numpy()


def synthetic_prepared_inputs(include_nowcast_features=False):
    forecasting = synthetic_model_data()
    nowcasting = synthetic_model_data(nowcasting=True)
    if include_nowcast_features:
        for offset, column in enumerate(loco.NOWCAST_FEATURES):
            nowcasting[column] = nowcasting["signal"] + offset / 1000
    return loco.prepare_model_inputs(forecasting, nowcasting, synthetic_lookup())


class SequentialRecordingRegressor:
    def __init__(self, call_index, **params):
        self.call_index = call_index
        self.params = params
        self.fit_X = None
        self.fit_y = None

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = pd.Series(y).reset_index(drop=True)
        return self

    def predict(self, X):
        return pd.Series(0.0, index=X.index).to_numpy()


class SequentialRecordingFactory:
    def __init__(self):
        self.models = []

    def __call__(self, **params):
        model = SequentialRecordingRegressor(len(self.models), **params)
        self.models.append(model)
        return model

    @property
    def layer2_models(self):
        return self.models[1::2]

    @property
    def max_depth_sequence(self):
        return [model.params["max_depth"] for model in self.models]


def _perfect_fold_predictions(data, held_out_country):
    held = loco.add_cumulative_targets(
        data.loc[data["country_code_3"] == held_out_country].copy()
    )
    result = held[["area_id", "date", "country_code_3"]].copy()
    result["source_row_index"] = held.index.to_numpy()
    result["fold_country"] = held_out_country
    result["source_overall_phase"] = held["overall_phase"].to_numpy()
    for phase in range(2, 6):
        result[f"phase{phase}_test"] = held[f"phase{phase}_worse"].to_numpy()
        result[f"phase{phase}_pred"] = held[f"phase{phase}_worse"].to_numpy()
    return loco.wide_predictions_to_phases(result)


def fake_forecasting_fold(data, held_out_country, *args, **kwargs):
    return _perfect_fold_predictions(data, held_out_country)


def fake_nowcasting_fold(
    forecasting, nowcasting, held_out_country, *args, **kwargs
):
    result = _perfect_fold_predictions(forecasting, held_out_country)
    result["phase3_layer1_pred"] = result["phase3_pred"]
    result["phase3_residual_pred"] = 0.0
    return result


class LeaveOneCountryOutTests(unittest.TestCase):
    def test_load_hyperparameters_can_preserve_estimator_default_n_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            general_path = Path(temp_dir) / "general.json"
            phase3_path = Path(temp_dir) / "phase3.json"
            general_path.write_text(json.dumps({"max_depth": 11}), encoding="utf-8")
            phase3_path.write_text(json.dumps({"max_depth": 9}), encoding="utf-8")

            general, phase3 = loco.load_hyperparameters(
                general_path,
                phase3_path,
                random_state=0,
                estimator_n_jobs=None,
            )

        self.assertEqual(general["random_state"], 0)
        self.assertEqual(phase3["random_state"], 0)
        self.assertNotIn("n_jobs", general)
        self.assertNotIn("n_jobs", phase3)

    def test_normalize_country_lookup_deduplicates_and_sorts(self):
        raw = pd.DataFrame(
            {
                "area_id": [2, 1, 2],
                "country_code_3": ["BBB", "AAA", "BBB"],
            }
        )

        observed = loco.normalize_country_lookup(raw)

        expected = pd.DataFrame(
            {"area_id": [1, 2], "country_code_3": ["AAA", "BBB"]}
        )
        pd.testing.assert_frame_equal(observed, expected)

    def test_normalize_country_lookup_rejects_conflicts(self):
        raw = pd.DataFrame(
            {"area_id": [1, 1], "country_code_3": ["AAA", "BBB"]}
        )

        with self.assertRaisesRegex(ValueError, "multiple countries"):
            loco.normalize_country_lookup(raw)

    def test_export_country_lookup_writes_only_two_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.csv"
            output = Path(temp_dir) / "lookup.csv"
            pd.DataFrame(
                {
                    "area_id": [2, 1],
                    "country_code_3": ["BBB", "AAA"],
                    "unused": [10, 20],
                }
            ).to_csv(source, index=False)

            observed = loco.export_country_lookup(source, output)

            self.assertEqual(
                observed.columns.tolist(), ["area_id", "country_code_3"]
            )
            pd.testing.assert_frame_equal(pd.read_csv(output), observed)

    def test_prepare_model_inputs_joins_by_key_not_row_position(self):
        forecasting = synthetic_model_data().iloc[[1, 0, 3, 2]].reset_index(
            drop=True
        )
        nowcasting = synthetic_model_data(nowcasting=True).iloc[
            [2, 3, 0, 1]
        ].reset_index(drop=True)

        prepared_forecasting, prepared_nowcasting = loco.prepare_model_inputs(
            forecasting, nowcasting, synthetic_lookup()
        )

        expected_keys = [
            (1, "2020-01-01"),
            (1, "2020-02-01"),
            (2, "2020-01-01"),
            (2, "2020-02-01"),
        ]
        self.assertEqual(
            list(zip(prepared_forecasting.area_id, prepared_forecasting.date)),
            expected_keys,
        )
        self.assertEqual(
            list(zip(prepared_nowcasting.area_id, prepared_nowcasting.date)),
            expected_keys,
        )
        self.assertNotIn("infra_index_m12", prepared_forecasting.columns)
        self.assertIn("infra_index_m12_l12", prepared_forecasting.columns)
        self.assertIn("infra_index_m12", prepared_nowcasting.columns)

    def test_country_masks_exclude_every_held_country_row_from_training(self):
        countries = pd.Series(["AAA", "AAA", "BBB", "CCC"])

        train_mask, test_mask = loco.country_masks(countries, "AAA")

        self.assertEqual(train_mask.tolist(), [False, False, True, True])
        self.assertEqual(test_mask.tolist(), [True, True, False, False])
        self.assertFalse((train_mask & test_mask).any())

    def test_wide_predictions_use_highest_phase_at_twenty_percent(self):
        wide = pd.DataFrame(
            {
                "phase2_test": [0.20, 0.30],
                "phase3_test": [0.20, 0.10],
                "phase4_test": [0.20, 0.00],
                "phase5_test": [0.20, 0.00],
                "phase2_pred": [0.204, 0.19],
                "phase3_pred": [0.204, 0.19],
                "phase4_pred": [0.204, 0.19],
                "phase5_pred": [0.204, 0.19],
            }
        )

        observed = loco.wide_predictions_to_phases(wide)

        self.assertEqual(observed["overall_phase"].tolist(), [5, 2])
        self.assertEqual(observed["overall_phase_pred"].tolist(), [5, 1])

    def test_nonpositive_cumulative_predictions_are_retained_and_flagged(self):
        wide = pd.DataFrame(
            {
                "phase2_test": [0.20],
                "phase3_test": [0.20],
                "phase4_test": [0.00],
                "phase5_test": [0.00],
                "phase2_pred": [-0.10],
                "phase3_pred": [-0.10],
                "phase4_pred": [0.00],
                "phase5_pred": [0.00],
            }
        )

        observed = loco.wide_predictions_to_phases(wide)

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed.loc[0, "overall_phase_pred"], 1)
        self.assertTrue(
            bool(observed.loc[0, "nonpositive_cumulative_prediction_sum"])
        )

    def test_metrics_leave_zero_denominators_undefined(self):
        predictions = pd.DataFrame(
            {
                "overall_phase": [1, 2],
                "overall_phase_pred": [1, 2],
                "phase3_test": [0.0, 0.0],
                "phase3_pred": [0.0, 0.1],
            }
        )

        metrics = loco.calculate_country_metrics(
            predictions, "Forecasting", "AAA"
        )

        self.assertTrue(pd.isna(metrics["phase3plus_precision"]))
        self.assertTrue(pd.isna(metrics["phase3plus_recall"]))
        self.assertEqual(
            metrics["precision_undefined_reason"], "no_predicted_phase3plus"
        )
        self.assertEqual(
            metrics["recall_undefined_reason"], "no_actual_phase3plus"
        )
        self.assertEqual(
            metrics["r2_undefined_reason"],
            "constant_actual_phase3plus_share",
        )

    def test_area_macro_helper_remains_available_as_a_diagnostic(self):
        forecasting = pd.DataFrame(
            {
                "area_id": [1, 1, 2, 2, 2],
                "date": [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                ],
                "overall_phase": [3, 1, 1, 1, 1],
                "overall_phase_pred": [3, 3, 1, 1, 1],
                "phase3_test": [0.4, 0.0, 0.0, 0.0, 0.0],
                "phase3_pred": [0.3, 0.1, 0.0, 0.0, 0.0],
            }
        )
        nowcasting = forecasting.copy()
        nowcasting.loc[nowcasting["area_id"].eq(1), "overall_phase_pred"] = [1, 1]
        nowcasting.loc[nowcasting["area_id"].eq(1), "phase3_pred"] = [0.2, 0.0]

        observed = loco.calculate_area_macro_metrics(forecasting, nowcasting)

        expected = pd.DataFrame(
            {
                "accuracy": [0.75, 0.75],
                "precision": [float("nan"), 0.5],
                "recall": [0.0, 1.0],
                "R2(p3)": [0.5, 0.75],
            },
            index=pd.Index(["Nowcasting", "Forecasting"], name="model"),
        )
        pd.testing.assert_frame_equal(observed, expected)

    def test_micro_metrics_pool_rows_instead_of_equal_weighting_areas(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 2, 2, 2],
                "date": [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                ],
                "overall_phase": [3, 3, 1, 1],
                "overall_phase_pred": [1, 3, 1, 1],
                "phase3_test": [0.4, 0.3, 0.0, 0.0],
                "phase3_pred": [0.0, 0.3, 0.0, 0.0],
            }
        )

        micro = loco.calculate_micro_metrics(predictions, predictions)
        area_macro = loco.calculate_area_macro_metrics(predictions, predictions)

        self.assertAlmostEqual(micro.loc["Forecasting", "accuracy"], 0.75)
        self.assertAlmostEqual(area_macro.loc["Forecasting", "accuracy"], 0.5)
        self.assertAlmostEqual(micro.loc["Forecasting", "precision"], 1.0)
        self.assertAlmostEqual(micro.loc["Forecasting", "recall"], 0.5)

    def test_existing_predictions_can_regenerate_micro_table(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 2, 2, 2],
                "date": [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                ],
                "overall_phase": [3, 3, 1, 1],
                "overall_phase_pred": [1, 3, 1, 1],
                "phase3_test": [0.4, 0.3, 0.0, 0.0],
                "phase3_pred": [0.0, 0.3, 0.0, 0.0],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            predictions.to_csv(
                output_dir / "leave_one_country_out_forecasting_predictions.csv",
                index=False,
            )
            predictions.to_csv(
                output_dir / "leave_one_country_out_nowcasting_predictions.csv",
                index=False,
            )

            output_path = loco.aggregate_existing_loco_predictions(output_dir)
            observed = pd.read_csv(output_path, index_col="model")

        self.assertEqual(
            output_path.name, "leave_one_country_out_micro_metrics.csv"
        )
        self.assertEqual(observed.shape, (2, 4))
        self.assertEqual(observed.index.tolist(), ["Nowcasting", "Forecasting"])
        self.assertEqual(
            observed.columns.tolist(), ["accuracy", "precision", "recall", "R2(p3)"]
        )
        self.assertAlmostEqual(observed.loc["Forecasting", "accuracy"], 0.75)
        self.assertAlmostEqual(observed.loc["Forecasting", "precision"], 1.0)
        self.assertAlmostEqual(observed.loc["Forecasting", "recall"], 0.5)

    def test_forecasting_fold_excludes_country_and_preserves_parameter_carryover(
        self,
    ):
        RecordingMeanRegressor.created = []
        data, _ = loco.prepare_model_inputs(
            synthetic_model_data(),
            synthetic_model_data(nowcasting=True),
            synthetic_lookup(),
        )
        data = loco.add_cumulative_targets(data)

        observed = loco.fit_forecasting_fold(
            data,
            "AAA",
            {"max_depth": 11, "random_state": 0},
            {"max_depth": 9, "random_state": 0},
            estimator_factory=RecordingMeanRegressor,
        )

        self.assertEqual(set(observed["country_code_3"]), {"AAA"})
        self.assertEqual(len(observed), 2)
        self.assertEqual(
            [model.params["max_depth"] for model in RecordingMeanRegressor.created],
            [11, 9, 9, 9],
        )
        for model in RecordingMeanRegressor.created:
            self.assertTrue(
                all(
                    data.loc[index, "country_code_3"] != "AAA"
                    for index in model.fit_index
                )
            )
            self.assertIn("fews_ipc_ha", model.fit_columns)

    def test_forecasting_split_excludes_multiple_areas_and_keeps_country_codes(self):
        RecordingMeanRegressor.created = []
        forecasting, _ = synthetic_prepared_inputs()
        extra = forecasting.loc[forecasting["area_id"].eq(2)].copy()
        extra["area_id"] = 3
        forecasting = pd.concat([forecasting, extra], ignore_index=True)
        forecasting = loco.add_cumulative_targets(forecasting)
        test_mask = forecasting["area_id"].isin([1, 2])
        train_mask = ~test_mask

        observed = loco.fit_forecasting_split(
            forecasting,
            train_mask,
            test_mask,
            "area_holdout_seed0",
            {"max_depth": 11, "random_state": 0},
            {"max_depth": 9, "random_state": 0},
            fold_column="fold_id",
            estimator_factory=RecordingMeanRegressor,
        )

        self.assertEqual(set(observed["area_id"]), {1, 2})
        self.assertEqual(set(observed["country_code_3"]), {"AAA", "BBB"})
        self.assertEqual(set(observed["fold_id"]), {"area_holdout_seed0"})
        for model in RecordingMeanRegressor.created:
            self.assertTrue(
                all(forecasting.loc[index, "area_id"] == 3 for index in model.fit_index)
            )

    def test_nowcasting_fold_aligns_in_sample_residuals_by_key(self):
        forecasting, nowcasting = synthetic_prepared_inputs(
            include_nowcast_features=True
        )
        forecasting = loco.add_cumulative_targets(forecasting)
        nowcasting = nowcasting.iloc[::-1].reset_index(drop=True)
        factory = SequentialRecordingFactory()

        observed = loco.fit_nowcasting_fold(
            forecasting,
            nowcasting,
            "AAA",
            {"max_depth": 11, "random_state": 0},
            {"max_depth": 9, "random_state": 0},
            estimator_factory=factory,
        )

        self.assertEqual(len(observed), 2)
        self.assertEqual(set(observed["country_code_3"]), {"AAA"})
        self.assertEqual(len(factory.layer2_models), 4)
        self.assertEqual(
            factory.max_depth_sequence,
            [11, 11, 9, 9, 9, 9, 9, 9],
        )
        train = forecasting.loc[forecasting["country_code_3"] != "AAA"]
        for phase, model in zip(range(2, 6), factory.layer2_models):
            expected_by_signal = train.set_index("signal")[f"phase{phase}_worse"]
            observed_signal = model.fit_X["CPI"].round(0)
            expected = observed_signal.map(expected_by_signal).reset_index(drop=True)
            pd.testing.assert_series_equal(
                model.fit_y, expected, check_names=False, check_dtype=False
            )

    def test_nowcasting_split_uses_separate_keyed_masks_for_multiple_areas(self):
        forecasting, nowcasting = synthetic_prepared_inputs(
            include_nowcast_features=True
        )
        forecast_extra = forecasting.loc[forecasting["area_id"].eq(2)].copy()
        forecast_extra["area_id"] = 3
        nowcast_extra = nowcasting.loc[nowcasting["area_id"].eq(2)].copy()
        nowcast_extra["area_id"] = 3
        forecasting = pd.concat([forecasting, forecast_extra], ignore_index=True)
        nowcasting = pd.concat([nowcasting, nowcast_extra], ignore_index=True)
        forecasting = loco.add_cumulative_targets(forecasting)
        nowcasting = nowcasting.iloc[::-1].reset_index(drop=True)
        train_mask = ~forecasting["area_id"].isin([1, 2])
        test_mask = ~train_mask
        now_train_mask = ~nowcasting["area_id"].isin([1, 2])
        now_test_mask = ~now_train_mask
        factory = SequentialRecordingFactory()

        observed = loco.fit_nowcasting_split(
            forecasting,
            nowcasting,
            train_mask,
            test_mask,
            now_train_mask,
            now_test_mask,
            "area_holdout_seed0",
            {"max_depth": 11, "random_state": 0},
            {"max_depth": 9, "random_state": 0},
            fold_column="fold_id",
            estimator_factory=factory,
        )

        self.assertEqual(set(observed["area_id"]), {1, 2})
        self.assertEqual(set(observed["country_code_3"]), {"AAA", "BBB"})
        self.assertEqual(set(observed["fold_id"]), {"area_holdout_seed0"})
        self.assertEqual(factory.max_depth_sequence, [11, 11, 9, 9, 9, 9, 9, 9])
        for model in factory.models:
            if model.fit_X is not None and "signal" in model.fit_X.columns:
                self.assertTrue(model.fit_X["signal"].isin([20.0, 21.0]).all())

    def test_run_loco_predictions_covers_each_source_row_once_per_model(self):
        forecasting, nowcasting = synthetic_prepared_inputs(
            include_nowcast_features=True
        )

        forecast_predictions, nowcast_predictions, metrics, audit = (
            loco.run_loco_predictions(
                forecasting,
                nowcasting,
                general_params={"random_state": 0},
                phase3_params={"random_state": 0},
                countries=["BBB", "AAA"],
                workers=1,
                forecasting_runner=fake_forecasting_fold,
                nowcasting_runner=fake_nowcasting_fold,
            )
        )

        self.assertEqual(len(forecast_predictions), len(forecasting))
        self.assertEqual(len(nowcast_predictions), len(nowcasting))
        self.assertFalse(
            forecast_predictions.duplicated(["area_id", "date"]).any()
        )
        self.assertFalse(nowcast_predictions.duplicated(["area_id", "date"]).any())
        self.assertEqual(len(metrics), 4)
        self.assertEqual(len(audit), 4)
        self.assertEqual(
            metrics[["model", "country_code_3"]].values.tolist(),
            [
                ["Forecasting", "AAA"],
                ["Forecasting", "BBB"],
                ["Nowcasting", "AAA"],
                ["Nowcasting", "BBB"],
            ],
        )

    def test_checkpoint_is_reused_only_for_an_exact_manifest_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            manifest = {
                "country": "AAA",
                "random_state": 0,
                "script_sha256": "abc",
            }
            forecast = pd.DataFrame(
                {"area_id": [1], "date": ["2020-01-01"]}
            )
            nowcast = forecast.copy()

            loco.save_country_checkpoint(
                checkpoint_dir, "AAA", manifest, forecast, nowcast
            )
            loaded = loco.load_country_checkpoint(
                checkpoint_dir, "AAA", manifest
            )
            stale = loco.load_country_checkpoint(
                checkpoint_dir,
                "AAA",
                {**manifest, "script_sha256": "changed"},
            )

            self.assertIsNotNone(loaded)
            loaded_forecast, loaded_nowcast = loaded
            pd.testing.assert_frame_equal(loaded_forecast, forecast)
            pd.testing.assert_frame_equal(loaded_nowcast, nowcast)
            self.assertIsNone(stale)

    def test_partial_run_rejects_default_production_output_directory(self):
        with self.assertRaisesRegex(ValueError, "non-default output directory"):
            loco.run_analysis(
                countries=["AAA"], output_dir=loco.DEFAULT_OUTPUT_DIR
            )

    def test_figure_has_two_panels_shared_limits_and_no_aggregate_point(self):
        metrics = pd.DataFrame(
            {
                "model": [
                    "Forecasting",
                    "Forecasting",
                    "Nowcasting",
                    "Nowcasting",
                ],
                "country_code_3": ["AAA", "BBB", "AAA", "BBB"],
                "phase3plus_precision": [0.6, pd.NA, 0.7, 0.8],
                "phase3plus_recall": [0.5, 0.4, 0.6, 0.9],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure = loco.create_precision_recall_figure(metrics)

        self.assertEqual(caught, [])
        self.assertEqual(len(figure.axes), 2)
        for axis in figure.axes:
            self.assertLess(axis.get_xlim()[0], 0.0)
            self.assertGreater(axis.get_xlim()[1], 1.0)
            self.assertLess(axis.get_ylim()[0], 0.0)
            self.assertGreater(axis.get_ylim()[1], 1.0)
            self.assertEqual(
                tuple(round(value, 1) for value in axis.get_xticks()),
                (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            )
            self.assertEqual(
                tuple(round(value, 1) for value in axis.get_yticks()),
                (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            )
        plotted_points = sum(
            len(collection.get_offsets())
            for axis in figure.axes
            for collection in axis.collections
        )
        self.assertEqual(plotted_points, 3)
        plt.close(figure)

    def test_figure_country_labels_do_not_overlap_or_leave_their_panel(self):
        metrics_path = (
            Path(__file__).resolve().parents[1]
            / "2.Source Code"
            / "produced_graph"
            / "leave_one_country_out_country_metrics.csv"
        )
        metrics = pd.read_csv(metrics_path)
        figure = loco.create_precision_recall_figure(metrics)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()

        for axis in figure.axes:
            model_name = axis.get_title(loc="left")
            panel = metrics.loc[metrics["model"].eq(model_name)].dropna(
                subset=["phase3plus_precision", "phase3plus_recall"]
            )
            country_codes = set(panel["country_code_3"])
            labels = [
                text for text in axis.texts if text.get_text() in country_codes
            ]
            self.assertEqual(len(labels), len(country_codes))

            bboxes = [
                mpl.text.Text.get_window_extent(label, renderer) for label in labels
            ]
            for bbox in bboxes:
                self.assertGreaterEqual(bbox.x0, axis.bbox.x0)
                self.assertLessEqual(bbox.x1, axis.bbox.x1)
                self.assertGreaterEqual(bbox.y0, axis.bbox.y0)
                self.assertLessEqual(bbox.y1, axis.bbox.y1)
            overlaps = [
                (labels[left].get_text(), labels[right].get_text())
                for left, left_bbox in enumerate(bboxes)
                for right, right_bbox in enumerate(bboxes[left + 1 :], left + 1)
                if left_bbox.overlaps(right_bbox)
            ]
            self.assertEqual(overlaps, [])
            marker_centers = axis.transData.transform(
                axis.collections[0].get_offsets()
            )
            marker_boxes = [
                mpl.transforms.Bbox.from_bounds(x - 4, y - 4, 8, 8)
                for x, y in marker_centers
            ]
            label_marker_overlaps = [
                label.get_text()
                for label, label_bbox in zip(labels, bboxes)
                if any(label_bbox.overlaps(marker_bbox) for marker_bbox in marker_boxes)
            ]
            self.assertEqual(label_marker_overlaps, [])

        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
