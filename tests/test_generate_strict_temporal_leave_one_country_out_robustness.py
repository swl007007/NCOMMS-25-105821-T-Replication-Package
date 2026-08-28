import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_leave_one_country_out_robustness as loco
import generate_strict_temporal_leave_one_country_out_robustness as strict


def strict_fixture():
    return pd.DataFrame(
        {
            "area_id": [10, 10, 20, 20, 30, 40],
            "date": [
                "2021-01-01",
                "2022-01-01",
                "2021-01-01",
                "2022-01-01",
                "2021-06-01",
                "2023-01-01",
            ],
            "country_code_3": ["AAA", "AAA", "BBB", "BBB", "CCC", "AAA"],
        }
    )


class StrictSplitTests(unittest.TestCase):
    def test_masks_assign_the_four_quadrants(self):
        train, test, excluded = strict.strict_temporal_masks(strict_fixture(), "AAA")

        self.assertEqual(train.tolist(), [False, False, True, False, True, False])
        self.assertEqual(test.tolist(), [False, True, False, False, False, True])
        self.assertEqual(excluded.tolist(), [True, False, False, True, False, False])
        self.assertTrue((train | test | excluded).all())
        self.assertFalse((train & test).any())

    def test_active_subset_preserves_source_indexes(self):
        data = strict_fixture()
        train, test, _ = strict.strict_temporal_masks(data, "AAA")

        active, active_train, active_test = strict.subset_for_complete_split(
            data, train, test
        )

        self.assertEqual(active.index.tolist(), [1, 2, 4, 5])
        self.assertTrue((active_train | active_test).all())
        self.assertFalse((active_train & active_test).any())


class MetricAggregationTests(unittest.TestCase):
    def test_r2_treats_report_precision_actual_shares_as_constant(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1777, 1777],
                "date": ["2022-03-01", "2022-09-01"],
                "country_code_3": ["AFG", "AFG"],
                "overall_phase": [4, 3],
                "overall_phase_pred": [3, 3],
                "phase3_test": [0.45, 0.44999999999999996],
                "phase3_pred": [0.41, 0.39],
                "nonpositive_cumulative_prediction_sum": [False, False],
            }
        )

        area = strict.calculate_area_metrics(predictions, predictions)
        observed = area.loc[area["model"].eq("Forecasting")].iloc[0]

        self.assertTrue(pd.isna(observed["R2(p3)"]))
        self.assertEqual(
            observed["r2_undefined_reason"],
            "constant_actual_phase3plus_share",
        )

    def test_prediction_serialization_preserves_metric_defining_float_values(self):
        predictions = pd.DataFrame(
            {
                "phase3_test": [
                    0.45000000000000001,
                    0.44999999999999996,
                ],
                "phase3_pred": [0.41, 0.39],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            strict._write_dataframe("forecasting_predictions", predictions, path)
            restored = pd.read_csv(path)

        self.assertEqual(restored["phase3_test"].nunique(), 2)

    def test_checkpoint_loading_preserves_metric_defining_float_values(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 1],
                "date": ["2022-01-01", "2022-02-01"],
                "country_code_3": ["AAA", "AAA"],
                "cutoff": [strict.CUTOFF, strict.CUTOFF],
                "phase3_test": [0.45000000000000001, 0.44999999999999996],
                "phase3_pred": [0.41, 0.39],
            }
        )
        manifest = {"country": "AAA"}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            strict.save_checkpoint(
                checkpoint_dir, "AAA", manifest, predictions, predictions
            )
            loaded = strict.load_checkpoint(
                checkpoint_dir,
                "AAA",
                manifest,
                predictions[["area_id", "date"]],
            )

        self.assertIsNotNone(loaded)
        self.assertEqual(
            loaded[0]["phase3_test"].tolist(), predictions["phase3_test"].tolist()
        )

    def test_primary_metrics_pool_rows_instead_of_equal_weighting_areas(self):
        self.assertEqual(
            strict.FINAL_FILENAMES["micro_metrics"],
            "strict_temporal_loco_micro_metrics.csv",
        )
        self.assertNotIn("area_macro_metrics", strict.FINAL_FILENAMES)
        forecast = pd.DataFrame(
            {
                "area_id": [1, 1, 2, 2, 2],
                "date": pd.date_range("2022-01-01", periods=5, freq="MS"),
                "country_code_3": ["AAA", "AAA", "BBB", "BBB", "BBB"],
                "overall_phase": [3, 1, 1, 1, 1],
                "overall_phase_pred": [3, 3, 1, 1, 1],
                "phase3_test": [0.4, 0.0, 0.0, 0.0, 0.0],
                "phase3_pred": [0.3, 0.1, 0.0, 0.0, 0.0],
                "nonpositive_cumulative_prediction_sum": [False] * 5,
            }
        )
        nowcast = forecast.copy()

        area = strict.calculate_area_metrics(forecast, nowcast)
        area_macro, denominators = strict.aggregate_area_metrics(area)
        micro = strict.calculate_micro_metrics(forecast, nowcast)

        self.assertEqual(len(area), 4)
        self.assertEqual(micro.index.tolist(), ["Nowcasting", "Forecasting"])
        self.assertEqual(
            micro.columns.tolist(), ["accuracy", "precision", "recall", "R2(p3)"]
        )
        self.assertAlmostEqual(area_macro.loc["Forecasting", "accuracy"], 0.75)
        self.assertAlmostEqual(micro.loc["Forecasting", "accuracy"], 0.8)
        self.assertAlmostEqual(micro.loc["Forecasting", "precision"], 0.5)
        self.assertAlmostEqual(micro.loc["Forecasting", "recall"], 1.0)
        self.assertEqual(denominators.loc["Forecasting", "area_count_total"], 2)

    def test_micro_accuracy_remains_exact_five_phase_accuracy(self):
        predictions = pd.DataFrame(
            {
                "overall_phase": [4, 1],
                "overall_phase_pred": [3, 1],
                "phase3_test": [0.6, 0.0],
                "phase3_pred": [0.5, 0.0],
            }
        )

        micro = strict.calculate_micro_metrics(predictions, predictions)

        self.assertAlmostEqual(micro.loc["Forecasting", "accuracy"], 0.5)
        self.assertAlmostEqual(micro.loc["Forecasting", "precision"], 1.0)
        self.assertAlmostEqual(micro.loc["Forecasting", "recall"], 1.0)

    def test_micro_r2_uses_report_precision_shares(self):
        predictions = pd.DataFrame(
            {
                "overall_phase": [4, 3],
                "overall_phase_pred": [3, 3],
                "phase3_test": [0.45, 0.44999999999999996],
                "phase3_pred": [0.41, 0.39],
            }
        )

        micro = strict.calculate_micro_metrics(predictions, predictions)

        self.assertTrue(pd.isna(micro.loc["Forecasting", "R2(p3)"]))

    def test_area_diagnostics_keep_metric_specific_denominators(self):
        area = pd.DataFrame(
            {
                "model": ["Nowcasting"] * 3 + ["Forecasting"] * 3,
                "area_id": [1, 2, 3, 1, 2, 3],
                "accuracy": [1.0, 0.5, 0.0, 1.0, 0.5, 0.0],
                "precision": [0.75, np.nan, np.nan, 1.0, 0.5, 0.25],
                "recall": [1.0, 0.0, np.nan, 1.0, 0.5, 0.25],
                "R2(p3)": [-9.0, np.nan, np.nan, 1.0, 0.5, 0.25],
            }
        )

        macro, denominators = strict.aggregate_area_metrics(area)

        self.assertEqual(
            denominators.loc["Nowcasting", "precision_area_count_defined"], 1
        )
        self.assertEqual(
            denominators.loc["Nowcasting", "recall_area_count_defined"], 2
        )
        self.assertEqual(denominators.loc["Nowcasting", "r2_area_count_defined"], 1)
        self.assertEqual(macro.loc["Nowcasting", "R2(p3)"], -9.0)


class LivePopulationTests(unittest.TestCase):
    def test_live_strict_test_union_matches_canonical_temporal_population(self):
        lookup = loco.load_country_lookup()
        forecasting, nowcasting = loco.prepare_model_inputs(
            pd.read_csv(loco.DEFAULT_FORECASTING_INPUT),
            pd.read_csv(loco.DEFAULT_NOWCASTING_INPUT),
            lookup,
        )
        test = pd.to_datetime(forecasting["date"]).ge(strict.CUTOFF)

        self.assertEqual(len(forecasting), 5575)
        self.assertEqual(len(nowcasting), 5575)
        self.assertEqual(int(test.sum()), 1170)
        self.assertEqual(forecasting.loc[test, "area_id"].nunique(), 646)
        self.assertEqual(forecasting.loc[test, "country_code_3"].nunique(), 27)
        self.assertEqual(
            strict.canonical_key_sha256(forecasting.loc[test]),
            strict.CANONICAL_TEST_KEY_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
