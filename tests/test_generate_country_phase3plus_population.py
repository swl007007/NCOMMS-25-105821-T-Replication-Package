import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_country_phase3plus_population as country_population


def canonical_predictions():
    restored = [3374, 3517, 3534, 3553, 3567]
    rows = np.arange(1170)
    actual = (rows % 5) + 1
    return pd.DataFrame(
        {
            "test_index": restored + list(range(1165)),
            "overall_phase": actual,
            "overall_phase_pred": np.where(actual >= 3, 3, 2),
            "phase3_pred": np.where(actual >= 3, 0.8, 0.1),
            "area_id": rows + 1,
            "date": ["2022-01-01"] * 1170,
            "lat": np.linspace(-20, 20, 1170),
            "lon": np.linspace(10, 50, 1170),
            "nowcast_predict": np.where(actual >= 3, 3, 2),
            "phase3_nowcast": np.where(actual >= 3, 0.85, 0.05),
        }
    )


class AggregateCountryPhase3PlusPopulationTests(unittest.TestCase):
    def test_sums_area_population_times_each_proxy_within_country_date(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 2, 1, 3],
                "date": ["2022-01-01", "2022-01-01", "2022-02-01", "2022-01-01"],
                "phase3_pred": [0.20, 0.40, 0.30, 0.50],
                "phase3_nowcast": [0.10, 0.50, 0.20, 0.25],
            }
        )
        population = pd.DataFrame(
            {
                "area_id": [1, 2, 1, 3],
                "date": ["2022-01-01", "2022-01-01", "2022-02-01", "2022-01-01"],
                "country_code_3": ["AAA", "AAA", "AAA", "BBB"],
                "country_en": ["Country A", "Country A", "Country A", "Country B"],
                "estimated_population": [100.0, 300.0, 110.0, 200.0],
                "overall_phase": [3, 3, 3, 3],
                "phase1_population": [50.0, 150.0, 55.0, 100.0],
                "phase2_population": [20.0, 60.0, 22.0, 40.0],
                "phase3_population": [20.0, 60.0, 22.0, 40.0],
                "phase4_population": [10.0, 30.0, 11.0, 20.0],
                "phase5_population": [0.0, 0.0, 0.0, 0.0],
            }
        )

        result = country_population.aggregate_country_phase3plus_population(
            predictions, population
        ).set_index(["country_code_3", "date"])

        january_a = result.loc[("AAA", pd.Timestamp("2022-01-01"))]
        self.assertEqual(january_a["source_geometry_count"], 2)
        self.assertEqual(january_a["population_profile_count"], 2)
        self.assertAlmostEqual(january_a["analyzed_population"], 400.0)
        self.assertAlmostEqual(january_a["actual_phase3plus_population"], 120.0)
        self.assertAlmostEqual(january_a["actual_phase3plus_share"], 0.30)
        self.assertAlmostEqual(january_a["forecast_phase3plus_population"], 140.0)
        self.assertAlmostEqual(january_a["forecast_phase3plus_share"], 0.35)
        self.assertAlmostEqual(january_a["nowcast_phase3plus_population"], 160.0)
        self.assertAlmostEqual(january_a["nowcast_phase3plus_share"], 0.40)

        february_a = result.loc[("AAA", pd.Timestamp("2022-02-01"))]
        self.assertEqual(february_a["source_geometry_count"], 1)
        self.assertEqual(february_a["population_profile_count"], 1)
        self.assertAlmostEqual(february_a["analyzed_population"], 110.0)
        self.assertAlmostEqual(february_a["actual_phase3plus_population"], 33.0)
        self.assertAlmostEqual(february_a["actual_phase3plus_share"], 0.30)
        self.assertAlmostEqual(february_a["forecast_phase3plus_population"], 33.0)
        self.assertAlmostEqual(february_a["nowcast_phase3plus_population"], 22.0)

    def test_rejects_population_lookup_with_duplicate_area_date_keys(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1],
                "date": ["2022-01-01"],
                "phase3_pred": [0.20],
                "phase3_nowcast": [0.10],
            }
        )
        population = pd.DataFrame(
            {
                "area_id": [1, 1],
                "date": ["2022-01-01", "2022-01-01"],
                "country_code_3": ["AAA", "AAA"],
                "country_en": ["Country A", "Country A"],
                "estimated_population": [100.0, 100.0],
                "overall_phase": [3, 3],
                "phase1_population": [50.0, 50.0],
                "phase2_population": [20.0, 20.0],
                "phase3_population": [20.0, 20.0],
                "phase4_population": [10.0, 10.0],
                "phase5_population": [0.0, 0.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate area_id/date keys"):
            country_population.aggregate_country_phase3plus_population(
                predictions, population
            )

    def test_ignores_invalid_population_rows_not_referenced_by_predictions(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1],
                "date": ["2022-01-01"],
                "phase3_pred": [0.20],
                "phase3_nowcast": [0.10],
            }
        )
        population = pd.DataFrame(
            {
                "area_id": [1, 99],
                "date": ["2022-01-01", "2016-01-01"],
                "country_code_3": ["AAA", None],
                "country_en": ["Country A", None],
                "estimated_population": [100.0, None],
                "overall_phase": [3, None],
                "phase1_population": [50.0, None],
                "phase2_population": [20.0, None],
                "phase3_population": [20.0, None],
                "phase4_population": [10.0, None],
                "phase5_population": [0.0, None],
            }
        )

        result = country_population.aggregate_country_phase3plus_population(
            predictions, population
        )

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result.loc[0, "forecast_phase3plus_population"], 20.0)
        self.assertAlmostEqual(result.loc[0, "nowcast_phase3plus_population"], 10.0)

    def test_counts_a_repeated_ipc_population_profile_once(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 2],
                "date": ["2022-01-01", "2022-01-01"],
                "phase3_pred": [0.20, 0.40],
                "phase3_nowcast": [0.10, 0.50],
            }
        )
        population = pd.DataFrame(
            {
                "area_id": [1, 2],
                "date": ["2022-01-01", "2022-01-01"],
                "country_code_3": ["AAA", "AAA"],
                "country_en": ["Country A", "Country A"],
                "estimated_population": [100.0, 100.0],
                "overall_phase": [3, 3],
                "phase1_population": [50.0, 50.0],
                "phase2_population": [20.0, 20.0],
                "phase3_population": [20.0, 20.0],
                "phase4_population": [10.0, 10.0],
                "phase5_population": [0.0, 0.0],
            }
        )

        result = country_population.aggregate_country_phase3plus_population(
            predictions, population
        )

        self.assertEqual(result.loc[0, "source_geometry_count"], 2)
        self.assertEqual(result.loc[0, "population_profile_count"], 1)
        self.assertAlmostEqual(result.loc[0, "analyzed_population"], 100.0)
        self.assertAlmostEqual(result.loc[0, "actual_phase3plus_population"], 30.0)
        self.assertAlmostEqual(result.loc[0, "actual_phase3plus_share"], 0.30)
        self.assertAlmostEqual(result.loc[0, "forecast_phase3plus_population"], 30.0)
        self.assertAlmostEqual(result.loc[0, "nowcast_phase3plus_population"], 30.0)


class CanonicalPredictionInputTests(unittest.TestCase):
    def test_loader_accepts_the_current_1170_contract(self):
        self.assertTrue(
            hasattr(country_population, "load_prediction_input"),
            "Country aggregation has no canonical prediction-input guard.",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "All_prediction.csv"
            canonical_predictions().to_csv(path, index=False)
            observed = country_population.load_prediction_input(path)
        self.assertEqual(len(observed), 1170)

    def test_loader_rejects_legacy_or_malformed_current_inputs(self):
        self.assertTrue(hasattr(country_population, "load_prediction_input"))
        source = canonical_predictions()
        cases = {
            "legacy row count": source.iloc[:1165].copy(),
            "duplicate key": source.assign(
                area_id=lambda frame: frame["area_id"].mask(frame.index == 1, 1)
            ),
            "missing restored index": source.assign(
                test_index=lambda frame: frame["test_index"].mask(
                    frame.index == 0, 999999
                )
            ),
            "missing prediction": source.drop(columns="phase3_nowcast"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, data in cases.items():
                with self.subTest(label=label):
                    path = Path(directory) / f"{label}.csv"
                    data.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        country_population.load_prediction_input(path)

    def test_explicit_diagnostic_loader_can_return_a_small_noncanonical_fixture(self):
        self.assertTrue(hasattr(country_population, "load_prediction_input"))
        fixture = pd.DataFrame(
            {
                "area_id": [1],
                "date": ["2022-01-01"],
                "phase3_pred": [0.2],
                "phase3_nowcast": [0.1],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            fixture.to_csv(path, index=False)
            observed = country_population.load_prediction_input(
                path, require_canonical=False
            )
        pd.testing.assert_frame_equal(observed, fixture)


if __name__ == "__main__":
    unittest.main()
