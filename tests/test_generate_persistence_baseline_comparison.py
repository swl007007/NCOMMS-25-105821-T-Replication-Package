import sys
import unittest
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_persistence_baseline_comparison as persistence


def observation(date, area_id, phase, lat, lon):
    shares = {f"phase{i}_percent": 0.0 for i in range(1, 6)}
    shares[f"phase{phase}_percent"] = 0.2 if phase > 1 else 1.0
    if phase > 1:
        shares["phase1_percent"] = 0.8
    return {
        "date": date,
        "area_id": area_id,
        "overall_phase": phase,
        "lat": lat,
        "lon": lon,
        **shares,
    }


class PersistenceBaselineTests(unittest.TestCase):
    def test_nowcast_uses_own_latest_history_even_after_a_long_gap(self):
        data = pd.DataFrame(
            [
                observation("2017-01-01", 1, 4, 0.0, 0.0),
                observation("2021-12-01", 2, 1, 0.01, 0.0),
                observation("2022-01-01", 1, 2, 0.0, 0.0),
            ]
        )
        countries = pd.DataFrame(
            {"area_id": [1, 2], "country_code_3": ["AAA", "AAA"]}
        )

        result = persistence.build_persistence_predictions(
            data, countries, horizon="nowcasting", target_start="2022-01-01"
        )

        row = result.loc[result["area_id"] == 1].iloc[0]
        self.assertEqual(row["predicted_phase"], 4)
        self.assertEqual(row["source_method"], "own_history")
        self.assertEqual(row["source_date"], "2017-01-01")

    def test_fallback_searches_older_dates_for_same_country_before_global_neighbor(self):
        data = pd.DataFrame(
            [
                observation("2022-01-01", 30, 2, 5.0, 0.0),
                observation("2022-02-01", 20, 5, 0.01, 0.0),
                observation("2022-03-01", 10, 1, 0.0, 0.0),
            ]
        )
        countries = pd.DataFrame(
            {
                "area_id": [10, 20, 30],
                "country_code_3": ["AAA", "BBB", "AAA"],
            }
        )

        result = persistence.build_persistence_predictions(
            data, countries, horizon="nowcasting", target_start="2022-03-01"
        )

        row = result.iloc[0]
        self.assertEqual(row["source_method"], "same_country_neighbor")
        self.assertEqual(row["source_area_id"], 30)
        self.assertEqual(row["source_date"], "2022-01-01")
        self.assertEqual(row["predicted_phase"], 2)

    def test_global_fallback_breaks_equal_distance_ties_by_smallest_area_id(self):
        data = pd.DataFrame(
            [
                observation("2022-02-01", 5, 5, 1.0, 0.0),
                observation("2022-02-01", 4, 2, -1.0, 0.0),
                observation("2022-03-01", 10, 1, 0.0, 0.0),
            ]
        )
        countries = pd.DataFrame(
            {
                "area_id": [4, 5, 10],
                "country_code_3": ["BBB", "BBB", "CCC"],
            }
        )

        result = persistence.build_persistence_predictions(
            data, countries, horizon="nowcasting", target_start="2022-03-01"
        )

        row = result.iloc[0]
        self.assertEqual(row["source_method"], "global_neighbor")
        self.assertEqual(row["source_area_id"], 4)
        self.assertEqual(row["predicted_phase"], 2)

    def test_forecasting_uses_information_on_issue_date_but_not_after_it(self):
        data = pd.DataFrame(
            [
                observation("2021-06-01", 1, 3, 0.0, 0.0),
                observation("2021-12-01", 1, 5, 0.0, 0.0),
                observation("2022-06-01", 1, 2, 0.0, 0.0),
            ]
        )
        countries = pd.DataFrame(
            {"area_id": [1], "country_code_3": ["AAA"]}
        )

        result = persistence.build_persistence_predictions(
            data, countries, horizon="forecasting", target_start="2022-01-01"
        )

        row = result.iloc[0]
        self.assertEqual(row["issue_date"], "2021-06-01")
        self.assertEqual(row["source_date"], "2021-06-01")
        self.assertEqual(row["predicted_phase"], 3)


if __name__ == "__main__":
    unittest.main()
