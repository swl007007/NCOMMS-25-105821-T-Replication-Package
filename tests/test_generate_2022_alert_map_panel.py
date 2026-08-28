import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

from generate_2022_alert_map_panel import add_top30_flags, prepare_area_records
import generate_2022_alert_map_panel as alert_panel


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


class PrepareAreaRecordsTests(unittest.TestCase):
    def test_preserves_forecast_max_row_for_all_phase_alerts(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 1, 2],
                "date": ["2022-01-01", "2022-02-01", "2022-01-01"],
                "phase3_pred": [0.90, 0.20, 0.40],
                "phase3_nowcast": [0.10, 0.80, 0.30],
                "overall_phase": [2, 4, 3],
                "overall_phase_pred": [3, 2, 3],
                "nowcast_predict": [2, 4, 3],
                "lat": [1.0, 1.0, 2.0],
                "lon": [10.0, 10.0, 20.0],
            }
        )
        actual = pd.DataFrame(
            {
                "area_id": [1, 1, 2],
                "date": ["2022-01-01", "2022-02-01", "2022-01-01"],
                "phase3_percent": [0.10, 0.50, 0.20],
                "phase4_percent": [0.00, 0.30, 0.10],
                "phase5_percent": [0.00, 0.10, 0.00],
            }
        )

        result = prepare_area_records(predictions, actual, year=2022).set_index("area_id")

        self.assertEqual(result.loc[1, "date"], pd.Timestamp("2022-01-01"))
        self.assertEqual(result.loc[1, "crisis_forecast"], 1)
        self.assertEqual(result.loc[1, "crisis_nowcast"], 0)
        self.assertEqual(result.loc[1, "crisis_actual"], 0)
        self.assertAlmostEqual(result.loc[1, "phase3_actual"], 0.90)

    def test_selects_each_annual_minimum_severity_independently(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 1, 1, 1, 2],
                "date": [
                    "2022-01-01",
                    "2022-02-01",
                    "2022-03-01",
                    "2021-12-01",
                    "2022-01-01",
                ],
                "phase3_pred": [0.10, 0.80, 0.60, 0.00, 0.40],
                "phase3_nowcast": [0.70, 0.05, 0.40, 0.00, 0.30],
            }
        )
        actual = pd.DataFrame(
            {
                "area_id": [1, 1, 1, 1, 2],
                "date": [
                    "2022-01-01",
                    "2022-02-01",
                    "2022-03-01",
                    "2021-12-01",
                    "2022-01-01",
                ],
                "phase3_percent": [0.50, 0.20, 0.03, 0.00, 0.20],
                "phase4_percent": [0.10, 0.05, 0.01, 0.00, 0.10],
                "phase5_percent": [0.00, 0.00, 0.01, 0.00, 0.00],
            }
        )

        result = alert_panel.prepare_minimum_severity_area_records(
            predictions, actual, year=2022
        ).set_index("area_id")

        self.assertEqual(result.index.tolist(), [1, 2])
        self.assertAlmostEqual(result.loc[1, "phase3_pred"], 0.10)
        self.assertAlmostEqual(result.loc[1, "phase3_nowcast"], 0.05)
        self.assertAlmostEqual(result.loc[1, "phase3_actual"], 0.05)
        self.assertAlmostEqual(result.loc[2, "phase3_pred"], 0.40)
        self.assertAlmostEqual(result.loc[2, "phase3_nowcast"], 0.30)
        self.assertAlmostEqual(result.loc[2, "phase3_actual"], 0.30)


class TopThirtyTests(unittest.TestCase):
    def test_uses_each_severity_proxy_70th_percentile_with_greater_equal(self):
        records = pd.DataFrame(
            {
                "phase3_pred": [0.10, 0.20, 0.30, 0.30],
                "phase3_nowcast": [0.40, 0.30, 0.20, 0.10],
                "phase3_actual": [0.10, 0.10, 0.10, 0.10],
            }
        )

        flagged, thresholds = add_top30_flags(records)

        self.assertAlmostEqual(thresholds["forecasting"], 0.30)
        self.assertAlmostEqual(thresholds["nowcasting"], 0.31)
        self.assertAlmostEqual(thresholds["actual"], 0.10)
        self.assertEqual(flagged["top30_forecast"].tolist(), [0, 0, 1, 1])
        self.assertEqual(flagged["top30_nowcast"].tolist(), [1, 0, 0, 0])
        self.assertEqual(flagged["top30_actual"].tolist(), [1, 1, 1, 1])


class FixedSeverityAlertTests(unittest.TestCase):
    def test_uses_same_severity_fields_as_top30_with_fixed_020_cutoff(self):
        records = pd.DataFrame(
            {
                "phase3_pred": [0.19, 0.20],
                "phase3_nowcast": [0.20, 0.19],
                "phase3_actual": [0.19, 0.20],
            }
        )

        try:
            flagged = alert_panel.add_fixed_severity_alert_flags(records, cutoff=0.20)
        except AttributeError:
            self.fail("severity-aligned fixed-cutoff alert function is missing")

        self.assertEqual(flagged["severity_alert_forecast"].tolist(), [0, 1])
        self.assertEqual(flagged["severity_alert_nowcast"].tolist(), [1, 0])
        self.assertEqual(flagged["severity_alert_actual"].tolist(), [0, 1])


class CanonicalPredictionInputTests(unittest.TestCase):
    def test_loader_accepts_the_current_1170_contract(self):
        self.assertTrue(
            hasattr(alert_panel, "load_prediction_input"),
            "Alert-map generation has no canonical prediction-input guard.",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "All_prediction.csv"
            canonical_predictions().to_csv(path, index=False)
            observed = alert_panel.load_prediction_input(path)
        self.assertEqual(len(observed), 1170)

    def test_loader_rejects_legacy_or_malformed_current_inputs(self):
        self.assertTrue(hasattr(alert_panel, "load_prediction_input"))
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
            "missing prediction": source.drop(columns="nowcast_predict"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, data in cases.items():
                with self.subTest(label=label):
                    path = Path(directory) / f"{label}.csv"
                    data.to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        alert_panel.load_prediction_input(path)

    def test_explicit_diagnostic_loader_can_return_a_small_noncanonical_fixture(self):
        self.assertTrue(hasattr(alert_panel, "load_prediction_input"))
        fixture = pd.DataFrame(
            {
                "area_id": [1],
                "date": ["2022-01-01"],
                "phase3_pred": [0.2],
                "phase3_nowcast": [0.1],
                "overall_phase": [3],
                "overall_phase_pred": [3],
                "nowcast_predict": [3],
                "lat": [10.0],
                "lon": [20.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            fixture.to_csv(path, index=False)
            observed = alert_panel.load_prediction_input(
                path, require_canonical=False
            )
        pd.testing.assert_frame_equal(observed, fixture)


if __name__ == "__main__":
    unittest.main()
