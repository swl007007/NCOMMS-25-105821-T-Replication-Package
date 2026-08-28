import sys
import unittest
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_multinomial_baseline_comparison as baseline


class MultinomialBaselineTests(unittest.TestCase):
    def test_derive_phase_labels_uses_cumulative_shares_and_highest_phase(self):
        data = pd.DataFrame(
            {
                "phase1_percent": [0.8, 0.8, 0.8, 0.8, 0.9],
                "phase2_percent": [0.0, 0.0, 0.0, 0.12, 0.04],
                "phase3_percent": [0.0, 0.0, 0.12, 0.08, 0.03],
                "phase4_percent": [0.0, 0.12, 0.08, 0.0, 0.02],
                "phase5_percent": [0.20, 0.08, 0.0, 0.0, 0.01],
            }
        )

        observed = baseline.derive_phase_labels(data)

        self.assertEqual(observed.tolist(), [5, 4, 3, 2, 1])

    def test_classification_metrics_apply_phase3_threshold(self):
        y_true = pd.Series([1, 2, 3, 4, 5, 2])
        y_pred = pd.Series([1, 3, 3, 2, 5, 2])

        metrics = baseline.calculate_classification_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["overall_accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["phase3plus_accuracy"], 4 / 6)
        self.assertAlmostEqual(metrics["phase3plus_recall"], 2 / 3)
        self.assertAlmostEqual(metrics["phase3plus_precision"], 2 / 3)

    def test_feature_selection_excludes_keys_and_outcome_components(self):
        data = pd.DataFrame(
            columns=[
                "date",
                "area_id",
                "overall_phase",
                "phase1_percent",
                "phase2_percent",
                "phase3_percent",
                "phase4_percent",
                "phase5_percent",
                "signal",
                "all_missing_in_training",
            ]
        )

        observed = baseline.select_feature_columns(data)

        self.assertEqual(observed, ["signal", "all_missing_in_training"])

    def test_temporal_masks_assign_boundary_date_to_test(self):
        dates = pd.Series(["2021-12-31", "2022-01-01", "2022-02-01"])

        train_mask, test_mask = baseline.temporal_masks(dates, "2022-01-01")

        self.assertEqual(train_mask.tolist(), [True, False, False])
        self.assertEqual(test_mask.tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
