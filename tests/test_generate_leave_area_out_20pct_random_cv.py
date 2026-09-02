import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "2.Source Code"
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))
sys.path.insert(0, str(TEST_DIR))

import generate_leave_area_out_20pct_random_cv as random_cv
from test_generate_leave_area_out_20pct_fivefold_robustness import (
    perfect_forecasting_runner,
    perfect_nowcasting_runner,
    synthetic_fivefold_data,
)


class RandomAreaFivefoldTests(unittest.TestCase):
    def _write_inputs(self, root: Path) -> dict[str, Path]:
        forecasting_path = root / "forecasting.csv"
        nowcasting_path = root / "nowcasting.csv"
        lookup_path = root / "lookup.csv"
        general_path = root / "general.json"
        phase3_path = root / "phase3.json"
        forecasting = synthetic_fivefold_data()
        nowcasting = synthetic_fivefold_data(nowcasting=True)
        forecasting.drop(columns="country_code_3").to_csv(forecasting_path, index=False)
        nowcasting.drop(columns="country_code_3").to_csv(nowcasting_path, index=False)
        pd.DataFrame(
            {
                "area_id": range(1, 11),
                "country_code_3": ["AAA"] * 5 + ["BBB"] * 5,
            }
        ).to_csv(lookup_path, index=False)
        general_path.write_text(json.dumps({"max_depth": 2}), encoding="utf-8")
        phase3_path.write_text(json.dumps({"max_depth": 3}), encoding="utf-8")
        return {
            "forecasting_path": forecasting_path,
            "nowcasting_path": nowcasting_path,
            "country_lookup_path": lookup_path,
            "general_params_path": general_path,
            "phase3_params_path": phase3_path,
        }

    def test_all_dates_random_area_cv_writes_fold_summary_without_pooled_metrics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            output_dir.mkdir()
            protected = output_dir / "leave_area_out_10pct_fixture.csv"
            protected.write_text("frozen\n", encoding="utf-8")

            artifacts = random_cv.run_analysis(
                **self._write_inputs(root),
                output_dir=output_dir,
                forecasting_runner=perfect_forecasting_runner,
                nowcasting_runner=perfect_nowcasting_runner,
            )

            self.assertEqual(set(artifacts), set(random_cv.FINAL_FILENAMES))
            self.assertTrue(all(path.is_file() for path in artifacts.values()))
            self.assertEqual(protected.read_text(encoding="utf-8"), "frozen\n")
            folds = pd.read_csv(artifacts["folds"])
            forecast = pd.read_csv(artifacts["forecasting_predictions"])
            fold_metrics = pd.read_csv(artifacts["fold_metrics"])
            summary = pd.read_csv(artifacts["metrics"])
            self.assertEqual(len(folds), 10)
            self.assertEqual(set(folds["fold_id"]), set(range(5)))
            self.assertEqual(folds["fold_id"].tolist(), [4, 2, 0, 3, 1, 4, 2, 3, 0, 1])
            self.assertEqual(len(forecast), 20)
            self.assertEqual(set(forecast["date"]), {"2021-01-01", "2022-01-01"})
            self.assertEqual(len(fold_metrics), 10)
            self.assertTrue(fold_metrics["train_excludes_held_areas"].all())
            self.assertTrue(fold_metrics["test_only_held_areas"].all())
            self.assertTrue(summary["aggregation"].eq("fold_mean_sample_sd").all())
            self.assertTrue(summary["n_folds"].eq(5).all())
            self.assertNotIn("n_test", summary)
            random_cv._validate_saved_artifacts(
                artifacts, pd.read_csv(self._write_inputs(root)["forecasting_path"])
            )

    def test_saved_validation_rejects_a_changed_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = random_cv.run_analysis(
                **self._write_inputs(root),
                output_dir=root / "output",
                forecasting_runner=perfect_forecasting_runner,
                nowcasting_runner=perfect_nowcasting_runner,
            )
            summary = pd.read_csv(artifacts["metrics"])
            summary.loc[0, "fold_mean_overall_accuracy"] = 0.0
            summary.to_csv(artifacts["metrics"], index=False)

            with self.assertRaises(ValueError):
                random_cv._validate_saved_artifacts(
                    artifacts, pd.read_csv(self._write_inputs(root)["forecasting_path"])
                )


if __name__ == "__main__":
    unittest.main()
