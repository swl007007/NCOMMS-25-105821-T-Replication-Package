import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_leave_area_out_10pct_robustness as area_holdout
import generate_leave_one_country_out_robustness as loco


def synthetic_area_data(nowcasting=False):
    rows = [
        (1, "2020-01-01", 3, [0.8, 0.0, 0.2, 0.0, 0.0], 10.0, "AAA"),
        (1, "2020-02-01", 4, [0.8, 0.0, 0.0, 0.2, 0.0], 11.0, "AAA"),
        (2, "2020-01-01", 2, [0.8, 0.2, 0.0, 0.0, 0.0], 20.0, "AAA"),
        (2, "2020-02-01", 3, [0.8, 0.0, 0.2, 0.0, 0.0], 21.0, "AAA"),
        (3, "2020-01-01", 1, [1.0, 0.0, 0.0, 0.0, 0.0], 30.0, "BBB"),
        (3, "2020-02-01", 2, [0.8, 0.2, 0.0, 0.0, 0.0], 31.0, "BBB"),
    ]
    records = []
    for area_id, date, phase, shares, signal, country in rows:
        record = {
            "area_id": area_id,
            "date": date,
            "country_code_3": country,
            "overall_phase": phase,
            "signal": signal,
            "fews_ipc_ha": phase,
        }
        record.update({f"phase{i}_percent": shares[i - 1] for i in range(1, 6)})
        if nowcasting:
            for offset, column in enumerate(loco.NOWCAST_FEATURES):
                record[column] = signal + offset / 1000
        records.append(record)
    return pd.DataFrame(records)


def perfect_forecasting_runner(
    forecasting,
    train_mask,
    test_mask,
    split_id,
    general_params,
    phase3_params,
    fold_column="fold_id",
):
    held = loco.add_cumulative_targets(forecasting.loc[test_mask].copy())
    result = held[["area_id", "date", "country_code_3"]].copy()
    result["source_row_index"] = held.index.to_numpy()
    result[fold_column] = split_id
    result["source_overall_phase"] = held["overall_phase"].to_numpy()
    for phase in range(2, 6):
        result[f"phase{phase}_test"] = held[f"phase{phase}_worse"].to_numpy()
        result[f"phase{phase}_pred"] = held[f"phase{phase}_worse"].to_numpy()
    return loco.wide_predictions_to_phases(result)


def perfect_nowcasting_runner(
    forecasting,
    nowcasting,
    train_mask,
    test_mask,
    now_train_mask,
    now_test_mask,
    split_id,
    general_params,
    phase3_params,
    fold_column="fold_id",
):
    result = perfect_forecasting_runner(
        forecasting,
        train_mask,
        test_mask,
        split_id,
        general_params,
        phase3_params,
        fold_column=fold_column,
    )
    result["phase3_layer1_pred"] = result["phase3_pred"]
    result["phase3_residual_pred"] = 0.0
    return result


class LeaveAreaOutRobustnessTests(unittest.TestCase):
    def test_allocate_country_quotas_uses_minimum_one_and_remaining_capacity(self):
        lookup = pd.DataFrame(
            {
                "area_id": [1, 2, 3, 4, 5, 6, 7],
                "country_code_3": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "CCC"],
            }
        )

        observed = area_holdout.allocate_country_area_quotas(lookup, sample_size=4)

        expected = pd.DataFrame(
            {
                "country_code_3": ["AAA", "BBB", "CCC"],
                "area_count": [4, 2, 1],
                "sample_quota": [2, 1, 1],
            }
        )
        pd.testing.assert_frame_equal(observed, expected)

    def test_real_seed_zero_sample_has_the_approved_reproducible_contract(self):
        lookup = pd.read_csv(REPO_ROOT / "1.Source Data" / "area_country_lookup.csv")

        observed = area_holdout.sample_stratified_areas(
            lookup, sample_fraction=0.10, random_state=0
        )

        payload = observed.to_csv(index=False, lineterminator="\n").encode("utf-8")
        self.assertEqual(len(observed), 120)
        self.assertEqual(observed["area_id"].nunique(), 120)
        self.assertEqual(observed["country_code_3"].nunique(), 29)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "b9cd7f069330ed5cdd0991ebe2c4b5a04d8ae418b1bc155010a1cb7d919f06b7",
        )

    def test_area_masks_hold_out_every_date_but_keep_other_same_country_areas(self):
        data = pd.DataFrame(
            {
                "area_id": [1, 1, 2, 2, 3],
                "country_code_3": ["AAA", "AAA", "AAA", "AAA", "BBB"],
                "date": ["2020-01", "2020-02", "2020-01", "2020-02", "2020-01"],
            }
        )
        sample = pd.DataFrame(
            {"area_id": [1, 3], "country_code_3": ["AAA", "BBB"]}
        )

        train_mask, test_mask = area_holdout.area_holdout_masks(data, sample)

        self.assertEqual(test_mask.tolist(), [True, True, False, False, True])
        self.assertEqual(train_mask.tolist(), [False, False, True, True, False])
        self.assertTrue(data.loc[train_mask, "country_code_3"].eq("AAA").any())
        self.assertFalse(set(data.loc[train_mask, "area_id"]) & set(sample["area_id"]))

    def test_pooled_metrics_use_all_test_rows(self):
        predictions = pd.DataFrame(
            {
                "overall_phase": [3, 4, 2, 1],
                "overall_phase_pred": [3, 2, 3, 1],
                "phase3_test": [0.3, 0.4, 0.1, 0.0],
                "phase3_pred": [0.3, 0.2, 0.2, 0.0],
            }
        )

        observed = area_holdout.calculate_pooled_metrics(predictions, "Forecasting")

        self.assertEqual(observed["n_test"], 4)
        self.assertEqual(observed["true_positive"], 1)
        self.assertEqual(observed["false_positive"], 1)
        self.assertEqual(observed["false_negative"], 1)
        self.assertEqual(observed["true_negative"], 1)
        self.assertAlmostEqual(observed["phase3plus_precision"], 0.5)
        self.assertAlmostEqual(observed["phase3plus_recall"], 0.5)
        self.assertAlmostEqual(observed["overall_accuracy"], 0.5)
        self.assertAlmostEqual(observed["phase3plus_r2"], 0.5)

    def test_run_predictions_covers_only_all_rows_of_sampled_areas(self):
        forecasting = synthetic_area_data()
        nowcasting = synthetic_area_data(nowcasting=True).iloc[::-1].reset_index(
            drop=True
        )
        sample = pd.DataFrame(
            {"area_id": [1, 3], "country_code_3": ["AAA", "BBB"]}
        )

        forecast, nowcast, metrics = area_holdout.run_area_holdout_predictions(
            forecasting,
            nowcasting,
            sample,
            general_params={"random_state": 0},
            phase3_params={"random_state": 0},
            workers=1,
            forecasting_runner=perfect_forecasting_runner,
            nowcasting_runner=perfect_nowcasting_runner,
        )

        self.assertEqual(set(forecast["area_id"]), {1, 3})
        self.assertEqual(set(nowcast["area_id"]), {1, 3})
        self.assertEqual(len(forecast), 4)
        self.assertEqual(len(nowcast), 4)
        self.assertFalse(forecast.duplicated(["area_id", "date"]).any())
        self.assertFalse(nowcast.duplicated(["area_id", "date"]).any())
        self.assertEqual(metrics["model"].tolist(), ["Forecasting", "Nowcasting"])
        self.assertEqual(metrics["n_test"].tolist(), [4, 4])
        self.assertEqual(metrics["n_test_areas"].tolist(), [2, 2])
        self.assertEqual(metrics["overall_accuracy"].tolist(), [1.0, 1.0])

    def test_metric_figure_has_four_panels_and_two_model_points_per_panel(self):
        metrics = pd.DataFrame(
            {
                "model": ["Forecasting", "Nowcasting"],
                "phase3plus_precision": [0.70, 0.75],
                "phase3plus_recall": [0.80, 0.85],
                "overall_accuracy": [0.60, 0.65],
                "phase3plus_r2": [0.20, -0.10],
                "n_test": [100, 100],
                "n_test_areas": [20, 20],
            }
        )

        figure = area_holdout.create_metric_figure(metrics)

        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            [
                "Phase 3+ precision",
                "Phase 3+ recall",
                "Overall-phase accuracy",
                "Phase 3+ share R²",
            ],
        )
        for axis in figure.axes:
            plotted_points = sum(
                len(collection.get_offsets()) for collection in axis.collections
            )
            self.assertEqual(plotted_points, 2)
        for axis in figure.axes[:3]:
            lower, upper = axis.get_ylim()
            self.assertGreater(lower, 0.0)
            self.assertLessEqual(upper, 1.0)
        r2_lower, r2_upper = figure.axes[3].get_ylim()
        self.assertLessEqual(r2_lower, -0.1)
        self.assertGreaterEqual(r2_upper, 0.2)
        self.assertLessEqual(r2_lower, 0.0)
        self.assertGreaterEqual(r2_upper, 0.0)
        self.assertIn("Shared 10% stratified area holdout", figure._suptitle.get_text())
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            numeric_annotations = [
                text
                for text in axis.texts
                if text.get_text().replace(".", "", 1).lstrip("-").isdigit()
            ]
            self.assertEqual(len(numeric_annotations), 2)
            for annotation in numeric_annotations:
                bbox = annotation.get_window_extent(renderer)
                self.assertGreaterEqual(bbox.x0, axis.bbox.x0)
                self.assertLessEqual(bbox.x1, axis.bbox.x1)
                self.assertGreaterEqual(bbox.y0, axis.bbox.y0)
                self.assertLessEqual(bbox.y1, axis.bbox.y1)
        plt.close(figure)

    def test_run_analysis_writes_the_complete_reproducible_artifact_set(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            forecasting_path = root / "forecasting.csv"
            nowcasting_path = root / "nowcasting.csv"
            lookup_path = root / "lookup.csv"
            general_path = root / "general.json"
            phase3_path = root / "phase3.json"
            output_dir = root / "output"
            synthetic_area_data().drop(columns="country_code_3").to_csv(
                forecasting_path, index=False
            )
            synthetic_area_data(nowcasting=True).drop(
                columns="country_code_3"
            ).to_csv(nowcasting_path, index=False)
            pd.DataFrame(
                {
                    "area_id": [1, 2, 3],
                    "country_code_3": ["AAA", "AAA", "BBB"],
                }
            ).to_csv(lookup_path, index=False)
            general_path.write_text(json.dumps({"max_depth": 11}), encoding="utf-8")
            phase3_path.write_text(json.dumps({"max_depth": 9}), encoding="utf-8")

            artifacts = area_holdout.run_analysis(
                forecasting_path=forecasting_path,
                nowcasting_path=nowcasting_path,
                country_lookup_path=lookup_path,
                general_params_path=general_path,
                phase3_params_path=phase3_path,
                output_dir=output_dir,
                sample_fraction=0.67,
                random_state=0,
                workers=1,
                forecasting_runner=perfect_forecasting_runner,
                nowcasting_runner=perfect_nowcasting_runner,
            )

            self.assertEqual(
                set(artifacts),
                {
                    "sample",
                    "metrics",
                    "forecasting_predictions",
                    "nowcasting_predictions",
                    "source_audit",
                    "jpg",
                    "png",
                    "pdf",
                },
            )
            self.assertTrue(all(path.exists() for path in artifacts.values()))
            sample = pd.read_csv(artifacts["sample"])
            metrics = pd.read_csv(artifacts["metrics"])
            audit = pd.read_csv(artifacts["source_audit"])
            self.assertEqual(len(sample), 2)
            self.assertEqual(len(metrics), 2)
            self.assertEqual(len(audit), 2)
            self.assertEqual(
                set(audit["evaluation_protocol"]),
                {"fixed_hyperparameter_area_holdout_10pct"},
            )
            self.assertTrue(audit["train_excludes_held_areas"].all())
            self.assertTrue(audit["test_only_held_areas"].all())
            self.assertEqual(set(audit["n_test_areas"]), {2})


if __name__ == "__main__":
    unittest.main()
