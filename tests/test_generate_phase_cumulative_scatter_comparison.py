import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parents[1] / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_phase_cumulative_scatter_comparison as scatter


def prediction_frame(offset=0.0):
    return pd.DataFrame(
        {
            "area_id": [1, 1, 2, 2],
            "date": ["2022-01-01", "2022-02-01", "2022-01-01", "2022-02-01"],
            "country_code_3": ["AAA", "AAA", "BBB", "BBB"],
            "source_row_index": [10, 11, 12, 13],
            "phase2_test": [0.8, 0.6, 0.4, 0.2],
            "phase2_pred": [0.7 + offset, 0.5 + offset, 0.3 + offset, 0.1 + offset],
            "phase3_test": [0.6, 0.4, 0.2, 0.1],
            "phase3_pred": [0.5 + offset, 0.3 + offset, 0.2 + offset, 0.1 + offset],
            "phase4_test": [0.3, 0.2, 0.1, 0.0],
            "phase4_pred": [0.25 + offset, 0.15 + offset, 0.05 + offset, 0.0],
            "phase5_test": [0.1, 0.0, 0.0, 0.0],
            "phase5_pred": [0.08 + offset, 0.01 + offset, 0.0, 0.0],
        }
    )


def contemporaneous_prediction_frame():
    return pd.DataFrame(
        {
            "source_row_index": [20, 21, 22, 23, 24],
            "area_id": [3, 3, 4, 4, 5],
            "date": [
                "2019-01-01",
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2022-02-01",
            ],
            "fold": [0, 1, 2, 3, 4],
            "evaluation_protocol": ["random_5fold_row_cv"] * 5,
            "evaluation_population": ["random_5fold_full_oof_5575"] * 5,
            "phase2_actual": [0.8, 0.6, 0.4, 0.2, 0.1],
            "phase2_contemporaneous": [0.711, 0.509, 0.289, 0.111, 0.001],
            "phase3_actual": [0.6, 0.4, 0.2, 0.1, 0.05],
            "phase3_contemporaneous": [0.511, 0.309, 0.189, 0.111, 0.041],
            "phase4_actual": [0.3, 0.2, 0.1, 0.0, 0.0],
            "phase4_contemporaneous": [0.261, 0.149, 0.049, 0.011, 0.001],
        }
    )


class PhaseCumulativeScatterTests(unittest.TestCase):
    def test_formal_contemporaneous_bundle_matches_frozen_contract(self):
        if not scatter.DEFAULT_CONTEMPORANEOUS_PREDICTIONS.is_file():
            self.skipTest("Formal random-CV bundle has not been regenerated yet.")
        predictions, audit = scatter.load_contemporaneous_artifacts(
            scatter.DEFAULT_CONTEMPORANEOUS_PREDICTIONS,
            scatter.DEFAULT_CONTEMPORANEOUS_AUDIT,
            production_run=True,
        )

        self.assertEqual(
            len(predictions), scatter.DEFAULT_EXPECTED_CONTEMPORANEOUS_ROWS
        )
        self.assertEqual(audit.loc[0, "evaluation_protocol"], "random_5fold_row_cv")
        self.assertEqual(audit.loc[0, "parameter_contract"], "notebook_effective_general_params_all_targets")

    def test_temporal_runner_forwards_estimator_default_jobs_to_both_models(self):
        calls = []

        def fake_fit(model_name, *args):
            calls.append((model_name, args[-1]))
            return model_name, prediction_frame()

        with mock.patch.object(scatter, "_fit_model_from_paths", side_effect=fake_fit):
            observed = scatter.run_temporal_predictions(
                workers=1,
                estimator_n_jobs=None,
            )

        self.assertEqual(list(observed), ["Forecasting", "Nowcasting"])
        self.assertEqual(calls, [("Forecasting", None), ("Nowcasting", None)])

    def test_temporal_masks_put_only_2022_rows_in_test(self):
        forecasting = pd.DataFrame(
            {"date": ["2021-12-01", "2022-01-01", "2022-11-01"]}
        )
        nowcasting = forecasting.iloc[::-1].reset_index(drop=True)

        train, test, now_train, now_test = scatter.temporal_split_masks(
            forecasting, nowcasting
        )

        self.assertEqual(train.tolist(), [True, False, False])
        self.assertEqual(test.tolist(), [False, True, True])
        self.assertEqual(now_train.tolist(), [False, False, True])
        self.assertEqual(now_test.tolist(), [True, True, False])

    def test_build_long_predictions_maps_requested_cumulative_phases(self):
        observed = scatter.build_long_predictions(
            {
                "Forecasting": prediction_frame(),
                "Nowcasting": prediction_frame(0.01),
            },
            expected_test_rows=4,
        )

        self.assertEqual(len(observed), 24)
        self.assertEqual(set(observed["phase"]), {2, 4, 5})
        self.assertEqual(set(observed["phase_label"]), {"Phase 2+", "Phase 4+", "Phase 5"})
        phase4 = observed.loc[
            observed["model"].eq("Forecasting") & observed["phase"].eq(4)
        ]
        self.assertEqual(phase4["actual"].tolist(), [0.3, 0.2, 0.1, 0.0])
        self.assertEqual(phase4["predicted"].tolist(), [0.25, 0.15, 0.05, 0.0])

    def test_build_long_predictions_rejects_reduced_evaluation_population(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 test rows"):
            scatter.build_long_predictions(
                {
                    "Forecasting": prediction_frame(),
                    "Nowcasting": prediction_frame().iloc[:3].copy(),
                },
                expected_test_rows=4,
            )

    def test_three_model_frames_preserve_separate_populations_and_round_predictions(self):
        observed = scatter.build_three_model_prediction_frames(
            {
                "Forecasting": prediction_frame(),
                "Nowcasting": prediction_frame(0.01),
            },
            contemporaneous_prediction_frame(),
            expected_test_rows=4,
            expected_contemporaneous_rows=5,
        )

        self.assertEqual(list(observed), list(scatter.THREE_MODEL_ORDER))
        contemporaneous = observed["Contemporaneous"]
        self.assertEqual(contemporaneous["source_row_index"].tolist(), [20, 21, 22, 23, 24])
        self.assertEqual(contemporaneous["phase2_pred"].tolist(), [0.71, 0.51, 0.29, 0.11, 0.0])
        self.assertEqual(contemporaneous["phase3_test"].tolist(), [0.6, 0.4, 0.2, 0.1, 0.05])
        self.assertEqual(
            set(contemporaneous["evaluation_population"]),
            {"random_5fold_full_oof_5575"},
        )

    def test_phase_figure_has_two_panels_r2_fit_and_identity_lines(self):
        plotting = scatter.build_long_predictions(
            {
                "Forecasting": prediction_frame(),
                "Nowcasting": prediction_frame(0.01),
            },
            expected_test_rows=4,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            figure = scatter.create_phase_scatter_figure(plotting, phase=4)

        self.assertEqual(caught, [])
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            ["Forecasting", "Nowcasting"],
        )
        for axis in figure.axes:
            self.assertTrue(any("R²" in text.get_text() for text in axis.texts))
            labels = {line.get_label() for line in axis.lines}
            self.assertIn("Linear fit", labels)
            self.assertIn("Perfect prediction (y=x)", labels)
            self.assertEqual(axis.get_xlim(), figure.axes[0].get_xlim())
            self.assertEqual(axis.get_ylim(), figure.axes[0].get_ylim())
        plt.close(figure)

    def test_phase_figure_uses_phase_specific_shared_scale(self):
        plotting = scatter.build_long_predictions(
            {
                "Forecasting": prediction_frame(),
                "Nowcasting": prediction_frame(0.01),
            },
            expected_test_rows=4,
        )

        figure = scatter.create_phase_scatter_figure(plotting, phase=4)

        self.assertLess(figure.axes[0].get_xlim()[1], 0.7)
        self.assertLess(figure.axes[0].get_ylim()[1], 0.7)
        self.assertEqual(figure.axes[0].get_xlim(), figure.axes[1].get_xlim())
        self.assertEqual(figure.axes[0].get_ylim(), figure.axes[1].get_ylim())
        plt.close(figure)

    def test_constant_predictions_are_labeled_without_linear_fit_legend(self):
        forecasting = prediction_frame()
        nowcasting = prediction_frame(0.01)
        forecasting["phase5_pred"] = 0.0
        nowcasting["phase5_pred"] = 0.0
        plotting = scatter.build_long_predictions(
            {"Forecasting": forecasting, "Nowcasting": nowcasting},
            expected_test_rows=4,
        )

        figure = scatter.create_phase_scatter_figure(plotting, phase=5)

        legend_labels = {text.get_text() for text in figure.legends[0].get_texts()}
        self.assertNotIn("Linear fit", legend_labels)
        for axis in figure.axes:
            self.assertTrue(
                any(
                    "Linear fit not estimable" in text.get_text()
                    for text in axis.texts
                )
            )
        plt.close(figure)

    def test_three_model_grid_has_nine_panels_and_fit_coefficients(self):
        frames = scatter.build_three_model_prediction_frames(
            {
                "Forecasting": prediction_frame(),
                "Nowcasting": prediction_frame(0.01),
            },
            contemporaneous_prediction_frame(),
            expected_test_rows=4,
            expected_contemporaneous_rows=5,
        )
        plotting = scatter.build_long_predictions(
            frames,
            expected_test_rows=4,
            model_order=scatter.THREE_MODEL_ORDER,
            phase_specs=scatter.THREE_MODEL_PHASE_SPECS,
            expected_rows_by_model={
                "Forecasting": 4,
                "Nowcasting": 4,
                "Contemporaneous": 5,
            },
        )
        metrics = scatter.calculate_three_model_grid_summary(plotting)

        figure = scatter.create_three_model_phase_grid(plotting, metrics)

        self.assertEqual(len(figure.axes), 9)
        self.assertEqual(
            [axis.get_title(loc="center") for axis in figure.axes[:3]],
            [
                "Forecasting\n(2022 temporal holdout)",
                "Nowcasting\n(2022 temporal holdout)",
                "Contemporaneous\n(random 5-fold CV)",
            ],
        )
        self.assertEqual(len(metrics), 9)
        self.assertEqual(set(metrics["phase"]), {2, 3, 4})
        self.assertTrue(metrics["fit_estimable"].all())
        for axis in figure.axes:
            annotations = "\n".join(text.get_text() for text in axis.texts)
            self.assertIn("Intercept", annotations)
            self.assertIn("Slope", annotations)
            self.assertIn("n =", annotations)
            labels = {line.get_label() for line in axis.lines}
            self.assertIn("Linear fit", labels)
            self.assertIn("Perfect prediction (y=x)", labels)
        for row_start in (0, 3, 6):
            row_axes = figure.axes[row_start : row_start + 3]
            self.assertEqual(len({axis.get_xlim() for axis in row_axes}), 1)
            self.assertEqual(len({axis.get_ylim() for axis in row_axes}), 1)
        plt.close(figure)

    def test_write_artifacts_exports_three_phases_tables_and_audit(self):
        predictions = {
            "Forecasting": prediction_frame(),
            "Nowcasting": prediction_frame(0.01),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifacts = scatter.write_artifacts(
                predictions,
                output_dir,
                expected_test_rows=4,
                expected_contemporaneous_rows=5,
                source_audit=pd.DataFrame(
                    {"model": ["Forecasting", "Nowcasting"], "rows": [4, 4]}
                ),
                contemporaneous_predictions=contemporaneous_prediction_frame(),
                contemporaneous_audit=pd.DataFrame(
                    {"model": ["Contemporaneous"], "test_rows": [4]}
                ),
            )

            expected_stems = {
                "phase2plus_actual_vs_predicted_forecasting_nowcasting",
                "phase4plus_actual_vs_predicted_forecasting_nowcasting",
                "phase5_actual_vs_predicted_forecasting_nowcasting",
            }
            png_shapes = set()
            for stem in expected_stems:
                for suffix in ("jpg", "png", "pdf"):
                    path = output_dir / f"{stem}.{suffix}"
                    self.assertEqual(artifacts[f"{stem}_{suffix}"], path)
                    self.assertTrue(path.exists())
                    self.assertGreater(path.stat().st_size, 0)
                    if suffix == "png":
                        png_shapes.add(plt.imread(path).shape)
            self.assertEqual(len(png_shapes), 1)

            for suffix in ("jpg", "png", "pdf"):
                path = output_dir / f"{scatter.THREE_MODEL_FIGURE_STEM}.{suffix}"
                self.assertEqual(artifacts[f"three_model_grid_{suffix}"], path)
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

            long_table = pd.read_csv(artifacts["predictions_csv"])
            metrics = pd.read_csv(artifacts["metrics_csv"])
            audit = pd.read_csv(artifacts["source_audit_csv"])
            three_model_long = pd.read_csv(artifacts["three_model_predictions_csv"])
            three_model_metrics = pd.read_csv(artifacts["three_model_metrics_csv"])
            self.assertEqual(len(long_table), 24)
            self.assertEqual(len(metrics), 6)
            self.assertEqual(len(audit), 3)
            self.assertEqual(set(metrics["n_test"]), {4})
            self.assertEqual(len(three_model_long), 39)
            self.assertEqual(len(three_model_metrics), 9)
            self.assertEqual(set(three_model_metrics["n_test"]), {4, 5})
            self.assertEqual(
                set(three_model_metrics["evaluation_protocol"]),
                {"fixed_2022_temporal_holdout", "random_5fold_row_cv"},
            )

    def test_duplicate_contemporaneous_source_row_does_not_write_legacy_artifacts(self):
        invalid = contemporaneous_prediction_frame()
        invalid.loc[0, "source_row_index"] = invalid.loc[1, "source_row_index"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with self.assertRaisesRegex(ValueError, "duplicate source rows"):
                scatter.write_artifacts(
                    {
                        "Forecasting": prediction_frame(),
                        "Nowcasting": prediction_frame(0.01),
                    },
                    output_dir,
                    expected_test_rows=4,
                    expected_contemporaneous_rows=5,
                    source_audit=pd.DataFrame(
                        {"model": ["Forecasting", "Nowcasting"], "rows": [4, 4]}
                    ),
                    contemporaneous_predictions=invalid,
                    contemporaneous_audit=pd.DataFrame(
                        {"model": ["Contemporaneous"], "test_rows": [4]}
                    ),
                )

            self.assertEqual(list(output_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
