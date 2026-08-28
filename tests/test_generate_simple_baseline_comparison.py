from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import os
import re

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-simple-baseline-tests")
)
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE = REPO_ROOT / "2.Source Code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))

import generate_simple_baseline_comparison as simple


def synthetic_metrics() -> pd.DataFrame:
    values = {
        ("Nowcasting", "Persistence"): (0.823, 0.745),
        ("Nowcasting", "Multinomial"): (0.795, 0.595),
        ("Nowcasting", "Ordered Probit"): (0.810, 0.680),
        ("Nowcasting", "Ensemble OLS"): (0.825, 0.770),
        ("Forecasting", "Persistence"): (0.853, 0.612),
        ("Forecasting", "Multinomial"): (0.791, 0.673),
        ("Forecasting", "Ordered Probit"): (0.805, 0.700),
        ("Forecasting", "Ensemble OLS"): (0.830, 0.750),
    }
    rows = []
    for task in simple.TASK_ORDER:
        for method in simple.METHOD_ORDER:
            precision, recall = values[(task, method)]
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "method_role": "baseline",
                    "phase3plus_precision": precision,
                    "phase3plus_recall": recall,
                    "n_test": 1170,
                    "test_key_sha256": "a" * 64,
                }
            )
        reference = simple.ORIGINAL_REFERENCES[task]
        rows.append(
            {
                "task": task,
                "method": simple.ORIGINAL_METHOD,
                "method_role": "original_reference",
                "phase3plus_precision": reference["phase3plus_precision"],
                "phase3plus_recall": reference["phase3plus_recall"],
                "n_test": 1170,
                "test_key_sha256": "a" * 64,
            }
        )
    return pd.DataFrame(rows)


class ConstantsAndPhaseTests(unittest.TestCase):
    def test_order_constants_are_exact(self):
        self.assertEqual(simple.TASK_ORDER, ("Nowcasting", "Forecasting"))
        self.assertEqual(
            simple.METHOD_ORDER,
            ("Persistence", "Multinomial", "Ordered Probit", "Ensemble OLS"),
        )
        self.assertEqual(
            simple.METRIC_ORDER,
            ("phase3plus_precision", "phase3plus_recall"),
        )
        self.assertEqual(simple.ORIGINAL_METHOD, "Main result")

    def test_evaluation_phase_uses_twenty_percent_cumulative_rule(self):
        data = pd.DataFrame(
            {
                "phase1_percent": [0.81, 0.69, 0.50, 0.40, 0.40],
                "phase2_percent": [0.19, 0.20, 0.19, 0.19, 0.19],
                "phase3_percent": [0.00, 0.11, 0.20, 0.19, 0.10],
                "phase4_percent": [0.00, 0.00, 0.11, 0.20, 0.10],
                "phase5_percent": [0.00, 0.00, 0.00, 0.02, 0.21],
            }
        )
        self.assertEqual(simple.derive_evaluation_phase(data).tolist(), [1, 2, 3, 4, 5])

    def test_temporal_masks_put_cutoff_in_test(self):
        train, test = simple.temporal_masks(
            pd.Series(["2021-12-01", "2022-01-01", "2022-02-01"])
        )
        self.assertEqual(train.tolist(), [True, False, False])
        self.assertEqual(test.tolist(), [False, True, True])

    def test_canonical_key_hash_is_row_order_invariant(self):
        data = pd.DataFrame(
            {
                "area_id": [2, 1],
                "date": ["2022-02-01", "2022-01-01"],
            }
        )
        self.assertEqual(
            simple.canonical_key_sha256(data),
            simple.canonical_key_sha256(data.iloc[::-1].reset_index(drop=True)),
        )


class PreprocessorTests(unittest.TestCase):
    def test_train_only_preprocessor_is_finite_and_full_rank(self):
        train = pd.DataFrame(
            {
                "signal": [1.0, 2.0, 3.0, 4.0, 5.0],
                "duplicate": [1.0, 2.0, 3.0, 4.0, 5.0],
                "linear": [2.0, 4.0, 6.0, 8.0, 10.0],
                "partial": [1.0, np.nan, 2.0, 4.0, 8.0],
                "constant": [7.0] * 5,
                "all_missing": [np.nan] * 5,
            }
        )
        test = pd.DataFrame(
            {
                "signal": [1000.0, 6.0],
                "duplicate": [1000.0, 6.0],
                "linear": [2000.0, 12.0],
                "partial": [np.nan, 16.0],
                "constant": [7.0, 7.0],
                "all_missing": [5.0, 6.0],
            }
        )
        fitted = simple.fit_numeric_preprocessor(
            train,
            list(train.columns),
            task="Forecasting",
            method="Ordered Probit",
            layer="direct",
        )
        train_matrix = fitted.transform(train)
        test_matrix = fitted.transform(test)
        self.assertTrue(np.isfinite(train_matrix).all())
        self.assertTrue(np.isfinite(test_matrix).all())
        self.assertEqual(np.linalg.matrix_rank(train_matrix), train_matrix.shape[1])
        self.assertFalse(np.any(np.ptp(train_matrix, axis=0) == 0))
        self.assertLess(train_matrix.shape[1], len(train.columns))
        reasons = set(fitted.manifest["exclusion_reason"].dropna())
        self.assertIn("all_missing_in_training", reasons)
        self.assertIn("zero_variance_after_imputation", reasons)
        self.assertTrue(
            {"exact_duplicate_after_scaling", "pivoted_qr_rank_dependent"}.intersection(
                reasons
            )
        )

    def test_ols_design_adds_exactly_one_intercept(self):
        matrix = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        design = simple.add_ols_intercept(matrix)
        self.assertTrue(np.allclose(design[:, 0], 1.0))
        self.assertEqual(design.shape[1], matrix.shape[1] + 1)
        self.assertEqual(np.linalg.matrix_rank(design), design.shape[1])


class StatisticalModelTests(unittest.TestCase):
    def test_ols_fit_predicts_known_linear_relation(self):
        x_train = np.arange(8, dtype=float).reshape(-1, 1)
        y_train = 2.0 + 3.0 * x_train[:, 0]
        x_test = np.array([[8.0], [9.0]])
        predictions, audit = simple.fit_ols_arrays(x_train, y_train, x_test)
        np.testing.assert_allclose(predictions, [26.0, 29.0], atol=1e-10)
        self.assertEqual(audit["design_rank"], 2)
        self.assertTrue(audit["parameters_finite"])

    def test_ordered_probit_returns_valid_five_class_probabilities(self):
        rng = np.random.default_rng(17)
        x_train = rng.normal(size=(400, 3))
        latent = x_train[:, 0] - 0.5 * x_train[:, 1] + rng.normal(
            scale=0.8, size=400
        )
        y_train = np.digitize(latent, [-1.2, -0.3, 0.5, 1.3]) + 1
        x_test = rng.normal(size=(20, 3))
        predicted, probabilities, audit = simple.fit_ordered_probit_arrays(
            x_train,
            y_train,
            x_test,
            optimizer="bfgs",
            maxiter=500,
        )
        self.assertEqual(probabilities.shape, (20, 5))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(np.isin(predicted, [1, 2, 3, 4, 5]).all())
        self.assertTrue(audit["converged"])
        self.assertTrue(audit["transformed_cutpoints_strictly_ordered"])

    def test_ordered_probit_rejects_missing_class(self):
        rng = np.random.default_rng(3)
        with self.assertRaisesRegex(ValueError, "five ordered classes"):
            simple.fit_ordered_probit_arrays(
                rng.normal(size=(100, 2)),
                np.tile([1, 2, 3, 4], 25),
                rng.normal(size=(5, 2)),
            )


class MetricTests(unittest.TestCase):
    def test_pooled_metrics_use_phase3plus_confusion_counts(self):
        predictions = pd.DataFrame(
            {
                "actual_phase": [1, 2, 3, 4, 5],
                "predicted_phase": [1, 3, 2, 4, 5],
            }
        )
        result = simple.calculate_pooled_metrics(predictions)
        self.assertEqual(
            (
                result["true_positive"],
                result["false_positive"],
                result["false_negative"],
                result["true_negative"],
            ),
            (2, 1, 1, 1),
        )
        self.assertAlmostEqual(result["phase3plus_precision"], 2 / 3)
        self.assertAlmostEqual(result["phase3plus_recall"], 2 / 3)
        self.assertAlmostEqual(result["overall_accuracy"], 3 / 5)


class FigureTests(unittest.TestCase):
    def test_figure_has_exact_panel_bar_and_reference_contract(self):
        figure = simple.create_simple_baseline_comparison_figure(synthetic_metrics())
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(
            [axis.get_title(loc="left") for axis in figure.axes],
            ["Phase 3+ precision", "Phase 3+ recall", "", ""],
        )
        self.assertEqual(
            [axis.get_ylabel() for axis in figure.axes],
            ["Nowcasting", "", "Forecasting", ""],
        )
        letters = [
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
            if text.get_text() in list("abcd")
        ]
        self.assertEqual(letters, list("abcd"))
        for axis in figure.axes:
            self.assertEqual(len(axis.patches), 4)
            reference_lines = [
                line
                for line in axis.lines
                if line.get_linestyle() == "--"
                and np.asarray(line.get_ydata()).shape == (2,)
                and np.allclose(line.get_ydata()[0], line.get_ydata()[1])
            ]
            self.assertEqual(len(reference_lines), 1)
            self.assertEqual(len(axis.collections), 1)
            marker = axis.collections[0]
            self.assertEqual(marker.get_offsets().shape, (1, 2))
            np.testing.assert_allclose(
                marker.get_facecolors()[0], matplotlib.colors.to_rgba("white")
            )
            np.testing.assert_allclose(
                marker.get_edgecolors()[0], matplotlib.colors.to_rgba("#5F5F5F")
            )
            self.assertEqual(axis.get_ylim(), (0.0, 1.0))
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            [*simple.METHOD_ORDER, simple.ORIGINAL_METHOD],
        )
        self.assertEqual(
            figure._suptitle.get_text(),
            "Simple baselines versus the main result: "
            "2022 temporal evaluation, 1,170 area-date rows",
        )
        simple.plt.close(figure)

    def test_figure_numeric_labels_do_not_overlap(self):
        figure = simple.create_simple_baseline_comparison_figure(synthetic_metrics())
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes:
            numeric_labels = [
                text
                for text in axis.texts
                if re.fullmatch(r"\d\.\d{3}", text.get_text())
            ]
            self.assertEqual(len(numeric_labels), 5)
            boxes = [text.get_window_extent(renderer) for text in numeric_labels]
            for index, first in enumerate(boxes):
                for second in boxes[index + 1 :]:
                    self.assertFalse(first.overlaps(second))
        simple.plt.close(figure)

    def test_figure_values_match_metrics(self):
        metrics = synthetic_metrics()
        figure = simple.create_simple_baseline_comparison_figure(metrics)
        for index, axis in enumerate(figure.axes):
            task = simple.TASK_ORDER[index // 2]
            metric = simple.METRIC_ORDER[index % 2]
            expected = (
                metrics.loc[
                    metrics["task"].eq(task)
                    & metrics["method"].isin(simple.METHOD_ORDER)
                ]
                .set_index("method")
                .loc[list(simple.METHOD_ORDER), metric]
                .to_numpy()
            )
            np.testing.assert_allclose(
                [patch.get_height() for patch in axis.patches], expected
            )
            reference = metrics.loc[
                metrics["task"].eq(task)
                & metrics["method"].eq(simple.ORIGINAL_METHOD),
                metric,
            ].iloc[0]
            dashed = next(line for line in axis.lines if line.get_linestyle() == "--")
            self.assertTrue(np.allclose(dashed.get_ydata(), [reference, reference]))
        simple.plt.close(figure)

    def test_save_figure_writes_three_formats(self):
        figure = simple.create_simple_baseline_comparison_figure(synthetic_metrics())
        with tempfile.TemporaryDirectory() as directory:
            paths = simple.save_simple_baseline_comparison_figure(
                figure, Path(directory)
            )
            self.assertEqual(paths["figure_jpg"].read_bytes()[:2], b"\xff\xd8")
            self.assertEqual(
                paths["figure_png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
            )
            self.assertEqual(paths["figure_pdf"].read_bytes()[:4], b"%PDF")
            for key in ("figure_jpg", "figure_png"):
                with Image.open(paths[key]) as image:
                    dpi = image.info.get("dpi")
                    self.assertIsNotNone(dpi)
                    self.assertAlmostEqual(float(dpi[0]), 300.0, delta=1.0)
                    self.assertAlmostEqual(float(dpi[1]), 300.0, delta=1.0)
            pdf_payload = paths["figure_pdf"].read_bytes()
            self.assertEqual(len(re.findall(rb"/Type\s*/Page\b", pdf_payload)), 1)
            self.assertNotIn(b"/Subtype /Image", pdf_payload)
            self.assertNotIn(b"/CreationDate", pdf_payload)
            self.assertNotIn(b"/ModDate", pdf_payload)
        simple.plt.close(figure)


class LiveInputContractTests(unittest.TestCase):
    def test_live_temporal_population_and_label_disagreement_contract(self):
        bundle = simple.load_prepared_inputs()
        self.assertEqual(len(bundle.forecasting), 5575)
        self.assertEqual(len(bundle.nowcasting), 5575)
        self.assertEqual(int(bundle.forecasting_test_mask.sum()), 1170)
        self.assertEqual(int(bundle.nowcasting_test_mask.sum()), 1170)
        self.assertEqual(bundle.source_label_disagreement_test_count, 5)
        self.assertEqual(
            simple.canonical_key_sha256(
                bundle.forecasting.loc[bundle.forecasting_test_mask]
            ),
            simple.canonical_key_sha256(
                bundle.nowcasting.loc[bundle.nowcasting_test_mask]
            ),
        )


class LiveAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = simple.load_prepared_inputs()

    def test_existing_adapters_reproduce_frozen_metrics(self):
        persistence_groups, _, _ = simple.fit_persistence_adapter(self.bundle)
        multinomial_groups, _, _ = simple.fit_multinomial_adapter(self.bundle)
        expected = {
            ("Persistence", "Forecasting"): (0.8526148969889065, 0.6120591581342435),
            ("Persistence", "Nowcasting"): (0.8228643216080402, 0.745164960182025),
            ("Multinomial", "Forecasting"): (0.7914438502673797, 0.6734926052332195),
            ("Multinomial", "Nowcasting"): (0.7948328267477204, 0.5949943117178612),
        }
        for method, groups in (
            ("Persistence", persistence_groups),
            ("Multinomial", multinomial_groups),
        ):
            for task, frame in groups.items():
                metrics = simple.calculate_pooled_metrics(frame)
                precision, recall = expected[(method, task)]
                self.assertAlmostEqual(metrics["phase3plus_precision"], precision)
                self.assertAlmostEqual(metrics["phase3plus_recall"], recall)
                self.assertEqual(len(frame), 1170)
                self.assertEqual(frame["test_key_sha256"].nunique(), 1)

    def test_ensemble_ols_preserves_components_and_complete_keys(self):
        groups, manifest, audit = simple.fit_ensemble_ols_adapter(self.bundle)
        self.assertFalse(manifest.empty)
        self.assertNotIn("evaluation_phase", set(manifest["feature"]))
        layer_counts = manifest.groupby(["task", "layer"], observed=True).size()
        self.assertEqual(
            int(layer_counts.loc[("Forecasting", "layer1_shared")]), 106
        )
        self.assertEqual(int(layer_counts.loc[("Nowcasting", "layer1_shared")]), 106)
        self.assertEqual(int(layer_counts.loc[("Nowcasting", "layer2_residual")]), 69)
        source_feature_counts = (
            audit.set_index("task")["source_feature_count"].astype(int).to_dict()
        )
        self.assertEqual(source_feature_counts["Forecasting"], 106)
        self.assertEqual(source_feature_counts["Nowcasting"], 175)
        self.assertEqual(set(groups), {"Forecasting", "Nowcasting"})
        for task, frame in groups.items():
            self.assertEqual(len(frame), 1170)
            self.assertEqual(
                simple.canonical_key_sha256(frame), self.bundle.test_key_sha256
            )
            for phase in range(2, 6):
                if task == "Forecasting":
                    np.testing.assert_allclose(
                        frame[f"phase{phase}_pred_raw"],
                        frame[f"phase{phase}_layer1_pred"],
                    )
                    self.assertTrue(
                        frame[f"phase{phase}_residual_pred"].isna().all()
                    )
                else:
                    np.testing.assert_allclose(
                        frame[f"phase{phase}_pred_raw"],
                        frame[f"phase{phase}_layer1_pred"]
                        + frame[f"phase{phase}_residual_pred"],
                        atol=1e-12,
                    )
                np.testing.assert_allclose(
                    frame[f"phase{phase}_pred_rounded"],
                    frame[f"phase{phase}_pred_raw"].round(2),
                )
        self.assertTrue(
            audit["out_of_range_prediction_cell_count"].astype(int).gt(0).all()
        )
        self.assertTrue(
            audit["cumulative_order_violation_row_count"].astype(int).gt(0).all()
        )


if __name__ == "__main__":
    unittest.main()
