from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "2.Source Code"
MODULE_PATH = SOURCE_DIR / "generate_direct_phase3_vs_phase45_rescue.py"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def load_module():
    if not MODULE_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "generate_direct_phase3_vs_phase45_rescue", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load rescue generator: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RescueTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rescue = load_module()

    def require(self, name):
        self.assertIsNotNone(
            self.rescue,
            f"Direct Phase-3-vs-Phase-4/5 generator does not exist: {MODULE_PATH}",
        )
        value = getattr(self.rescue, name, None)
        self.assertIsNotNone(value, f"Missing generator contract: {name}")
        return value


class FrozenContractTests(RescueTestCase):
    def test_exact_artifact_candidate_and_method_contracts(self):
        artifacts = self.require("EXPECTED_ARTIFACTS")
        self.assertEqual(len(artifacts), 30)
        self.assertEqual(len(set(artifacts)), 30)
        self.assertTrue(
            all(name.startswith("direct_phase3_vs_phase45_rescue_") for name in artifacts)
        )
        self.assertEqual(
            self.require("CANDIDATE_ORDER"),
            ("unweighted", "sqrt_balance", "full_balance"),
        )
        self.assertEqual(
            self.require("METHOD_ORDER"),
            (
                "frozen_base",
                "legacy_direct_exact_phase4_050",
                "direct_phase45_unweighted",
                "direct_phase45_sqrt_balance",
                "direct_phase45_full_balance",
            ),
        )
        joined = "\n".join([*artifacts, *self.require("METHOD_ORDER")])
        self.assertNotIn("deployment_aligned", joined)
        self.assertNotIn("single_score", joined)
        self.assertNotIn("no_rescue_sentinel", joined)

    def test_temporal_folds_are_exact_and_forward(self):
        folds = self.require("BASE_FOLDS")
        self.assertEqual(
            [(fold.fold_id, fold.training_years, fold.validation_year) for fold in folds],
            [
                ("B1", (2017,), 2018),
                ("B2", (2017, 2018), 2019),
                ("B3", (2017, 2018, 2019), 2020),
                ("B4", (2017, 2018, 2019, 2020), 2021),
            ],
        )

    def test_model_and_table_row_arithmetic_is_frozen(self):
        self.assertEqual(len(self.require("MODEL_BASENAMES")), 9)
        self.assertEqual(self.require("EXPECTED_OOF_ROWS"), 22560)
        self.assertEqual(self.require("EXPECTED_BENCHMARK_ROWS_LONG"), 11700)
        self.assertEqual(self.require("EXPECTED_OOF_STABILITY_ROWS"), 30)
        self.assertEqual(self.require("EXPECTED_COUNTRY_METRIC_ROWS"), 270)
        self.assertEqual(self.require("EXPECTED_BINARY_CONFUSION_ROWS"), 80)
        self.assertEqual(self.require("EXPECTED_FIVE_CLASS_CONFUSION_ROWS"), 250)
        self.assertEqual(self.require("EXPECTED_FEATURE_MANIFEST_ROWS"), 281)
        self.assertEqual(self.require("EXPECTED_MODEL_AUDIT_ROWS"), 45)
        self.assertEqual(self.require("EXPECTED_BOOTSTRAP_DRAW_ROWS"), 16000)
        self.assertEqual(self.require("EXPECTED_BOOTSTRAP_SUMMARY_ROWS"), 8)


class TargetWeightAndModelTests(RescueTestCase):
    def test_phase4_and_phase5_are_positive(self):
        build_target = self.require("build_severe_rescue_target")
        result = build_target(pd.Series([1, 2, 3, 4, 5]))
        self.assertEqual(result.tolist(), [0, 0, 0, 1, 1])

    def test_candidate_positive_weights_use_fold_local_ratio(self):
        positive_weight = self.require("candidate_positive_weight")
        ratio = 16.0
        self.assertEqual(positive_weight("unweighted", ratio), 1.0)
        self.assertEqual(positive_weight("sqrt_balance", ratio), 4.0)
        self.assertEqual(positive_weight("full_balance", ratio), 16.0)
        with self.assertRaises(ValueError):
            positive_weight("unknown", ratio)

    def test_row_weights_apply_only_to_positive_rows(self):
        build_weights = self.require("build_candidate_sample_weight")
        target = pd.Series([0, 1, 0, 1], dtype=np.uint8)
        weights, audit = build_weights(target, "sqrt_balance")
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(audit["negative_count"], 2)
        self.assertEqual(audit["positive_count"], 2)
        self.assertEqual(audit["class_ratio"], 1.0)
        target = pd.Series([0, 0, 0, 1], dtype=np.uint8)
        weights, audit = build_weights(target, "full_balance")
        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0, 3.0])
        self.assertEqual(audit["positive_row_weight"], 3.0)

    def test_nowcasting_layers_receive_identical_sample_weight(self):
        fit_bundle = self.require("fit_direct_candidate_models")

        class RecordingClassifier:
            def __init__(self, **parameters):
                self.parameters = parameters
                self.weights = None

            def fit(self, matrix, target, sample_weight=None):
                self.weights = np.asarray(sample_weight, dtype=float).copy()
                return self

            def predict_proba(self, matrix):
                score = np.full(len(matrix), 0.25)
                return np.column_stack([1.0 - score, score])

        class RecordingRegressor:
            def __init__(self, **parameters):
                self.parameters = parameters
                self.weights = None
                self.target = None

            def fit(self, matrix, target, sample_weight=None):
                self.weights = np.asarray(sample_weight, dtype=float).copy()
                self.target = np.asarray(target, dtype=float).copy()
                return self

            def predict(self, matrix):
                return np.zeros(len(matrix))

        weights = np.array([1.0, 2.0, 1.0, 2.0])
        result = fit_bundle(
            task="Nowcasting",
            layer1_matrix=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}),
            layer2_matrix=pd.DataFrame({"z": [1.0, 1.0, 2.0, 2.0]}),
            target=pd.Series([0, 1, 0, 1], dtype=np.uint8),
            sample_weight=weights,
            classifier_factory=RecordingClassifier,
            regressor_factory=RecordingRegressor,
            classifier_parameters={"scale_pos_weight": 1},
            regressor_parameters={"scale_pos_weight": 1},
        )
        np.testing.assert_array_equal(result["layer1_model"].weights, weights)
        np.testing.assert_array_equal(result["layer2_model"].weights, weights)
        np.testing.assert_allclose(result["residual_target"], [-0.25, 0.75, -0.25, 0.75])
        self.assertEqual(result["layer1_model"].parameters["scale_pos_weight"], 1)
        self.assertEqual(result["layer2_model"].parameters["scale_pos_weight"], 1)


class MetricSelectionAndConfusionTests(RescueTestCase):
    def test_binary_metrics_include_exact_f2_and_metric_specific_na(self):
        metrics = self.require("binary_metric_bundle")(
            np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])
        )
        self.assertEqual(metrics["true_positive"], 1)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertAlmostEqual(metrics["f2"], 0.5)
        no_predictions = self.require("binary_metric_bundle")(
            np.array([0, 0]), np.array([0, 0])
        )
        self.assertTrue(math.isnan(no_predictions["precision"]))
        self.assertTrue(math.isnan(no_predictions["recall"]))
        self.assertTrue(math.isnan(no_predictions["f1"]))
        self.assertTrue(math.isnan(no_predictions["f2"]))

    def test_rescue_changes_only_phase3_to_phase4(self):
        apply_rescue = self.require("apply_phase45_rescue")
        base = np.array([1, 2, 3, 3, 4, 5])
        score = np.array([1.0, 1.0, 0.49, 0.50, 1.0, 1.0])
        result = apply_rescue(base, score, 0.5)
        self.assertEqual(result.tolist(), [1, 2, 3, 4, 4, 5])
        np.testing.assert_array_equal(base >= 3, result >= 3)

    def candidate_oof(self, candidate, scores):
        return pd.DataFrame(
            {
                "task": "Forecasting",
                "candidate_id": candidate,
                "base_oof_fold": ["B1"] * 6,
                "area_id": np.arange(1, 7),
                "date": pd.to_datetime(["2018-01-01"] * 6),
                "reconstructed_overall_phase": [4, 5, 3, 2, 1, 3],
                "base_overall_phase_pred": [3] * 6,
                "in_base_phase3_gate": [True] * 6,
                "severe_rescue_target": [1, 1, 0, 0, 0, 0],
                "direct_phase45_score": scores,
            }
        )

    def test_frontier_contains_above_max_reference_and_all_unique_scores(self):
        build = self.require("build_threshold_frontier")
        frame = self.candidate_oof("unweighted", [0.9, 0.8, 0.7, 0.2, 0.1, 0.7])
        frontier = build(frame)
        self.assertEqual(len(frontier), 6)
        self.assertEqual(int(frontier["is_above_max_reference"].sum()), 1)
        sentinel = frontier.loc[frontier["is_above_max_reference"]].iloc[0]
        self.assertGreater(float(sentinel["threshold"]), 0.9)
        self.assertEqual(int(sentinel["total_promotions"]), 0)

    def test_f2_first_policy_selection_and_no_effective_branch(self):
        build = self.require("build_threshold_frontier")
        select = self.require("select_task_policies")
        frontiers = []
        for candidate, scores in (
            ("unweighted", [0.9, 0.8, 0.7, 0.2, 0.1, 0.6]),
            ("sqrt_balance", [0.95, 0.7, 0.8, 0.3, 0.2, 0.6]),
            ("full_balance", [0.99, 0.98, 0.97, 0.96, 0.95, 0.94]),
        ):
            frontiers.append(build(self.candidate_oof(candidate, scores)))
        marked, policies = select(pd.concat(frontiers, ignore_index=True))
        self.assertEqual(len(policies), 3)
        self.assertEqual(int(policies["primary_selected"].sum()), 1)
        self.assertEqual(set(policies["selection_status"]), {"selected"})
        self.assertEqual(int(marked["within_candidate_selected"].sum()), 3)

        ineffective = []
        for candidate in self.require("CANDIDATE_ORDER"):
            frame = self.candidate_oof(
                candidate, [0.1, 0.2, 0.9, 0.8, 0.7, 0.6]
            )
            frame["reconstructed_overall_phase"] = [3, 3, 3, 2, 1, 3]
            frame["severe_rescue_target"] = 0
            ineffective.append(build(frame))
        _, no_effect = select(pd.concat(ineffective, ignore_index=True))
        self.assertEqual(set(no_effect["selection_status"]), {"no_effective_rescue"})
        self.assertEqual(int(no_effect["primary_selected"].sum()), 0)
        self.assertTrue(no_effect["is_above_max_reference"].all())

    def test_confusion_builders_keep_all_zero_cells(self):
        binary = self.require("build_binary_confusion_cells")(
            np.array([0, 0, 1]), np.array([0, 0, 0])
        )
        self.assertEqual(len(binary), 4)
        self.assertEqual(int(binary["count"].sum()), 3)
        five = self.require("build_five_class_confusion_cells")(
            np.array([3, 4, 5]), np.array([3, 4, 4])
        )
        self.assertEqual(len(five), 25)
        self.assertEqual(int(five["count"].sum()), 3)


class LiveDataAndIsolationTests(RescueTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if cls.rescue is not None:
            cls.forecasting, cls.nowcasting, cls.layer1_features = (
                cls.rescue.load_prepared_inputs()
            )

    def test_phase345_training_population_and_features_are_exact(self):
        population, target = self.require("build_phase345_training_population")(
            self.forecasting
        )
        self.assertEqual(len(population), 2530)
        self.assertEqual(int(target.sum()), 381)
        self.assertEqual(
            population["reconstructed_overall_phase"].value_counts().to_dict(),
            {3: 2149, 4: 369, 5: 12},
        )
        self.assertEqual(len(self.layer1_features), 106)
        self.assertEqual(len(self.require("NOWCAST_FEATURES")), 69)

    def test_predecessor_evidence_is_complete_and_keyed(self):
        oof = self.require("load_frozen_oof_gate_evidence")()
        benchmark = self.require("load_frozen_benchmark_evidence")()
        self.assertEqual(len(oof), 7520)
        self.assertEqual(len(benchmark), 2340)
        self.assertFalse(oof.duplicated(["task", "area_id", "date"]).any())
        self.assertFalse(benchmark.duplicated(["task", "area_id", "date"]).any())
        self.assertTrue(
            np.array_equal(
                oof["in_base_phase3_gate"].to_numpy(dtype=bool),
                oof["base_overall_phase_pred"].to_numpy(dtype=int) == 3,
            )
        )

    def test_feature_manifest_has_shared_281_row_contract(self):
        manifest = self.require("build_feature_manifest")(
            self.forecasting, self.nowcasting, self.layer1_features
        )
        self.assertEqual(len(manifest), 281)
        self.assertFalse(
            manifest.duplicated(["task", "model_component", "feature_order"]).any()
        )

    def test_nonempty_output_is_rejected_before_formal_environment_check(self):
        run = self.require("run_generation")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            with patch.object(
                self.rescue,
                "assert_formal_environment",
                side_effect=AssertionError("environment check must not run"),
            ):
                with self.assertRaises(FileExistsError):
                    run(output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_protected_manifest_excludes_only_the_requested_new_target(self):
        manifest = self.require("protected_artifact_manifest_sha256")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "old" / "artifact.txt"
            protected.parent.mkdir()
            protected.write_text("old", encoding="utf-8")
            target = root / "new"
            target.mkdir()
            (target / "artifact.txt").write_text("new-a", encoding="utf-8")
            with patch.object(self.rescue, "PRODUCED_GRAPH_DIR", root):
                before = manifest(excluded_paths=(target,))
                (target / "artifact.txt").write_text("new-b", encoding="utf-8")
                self.assertEqual(before, manifest(excluded_paths=(target,)))
                protected.write_text("changed", encoding="utf-8")
                self.assertNotEqual(before, manifest(excluded_paths=(target,)))

    def test_custom_target_does_not_exclude_default_formal_directory(self):
        manifest = self.require("protected_artifact_manifest_sha256")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "direct_phase3_vs_phase45_rescue"
            default.mkdir()
            protected = default / "formal.txt"
            protected.write_text("formal-a", encoding="utf-8")
            custom = root / "repeat"
            custom.mkdir()
            (custom / "repeat.txt").write_text("repeat-a", encoding="utf-8")
            with (
                patch.object(self.rescue, "PRODUCED_GRAPH_DIR", root),
                patch.object(self.rescue, "DEFAULT_OUTPUT_DIR", default),
            ):
                before = manifest(excluded_paths=(custom,))
                protected.write_text("formal-b", encoding="utf-8")
                self.assertNotEqual(before, manifest(excluded_paths=(custom,)))

    def test_byte_identity_gate_rejects_one_byte_difference(self):
        compare = self.require("assert_byte_identical_artifacts")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "a.txt").write_bytes(b"same")
            (second / "a.txt").write_bytes(b"same")
            self.assertEqual(
                compare(first, second, expected_names=("a.txt",)),
                {"a.txt": self.rescue.bytes_sha256(b"same")},
            )
            (second / "a.txt").write_bytes(b"same!")
            with self.assertRaises(RuntimeError):
                compare(first, second, expected_names=("a.txt",))

    def test_windows_directory_cleanup_retries_transient_lock(self):
        cleanup = self.require("_rmtree_with_retry")
        with (
            patch.object(
                self.rescue.shutil,
                "rmtree",
                side_effect=[PermissionError("locked"), None],
            ) as mocked,
            patch.object(self.rescue.time, "sleep") as sleep,
            patch.object(Path, "exists", return_value=True),
        ):
            cleanup(Path("temporary-staging"), attempts=2)
        self.assertEqual(mocked.call_count, 2)
        sleep.assert_called_once_with(0.25)

    def test_main_comparison_suptitle_does_not_overlap_legends(self):
        render = self.require("render_main_comparison_figure")
        rows = []
        for task in self.require("TASK_ORDER"):
            for method_index, method in enumerate(self.require("METHOD_ORDER")):
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "primary_selected": method_index == 3,
                        "phase45_precision": 0.10 + 0.05 * method_index,
                        "phase45_recall": 0.20 + 0.10 * method_index,
                        "phase45_f1": 0.15 + 0.06 * method_index,
                        "phase45_f2": 0.18 + 0.08 * method_index,
                        "true_rescue_actual_phase4": 10 + method_index,
                        "true_rescue_actual_phase5": method_index % 2,
                        "false_promotion_actual_phase1": method_index,
                        "false_promotion_actual_phase2": 2 * method_index,
                        "false_promotion_actual_phase3": 3 * method_index,
                    }
                )
        captured = {}

        def inspect_figure(figure, *_args, **_kwargs):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            title_box = figure._suptitle.get_window_extent(renderer=renderer)
            legend_boxes = [
                axis.get_legend().get_window_extent(renderer=renderer)
                for axis in figure.axes
                if axis.get_legend() is not None
            ]
            captured["overlap"] = any(title_box.overlaps(box) for box in legend_boxes)
            self.rescue.plt.close(figure)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(self.rescue, "_save_figure", side_effect=inspect_figure),
        ):
            root = Path(directory)
            render(pd.DataFrame(rows), root / "figure.pdf", root / "figure.png")
        self.assertFalse(captured["overlap"])


if __name__ == "__main__":
    unittest.main()
