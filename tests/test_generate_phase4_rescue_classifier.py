from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-cache-phase4-rescue-tests")
)

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE = REPO_ROOT / "2.Source Code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))

import generate_phase4_rescue_classifier as rescue


class FrozenContractTests(unittest.TestCase):
    def test_exact_artifact_and_method_contracts(self):
        self.assertEqual(len(rescue.EXPECTED_ARTIFACTS), 20)
        self.assertEqual(len(set(rescue.EXPECTED_ARTIFACTS)), 20)
        self.assertEqual(
            rescue.MAIN_METHOD_ORDER,
            ("frozen_base", "direct_phase34_xgboost", "xgboost"),
        )
        self.assertEqual(rescue.DIRECT_THRESHOLD, 0.5)

    def test_temporal_folds_are_exact_and_strictly_forward(self):
        self.assertEqual(
            [(fold.fold_id, fold.training_years, fold.validation_year) for fold in rescue.BASE_FOLDS],
            [
                ("B1", (2017,), 2018),
                ("B2", (2017, 2018), 2019),
                ("B3", (2017, 2018, 2019), 2020),
                ("B4", (2017, 2018, 2019, 2020), 2021),
            ],
        )
        self.assertEqual(
            [(fold.fold_id, fold.training_years, fold.validation_year) for fold in rescue.META_FOLDS],
            [
                ("A1", (2018,), 2019),
                ("A2", (2018, 2019), 2020),
                ("A3", (2018, 2019, 2020), 2021),
            ],
        )

    def test_auxiliary_grid_contains_exactly_24_unique_candidates(self):
        candidates = rescue.get_auxiliary_parameter_candidates()
        self.assertEqual(len(candidates), 24)
        serialized = {tuple(sorted(candidate.items())) for candidate in candidates}
        self.assertEqual(len(serialized), 24)
        self.assertEqual({candidate["max_depth"] for candidate in candidates}, {1, 2, 3})
        self.assertEqual({candidate["min_child_weight"] for candidate in candidates}, {5, 20})

    def test_rescue_feature_orders_are_exact(self):
        self.assertEqual(len(rescue.FORECAST_RESCUE_FEATURES), 7)
        self.assertEqual(len(rescue.NOWCAST_RESCUE_FEATURES), 15)
        self.assertEqual(
            rescue.FORECAST_RESCUE_FEATURES,
            (
                "phase2_pred_raw", "phase3_pred_raw", "phase4_pred_raw",
                "phase5_pred_raw", "phase3_margin_020", "phase4_margin_020",
                "phase3_minus_phase4",
            ),
        )


class PhaseAndCascadeTests(unittest.TestCase):
    def test_reconstructed_phase_uses_highest_cumulative_twenty_percent_rule(self):
        frame = pd.DataFrame(
            {
                "phase1_percent": [0.81, 0.69, 0.50, 0.40, 0.40],
                "phase2_percent": [0.19, 0.20, 0.19, 0.19, 0.19],
                "phase3_percent": [0.00, 0.11, 0.20, 0.19, 0.10],
                "phase4_percent": [0.00, 0.00, 0.11, 0.20, 0.10],
                "phase5_percent": [0.00, 0.00, 0.00, 0.02, 0.21],
            }
        )
        self.assertEqual(rescue.reconstruct_overall_phase(frame).tolist(), [1, 2, 3, 4, 5])

    def test_prediction_phase_uses_rounded_scores(self):
        frame = pd.DataFrame(
            {
                "phase2_pred_rounded": [0.19, 0.20, 0.20],
                "phase3_pred_rounded": [0.19, 0.19, 0.20],
                "phase4_pred_rounded": [0.19, 0.19, 0.19],
                "phase5_pred_rounded": [0.19, 0.19, 0.19],
            }
        )
        self.assertEqual(rescue.phase_from_rounded_predictions(frame).tolist(), [1, 2, 3])

    def test_rescue_changes_only_three_to_four(self):
        base = np.array([1, 2, 3, 3, 4, 5])
        score = np.array([1.0, 1.0, 0.49, 0.50, 1.0, 1.0])
        result = rescue.apply_phase4_rescue(base, score, 0.5)
        self.assertEqual(result.tolist(), [1, 2, 3, 4, 4, 5])
        np.testing.assert_array_equal(base >= 3, result >= 3)

    def test_direct_nowcasting_score_is_unclipped_sum(self):
        combined = rescue.combine_direct_nowcasting_scores(
            np.array([0.9, 0.1]), np.array([0.4, -0.5])
        )
        np.testing.assert_allclose(combined, [1.3, -0.4], rtol=0, atol=0)

    def test_invalid_postclassification_change_is_rejected(self):
        with self.assertRaises(ValueError):
            rescue.assert_postclassification_invariants([2, 3], [3, 4])


class ThresholdSelectionTests(unittest.TestCase):
    def validation_frame(self, scores):
        actual = [3, 3, 3, 4, 4, 2, 1, 5]
        return pd.DataFrame(
            {
                "reconstructed_overall_phase": actual,
                "base_overall_phase_pred": [3] * len(actual),
                "score": scores,
            }
        )

    def test_threshold_selects_eligible_high_precision_rescue(self):
        frame = self.validation_frame([0.1, 0.2, 0.3, 0.95, 0.90, 0.1, 0.1, 0.1])
        search, selected = rescue.select_promotion_threshold(
            "Forecasting", "xgboost", frame, "score"
        )
        self.assertFalse(bool(selected["no_rescue_sentinel"]))
        self.assertTrue(bool(selected["eligible"]))
        self.assertEqual(int(search["selected"].sum()), 1)
        self.assertGreaterEqual(float(selected["phase4_precision"]), 0.30)

    def test_threshold_uses_no_rescue_when_no_candidate_is_eligible(self):
        frame = self.validation_frame([0.5] * 8)
        _, selected = rescue.select_promotion_threshold(
            "Nowcasting", "single_score", frame, "score"
        )
        self.assertTrue(bool(selected["no_rescue_sentinel"]))
        self.assertFalse(bool(selected["eligible"]))
        self.assertEqual(selected["evaluation_status"], "selected_no_rescue")


class LiveDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasting, cls.nowcasting, cls.layer1_features = rescue.load_prepared_inputs()

    def test_source_population_and_feature_counts_are_frozen(self):
        self.assertEqual(len(self.forecasting), 5575)
        self.assertEqual(len(self.nowcasting), 5575)
        self.assertEqual(len(self.forecasting.loc[self.forecasting["date"] < rescue.CUTOFF]), 4405)
        self.assertEqual(len(self.layer1_features), 106)
        self.assertEqual(len(rescue.NOWCAST_FEATURES), 69)

    def test_direct_population_is_exact_reconstructed_phase34_universe(self):
        population, target = rescue.build_direct_phase34_population(self.forecasting)
        self.assertEqual(len(population), 2518)
        self.assertEqual(int(target.sum()), 369)
        self.assertEqual(population["reconstructed_overall_phase"].value_counts().to_dict(), {3: 2149, 4: 369})

    def test_native_missingness_is_preserved_and_infinity_absent(self):
        population, _ = rescue.build_direct_phase34_population(self.forecasting)
        forecasting_matrix = population[self.layer1_features]
        keyed_nowcasting = self.nowcasting.merge(
            population[["area_id", "date"]], on=["area_id", "date"],
            how="inner", validate="one_to_one",
        ).sort_values(["area_id", "date"], kind="mergesort")
        nowcasting_matrix = keyed_nowcasting[list(rescue.NOWCAST_FEATURES)]
        self.assertGreater(int(forecasting_matrix.isna().sum().sum()), 0)
        self.assertGreater(int(nowcasting_matrix.isna().sum().sum()), 0)
        self.assertFalse(np.isinf(forecasting_matrix.to_numpy(dtype=float)).any())
        self.assertFalse(np.isinf(nowcasting_matrix.to_numpy(dtype=float)).any())

    def test_feature_manifest_has_exact_303_rows(self):
        manifest = rescue.build_feature_manifest(
            self.forecasting, self.nowcasting, self.layer1_features
        )
        self.assertEqual(len(manifest), 303)
        self.assertEqual(
            manifest.groupby(["task", "model_component"]).size().to_dict(),
            {
                ("Forecasting", "direct_forecasting_classifier"): 106,
                ("Forecasting", "rescue_classifier"): 7,
                ("Nowcasting", "direct_nowcasting_layer1_classifier"): 106,
                ("Nowcasting", "direct_nowcasting_layer2_regressor"): 69,
                ("Nowcasting", "rescue_classifier"): 15,
            },
        )

    def test_frozen_hashes_and_benchmark_metrics_recompute(self):
        rescue.verify_frozen_source_hashes()
        base = rescue.load_and_validate_frozen_base_predictions()
        references = rescue.frozen_main.classification_references()
        for task in rescue.TASK_ORDER:
            frame = base[task].rename(
                columns={
                    "overall_phase": "reconstructed_overall_phase",
                    "overall_phase_pred": "base_overall_phase_pred",
                }
            )
            record = rescue.calculate_metric_record(
                "benchmark", task, "frozen_base", frame,
                "base_overall_phase_pred", "base_overall_phase_pred", None,
                "frozen_reference",
            )
            self.assertEqual(record["phase3plus_precision"], references[task]["phase3plus_precision"])
            self.assertEqual(record["phase3plus_recall"], references[task]["phase3plus_recall"])
            self.assertAlmostEqual(record["phase3plus_r2"], references[task]["phase3plus_r2"], places=12)
            self.assertEqual(record["accuracy"], references[task]["overall_accuracy"])

    def test_frozen_benchmark_outcomes_match_source_tables_by_key(self):
        base = rescue.load_and_validate_frozen_base_predictions()
        rescue.validate_frozen_benchmark_outcomes(self.forecasting, self.nowcasting, base)
        drifted = {task: frame.copy() for task, frame in base.items()}
        drifted["Forecasting"].loc[0, "phase3_test"] += 1e-6
        with self.assertRaises(ValueError):
            rescue.validate_frozen_benchmark_outcomes(
                self.forecasting, self.nowcasting, drifted
            )


class ArtifactHelperTests(unittest.TestCase):
    def synthetic_benchmark(self):
        rows = []
        actual = np.tile(np.array([1, 2, 3, 4, 5]), 234)[:1170]
        for task in rescue.TASK_ORDER:
            for index, phase in enumerate(actual):
                rows.append(
                    {
                        "task": task,
                        "area_id": index + 1,
                        "date": pd.Timestamp("2022-01-01"),
                        "country_code_3": f"C{index % 27:02d}",
                        "reconstructed_overall_phase": int(phase),
                        "base_overall_phase_pred": int(phase),
                        "single_score_overall_phase_pred": int(phase),
                        "direct_overall_phase_pred": int(phase),
                        "rescued_overall_phase_pred": int(phase),
                    }
                )
        return pd.DataFrame(rows)

    def test_confusion_source_has_all_200_cells(self):
        confusion = rescue.build_confusion_source(self.synthetic_benchmark())
        self.assertEqual(len(confusion), 200)
        self.assertTrue((confusion.groupby(["task", "method"])["count"].sum() == 1170).all())

    def test_nonempty_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.mkdir()
            (target / "sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                rescue.validate_generation_target(target)
            self.assertEqual((target / "sentinel.txt").read_text(encoding="utf-8"), "keep")

    def test_protected_manifest_always_excludes_formal_artifact_set(self):
        with tempfile.TemporaryDirectory() as directory:
            produced_graph = Path(directory) / "produced_graph"
            formal_target = produced_graph / "phase4_rescue_classifier"
            formal_target.mkdir(parents=True)
            (produced_graph / "protected.csv").write_text("stable\n", encoding="utf-8")
            generated = formal_target / "generated.csv"
            generated.write_text("first\n", encoding="utf-8")
            with (
                patch.object(rescue, "PRODUCED_GRAPH_DIR", produced_graph),
                patch.object(rescue, "DEFAULT_OUTPUT_DIR", formal_target),
            ):
                first = rescue.protected_artifact_manifest_sha256()
                generated.write_text("second\n", encoding="utf-8")
                second = rescue.protected_artifact_manifest_sha256(Path(directory) / "repeat")
            self.assertEqual(first, second)

    def test_finalize_source_audit_records_measured_after_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            row = {column: np.nan for column in rescue.SOURCE_AUDIT_COLUMNS}
            row.update(
                {
                    "run_status": "complete",
                    "protected_manifest_sha256_before": "before",
                    "protected_manifest_sha256_after": "before",
                    "protected_manifest_match": True,
                    "artifact_manifest_json": "{}",
                    "artifact_manifest_sha256": rescue.manifest_sha256({}),
                }
            )
            rescue._write_csv(
                pd.DataFrame([row], columns=rescue.SOURCE_AUDIT_COLUMNS),
                target / "phase4_rescue_source_audit.csv",
                rescue.SOURCE_AUDIT_COLUMNS,
            )
            rescue.finalize_source_audit(target, "measured-after")
            finalized = pd.read_csv(target / "phase4_rescue_source_audit.csv")
            self.assertEqual(
                finalized.loc[0, "protected_manifest_sha256_after"], "measured-after"
            )
            self.assertFalse(bool(finalized.loc[0, "protected_manifest_match"]))

    def test_main_figure_is_vector_pdf_and_exact_600_dpi_canvas(self):
        confusion = rescue.build_confusion_source(self.synthetic_benchmark())
        rows = []
        for task in rescue.TASK_ORDER:
            for method in rescue.MAIN_METHOD_ORDER:
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "display_status": "test",
                        "n_rows": 1170,
                        "phase3plus_precision": 0.8,
                        "phase3plus_recall": 0.9,
                        "phase3plus_r2": 0.25,
                        "accuracy": 0.65,
                        "phase3plus_precision_delta_from_base": 0.0,
                        "phase3plus_recall_delta_from_base": 0.0,
                        "phase3plus_r2_delta_from_base": 0.0,
                        "accuracy_delta_from_base": 0.0,
                    }
                )
        main = pd.DataFrame(rows, columns=rescue.MAIN_METRIC_COLUMNS)
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "figure.pdf"
            png = Path(directory) / "figure.png"
            rescue.render_main_comparison_figure(confusion, main, pdf, png)
            payload = pdf.read_bytes()
            self.assertTrue(payload.startswith(b"%PDF"))
            self.assertNotIn(b"/Subtype /Image", payload)
            self.assertNotIn(b"CreationDate", payload)
            self.assertNotIn(b"ModDate", payload)
            width, height, dpi = rescue._read_png_contract(png)
            self.assertEqual((width, height), (7200, 3900))
            self.assertIsNotNone(dpi)
            self.assertLessEqual(abs(float(dpi) - 600.0), 1.0)

    def test_canonical_key_hash_is_order_invariant(self):
        data = pd.DataFrame(
            {"area_id": [2, 1], "date": ["2022-02-01", "2022-01-01"]}
        )
        self.assertEqual(
            rescue.canonical_key_sha256(data),
            rescue.canonical_key_sha256(data.iloc[::-1].reset_index(drop=True)),
        )


if __name__ == "__main__":
    unittest.main()
