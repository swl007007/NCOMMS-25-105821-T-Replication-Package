import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "2.Source Code"
MODULE_PATH = SOURCE_DIR / "generate_all_prediction_temporal_test_evaluation.py"
sys.path.insert(0, str(SOURCE_DIR))


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(
            "The 1,170 temporal-test evaluator does not exist yet: " f"{MODULE_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "generate_all_prediction_temporal_test_evaluation", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_predictions():
    restored = [3374, 3517, 3534, 3553, 3567]
    test_indices = restored + list(range(1165))
    row_number = np.arange(1170)
    actual = (row_number % 5) + 1
    forecasting = np.where(actual >= 3, 3, 2)
    nowcasting = np.where(actual >= 4, 4, np.where(actual >= 3, 3, 2))
    contemporaneous = np.where(actual >= 3, 3, 2)
    return pd.DataFrame(
        {
            "test_index": test_indices,
            "overall_phase": actual,
            "overall_phase_pred": forecasting,
            "phase3_pred": np.where(actual >= 3, 0.8, 0.1),
            "area_id": row_number + 1,
            "date": ["2022-01-01"] * 1170,
            "lat": np.linspace(-30.0, 30.0, 1170),
            "lon": np.linspace(10.0, 50.0, 1170),
            "nowcast_predict": nowcasting,
            "phase3_nowcast": np.where(actual >= 3, 0.85, 0.05),
            "contemporaneous_predict": contemporaneous,
        }
    )


def write_predictions(directory, data=None):
    path = Path(directory) / "All_prediction.csv"
    (canonical_predictions() if data is None else data).to_csv(path, index=False)
    return path


def contemporaneous_oof_predictions(rows=10):
    source_row_index = np.arange(rows)
    actual = (source_row_index % 5) + 1
    predicted = np.where(actual >= 4, 4, np.where(actual >= 3, 3, 2))
    return pd.DataFrame(
        {
            "source_row_index": source_row_index,
            "area_id": source_row_index + 10000,
            "date": ["2021-01-01"] * rows,
            "fold": source_row_index % 5,
            "source_overall_phase": actual,
            "overall_phase": actual,
            "phase2_actual": np.where(actual >= 2, 0.4, 0.1),
            "phase2_contemporaneous": np.where(predicted >= 2, 0.4, 0.1),
            "phase3_actual": np.where(actual >= 3, 0.4, 0.1),
            "phase3_contemporaneous": np.where(predicted >= 3, 0.4, 0.1),
            "phase4_actual": np.where(actual >= 4, 0.4, 0.1),
            "phase4_contemporaneous": np.where(predicted >= 4, 0.4, 0.1),
            "phase5_actual": np.where(actual >= 5, 0.4, 0.1),
            "phase5_contemporaneous": np.where(predicted >= 5, 0.4, 0.1),
            "contemporaneous_predict": predicted,
            "evaluation_protocol": ["random_5fold_row_cv"] * rows,
            "evaluation_population": ["random_5fold_full_oof_5575"] * rows,
            "shuffle_seed": [0] * rows,
            "estimator_random_state": [0] * rows,
        }
    )


def write_contemporaneous_oof(directory, rows=10):
    path = Path(directory) / "contemporaneous.csv"
    contemporaneous_oof_predictions(rows).to_csv(path, index=False)
    return path


class LoaderContractTests(unittest.TestCase):
    def test_general_evaluation_entrypoint_defaults_to_canonical_1170_csv(self):
        general_path = SOURCE_DIR / "generate_all_prediction_evaluation.py"
        spec = importlib.util.spec_from_file_location(
            "generate_all_prediction_evaluation_current", general_path
        )
        general = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(general)

        with tempfile.TemporaryDirectory() as directory:
            input_path = write_predictions(directory)
            y_true, predictions = general.load_predictions(input_path)

        self.assertEqual(general.DEFAULT_INPUT_PATH.name, "All_prediction.csv")
        self.assertEqual(len(y_true), 1170)
        self.assertEqual(
            set(predictions), {"Forecasting", "Nowcasting"}
        )

    def test_legacy_named_entrypoint_accepts_the_current_1170_contract(self):
        legacy_path = SOURCE_DIR / "generate_all_prediction_1165_evaluation.py"
        spec = importlib.util.spec_from_file_location(
            "generate_all_prediction_1165_evaluation", legacy_path
        )
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)

        with tempfile.TemporaryDirectory() as directory:
            sidecar_path = write_contemporaneous_oof(directory)
            evaluations = legacy.load_predictions(
                write_predictions(directory), sidecar_path, 10
            )

        self.assertEqual(len(evaluations["Forecasting"]["y_true"]), 1170)
        self.assertEqual(set(evaluations), {"Forecasting", "Nowcasting", "Contemporaneous"})
        self.assertEqual(legacy.OUTPUT_PREFIX, "all_prediction_temporal_test")

    def test_constants_are_the_current_temporal_contract(self):
        module = load_module()
        self.assertEqual(module.EXPECTED_ROWS, 1170)
        self.assertEqual(module.OUTPUT_PREFIX, "all_prediction_temporal_test")
        self.assertEqual(module.ID_COLUMN, "test_index")
        self.assertEqual(module.KEY_COLUMNS, ["area_id", "date"])
        self.assertEqual(
            module.PREDICTION_COLUMNS,
            {
                "Forecasting": "overall_phase_pred",
                "Nowcasting": "nowcast_predict",
                "Contemporaneous": "contemporaneous_predict",
            },
        )

    def test_load_predictions_accepts_the_complete_1170_contract(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            input_path = write_predictions(directory)
            sidecar_path = write_contemporaneous_oof(directory)
            evaluations = module.load_predictions(input_path, sidecar_path, 10)

        self.assertEqual(set(evaluations), {"Forecasting", "Nowcasting", "Contemporaneous"})
        self.assertEqual(len(evaluations["Forecasting"]["y_true"]), 1170)
        self.assertEqual(len(evaluations["Contemporaneous"]["y_true"]), 10)
        self.assertEqual(
            evaluations["Contemporaneous"]["evaluation_protocol"],
            "random_5fold_row_cv",
        )

    def test_load_predictions_keeps_random_cv_sidecar_independent(self):
        module = load_module()
        source = canonical_predictions()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = write_predictions(directory, source)
            sidecar_path = write_contemporaneous_oof(directory)
            evaluations = module.load_predictions(input_path, sidecar_path, 10)

        self.assertEqual(
            len(evaluations["Contemporaneous"]["y_pred"]), 10
        )
        self.assertEqual(len(evaluations["Forecasting"]["y_pred"]), 1170)

    def test_rejects_1169_and_1171_rows(self):
        module = load_module()
        source = canonical_predictions()
        with tempfile.TemporaryDirectory() as directory:
            short = write_predictions(directory, source.iloc[:-1].copy())
            with self.assertRaisesRegex(ValueError, "Expected 1,170 prediction rows"):
                module.load_predictions(short, write_contemporaneous_oof(directory))

            extra_data = pd.concat([source, source.iloc[[0]]], ignore_index=True)
            extra_data.loc[1170, "test_index"] = 999999
            extra_data.loc[1170, "area_id"] = 999999
            long = write_predictions(directory, extra_data)
            with self.assertRaisesRegex(ValueError, "Expected 1,170 prediction rows"):
                module.load_predictions(long, write_contemporaneous_oof(directory))

    def test_rejects_duplicate_or_missing_test_index_and_duplicate_keys(self):
        module = load_module()
        source = canonical_predictions()
        with tempfile.TemporaryDirectory() as directory:
            duplicate_id = source.copy()
            duplicate_id.loc[1, "test_index"] = duplicate_id.loc[0, "test_index"]
            with self.assertRaisesRegex(ValueError, "test_index.*unique"):
                module.load_predictions(
                    write_predictions(directory, duplicate_id),
                    write_contemporaneous_oof(directory),
                )

            missing_id = source.copy()
            missing_id.loc[1, "test_index"] = np.nan
            with self.assertRaisesRegex(ValueError, "test_index.*complete"):
                module.load_predictions(
                    write_predictions(directory, missing_id),
                    write_contemporaneous_oof(directory),
                )

            duplicate_key = source.copy()
            duplicate_key.loc[1, ["area_id", "date"]] = duplicate_key.loc[
                0, ["area_id", "date"]
            ].to_numpy()
            with self.assertRaisesRegex(ValueError, "keys are not unique"):
                module.load_predictions(
                    write_predictions(directory, duplicate_key),
                    write_contemporaneous_oof(directory),
                )

    def test_rejects_missing_restored_indices_invalid_labels_and_columns(self):
        module = load_module()
        source = canonical_predictions()
        with tempfile.TemporaryDirectory() as directory:
            missing_restored = source.copy()
            missing_restored.loc[0, "test_index"] = 999999
            with self.assertRaisesRegex(ValueError, "restored indices"):
                module.load_predictions(
                    write_predictions(directory, missing_restored),
                    write_contemporaneous_oof(directory),
                )

            fractional = source.copy()
            fractional["overall_phase_pred"] = fractional[
                "overall_phase_pred"
            ].astype(float)
            fractional.loc[0, "overall_phase_pred"] = 2.5
            with self.assertRaisesRegex(ValueError, "integer phase"):
                module.load_predictions(
                    write_predictions(directory, fractional),
                    write_contemporaneous_oof(directory),
                )

            outside = source.copy()
            outside.loc[0, "nowcast_predict"] = 6
            with self.assertRaisesRegex(ValueError, "outside 1--5"):
                module.load_predictions(
                    write_predictions(directory, outside),
                    write_contemporaneous_oof(directory),
                )

            missing_column = source.drop(columns="nowcast_predict")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                module.load_predictions(
                    write_predictions(directory, missing_column),
                    write_contemporaneous_oof(directory),
                )


class ContemporaneousGenerationTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("xgboost") is not None,
        "XGBoost is required for the live contemporaneous generator check.",
    )
    def test_live_generator_produces_full_oof_random_cv_contract(self):
        general_path = SOURCE_DIR / "generate_all_prediction_evaluation.py"
        spec = importlib.util.spec_from_file_location(
            "generate_all_prediction_evaluation_live", general_path
        )
        general = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(general)

        predictions, audit = general.generate_contemporaneous_random_cv_predictions()

        self.assertEqual(len(predictions), 5575)
        self.assertEqual(
            predictions.columns.tolist(), general.CONTEMPORANEOUS_PREDICTION_COLUMNS
        )
        self.assertTrue(predictions["source_row_index"].is_unique)
        self.assertFalse(predictions.duplicated(["area_id", "date"]).any())
        self.assertEqual(predictions["area_id"].nunique(), 1198)
        self.assertEqual(set(predictions["source_row_index"]), set(range(5575)))
        self.assertEqual(
            predictions["fold"].value_counts().sort_index().tolist(),
            [1115] * 5,
        )
        self.assertEqual(
            set(predictions["contemporaneous_predict"].unique()).difference(
                {1, 2, 3, 4, 5}
            ),
            set(),
        )
        self.assertEqual(audit.loc[0, "oof_rows"], 5575)
        self.assertEqual(audit.loc[0, "feature_count"], 174)
        self.assertEqual(audit.loc[0, "evaluation_protocol"], "random_5fold_row_cv")
        self.assertTrue(bool(audit.loc[0, "kfolds_predictor_included"]))
        self.assertIn("not_exact_historical", audit.loc[0, "rerun_interpretation"])

    def test_mixed_protocol_run_regenerates_random_cv_sidecar(self):
        module = load_module()
        source = canonical_predictions()
        sidecar = contemporaneous_oof_predictions()

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = write_predictions(directory, source)
            output_dir = directory / "output"
            output_dir.mkdir()
            sidecar_path = (
                output_dir / f"{module.CONTEMPORANEOUS_PREDICTION_STEM}.csv"
            )
            def write_fresh(output_path, *_args, **_kwargs):
                fresh_path = (
                    Path(output_path)
                    / f"{module.CONTEMPORANEOUS_PREDICTION_STEM}.csv"
                )
                sidecar.to_csv(fresh_path, index=False)
                audit_path = (
                    Path(output_path)
                    / "all_prediction_contemporaneous_random_cv_source_audit.csv"
                )
                pd.DataFrame({"model": ["Contemporaneous"]}).to_csv(
                    audit_path, index=False
                )
                return {
                    "contemporaneous_predictions_csv": fresh_path,
                    "contemporaneous_audit_csv": audit_path,
                }

            with mock.patch.object(
                module,
                "write_contemporaneous_random_cv_artifacts",
                side_effect=write_fresh,
            ) as writer, mock.patch.object(
                module,
                "load_contemporaneous_random_cv_predictions",
                side_effect=lambda path, **kwargs: sidecar,
            ):
                module.run_analysis(input_path, output_dir)

        writer.assert_called_once()


class EvaluationArtifactTests(unittest.TestCase):
    def test_undefined_class_specific_precision_is_nan_with_definition_flag(self):
        module = load_module()
        per_class = pd.DataFrame(
            [
                {
                    "task": task,
                    "display_label": (
                        "Contemporaneous (random CV)"
                        if task == "Contemporaneous"
                        else task
                    ),
                    "evaluation_protocol": (
                        "random_5fold_row_cv"
                        if task == "Contemporaneous"
                        else "fixed_2022_temporal_holdout"
                    ),
                    "evaluation_population": (
                        "random_5fold_full_oof_5575"
                        if task == "Contemporaneous"
                        else "canonical_1170_temporal_test"
                    ),
                    "n_observations": 3,
                    "phase": phase,
                    "precision": 0.0 if phase == 1 else 0.5,
                    "recall": 0.25,
                    "f1": 0.0,
                    "support": 10,
                }
                for task in ("Forecasting", "Nowcasting", "Contemporaneous")
                for phase in range(1, 6)
            ]
        )
        evaluations = {
            "Forecasting": {"y_pred": pd.Series([2, 3, 3])},
            "Nowcasting": {"y_pred": pd.Series([2, 3, 4])},
            "Contemporaneous": {"y_pred": pd.Series([2, 3, 3])},
        }

        result = module.build_class_specific_precision_recall(per_class, evaluations)

        phase1 = result.loc[
            result["task"].eq("Forecasting") & result["phase"].eq(1)
        ].iloc[0]
        self.assertEqual(phase1["predicted_support"], 0)
        self.assertFalse(bool(phase1["precision_defined"]))
        self.assertTrue(np.isnan(phase1["precision_class_specific"]))
        self.assertEqual(phase1["precision_used_for_macro"], 0.0)

    def test_class_specific_figure_caption_marks_mixed_protocols(self):
        module = load_module()
        class_metrics = pd.DataFrame(
            [
                {
                    "task": task,
                    "display_label": (
                        "Contemporaneous (random CV)"
                        if task == "Contemporaneous"
                        else task
                    ),
                    "evaluation_protocol": (
                        "random_5fold_row_cv"
                        if task == "Contemporaneous"
                        else "fixed_2022_temporal_holdout"
                    ),
                    "evaluation_population": (
                        "random_5fold_full_oof_5575"
                        if task == "Contemporaneous"
                        else "canonical_1170_temporal_test"
                    ),
                    "n_observations": (
                        5575 if task == "Contemporaneous" else 1170
                    ),
                    "phase": phase,
                    "actual_support": 234,
                    "predicted_support": 0 if phase in {1, 4, 5} else 585,
                    "precision_defined": phase in {2, 3},
                    "precision_class_specific": (
                        np.nan if phase in {1, 4, 5} else 0.7
                    ),
                    "precision_used_for_macro": (
                        0.0 if phase in {1, 4, 5} else 0.7
                    ),
                    "recall": 0.6,
                }
                for task in ("Forecasting", "Nowcasting", "Contemporaneous")
                for phase in range(1, 6)
            ]
        )

        figure = module.create_class_specific_precision_recall_figure(class_metrics)
        try:
            self.assertEqual(len(figure.axes), 3)
            self.assertEqual(
                [axis.get_title(loc="left") for axis in figure.axes[:2]],
                ["Class-specific precision", "Class-specific recall"],
            )
            caption = " ".join(text.get_text() for text in figure.axes[-1].texts)
            self.assertIn("n = 1,170", caption)
            self.assertIn("n = 5,575", caption)
            self.assertIn("not directly comparable", caption)
            self.assertIn("n.d.", caption)
        finally:
            plt.close(figure)

    def test_run_analysis_writes_evaluation_family_with_task_specific_totals(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            input_path = write_predictions(directory)
            output_dir = directory / "output"

            sidecar = contemporaneous_oof_predictions()

            def write_sidecar(output_path, *_args, **_kwargs):
                prediction_path = (
                    Path(output_path)
                    / "all_prediction_contemporaneous_random_cv_predictions.csv"
                )
                audit_path = (
                    Path(output_path)
                    / "all_prediction_contemporaneous_random_cv_source_audit.csv"
                )
                sidecar.to_csv(prediction_path, index=False)
                pd.DataFrame({"model": ["Contemporaneous"]}).to_csv(
                    audit_path, index=False
                )
                return {
                    "contemporaneous_predictions_csv": prediction_path,
                    "contemporaneous_audit_csv": audit_path,
                }

            with mock.patch.object(
                module,
                "write_contemporaneous_random_cv_artifacts",
                side_effect=write_sidecar,
            ), mock.patch.object(
                module,
                "load_contemporaneous_random_cv_predictions",
                side_effect=lambda path, **kwargs: sidecar,
            ):
                paths = module.run_analysis(input_path, output_dir)

            self.assertEqual(len(paths), 15)
            self.assertEqual(
                set(paths),
                {
                    "metrics_csv",
                    "per_class_csv",
                    "confusion_csv",
                    "class_precision_recall_csv",
                    "metrics_jpg",
                    "metrics_png",
                    "metrics_pdf",
                    "confusion_jpg",
                    "confusion_png",
                    "confusion_pdf",
                    "class_precision_recall_jpg",
                    "class_precision_recall_png",
                    "class_precision_recall_pdf",
                    "contemporaneous_predictions_csv",
                    "contemporaneous_audit_csv",
                },
            )
            for path in paths.values():
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)
                self.assertTrue(
                    path.name.startswith("all_prediction_temporal_test_")
                    or path.name.startswith("all_prediction_contemporaneous_random_cv_")
                )

            metrics = pd.read_csv(paths["metrics_csv"])
            per_class = pd.read_csv(paths["per_class_csv"])
            confusion = pd.read_csv(paths["confusion_csv"])
            class_specific = pd.read_csv(paths["class_precision_recall_csv"])

        self.assertEqual(
            metrics.columns.tolist(),
            [
                "task",
                "display_label",
                "evaluation_protocol",
                "evaluation_population",
                "n_observations",
                "macro_label_set",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "zero_division",
                "phase_1_support",
                "phase_2_support",
                "phase_3_support",
                "phase_4_support",
                "phase_5_support",
            ],
        )
        self.assertEqual(
            per_class.columns.tolist(),
            [
                "task",
                "display_label",
                "evaluation_protocol",
                "evaluation_population",
                "n_observations",
                "phase",
                "precision",
                "recall",
                "f1",
                "support",
            ],
        )
        self.assertEqual(
            confusion.columns.tolist(),
            [
                "task",
                "display_label",
                "evaluation_protocol",
                "evaluation_population",
                "n_observations",
                "actual_phase",
                "predicted_phase",
                "count",
                "actual_phase_support",
                "row_percentage",
            ],
        )
        self.assertEqual(
            class_specific.columns.tolist(),
            [
                "task",
                "display_label",
                "evaluation_protocol",
                "evaluation_population",
                "n_observations",
                "phase",
                "actual_support",
                "predicted_support",
                "precision_defined",
                "precision_class_specific",
                "precision_used_for_macro",
                "recall",
            ],
        )
        expected = {"Forecasting": 1170, "Nowcasting": 1170, "Contemporaneous": 10}
        self.assertEqual(metrics.set_index("task")["n_observations"].to_dict(), expected)
        self.assertEqual(per_class.groupby("task")["support"].sum().to_dict(), expected)
        self.assertEqual(confusion.groupby("task")["count"].sum().to_dict(), expected)
        self.assertEqual(
            class_specific.groupby("task")["actual_support"].sum().to_dict(),
            expected,
        )
        self.assertEqual(
            class_specific.groupby("task")["predicted_support"].sum().to_dict(),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
