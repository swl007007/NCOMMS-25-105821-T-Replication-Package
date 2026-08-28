from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "2.Source Code"
MODULE_PATH = SOURCE_DIR / "generate_nowcasting_calibration.py"
sys.path.insert(0, str(SOURCE_DIR))


def load_module():
    if not MODULE_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "generate_nowcasting_calibration", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot load calibration generator: {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CalibrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_module()

    def require_function(self, name):
        self.assertIsNotNone(
            self.calibration,
            f"Calibration generator does not exist: {MODULE_PATH}",
        )
        function = getattr(self.calibration, name, None)
        self.assertTrue(callable(function), f"Missing calibration function: {name}")
        return function


class ProjectionTests(CalibrationTestCase):
    def test_projection_is_identity_for_valid_scores(self):
        project = self.require_function("project_cumulative_scores")
        raw = np.array([[0.9, 0.8, 0.7, 0.6]])

        projected = project(raw)

        np.testing.assert_allclose(projected, raw, rtol=0, atol=1e-12)

    def test_projection_pools_violating_blocks_and_applies_bounds(self):
        project = self.require_function("project_cumulative_scores")
        raw = np.array(
            [
                [0.8, 0.9, 0.4, 0.5],
                [1.2, 0.8, 0.9, -0.1],
                [-0.2, 0.4, 0.3, 0.1],
            ]
        )
        expected = np.array(
            [
                [0.85, 0.85, 0.45, 0.45],
                [1.0, 0.85, 0.85, 0.0],
                [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 0.1],
            ]
        )

        projected = project(raw)

        np.testing.assert_allclose(projected, expected, rtol=0, atol=1e-12)
        self.assertTrue(np.all(projected[:, :-1] >= projected[:, 1:]))
        self.assertTrue(np.all((projected >= 0.0) & (projected <= 1.0)))


class MappingAndMetricTests(CalibrationTestCase):
    def test_transform_offsets_phase3_and_phase4_before_projection(self):
        transform = self.require_function("transform_scores")
        raw = np.array([[0.70, 0.25, 0.15, 0.00]])

        result = transform(raw, -0.06, 0.05)

        np.testing.assert_allclose(
            result["adjusted"], [[0.70, 0.19, 0.20, 0.00]], rtol=0, atol=1e-12
        )
        self.assertEqual(result["predicted_phase"].tolist(), [4])

    def test_metrics_use_exact_phase4_and_binary_phase3plus(self):
        calculate = self.require_function("calculate_calibration_metrics")
        actual = np.array([2, 3, 4, 4])
        predicted = np.array([3, 3, 4, 3])

        metrics = calculate(actual, predicted)

        self.assertAlmostEqual(metrics["ordinal_mae"], 0.50)
        self.assertEqual(metrics["phase4_support"], 2)
        self.assertAlmostEqual(metrics["phase4_recall"], 0.5)
        self.assertAlmostEqual(metrics["phase3plus_precision"], 0.75)
        self.assertAlmostEqual(metrics["phase3plus_recall"], 1.0)


class TemporalSplitTests(CalibrationTestCase):
    def test_default_variant_remains_the_completed_h2_contract(self):
        get_variant = self.require_function("get_experiment_variant")
        self.assertEqual(
            self.calibration.DEFAULT_EXPERIMENT_VARIANT, "2021h2_2022"
        )
        variant = get_variant("2021h2_2022")
        self.assertEqual(variant.train_end, pd.Timestamp("2021-06-30"))
        self.assertEqual(variant.calibration_start, pd.Timestamp("2021-07-01"))
        self.assertEqual(variant.calibration_end, pd.Timestamp("2021-12-31"))
        self.assertEqual(variant.test_start, pd.Timestamp("2022-01-01"))
        self.assertEqual(
            (variant.train_rows, variant.calibration_rows, variant.test_rows),
            (3909, 496, 1170),
        )
        self.assertFalse(variant.sensitivity_analysis)

    def test_unknown_variant_fails(self):
        get_variant = self.require_function("get_experiment_variant")
        with self.assertRaisesRegex(ValueError, "experiment variant"):
            get_variant("unknown")

    def test_boundaries_are_complete_and_disjoint(self):
        build_masks = self.require_function("build_temporal_masks")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2021-06-01", "2021-07-01", "2021-12-01", "2022-01-01"]
                )
            }
        )

        masks = build_masks(frame)

        self.assertEqual(masks["train"].tolist(), [True, False, False, False])
        self.assertEqual(masks["calibration"].tolist(), [False, True, True, False])
        self.assertEqual(masks["test"].tolist(), [False, False, False, True])
        coverage = sum(mask.astype(int) for mask in masks.values())
        self.assertTrue(coverage.eq(1).all())

    def test_full2021_boundaries_are_complete_and_disjoint(self):
        build_masks = self.require_function("build_temporal_masks")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2020-12-01",
                        "2021-01-01",
                        "2021-12-01",
                        "2022-01-01",
                    ]
                )
            }
        )

        masks = build_masks(frame, "full2021_2022")

        self.assertEqual(masks["train"].tolist(), [True, False, False, False])
        self.assertEqual(masks["calibration"].tolist(), [False, True, True, False])
        self.assertEqual(masks["test"].tolist(), [False, False, False, True])
        coverage = sum(mask.astype(int) for mask in masks.values())
        self.assertTrue(coverage.eq(1).all())

    def test_unparseable_date_fails(self):
        build_masks = self.require_function("build_temporal_masks")

        with self.assertRaisesRegex(ValueError, "date"):
            build_masks(pd.DataFrame({"date": ["bad"]}))

    def test_empty_split_fails(self):
        build_masks = self.require_function("build_temporal_masks")

        with self.assertRaisesRegex(ValueError, "complete and disjoint"):
            build_masks(pd.DataFrame({"date": ["2022-01-01"]}))


class RecordingRegressor:
    instances = []

    def __init__(self, **params):
        self.params = params
        self.fit_rows = None
        self.value = None
        RecordingRegressor.instances.append(self)

    def fit(self, features, target):
        self.fit_rows = len(features)
        self.value = float(np.mean(target))
        return self

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def prepared_three_split_fixture():
    dates = pd.to_datetime(
        [
            "2021-05-01",
            "2021-06-01",
            "2021-07-01",
            "2021-12-01",
            "2022-01-01",
            "2022-02-01",
        ]
    )
    phases = np.array([2, 3, 2, 4, 3, 4])
    cumulative = {
        2: (0.30, 0.10, 0.00, 0.00),
        3: (0.50, 0.30, 0.10, 0.00),
        4: (0.60, 0.40, 0.30, 0.00),
    }
    target_values = np.array([cumulative[int(phase)] for phase in phases])
    forecasting = pd.DataFrame(
        {
            "area_id": np.arange(1, 7),
            "date": dates,
            "country_code_3": ["AAA"] * 6,
            "overall_phase": phases,
            "phase2_worse": target_values[:, 0],
            "phase3_worse": target_values[:, 1],
            "phase4_worse": target_values[:, 2],
            "phase5_worse": target_values[:, 3],
            "layer1_feature": np.arange(6, dtype=float),
        }
    )
    nowcasting = forecasting.loc[
        :, ["area_id", "date", "country_code_3"]
    ].copy()
    nowcasting["layer2_feature"] = np.arange(10, 16, dtype=float)
    return forecasting, nowcasting


class ModelBranchTests(CalibrationTestCase):
    def setUp(self):
        RecordingRegressor.instances.clear()

    def test_eight_models_fit_once_and_predict_both_windows(self):
        fit_branch = self.require_function("fit_calibration_branch")
        build_masks = self.require_function("build_temporal_masks")
        forecasting, nowcasting = prepared_three_split_fixture()

        result = fit_branch(
            forecasting,
            nowcasting,
            build_masks(forecasting),
            build_masks(nowcasting),
            layer1_features=("layer1_feature",),
            layer2_features=("layer2_feature",),
            general_params={"random_state": 0},
            phase3_params={"random_state": 0},
            estimator_factory=RecordingRegressor,
        )

        self.assertEqual(len(RecordingRegressor.instances), 8)
        self.assertTrue(
            all(model.fit_rows == 2 for model in RecordingRegressor.instances)
        )
        self.assertEqual(
            result.groupby("split", observed=True).size().to_dict(),
            {"calibration": 2, "test": 2},
        )
        self.assertEqual(result["area_id"].nunique(), 4)
        self.assertTrue(
            np.isfinite(
                result[
                    [f"phase{phase}_pred_raw" for phase in (2, 3, 4, 5)]
                ].to_numpy(dtype=float)
            ).all()
        )

    def test_split_key_mismatch_fails_before_model_fit(self):
        fit_branch = self.require_function("fit_calibration_branch")
        build_masks = self.require_function("build_temporal_masks")
        forecasting, nowcasting = prepared_three_split_fixture()
        nowcasting.loc[nowcasting["date"].eq(pd.Timestamp("2021-07-01")), "area_id"] = 99

        with self.assertRaisesRegex(ValueError, "calibration keys differ"):
            fit_branch(
                forecasting,
                nowcasting,
                build_masks(forecasting),
                build_masks(nowcasting),
                layer1_features=("layer1_feature",),
                layer2_features=("layer2_feature",),
                general_params={"random_state": 0},
                phase3_params={"random_state": 0},
                estimator_factory=RecordingRegressor,
            )
        self.assertEqual(RecordingRegressor.instances, [])


class LiveInputContractTests(CalibrationTestCase):
    def test_live_inputs_have_the_approved_three_way_counts(self):
        load_inputs = self.require_function("load_prepared_inputs")
        build_masks = self.require_function("build_temporal_masks")
        validate_counts = self.require_function("validate_production_counts")
        source_data = ROOT / "1.Source Data"

        forecasting, nowcasting = load_inputs(
            source_data / "Forecasting_Analysis_010825.csv",
            source_data / "Nowcasting_Analysis_010825.csv",
            source_data / "area_country_lookup.csv",
        )
        forecasting_masks = build_masks(forecasting)
        nowcasting_masks = build_masks(nowcasting)
        validate_counts(forecasting, forecasting_masks)
        validate_counts(nowcasting, nowcasting_masks)

        self.assertEqual(len(forecasting), 5575)
        self.assertEqual(len(nowcasting), 5575)
        self.assertEqual(
            {name: int(mask.sum()) for name, mask in forecasting_masks.items()},
            {"train": 3909, "calibration": 496, "test": 1170},
        )

    def test_live_inputs_have_the_full2021_sensitivity_contract(self):
        load_inputs = self.require_function("load_prepared_inputs")
        build_masks = self.require_function("build_temporal_masks")
        validate_counts = self.require_function("validate_production_counts")
        key_hash = self.require_function("canonical_key_sha256")
        source_data = ROOT / "1.Source Data"

        forecasting, nowcasting = load_inputs(
            source_data / "Forecasting_Analysis_010825.csv",
            source_data / "Nowcasting_Analysis_010825.csv",
            source_data / "area_country_lookup.csv",
        )
        forecasting_masks = build_masks(forecasting, "full2021_2022")
        nowcasting_masks = build_masks(nowcasting, "full2021_2022")
        validate_counts(forecasting, forecasting_masks, "full2021_2022")
        validate_counts(nowcasting, nowcasting_masks, "full2021_2022")

        self.assertEqual(
            {name: int(mask.sum()) for name, mask in forecasting_masks.items()},
            {"train": 3446, "calibration": 959, "test": 1170},
        )
        self.assertEqual(
            {
                name: int(forecasting.loc[mask, "area_id"].nunique())
                for name, mask in forecasting_masks.items()
            },
            {"train": 1029, "calibration": 566, "test": 646},
        )
        calibration = forecasting.loc[forecasting_masks["calibration"]]
        months = pd.to_datetime(calibration["date"]).dt.to_period("M").astype(str)
        self.assertEqual(months.nunique(), 11)
        self.assertNotIn("2021-08", set(months))
        self.assertEqual(
            calibration["overall_phase"].value_counts().sort_index().to_dict(),
            {1: 9, 2: 392, 3: 510, 4: 48},
        )
        reconstructed = self.calibration.loco._phase_from_cumulative(
            calibration, "worse"
        )
        self.assertEqual(
            pd.Series(reconstructed).value_counts().sort_index().to_dict(),
            {1: 9, 2: 390, 3: 512, 4: 48},
        )
        self.assertEqual(
            int(
                np.sum(
                    reconstructed
                    != calibration["overall_phase"].to_numpy(dtype=int)
                )
            ),
            2,
        )
        self.assertEqual(
            key_hash(forecasting.loc[forecasting_masks["test"]]),
            "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2",
        )


def raw_prediction_fixture(actual_phase, raw_scores, split="calibration"):
    actual = np.asarray(actual_phase, dtype=int)
    raw = np.asarray(raw_scores, dtype=float)
    cumulative = {
        1: (0.10, 0.00, 0.00, 0.00),
        2: (0.30, 0.10, 0.00, 0.00),
        3: (0.50, 0.30, 0.10, 0.00),
        4: (0.60, 0.40, 0.30, 0.00),
        5: (0.70, 0.50, 0.40, 0.30),
    }
    actual_scores = np.array([cumulative[int(phase)] for phase in actual])
    start = "2021-07-01" if split == "calibration" else "2022-01-01"
    frame = pd.DataFrame(
        {
            "split": split,
            "area_id": np.arange(1, len(actual) + 1),
            "date": pd.date_range(start, periods=len(actual), freq="MS"),
            "country_code_3": ["AAA"] * len(actual),
            "source_row_index": np.arange(len(actual)),
            "source_overall_phase": actual,
            "overall_phase": actual,
        }
    )
    for index, phase in enumerate((2, 3, 4, 5)):
        frame[f"phase{phase}_test"] = actual_scores[:, index]
        frame[f"phase{phase}_pred_raw"] = raw[:, index]
        frame[f"phase{phase}_layer1_pred"] = raw[:, index]
        frame[f"phase{phase}_residual_pred"] = 0.0
    return frame


def correctable_raw_fixture(split="calibration"):
    return raw_prediction_fixture(
        actual_phase=[2, 3, 4, 4],
        raw_scores=[
            [0.70, 0.25, 0.00, 0.00],
            [0.70, 0.35, 0.00, 0.00],
            [0.70, 0.25, 0.15, 0.00],
            [0.70, 0.25, 0.15, 0.00],
        ],
        split=split,
    )


def tied_candidate_table():
    shared = {
        "ordinal_mae": 0.25,
        "balanced_accuracy": 0.75,
        "phase4_recall": 0.50,
        "distribution_discrepancy": 0.20,
        "offset_l1": 0.03,
        "accepted": True,
    }
    return pd.DataFrame(
        [
            {"delta3": -0.02, "delta4": 0.01, **shared},
            {"delta3": -0.01, "delta4": 0.02, **shared},
        ]
    )


EXPECTED_PREDICTION_COLUMNS = [
    "split",
    "area_id",
    "date",
    "country_code_3",
    "source_row_index",
    "source_overall_phase",
    "overall_phase",
    *[f"phase{phase}_test" for phase in (2, 3, 4, 5)],
    *[f"phase{phase}_pred_raw" for phase in (2, 3, 4, 5)],
    *[
        column
        for phase in (2, 3, 4, 5)
        for column in (
            f"phase{phase}_layer1_pred",
            f"phase{phase}_residual_pred",
        )
    ],
    *[f"phase{phase}_identity_projected" for phase in (2, 3, 4, 5)],
    *[f"phase{phase}_identity_rounded" for phase in (2, 3, 4, 5)],
    *[f"phase{phase}_pred_adjusted" for phase in (2, 3, 4, 5)],
    *[f"phase{phase}_pred_projected" for phase in (2, 3, 4, 5)],
    *[f"phase{phase}_pred_rounded" for phase in (2, 3, 4, 5)],
    "identity_overall_phase_pred",
    "calibrated_overall_phase_pred",
    "selected_delta3",
    "selected_delta4",
]

EXPECTED_METRICS_COLUMNS = [
    "split",
    "variant",
    "n_rows",
    "ordinal_mae",
    "balanced_accuracy",
    "phase4_support",
    "phase4_recall",
    "phase3plus_precision",
    "phase3plus_recall",
    "distribution_discrepancy",
    *[f"actual_phase{phase}_count" for phase in range(1, 6)],
    *[f"predicted_phase{phase}_count" for phase in range(1, 6)],
]

EXPECTED_SOURCE_AUDIT_COLUMNS = [
    "evaluation_population_id",
    "source_rows",
    "train_rows",
    "calibration_rows",
    "test_rows",
    "train_end",
    "calibration_start",
    "calibration_end",
    "test_start",
    "calibration_key_sha256",
    "test_key_sha256",
    "delta3",
    "delta4",
    "identity_selected",
    "random_state",
    "estimator_n_jobs",
    "estimator_uses_default_n_jobs",
    "model_workers",
    "python_version",
    "numpy_version",
    "pandas_version",
    "scikit_learn_version",
    "xgboost_version",
    "platform",
    "protected_manifest_sha256_before",
    "protected_manifest_sha256_after",
    "protected_manifest_match",
    "generator_path",
    "generator_sha256",
    "forecasting_input_path",
    "forecasting_input_sha256",
    "nowcasting_input_path",
    "nowcasting_input_sha256",
    "country_lookup_path",
    "country_lookup_sha256",
    "general_params_path",
    "general_params_sha256",
    "phase3_params_path",
    "phase3_params_sha256",
    "grid_path",
    "grid_sha256",
    "predictions_path",
    "predictions_sha256",
    "metrics_path",
    "metrics_sha256",
]
FULL2021_SOURCE_AUDIT_EXTRA_COLUMNS = [
    "experiment_variant",
    "sensitivity_analysis",
    "h2_artifact_manifest_sha256_before",
    "h2_artifact_manifest_sha256_after",
]


class OffsetSearchTests(CalibrationTestCase):
    def test_grid_contains_exactly_441_candidates_and_identity(self):
        build_grid = self.require_function("build_offset_grid")

        grid = build_grid()

        self.assertEqual(len(grid), 441)
        self.assertEqual(grid[["delta3", "delta4"]].drop_duplicates().shape[0], 441)
        self.assertEqual(
            len(grid[(grid["delta3"] == 0.0) & (grid["delta4"] == 0.0)]), 1
        )
        self.assertEqual(grid["delta3"].min(), -0.20)
        self.assertEqual(grid["delta4"].max(), 0.20)

    def test_search_selects_minimal_candidate_that_corrects_phase3_and_phase4(self):
        search = self.require_function("search_offsets")

        grid, selected = search(correctable_raw_fixture())

        self.assertTrue(bool(selected["accepted"]))
        self.assertEqual(float(selected["delta3"]), -0.06)
        self.assertEqual(float(selected["delta4"]), 0.05)
        self.assertEqual(int(grid["selected"].sum()), 1)
        self.assertEqual(float(selected["ordinal_mae"]), 0.0)
        self.assertEqual(float(selected["phase4_recall"]), 1.0)

    def test_identity_is_selected_when_mae_cannot_strictly_improve(self):
        search = self.require_function("search_offsets")
        calibration_rows = raw_prediction_fixture(
            actual_phase=[2, 3, 4],
            raw_scores=[
                [0.70, 0.10, 0.00, 0.00],
                [0.70, 0.30, 0.00, 0.00],
                [0.70, 0.30, 0.30, 0.00],
            ],
        )

        grid, selected = search(calibration_rows)

        self.assertEqual(
            (float(selected["delta3"]), float(selected["delta4"])),
            (0.0, 0.0),
        )
        self.assertFalse(bool(selected["accepted"]))
        self.assertEqual(int(grid["selected"].sum()), 1)

    def test_final_tie_prefers_delta3_descending_then_delta4_ascending(self):
        select_index = self.require_function("select_candidate_index")
        tied = tied_candidate_table()

        selected_index = select_index(tied)

        self.assertEqual(
            (tied.loc[selected_index, "delta3"], tied.loc[selected_index, "delta4"]),
            (-0.01, 0.02),
        )

    def test_search_rejects_noncalibration_rows(self):
        search = self.require_function("search_offsets")

        with self.assertRaisesRegex(ValueError, "calibration rows"):
            search(correctable_raw_fixture(split="test"))


class OutputTableTests(CalibrationTestCase):
    def test_prediction_and_metrics_tables_keep_variants_separate(self):
        search = self.require_function("search_offsets")
        build_predictions = self.require_function("build_prediction_table")
        build_metrics = self.require_function("build_metrics_table")
        calibration_raw = correctable_raw_fixture(split="calibration")
        test_raw = correctable_raw_fixture(split="test")
        _, selected = search(calibration_raw)
        raw = pd.concat([calibration_raw, test_raw], ignore_index=True)

        predictions = build_predictions(
            raw, float(selected["delta3"]), float(selected["delta4"])
        )
        metrics = build_metrics(predictions)

        self.assertEqual(predictions.columns.tolist(), EXPECTED_PREDICTION_COLUMNS)
        self.assertEqual(metrics.columns.tolist(), EXPECTED_METRICS_COLUMNS)
        self.assertFalse(predictions.duplicated(["split", "area_id", "date"]).any())
        np.testing.assert_allclose(
            predictions[[f"phase{phase}_pred_raw" for phase in (2, 3, 4, 5)]],
            raw[[f"phase{phase}_pred_raw" for phase in (2, 3, 4, 5)]],
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            list(metrics[["split", "variant"]].itertuples(index=False, name=None)),
            [
                ("calibration", "identity"),
                ("calibration", "selected"),
                ("test", "identity"),
                ("test", "selected"),
            ],
        )
        self.assertEqual(predictions["selected_delta3"].unique().tolist(), [-0.06])
        self.assertEqual(predictions["selected_delta4"].unique().tolist(), [0.05])
        for row in metrics.itertuples(index=False):
            subset = predictions.loc[predictions["split"].eq(row.split)]
            predicted_column = (
                "identity_overall_phase_pred"
                if row.variant == "identity"
                else "calibrated_overall_phase_pred"
            )
            recomputed = self.calibration.calculate_calibration_metrics(
                subset["overall_phase"].to_numpy(dtype=int),
                subset[predicted_column].to_numpy(dtype=int),
            )
            for name, value in recomputed.items():
                self.assertAlmostEqual(getattr(row, name), value)


def artifact_frames(calibration):
    calibration_raw = correctable_raw_fixture(split="calibration")
    test_raw = correctable_raw_fixture(split="test")
    grid, selected = calibration.search_offsets(calibration_raw)
    raw = pd.concat([calibration_raw, test_raw], ignore_index=True)
    predictions = calibration.build_prediction_table(
        raw, float(selected["delta3"]), float(selected["delta4"])
    )
    metrics = calibration.build_metrics_table(predictions)
    return grid, selected, predictions, metrics


def source_frame_for_predictions(predictions):
    train = pd.DataFrame(
        {
            "area_id": [9001, 9002],
            "date": pd.to_datetime(["2021-05-01", "2021-06-01"]),
        }
    )
    evaluated = predictions[["area_id", "date"]].copy()
    return pd.concat([train, evaluated], ignore_index=True)


class ArtifactTests(CalibrationTestCase):
    def test_source_audit_schema_is_variant_specific(self):
        columns = self.require_function("source_audit_columns")

        self.assertEqual(columns("2021h2_2022"), EXPECTED_SOURCE_AUDIT_COLUMNS)
        self.assertEqual(
            columns("full2021_2022"),
            [
                *EXPECTED_SOURCE_AUDIT_COLUMNS,
                *FULL2021_SOURCE_AUDIT_EXTRA_COLUMNS,
            ],
        )

    def test_h2_artifact_manifest_covers_exactly_the_four_completed_csvs(self):
        manifest_hash = self.require_function("h2_artifact_manifest_sha256")
        expected = {
            "calibration_grid.csv",
            "calibration_predictions.csv",
            "calibration_metrics.csv",
            "calibration_source_audit.csv",
        }

        self.assertEqual(
            {path.name for path in self.calibration.H2_ARTIFACT_PATHS}, expected
        )
        self.assertTrue(
            all(path.is_file() for path in self.calibration.H2_ARTIFACT_PATHS)
        )
        self.assertRegex(manifest_hash(), r"^[0-9a-f]{64}$")

    def test_h2_manifest_changes_when_a_fixture_file_changes(self):
        manifest_hash = self.require_function("h2_artifact_manifest_sha256")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = tuple(
                root / name for name in ("a.csv", "b.csv", "c.csv", "d.csv")
            )
            for index, path in enumerate(paths):
                path.write_text(f"value\n{index}\n", encoding="utf-8")
            with patch.object(self.calibration, "H2_ARTIFACT_PATHS", paths):
                before = manifest_hash()
                paths[0].write_text("value\nchanged\n", encoding="utf-8")
                after = manifest_hash()

        self.assertNotEqual(before, after)

    def test_variant_default_directories_are_distinct(self):
        resolve_output = self.require_function("resolve_output_dir")

        h2 = resolve_output("2021h2_2022", None)
        full = resolve_output("full2021_2022", None)

        self.assertEqual(h2.name, "nowcasting_calibration")
        self.assertEqual(full.name, "nowcasting_calibration_full2021")
        self.assertNotEqual(h2.resolve(), full.resolve())

    def test_generation_target_rejects_cross_variant_and_nonempty_directories(self):
        validate_target = self.require_function("validate_generation_target")
        h2_dir = self.calibration.get_experiment_variant(
            "2021h2_2022"
        ).default_output_dir
        full_dir = self.calibration.get_experiment_variant(
            "full2021_2022"
        ).default_output_dir

        with self.assertRaisesRegex(ValueError, "output directory"):
            validate_target("full2021_2022", h2_dir)
        with self.assertRaisesRegex(ValueError, "output directory"):
            validate_target("2021h2_2022", full_dir)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "empty"):
                validate_target("full2021_2022", target)
            self.assertEqual(
                (target / "sentinel.txt").read_text(encoding="utf-8"), "keep"
            )

    def test_full2021_source_audit_records_sensitivity_and_h2_hashes(self):
        build_audit = self.require_function("build_source_audit")
        grid, selected, predictions, _ = artifact_frames(self.calibration)
        source = source_frame_for_predictions(predictions)
        masks = self.calibration.build_temporal_masks(source)
        source_data = ROOT / "1.Source Data"
        source_code = ROOT / "2.Source Code"

        with tempfile.TemporaryDirectory() as directory:
            paths = self.calibration.write_model_outputs(
                grid,
                predictions,
                self.calibration.build_metrics_table(predictions),
                Path(directory),
            )
            audit = build_audit(
                forecasting=source,
                forecasting_masks=masks,
                predictions=predictions,
                selected=selected,
                model_output_paths=paths,
                input_paths={
                    "forecasting": source_data / "Forecasting_Analysis_010825.csv",
                    "nowcasting": source_data / "Nowcasting_Analysis_010825.csv",
                    "country_lookup": source_data / "area_country_lookup.csv",
                    "general_params": source_code / "forecasting_hyperparameters.json",
                    "phase3_params": source_code
                    / "forecasting_hyperparameters_p3.json",
                },
                protected_before="protected",
                protected_after="protected",
                experiment_variant="full2021_2022",
                h2_before="h2hash",
                h2_after="h2hash",
            )

        self.assertEqual(audit.loc[0, "experiment_variant"], "full2021_2022")
        self.assertTrue(bool(audit.loc[0, "sensitivity_analysis"]))
        self.assertEqual(
            audit.loc[0, "h2_artifact_manifest_sha256_before"], "h2hash"
        )
        self.assertEqual(
            audit.loc[0, "h2_artifact_manifest_sha256_after"], "h2hash"
        )

    def test_full2021_audit_validation_rejects_wrong_variant_or_h2_drift(self):
        build_audit = self.require_function("build_source_audit")
        write_audit = self.require_function("write_source_audit")
        validate_artifacts = self.require_function("validate_written_artifacts")
        protected_hash = self.require_function("protected_artifact_manifest_sha256")
        h2_hash = self.require_function("h2_artifact_manifest_sha256")
        grid, selected, predictions, metrics = artifact_frames(self.calibration)
        source = source_frame_for_predictions(predictions)
        masks = self.calibration.build_temporal_masks(source)
        source_data = ROOT / "1.Source Data"
        source_code = ROOT / "2.Source Code"

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = self.calibration.write_model_outputs(
                grid, predictions, metrics, output_dir
            )
            protected = protected_hash(output_dir)
            h2 = h2_hash()
            audit = build_audit(
                forecasting=source,
                forecasting_masks=masks,
                predictions=predictions,
                selected=selected,
                model_output_paths=paths,
                input_paths={
                    "forecasting": source_data / "Forecasting_Analysis_010825.csv",
                    "nowcasting": source_data / "Nowcasting_Analysis_010825.csv",
                    "country_lookup": source_data / "area_country_lookup.csv",
                    "general_params": source_code / "forecasting_hyperparameters.json",
                    "phase3_params": source_code
                    / "forecasting_hyperparameters_p3.json",
                },
                protected_before=protected,
                protected_after=protected,
                experiment_variant="full2021_2022",
                h2_before=h2,
                h2_after=h2,
            )
            paths["source_audit"] = write_audit(
                audit, output_dir, "full2021_2022"
            )
            validate_artifacts(paths, "full2021_2022")

            wrong_variant = audit.copy()
            wrong_variant.loc[0, "experiment_variant"] = "2021h2_2022"
            paths["source_audit"] = write_audit(
                wrong_variant, output_dir, "full2021_2022"
            )
            with self.assertRaisesRegex(ValueError, "experiment variant"):
                validate_artifacts(paths, "full2021_2022")

            h2_drift = audit.copy()
            h2_drift.loc[0, "h2_artifact_manifest_sha256_after"] = "different"
            paths["source_audit"] = write_audit(
                h2_drift, output_dir, "full2021_2022"
            )
            with self.assertRaisesRegex(ValueError, "H2 artifact"):
                validate_artifacts(paths, "full2021_2022")

    def test_writer_creates_only_four_expected_csvs_and_roundtrips(self):
        write_outputs = self.require_function("write_model_outputs")
        write_audit = self.require_function("write_source_audit")
        validate_model_outputs = self.require_function(
            "validate_written_model_outputs"
        )
        grid, _, predictions, metrics = artifact_frames(self.calibration)
        audit = pd.DataFrame(
            [{column: "fixture" for column in EXPECTED_SOURCE_AUDIT_COLUMNS}]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = write_outputs(grid, predictions, metrics, output_dir)
            paths["source_audit"] = write_audit(audit, output_dir)

            self.assertEqual(
                set(paths), {"grid", "predictions", "metrics", "source_audit"}
            )
            self.assertEqual(
                {path.name for path in output_dir.iterdir()},
                {
                    "calibration_grid.csv",
                    "calibration_predictions.csv",
                    "calibration_metrics.csv",
                    "calibration_source_audit.csv",
                },
            )
            validate_model_outputs(paths)
            reread_audit = pd.read_csv(paths["source_audit"], keep_default_na=False)
            self.assertEqual(
                reread_audit.columns.tolist(), EXPECTED_SOURCE_AUDIT_COLUMNS
            )
            self.assertEqual(len(reread_audit), 1)

    def test_repeated_writes_to_same_directory_are_byte_identical(self):
        write_outputs = self.require_function("write_model_outputs")
        write_audit = self.require_function("write_source_audit")
        file_hash = self.require_function("file_sha256")
        grid, _, predictions, metrics = artifact_frames(self.calibration)
        audit = pd.DataFrame(
            [{column: "fixture" for column in EXPECTED_SOURCE_AUDIT_COLUMNS}]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            first = write_outputs(grid, predictions, metrics, output_dir)
            first["source_audit"] = write_audit(audit, output_dir)
            first_hashes = {name: file_hash(path) for name, path in first.items()}
            second = write_outputs(grid, predictions, metrics, output_dir)
            second["source_audit"] = write_audit(audit, output_dir)

            self.assertEqual(
                first_hashes,
                {name: file_hash(path) for name, path in second.items()},
            )

    def test_protected_manifest_is_precise_nonrecursive_allowlist(self):
        protected_paths = self.require_function("protected_artifact_paths")

        paths = protected_paths()
        relative = {path.relative_to(ROOT).as_posix() for path in paths}

        self.assertIn("2.Source Code/Table1_Forecasting_main.ipynb", relative)
        self.assertIn("1.Source Data/Forecasting_Analysis_010825.csv", relative)
        self.assertIn(
            "2.Source Code/produced_graph/all_prediction_temporal_test_source_audit.csv",
            relative,
        )
        self.assertTrue(
            all(
                not path.startswith(
                    "2.Source Code/produced_graph/.leave_one_country_out_checkpoints/"
                )
                for path in relative
            )
        )
        self.assertNotIn(
            "2.Source Code/produced_graph/spatial_feature_comparison_metrics.csv",
            relative,
        )
        prefix_files = {
            path.name
            for path in (ROOT / "2.Source Code" / "produced_graph").iterdir()
            if path.is_file()
            and path.name.startswith("all_prediction_temporal_test_")
        }
        self.assertEqual(
            {
                path.name
                for path in paths
                if path.parent == ROOT / "2.Source Code" / "produced_graph"
            },
            prefix_files,
        )

    def test_source_audit_and_full_validation_bind_saved_outputs(self):
        write_outputs = self.require_function("write_model_outputs")
        write_audit = self.require_function("write_source_audit")
        build_audit = self.require_function("build_source_audit")
        validate_artifacts = self.require_function("validate_written_artifacts")
        protected_hash = self.require_function("protected_artifact_manifest_sha256")
        grid, selected, predictions, metrics = artifact_frames(self.calibration)
        source = source_frame_for_predictions(predictions)
        masks = self.calibration.build_temporal_masks(source)
        source_data = ROOT / "1.Source Data"
        source_code = ROOT / "2.Source Code"

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = write_outputs(grid, predictions, metrics, output_dir)
            before = protected_hash(output_dir)
            audit = build_audit(
                forecasting=source,
                forecasting_masks=masks,
                predictions=predictions,
                selected=selected,
                model_output_paths=paths,
                input_paths={
                    "forecasting": source_data / "Forecasting_Analysis_010825.csv",
                    "nowcasting": source_data / "Nowcasting_Analysis_010825.csv",
                    "country_lookup": source_data / "area_country_lookup.csv",
                    "general_params": source_code / "forecasting_hyperparameters.json",
                    "phase3_params": source_code
                    / "forecasting_hyperparameters_p3.json",
                },
                protected_before=before,
                protected_after=before,
            )
            paths["source_audit"] = write_audit(audit, output_dir)

            validate_artifacts(paths)

    def test_formal_environment_gate_accepts_live_lineage_and_rejects_drift(self):
        environment = self.require_function("formal_environment_record")
        assert_environment = self.require_function("assert_formal_environment")
        observed = environment()

        assert_environment(observed)
        drifted = dict(observed)
        drifted["xgboost_version"] = "different"
        with self.assertRaisesRegex(RuntimeError, "formal Windows lineage"):
            assert_environment(drifted)


class CliTests(CalibrationTestCase):
    def test_cli_selects_full2021_default_directory(self):
        args = self.calibration.parse_args(
            ["--experiment-variant", "full2021_2022"]
        )

        self.assertEqual(args.experiment_variant, "full2021_2022")
        self.assertEqual(args.output_dir.name, "nowcasting_calibration_full2021")

    def test_cli_default_remains_h2(self):
        args = self.calibration.parse_args([])

        self.assertEqual(args.experiment_variant, "2021h2_2022")
        self.assertEqual(args.output_dir.name, "nowcasting_calibration")

    def test_nonempty_target_fails_before_environment_or_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "sentinel.txt").write_text("keep", encoding="utf-8")
            with patch.object(
                self.calibration, "assert_formal_environment"
            ) as environment:
                with patch.object(
                    self.calibration, "load_prepared_inputs"
                ) as load_inputs:
                    with self.assertRaisesRegex(FileExistsError, "empty"):
                        self.calibration.run_generation(
                            experiment_variant="full2021_2022",
                            output_dir=target,
                        )
            environment.assert_not_called()
            load_inputs.assert_not_called()

    def test_cli_defaults_and_prints_exactly_four_paths(self):
        parse_args = self.require_function("parse_args")
        main = self.require_function("main")
        args = parse_args([])

        self.assertEqual(args.output_dir.name, "nowcasting_calibration")
        fake_paths = {
            "grid": Path("grid.csv"),
            "predictions": Path("predictions.csv"),
            "metrics": Path("metrics.csv"),
            "source_audit": Path("source_audit.csv"),
        }
        stream = io.StringIO()
        with patch.object(
            self.calibration, "run_generation", return_value=fake_paths
        ) as run_generation:
            with redirect_stdout(stream):
                exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stream.getvalue().splitlines(),
            [f"{name}: {path}" for name, path in fake_paths.items()],
        )
        run_generation.assert_called_once_with(
            forecasting_path=args.forecasting_input,
            nowcasting_path=args.nowcasting_input,
            country_lookup_path=args.country_lookup,
            general_params_path=args.general_params,
            phase3_params_path=args.phase3_params,
            output_dir=args.output_dir,
            experiment_variant=args.experiment_variant,
        )


if __name__ == "__main__":
    unittest.main()
