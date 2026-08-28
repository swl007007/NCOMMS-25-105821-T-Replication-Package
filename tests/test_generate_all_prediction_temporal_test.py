import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "2.Source Code"
MODULE_PATH = SOURCE_DIR / "generate_all_prediction_temporal_test.py"
sys.path.insert(0, str(SOURCE_DIR))


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(
            "The authoritative temporal-test generator does not exist yet: "
            f"{MODULE_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "generate_all_prediction_temporal_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_frame():
    return pd.DataFrame(
        {
            "area_id": [1, 1, 2, 2],
            "date": ["2021-12-01", "2022-01-01", "2021-11-01", "2022-02-01"],
            "overall_phase": [2, 4, 1, 3],
            "phase1_percent": [0.70, 0.45, 0.90, 0.15],
            "phase2_percent": [0.20, 0.20, 0.05, 0.20],
            "phase3_percent": [0.10, 0.25, 0.05, 0.30],
            "phase4_percent": [0.00, 0.10, 0.00, 0.25],
            "phase5_percent": [0.00, 0.00, 0.00, 0.10],
            "lat": [10.0, 10.0, 20.0, 20.0],
            "lon": [30.0, 30.0, 40.0, 40.0],
        },
        index=[100, 101, 102, 103],
    )


def forecasting_predictions():
    return pd.DataFrame(
        {
            "source_row_index": [101, 103],
            "area_id": [1, 2],
            "date": ["2022-01-01", "2022-02-01"],
            "overall_phase": [3, 4],
            "overall_phase_pred": [3, 4],
            "phase3_pred": [0.35, 0.60],
        }
    )


def nowcasting_predictions():
    return pd.DataFrame(
        {
            "source_row_index": [103, 101],
            "area_id": [2, 1],
            "date": ["2022-02-01", "2022-01-01"],
            "overall_phase": [4, 3],
            "overall_phase_pred": [4, 2],
            "phase3_pred": [0.55, 0.25],
        }
    )


def forecasting_source():
    return pd.DataFrame(
        {
            "area_id": [2, 1],
            "date": ["2022-02-01", "2022-01-01"],
            "overall_phase": [3, 4],
            "phase1_percent": [0.15, 0.45],
            "phase2_percent": [0.20, 0.20],
            "phase3_percent": [0.30, 0.25],
            "phase4_percent": [0.25, 0.10],
            "phase5_percent": [0.10, 0.00],
            "lat": [20.0, 10.0],
            "lon": [40.0, 30.0],
            "country_code_3": ["BBB", "AAA"],
        },
        index=[103, 101],
    )


def assembled_fixture(module):
    return module.assemble_all_prediction(
        forecasting_predictions(), nowcasting_predictions(), forecasting_source()
    )


class ContractAndHashTests(unittest.TestCase):
    def test_contract_constants_and_schema(self):
        module = load_module()

        self.assertEqual(module.EXPECTED_SOURCE_ROWS, 5575)
        self.assertEqual(module.EXPECTED_TRAIN_ROWS, 4405)
        self.assertEqual(module.EXPECTED_TEST_ROWS, 1170)
        self.assertEqual(module.EXPECTED_TEST_AREAS, 646)
        self.assertEqual(module.CUTOFF, "2022-01-01")
        self.assertEqual(module.POPULATION_ID, "model_temporal_2022_1170")
        self.assertEqual(module.CANONICAL_KEY, ["area_id", "date"])
        self.assertEqual(
            module.CANONICAL_OUTPUT_COLUMNS,
            [
                "test_index",
                "overall_phase",
                "overall_phase_pred",
                "phase3_pred",
                "area_id",
                "date",
                "lat",
                "lon",
                "nowcast_predict",
                "phase3_nowcast",
            ],
        )

    def test_canonical_key_hash_is_order_invariant_and_date_normalized(self):
        module = load_module()
        strings = pd.DataFrame(
            {"area_id": [2, 1], "date": ["2022-02-01", "2022-01-01"]}
        )
        timestamps = strings.iloc[::-1].copy()
        timestamps["date"] = pd.to_datetime(timestamps["date"])

        self.assertEqual(
            module.canonical_key_sha256(strings),
            "9fc4a03a5106b402a18c59c0695291273fe2db2fb946af959881e2ac6c0a2736",
        )
        self.assertEqual(
            module.canonical_key_sha256(strings),
            module.canonical_key_sha256(timestamps),
        )

    def test_canonical_key_hash_supports_legacy_pandas_line_terminator_name(self):
        module = load_module()
        strings = pd.DataFrame(
            {"area_id": [2, 1], "date": ["2022-02-01", "2022-01-01"]}
        )
        original_to_csv = pd.DataFrame.to_csv
        calls = []

        def legacy_to_csv(frame, *args, **kwargs):
            calls.append(kwargs.copy())
            if "lineterminator" in kwargs:
                raise TypeError(
                    "to_csv() got an unexpected keyword argument 'lineterminator'"
                )
            kwargs["lineterminator"] = kwargs.pop("line_terminator")
            return original_to_csv(frame, *args, **kwargs)

        with mock.patch.object(pd.DataFrame, "to_csv", new=legacy_to_csv):
            try:
                observed = module.canonical_key_sha256(strings)
            except TypeError as error:
                self.fail(f"Canonical hashing does not support legacy pandas: {error}")

        self.assertEqual(
            observed,
            "9fc4a03a5106b402a18c59c0695291273fe2db2fb946af959881e2ac6c0a2736",
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["lineterminator"], "\n")
        self.assertEqual(calls[1]["line_terminator"], "\n")

    def test_canonical_key_hash_rejects_normalized_duplicates_and_intraday_dates(self):
        module = load_module()
        duplicate = pd.DataFrame(
            {
                "area_id": [1, 1],
                "date": ["2022-01-01", "2022-01-01 00:00:00"],
            }
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            module.canonical_key_sha256(duplicate)

        intraday = pd.DataFrame(
            {"area_id": [1], "date": ["2022-01-01 12:00:00"]}
        )
        with self.assertRaisesRegex(ValueError, "midnight"):
            module.canonical_key_sha256(intraday)


class TemporalPopulationTests(unittest.TestCase):
    def validate_synthetic(self, forecasting, nowcasting):
        module = load_module()
        with (
            mock.patch.object(module, "EXPECTED_SOURCE_ROWS", 4),
            mock.patch.object(module, "EXPECTED_TRAIN_ROWS", 2),
            mock.patch.object(module, "EXPECTED_TEST_ROWS", 2),
            mock.patch.object(module, "EXPECTED_TEST_AREAS", 2),
        ):
            return module.validate_temporal_population(
                forecasting, nowcasting, cutoff="2022-01-01"
            )

    def test_valid_population_returns_complete_masks_for_each_source_order(self):
        forecasting = source_frame()
        nowcasting = forecasting.iloc[::-1].copy()

        train, test, now_train, now_test = self.validate_synthetic(
            forecasting, nowcasting
        )

        self.assertEqual(train.tolist(), [True, False, True, False])
        self.assertEqual(test.tolist(), [False, True, False, True])
        self.assertEqual(now_train.tolist(), [False, True, False, True])
        self.assertEqual(now_test.tolist(), [True, False, True, False])

    def test_rejects_wrong_source_or_test_population_without_filtering(self):
        forecasting = source_frame()
        with self.assertRaisesRegex(ValueError, "Expected 4 source rows"):
            self.validate_synthetic(forecasting.iloc[:3].copy(), forecasting)

        changed_date = forecasting.copy()
        changed_date.loc[103, "date"] = "2021-10-01"
        with self.assertRaisesRegex(ValueError, "Expected 2 test rows"):
            self.validate_synthetic(forecasting, changed_date)

    def test_rejects_duplicate_missing_or_cross_model_keys(self):
        forecasting = source_frame()
        duplicate = forecasting.copy()
        duplicate.loc[103, ["area_id", "date"]] = [1, "2022-01-01"]
        with self.assertRaisesRegex(ValueError, "not unique"):
            self.validate_synthetic(duplicate, forecasting)

        missing = forecasting.copy()
        missing.loc[103, "area_id"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            self.validate_synthetic(missing, forecasting)

        different = forecasting.copy()
        different.loc[103, "area_id"] = 99
        with self.assertRaisesRegex(ValueError, "key sets differ"):
            self.validate_synthetic(forecasting, different)

    def test_rejects_missing_coordinates_or_outcomes(self):
        forecasting = source_frame()
        missing_coordinate = forecasting.copy()
        missing_coordinate.loc[101, "lat"] = np.nan
        with self.assertRaisesRegex(ValueError, "coordinates"):
            self.validate_synthetic(missing_coordinate, forecasting)

        missing_outcome = forecasting.copy()
        missing_outcome.loc[101, "phase3_percent"] = np.nan
        with self.assertRaisesRegex(ValueError, "outcome"):
            self.validate_synthetic(missing_outcome, forecasting)


class AssemblyAndArtifactTests(unittest.TestCase):
    def test_keyed_assembly_preserves_canonical_output_and_source_truth_lineage(self):
        module = load_module()

        result = assembled_fixture(module)

        self.assertEqual(
            result.columns[: len(module.CANONICAL_OUTPUT_COLUMNS)].tolist(),
            module.CANONICAL_OUTPUT_COLUMNS,
        )
        self.assertEqual(result["test_index"].tolist(), [101, 103])
        self.assertEqual(result["area_id"].tolist(), [1, 2])
        self.assertEqual(result["overall_phase"].tolist(), [3, 4])
        self.assertEqual(result["source_overall_phase"].tolist(), [4, 3])
        self.assertEqual(result["nowcast_predict"].tolist(), [2, 4])
        self.assertEqual(result["phase3_nowcast"].tolist(), [0.25, 0.55])
        self.assertEqual(result["lat"].tolist(), [10.0, 20.0])
        self.assertEqual(result["test_index"].tolist(), result["source_row_index"].tolist())

    def test_keyed_assembly_rejects_model_key_or_truth_drift(self):
        module = load_module()
        nowcasting = nowcasting_predictions()
        nowcasting.loc[nowcasting["area_id"].eq(2), "area_id"] = 9
        with self.assertRaisesRegex(ValueError, "key sets differ"):
            module.assemble_all_prediction(
                forecasting_predictions(), nowcasting, forecasting_source()
            )

        nowcasting = nowcasting_predictions()
        nowcasting.loc[nowcasting["area_id"].eq(2), "overall_phase"] = 2
        with self.assertRaisesRegex(ValueError, "truth values differ"):
            module.assemble_all_prediction(
                forecasting_predictions(), nowcasting, forecasting_source()
            )

    def test_truth_disagreement_audit_preserves_both_label_definitions(self):
        module = load_module()
        combined = assembled_fixture(module)

        audit = module.build_truth_disagreements(combined)

        self.assertEqual(audit["test_index"].tolist(), [101, 103])
        first = audit.loc[audit["test_index"].eq(101)].iloc[0]
        second = audit.loc[audit["test_index"].eq(103)].iloc[0]
        self.assertEqual(first["source_overall_phase"], 4)
        self.assertEqual(first["overall_phase"], 3)
        self.assertTrue(bool(first["truth_disagreement"]))
        self.assertEqual(second["source_overall_phase"], 3)
        self.assertEqual(second["overall_phase"], 4)
        self.assertTrue(bool(second["truth_disagreement"]))

    def test_canonical_artifact_validation_enforces_rows_keys_indices_and_values(self):
        module = load_module()
        canonical = assembled_fixture(module).loc[:, module.CANONICAL_OUTPUT_COLUMNS]
        with mock.patch.object(module, "RESTORED_TEST_INDICES", frozenset({101, 103})):
            observed = module.validate_canonical_prediction_artifact(
                canonical, expected_rows=2
            )
            self.assertEqual(len(observed), 2)

            with self.assertRaisesRegex(ValueError, "Expected 2 rows"):
                module.validate_canonical_prediction_artifact(
                    canonical.iloc[:1].copy(), expected_rows=2
                )

            duplicate = pd.concat([canonical.iloc[[0]], canonical.iloc[[0]]], ignore_index=True)
            with self.assertRaisesRegex(ValueError, "not unique"):
                module.validate_canonical_prediction_artifact(duplicate, expected_rows=2)

            missing_index = canonical.copy()
            missing_index.loc[1, "test_index"] = 999
            with self.assertRaisesRegex(ValueError, "restored indices"):
                module.validate_canonical_prediction_artifact(
                    missing_index, expected_rows=2
                )

            bad_phase = canonical.copy()
            bad_phase["overall_phase"] = bad_phase["overall_phase"].astype(float)
            bad_phase.loc[1, "overall_phase"] = 3.5
            with self.assertRaisesRegex(ValueError, "integer phase"):
                module.validate_canonical_prediction_artifact(bad_phase, expected_rows=2)

            bad_prediction = canonical.copy()
            bad_prediction.loc[1, "phase3_nowcast"] = np.inf
            with self.assertRaisesRegex(ValueError, "finite"):
                module.validate_canonical_prediction_artifact(
                    bad_prediction, expected_rows=2
                )

    def test_staged_writer_roundtrips_without_formal_promotion(self):
        module = load_module()
        combined = assembled_fixture(module)
        disagreements = module.build_truth_disagreements(combined)
        calibration = pd.DataFrame(
            [{"record_type": "summary", "formal_replacement_allowed": True}]
        )
        formal_path = ROOT / "1.Source Data/All_prediction_truth_disagreements.csv"
        formal_before = formal_path.read_bytes() if formal_path.exists() else None

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(module, "EXPECTED_TEST_ROWS", 2),
                mock.patch.object(
                    module, "RESTORED_TEST_INDICES", frozenset({101, 103})
                ),
            ):
                output = module.write_staged_artifacts(
                    combined=combined,
                    truth_disagreements=disagreements,
                    overlap_calibration=calibration,
                    staging_dir=Path(directory),
                )

            self.assertEqual(
                set(output),
                {
                    "all_prediction",
                    "df_vis_nowcast",
                    "truth_disagreements",
                    "overlap_calibration",
                },
            )
            canonical = pd.read_csv(output["all_prediction"])
            self.assertEqual(canonical.columns.tolist(), module.CANONICAL_OUTPUT_COLUMNS)
            self.assertEqual(len(canonical), 2)
            visualization = pd.read_csv(output["df_vis_nowcast"])
            self.assertEqual(len(visualization), 2)
            self.assertEqual(set(visualization["_merge"]), {"both"})
            formal_after = formal_path.read_bytes() if formal_path.exists() else None
            self.assertEqual(formal_after, formal_before)


class GenerationExecutionTests(unittest.TestCase):
    def test_generation_is_idempotent_without_a_legacy_1165_dependency(self):
        module = load_module()
        source = source_frame().reset_index(drop=True)
        forecast = forecasting_predictions().replace({101: 1, 103: 3})
        nowcast = nowcasting_predictions().replace({101: 1, 103: 3})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecasting_path = root / "forecasting.csv"
            nowcasting_path = root / "nowcasting.csv"
            lookup_path = root / "lookup.csv"
            general_path = root / "general.json"
            phase3_path = root / "phase3.json"
            source.to_csv(forecasting_path, index=False)
            source.to_csv(nowcasting_path, index=False)
            pd.DataFrame(
                {"area_id": [1, 2], "country_code_3": ["AAA", "BBB"]}
            ).to_csv(lookup_path, index=False)
            general_path.write_text("{}", encoding="utf-8")
            phase3_path.write_text("{}", encoding="utf-8")

            with (
                mock.patch.object(module, "EXPECTED_SOURCE_ROWS", 4),
                mock.patch.object(module, "EXPECTED_TRAIN_ROWS", 2),
                mock.patch.object(module, "EXPECTED_TEST_ROWS", 2),
                mock.patch.object(module, "EXPECTED_TEST_AREAS", 2),
                mock.patch.object(module, "RESTORED_TEST_INDICES", frozenset({1, 3})),
                mock.patch.object(
                    module.temporal,
                    "run_temporal_predictions",
                    return_value={"Forecasting": forecast, "Nowcasting": nowcast},
                ) as runner,
            ):
                first = module.run_generation(
                    forecasting_path=forecasting_path,
                    nowcasting_path=nowcasting_path,
                    country_lookup_path=lookup_path,
                    general_params_path=general_path,
                    phase3_params_path=phase3_path,
                    legacy_input_path=None,
                    staging_dir=root / "output",
                    workers=1,
                    estimator_n_jobs=None,
                )
                second = module.run_generation(
                    forecasting_path=forecasting_path,
                    nowcasting_path=nowcasting_path,
                    country_lookup_path=lookup_path,
                    general_params_path=general_path,
                    phase3_params_path=phase3_path,
                    legacy_input_path=None,
                    staging_dir=root / "output",
                    workers=1,
                    estimator_n_jobs=None,
                )

            self.assertEqual(len(pd.read_csv(first["all_prediction"])), 2)
            self.assertEqual(first["all_prediction"], second["all_prediction"])
            self.assertNotIn("overlap_calibration", first)
            self.assertEqual(runner.call_count, 2)
            for call in runner.call_args_list:
                self.assertEqual(call.kwargs["workers"], 1)
                self.assertIsNone(call.kwargs["estimator_n_jobs"])


class OverlapCalibrationTests(unittest.TestCase):
    def staged_and_legacy(self, module):
        staged = assembled_fixture(module).loc[:, module.CANONICAL_OUTPUT_COLUMNS]
        restored = staged.iloc[[1]].copy()
        restored.loc[:, "test_index"] = 105
        restored.loc[:, "area_id"] = 5
        restored.loc[:, "date"] = "2022-03-01"
        staged = pd.concat([staged, restored], ignore_index=True)
        legacy = staged.iloc[:2].copy()
        return staged, legacy

    def test_exact_legacy_overlap_allows_formal_replacement(self):
        module = load_module()
        staged, legacy = self.staged_and_legacy(module)
        with (
            mock.patch.object(module, "EXPECTED_TEST_ROWS", 3),
            mock.patch.object(module, "RESTORED_TEST_INDICES", frozenset({105})),
        ):
            calibration = module.build_overlap_calibration(staged, legacy)

        summary = calibration.loc[calibration["record_type"].eq("summary")].iloc[0]
        forecasting = calibration.loc[
            calibration["record_type"].eq("model_summary")
            & calibration["model"].eq("Forecasting")
        ].iloc[0]
        self.assertEqual(summary["overlap_rows"], 2)
        self.assertEqual(summary["restored_rows"], 1)
        self.assertFalse(bool(summary["prediction_drift_detected"]))
        self.assertTrue(bool(summary["formal_replacement_allowed"]))
        self.assertEqual(forecasting["phase_prediction_match_rows"], 2)
        self.assertEqual(forecasting["phase3_max_abs_difference"], 0.0)

    def test_prediction_drift_blocks_promotion_and_reports_differing_keys_and_metrics(self):
        module = load_module()
        staged, legacy = self.staged_and_legacy(module)
        staged.loc[staged["test_index"].eq(101), "nowcast_predict"] = 5
        staged.loc[staged["test_index"].eq(101), "phase3_nowcast"] = 0.95
        with (
            mock.patch.object(module, "EXPECTED_TEST_ROWS", 3),
            mock.patch.object(module, "RESTORED_TEST_INDICES", frozenset({105})),
        ):
            calibration = module.build_overlap_calibration(staged, legacy)

        summary = calibration.loc[calibration["record_type"].eq("summary")].iloc[0]
        nowcasting = calibration.loc[
            calibration["record_type"].eq("model_summary")
            & calibration["model"].eq("Nowcasting")
        ].iloc[0]
        differences = calibration.loc[
            calibration["record_type"].eq("prediction_difference")
        ]
        self.assertTrue(bool(summary["prediction_drift_detected"]))
        self.assertFalse(bool(summary["formal_replacement_allowed"]))
        self.assertEqual(nowcasting["phase_prediction_match_rows"], 1)
        self.assertGreater(nowcasting["phase3_max_abs_difference"], 0.0)
        self.assertIn("accuracy_delta", calibration.columns)
        self.assertEqual(differences[["area_id", "date"]].values.tolist(), [[1, "2022-01-01"]])
        self.assertEqual(
            json.loads(differences.iloc[0]["differing_models_json"]), ["Nowcasting"]
        )

    def test_promotion_gate_passes_exact_overlap_and_blocks_any_drift(self):
        module = load_module()
        self.assertTrue(
            hasattr(module, "assert_promotion_authorized"),
            "The fail-closed formal-promotion gate is not implemented.",
        )
        exact = pd.DataFrame(
            [
                {
                    "record_type": "summary",
                    "formal_replacement_allowed": True,
                }
            ]
        )
        module.assert_promotion_authorized(exact)

        drift = exact.copy()
        drift.loc[0, "formal_replacement_allowed"] = False
        with self.assertRaisesRegex(RuntimeError, "legacy Windows/XGBoost"):
            module.assert_promotion_authorized(drift)


class LiveTemporalContractTests(unittest.TestCase):
    def test_live_sources_match_the_fixed_2022_temporal_population(self):
        module = load_module()
        forecasting = pd.read_csv(
            ROOT / "1.Source Data/Forecasting_Analysis_010825.csv"
        )
        nowcasting = pd.read_csv(ROOT / "1.Source Data/Nowcasting_Analysis_010825.csv")
        canonical = pd.read_csv(ROOT / "1.Source Data/All_prediction.csv")

        train, test, now_train, now_test = module.validate_temporal_population(
            forecasting, nowcasting, cutoff=module.CUTOFF
        )
        test_rows = forecasting.loc[test].copy()
        test_rows["date"] = pd.to_datetime(test_rows["date"]).dt.strftime("%Y-%m-%d")
        now_test_rows = nowcasting.loc[now_test].copy()
        now_test_rows["date"] = pd.to_datetime(now_test_rows["date"]).dt.strftime(
            "%Y-%m-%d"
        )

        self.assertEqual((int(train.sum()), int(test.sum())), (4405, 1170))
        self.assertEqual((int(now_train.sum()), int(now_test.sum())), (4405, 1170))
        self.assertEqual(test_rows["area_id"].nunique(), 646)
        self.assertEqual(test_rows["date"].min(), "2022-01-01")
        self.assertEqual(test_rows["date"].max(), "2022-11-01")
        self.assertEqual(
            module.canonical_key_sha256(test_rows),
            "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2",
        )
        self.assertEqual(
            module.canonical_key_sha256(test_rows),
            module.canonical_key_sha256(now_test_rows),
        )
        self.assertEqual(
            set(module.RESTORED_TEST_INDICES),
            {3374, 3517, 3534, 3553, 3567},
        )
        validated = module.validate_canonical_prediction_artifact(canonical)
        self.assertEqual(len(validated), 1170)
        self.assertEqual(validated["area_id"].nunique(), 646)
        self.assertTrue(
            set(module.RESTORED_TEST_INDICES).issubset(validated["test_index"])
        )
        self.assertEqual(
            module.canonical_key_sha256(validated),
            "288059e7dad989ecfa5e634c01b6ec39282a81e22f17aa28625bad76455fe6c2",
        )


if __name__ == "__main__":
    unittest.main()
