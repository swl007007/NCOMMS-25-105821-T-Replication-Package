import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "2.Source Code"
sys.path.insert(0, str(SOURCE_DIR))

import generate_leave_area_out_20pct_fivefold_robustness as fivefold
import generate_leave_one_country_out_robustness as loco


def synthetic_fivefold_data(nowcasting=False):
    records = []
    for area_id in range(1, 11):
        phase = 2 + area_id % 3
        shares = {
            1: 0.8,
            2: 0.2 if phase == 2 else 0.0,
            3: 0.2 if phase == 3 else 0.0,
            4: 0.2 if phase == 4 else 0.0,
            5: 0.0,
        }
        for date in ("2021-01-01", "2022-01-01"):
            record = {
                "area_id": area_id,
                "date": date,
                "country_code_3": "AAA" if area_id <= 5 else "BBB",
                "overall_phase": phase,
                "fews_ipc_ha": float(phase),
                "signal": float(area_id),
            }
            record.update({f"phase{i}_percent": shares[i] for i in range(1, 6)})
            if nowcasting:
                record.update(
                    {column: float(area_id) for column in loco.NOWCAST_FEATURES}
                )
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


class StrictTemporalAreaSplitTests(unittest.TestCase):
    def test_build_area_folds_assigns_each_area_once_with_the_fixed_seed(self):
        canonical = pd.DataFrame(
            {"area_id": range(1, 647), "date": ["2022-01-01"] * 646}
        )

        folds = fivefold.build_area_folds(canonical)

        self.assertEqual(len(folds), 646)
        self.assertEqual(folds["area_id"].nunique(), 646)
        self.assertEqual(set(folds["fold_id"]), set(range(5)))
        self.assertTrue(folds.groupby("area_id")["fold_id"].nunique().eq(1).all())
        self.assertEqual(folds["fold_id"].value_counts().sort_index().tolist(), [130, 129, 129, 129, 129])
        self.assertEqual(folds.loc[:9, "fold_id"].tolist(), [1, 0, 3, 1, 2, 2, 1, 2, 0, 4])

    def test_strict_temporal_area_masks_remove_held_history_and_nonheld_2022_rows(self):
        data = pd.DataFrame(
            {
                "area_id": [1, 1, 2, 2, 3, 3],
                "date": [
                    "2021-01-01",
                    "2022-01-01",
                    "2021-01-01",
                    "2022-01-01",
                    "2021-01-01",
                    "2022-01-01",
                ],
            }
        )

        frame, train, test = fivefold.strict_temporal_area_masks(
            data, {1}, pd.Timestamp("2022-01-01")
        )

        self.assertEqual(set(frame.loc[train, "area_id"]), {2, 3})
        self.assertEqual(set(frame.loc[test, "area_id"]), {1})
        self.assertTrue(frame.loc[train, "date"].lt("2022-01-01").all())
        self.assertTrue(frame.loc[test, "date"].ge("2022-01-01").all())
        self.assertEqual(len(frame), 3)


class FivefoldPredictionTests(unittest.TestCase):
    def test_run_fivefold_predictions_is_temporal_area_disjoint_and_covers_canonical_keys(self):
        forecasting = synthetic_fivefold_data()
        nowcasting = synthetic_fivefold_data(nowcasting=True).iloc[::-1].reset_index(
            drop=True
        )
        canonical = forecasting.loc[
            forecasting["date"].eq("2022-01-01"), ["area_id", "date"]
        ].copy()
        canonical["date"] = pd.to_datetime(canonical["date"])
        canonical_keys = set(map(tuple, canonical.to_numpy()))

        def checked_forecasting_runner(*args, **kwargs):
            frame, train_mask, test_mask = args[:3]
            self.assertIn("fews_ipc_ha", frame)
            self.assertTrue(frame.loc[train_mask, "date"].lt(fivefold.CUTOFF).all())
            self.assertTrue(frame.loc[test_mask, "date"].ge(fivefold.CUTOFF).all())
            self.assertTrue(
                set(frame.loc[train_mask, "area_id"])
                .isdisjoint(set(frame.loc[test_mask, "area_id"]))
            )
            return perfect_forecasting_runner(*args, **kwargs)

        forecast, nowcast, fold_metrics = fivefold.run_fivefold_predictions(
            forecasting,
            nowcasting,
            canonical,
            general_params={"max_depth": 2},
            phase3_params={"max_depth": 3},
            forecasting_runner=checked_forecasting_runner,
            nowcasting_runner=perfect_nowcasting_runner,
        )

        self.assertEqual(len(forecast), 10)
        self.assertFalse(forecast.duplicated(["area_id", "date"]).any())
        self.assertEqual(set(forecast["fold_id"]), set(range(5)))
        self.assertEqual(set(map(tuple, forecast[["area_id", "date"]].to_numpy())), canonical_keys)
        self.assertEqual(len(fold_metrics), 10)
        self.assertTrue(fold_metrics["train_excludes_held_areas"].all())
        self.assertEqual(set(fold_metrics["n_train"]), {8})
        pd.testing.assert_frame_equal(
            forecast[["area_id", "date", "fold_id"]]
            .sort_values(["area_id", "date"])
            .reset_index(drop=True),
            nowcast[["area_id", "date", "fold_id"]]
            .sort_values(["area_id", "date"])
            .reset_index(drop=True),
        )


class FivefoldArtifactTests(unittest.TestCase):
    def _write_inputs(self, root):
        forecasting_path = root / "forecasting.csv"
        nowcasting_path = root / "nowcasting.csv"
        canonical_path = root / "canonical.csv"
        lookup_path = root / "lookup.csv"
        general_path = root / "general.json"
        phase3_path = root / "phase3.json"
        forecasting = synthetic_fivefold_data()
        nowcasting = synthetic_fivefold_data(nowcasting=True)
        forecasting.drop(columns="country_code_3").to_csv(forecasting_path, index=False)
        nowcasting.drop(columns="country_code_3").to_csv(nowcasting_path, index=False)
        forecasting.loc[
            forecasting["date"].eq("2022-01-01"), ["area_id", "date"]
        ].to_csv(canonical_path, index=False)
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
            "canonical_test_path": canonical_path,
            "country_lookup_path": lookup_path,
            "general_params_path": general_path,
            "phase3_params_path": phase3_path,
        }

    def test_run_analysis_writes_exactly_six_validated_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            protected = output_dir / "leave_area_out_10pct_fixture.csv"
            output_dir.mkdir()
            protected.write_text("frozen\n", encoding="utf-8")
            inputs = self._write_inputs(root)

            artifacts = fivefold.run_analysis(
                **inputs,
                output_dir=output_dir,
                forecasting_runner=perfect_forecasting_runner,
                nowcasting_runner=perfect_nowcasting_runner,
            )

            self.assertEqual(
                set(artifacts),
                {
                    "folds",
                    "forecasting_predictions",
                    "nowcasting_predictions",
                    "fold_metrics",
                    "metrics",
                    "source_audit",
                },
            )
            self.assertTrue(all(path.is_file() for path in artifacts.values()))
            self.assertEqual(protected.read_text(encoding="utf-8"), "frozen\n")
            folds = pd.read_csv(artifacts["folds"])
            summary = pd.read_csv(artifacts["metrics"])
            fold_metrics = pd.read_csv(artifacts["fold_metrics"])
            audit = pd.read_csv(artifacts["source_audit"])
            self.assertEqual(len(folds), 10)
            self.assertEqual(set(folds["fold_id"]), set(range(5)))
            self.assertEqual(len(fold_metrics), 10)
            self.assertTrue(summary["aggregation"].eq("pooled_oof").all())
            self.assertEqual(summary["n_test"].tolist(), [10, 10])
            self.assertIn("fold_mean_overall_accuracy", summary.columns)
            self.assertIn("fold_sd_overall_accuracy", summary.columns)
            self.assertEqual(len(audit), 1)
            self.assertIn("forecasting_predictions_sha256", audit.columns)

    def test_saved_artifact_validation_rejects_a_changed_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_dir = root / "output"
            inputs = self._write_inputs(root)
            artifacts = fivefold.run_analysis(
                **inputs,
                output_dir=output_dir,
                forecasting_runner=perfect_forecasting_runner,
                nowcasting_runner=perfect_nowcasting_runner,
            )
            summary = pd.read_csv(artifacts["metrics"])
            summary.loc[0, "overall_accuracy"] = 0.0
            summary.to_csv(artifacts["metrics"], index=False)

            with self.assertRaises(ValueError):
                fivefold._validate_saved_artifacts(
                    artifacts, pd.read_csv(inputs["canonical_test_path"])
                )


if __name__ == "__main__":
    unittest.main()
