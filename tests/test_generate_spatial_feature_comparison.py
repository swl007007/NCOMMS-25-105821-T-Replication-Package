from __future__ import annotations

import io
import hashlib
import re
import sys
import tempfile
import unittest
import warnings
from concurrent.futures import Future
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CODE = REPO_ROOT / "2.Source Code"
if str(SOURCE_CODE) not in sys.path:
    sys.path.insert(0, str(SOURCE_CODE))

import generate_spatial_feature_comparison as spatial


class RunConfigurationTests(unittest.TestCase):
    def test_condition_and_model_orders_are_exact(self):
        self.assertEqual(
            spatial.CONDITION_ORDER,
            (
                "baseline_with_lat_lon",
                "no_lat_lon",
                "knn5_spatial_means",
                "d200_spatial_means",
            ),
        )
        self.assertEqual(spatial.EXPERIMENT_CONDITIONS, spatial.CONDITION_ORDER[1:])
        self.assertEqual(spatial.MODEL_ORDER, ("Forecasting", "Nowcasting"))
        self.assertEqual(
            spatial.CONDITION_LABELS,
            {
                "baseline_with_lat_lon": "Baseline with latitude/longitude",
                "no_lat_lon": "No latitude/longitude",
                "knn5_spatial_means": "KNN-5 spatial means",
                "d200_spatial_means": "200 km spatial means",
            },
        )

    def test_default_run_is_formal(self):
        conditions, production_run = spatial.validate_run_configuration(
            None,
            spatial.DEFAULT_OUTPUT_DIR,
            random_state=0,
            workers=2,
        )
        self.assertEqual(conditions, spatial.CONDITION_ORDER)
        self.assertTrue(production_run)

    def test_subset_requires_nondefault_output_directory(self):
        with self.assertRaisesRegex(ValueError, "non-default output"):
            spatial.validate_run_configuration(
                ["no_lat_lon"],
                spatial.DEFAULT_OUTPUT_DIR,
                random_state=0,
                workers=1,
            )

    def test_nonzero_seed_requires_nondefault_output_directory(self):
        with self.assertRaisesRegex(ValueError, "non-default output"):
            spatial.validate_run_configuration(
                list(spatial.CONDITION_ORDER),
                spatial.DEFAULT_OUTPUT_DIR,
                random_state=7,
                workers=1,
            )

    def test_unknown_condition_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown condition"):
            spatial.validate_run_configuration(
                ["bogus"], Path("/tmp/spatial-bogus"), 0, 1
            )

    def test_duplicate_condition_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            spatial.validate_run_configuration(
                ["no_lat_lon", "no_lat_lon"], Path("/tmp/spatial-dup"), 0, 1
            )

    def test_zero_workers_fails(self):
        with self.assertRaisesRegex(ValueError, "workers"):
            spatial.validate_run_configuration(
                ["no_lat_lon"], Path("/tmp/spatial-workers"), 0, 0
            )

    def test_empty_conditions_fail(self):
        with self.assertRaisesRegex(ValueError, "at least one condition"):
            spatial.validate_run_configuration(
                [], Path("/tmp/spatial-empty"), 0, 1
            )

    def test_selected_conditions_are_returned_in_canonical_order(self):
        selected, production_run = spatial.validate_run_configuration(
            ["d200_spatial_means", "no_lat_lon"],
            Path("/tmp/spatial-order"),
            random_state=0,
            workers=1,
        )
        self.assertEqual(selected, ("no_lat_lon", "d200_spatial_means"))
        self.assertFalse(production_run)

    def test_parse_args_uses_approved_defaults(self):
        arguments = spatial.parse_args([])
        self.assertEqual(arguments.forecasting_input, spatial.DEFAULT_FORECASTING_INPUT)
        self.assertEqual(arguments.nowcasting_input, spatial.DEFAULT_NOWCASTING_INPUT)
        self.assertEqual(arguments.country_lookup, spatial.DEFAULT_COUNTRY_LOOKUP)
        self.assertEqual(arguments.general_params, spatial.DEFAULT_GENERAL_PARAMS)
        self.assertEqual(arguments.phase3_params, spatial.DEFAULT_PHASE3_PARAMS)
        self.assertEqual(arguments.output_dir, spatial.DEFAULT_OUTPUT_DIR)
        self.assertIsNone(arguments.conditions)
        self.assertEqual(arguments.workers, spatial.DEFAULT_WORKERS)
        self.assertEqual(arguments.random_state, spatial.DEFAULT_RANDOM_STATE)

    def test_parse_args_accepts_approved_condition_values(self):
        arguments = spatial.parse_args(
            [
                "--conditions",
                "d200_spatial_means",
                "no_lat_lon",
                "--workers",
                "1",
            ]
        )
        self.assertEqual(
            arguments.conditions,
            ["d200_spatial_means", "no_lat_lon"],
        )
        self.assertEqual(arguments.workers, 1)

    def test_help_exposes_exact_approved_option_surface(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised:
            with redirect_stdout(output):
                spatial.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        option_names = set(re.findall(r"--[a-z][a-z0-9-]*", output.getvalue()))
        self.assertEqual(
            option_names,
            {
                "--help",
                "--forecasting-input",
                "--nowcasting-input",
                "--country-lookup",
                "--general-params",
                "--phase3-params",
                "--output-dir",
                "--conditions",
                "--workers",
                "--random-state",
            },
        )


EXPECTED_LAYER1_STATIC_FEATURES = (
    "elevation",
    "market_access",
    "nitrogen_5-15cm_mean",
    "phh2o_5-15cm_mean",
    "cec_5-15cm_mean",
    "cfvo_5-15cm_mean",
    "soc_5-15cm_mean",
    "aez_groupid_4000",
    "aez_groupid_7000",
    "aez_groupid_9000",
    "aez_groupid_10000",
    "aez_groupid_12000",
    "aez_groupid_17000",
    "aez_groupid_19000",
    "aez_groupid_25000",
    "aez_groupid_30000",
    "aez_groupid_31000",
    "aez_groupid_32000",
    "aez_groupid_33000",
    "aez_groupid_34000",
    "aez_groupid_36000",
    "aez_groupid_40000",
    "aez_groupid_43000",
    "slope",
    "cropland",
    "rangeland",
    "area",
    "es_urban_pop",
    "urban_area",
    "distance_to_river",
    "ruggedness_index",
)


class InputManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecasting, cls.nowcasting, cls.lookup = spatial.load_prepared_inputs(
            spatial.DEFAULT_FORECASTING_INPUT,
            spatial.DEFAULT_NOWCASTING_INPUT,
            spatial.DEFAULT_COUNTRY_LOOKUP,
        )

    def test_live_feature_and_coordinate_contracts(self):
        layer1 = spatial.loco.select_layer1_features(self.forecasting)
        manifest = spatial.build_feature_manifest(layer1)
        coordinates = spatial.build_coordinate_table(self.forecasting, self.nowcasting)
        coordinate_validation = spatial.build_coordinate_validation_record(coordinates)

        self.assertEqual(len(self.forecasting), 5575)
        self.assertEqual(len(self.nowcasting), 5575)
        self.assertEqual(len(layer1), 106)
        self.assertEqual(len(coordinates), 1198)
        self.assertEqual(spatial.LAYER1_STATIC_FEATURES, EXPECTED_LAYER1_STATIC_FEATURES)
        self.assertEqual(manifest.columns.tolist(), list(spatial.FEATURE_MANIFEST_COLUMNS))
        self.assertEqual(manifest.shape, (175, 8))
        self.assertEqual(
            manifest.groupby(["layer", "feature_time_type"]).size().to_dict(),
            {
                ("layer1_shared", "coordinate"): 2,
                ("layer1_shared", "dynamic"): 73,
                ("layer1_shared", "static"): 31,
                ("nowcasting_layer2", "dynamic"): 69,
            },
        )
        self.assertEqual(
            manifest.loc[
                manifest["layer"].eq("layer1_shared"), "original_feature"
            ].tolist(),
            layer1,
        )
        self.assertEqual(
            manifest.loc[
                manifest["layer"].eq("nowcasting_layer2"), "original_feature"
            ].tolist(),
            list(spatial.loco.NOWCAST_FEATURES),
        )
        coordinates_manifest = manifest.loc[
            manifest["original_feature"].isin(["lat", "lon"])
        ]
        self.assertFalse(coordinates_manifest["neighbor_eligible"].any())
        self.assertTrue(coordinates_manifest["knn5_feature_name"].isna().all())
        self.assertTrue(coordinates_manifest["d200_feature_name"].isna().all())
        self.assertTrue(coordinate_validation["coordinates_complete_passed"])
        self.assertTrue(
            coordinate_validation["coordinates_unique_within_area_passed"]
        )
        self.assertTrue(
            coordinate_validation["coordinates_cross_table_equal_passed"]
        )
        self.assertTrue(coordinate_validation["coordinate_area_count_passed"])

    def test_coordinate_and_hash_outputs_ignore_input_row_order(self):
        shuffled_f = self.forecasting.sample(frac=1.0, random_state=11)
        shuffled_n = self.nowcasting.sample(frac=1.0, random_state=12)
        left = spatial.build_coordinate_table(self.forecasting, self.nowcasting)
        right = spatial.build_coordinate_table(shuffled_f, shuffled_n)
        pd.testing.assert_frame_equal(left, right)

        columns = ["area_id", "date", "fews_ipc_ha"]
        self.assertEqual(
            spatial.canonical_dataframe_sha256(
                self.forecasting, ["area_id", "date"], columns
            ),
            spatial.canonical_dataframe_sha256(
                shuffled_f, ["area_id", "date"], columns
            ),
        )
        self.assertEqual(
            spatial.canonical_key_sha256(self.forecasting),
            spatial.canonical_key_sha256(shuffled_f),
        )

    def test_canonical_hash_rejects_missing_duplicate_or_empty_keys(self):
        duplicate = self.forecasting.iloc[[0, 0]].copy()
        with self.assertRaisesRegex(ValueError, "not unique"):
            spatial.canonical_dataframe_sha256(
                duplicate,
                ["area_id", "date"],
                ["area_id", "date", "fews_ipc_ha"],
            )

        missing = self.forecasting.iloc[[0]].copy()
        missing["area_id"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            spatial.canonical_dataframe_sha256(
                missing,
                ["area_id", "date"],
                ["area_id", "date", "fews_ipc_ha"],
            )

        with self.assertRaisesRegex(ValueError, "at least one key"):
            spatial.canonical_dataframe_sha256(
                self.forecasting.iloc[[0]],
                [],
                ["fews_ipc_ha"],
            )

    def test_canonical_hash_normalizes_string_and_timestamp_dates(self):
        strings = pd.DataFrame(
            {"area_id": [2, 1], "date": ["2022-02-01", "2022-01-01"], "x": [2.0, 1.0]}
        )
        timestamps = strings.copy()
        timestamps["date"] = pd.to_datetime(timestamps["date"])
        self.assertEqual(
            spatial.canonical_dataframe_sha256(
                strings, ["area_id", "date"], ["area_id", "date", "x"]
            ),
            spatial.canonical_dataframe_sha256(
                timestamps, ["area_id", "date"], ["area_id", "date", "x"]
            ),
        )

    def test_canonical_hash_rejects_duplicates_created_by_date_normalization(self):
        duplicate_after_normalization = pd.DataFrame(
            {
                "area_id": [1, 1],
                "date": ["2022-01-01", "2022-01-01 00:00:00"],
                "x": [1.0, 2.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            spatial.canonical_dataframe_sha256(
                duplicate_after_normalization,
                ["area_id", "date"],
                ["area_id", "date", "x"],
            )

    def test_canonical_hash_rejects_missing_columns_and_nonmonthly_dates(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            spatial.canonical_dataframe_sha256(
                self.forecasting.iloc[[0]],
                ["area_id", "date"],
                ["area_id", "date", "not_a_column"],
            )
        intraday = pd.DataFrame(
            {"area_id": [1], "date": ["2022-01-01 12:00:00"], "x": [1.0]}
        )
        with self.assertRaisesRegex(ValueError, "midnight"):
            spatial.canonical_dataframe_sha256(
                intraday, ["area_id", "date"], ["area_id", "date", "x"]
            )

    def test_manifest_rejects_changed_live_feature_contracts(self):
        layer1 = spatial.loco.select_layer1_features(self.forecasting)
        with self.assertRaisesRegex(ValueError, "106"):
            spatial.build_feature_manifest(layer1[:-1])
        with self.assertRaisesRegex(ValueError, "ordered 69"):
            spatial.build_feature_manifest(
                layer1, tuple(reversed(spatial.loco.NOWCAST_FEATURES))
            )

    def test_coordinate_builder_fails_on_each_invalid_contract(self):
        forecasting = pd.DataFrame(
            {
                "area_id": [1, 1, 2],
                "lat": [0.0, 0.0, 1.0],
                "lon": [10.0, 10.0, 11.0],
            }
        )
        nowcasting = forecasting.copy()

        missing = forecasting.copy()
        missing.loc[0, "lat"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            spatial.build_coordinate_table(missing, nowcasting, expected_area_count=2)

        conflicting = forecasting.copy()
        conflicting.loc[1, "lon"] = 99.0
        with self.assertRaisesRegex(ValueError, "non-unique"):
            spatial.build_coordinate_table(
                conflicting, nowcasting, expected_area_count=2
            )

        different = nowcasting.copy()
        different.loc[different["area_id"].eq(2), "lat"] = 2.0
        with self.assertRaisesRegex(ValueError, "coordinates differ"):
            spatial.build_coordinate_table(
                forecasting, different, expected_area_count=2
            )

        with self.assertRaisesRegex(ValueError, "Expected 3"):
            spatial.build_coordinate_table(
                forecasting, nowcasting, expected_area_count=3
            )


class SpatialWeightTests(unittest.TestCase):
    def setUp(self):
        self.coordinates = pd.DataFrame(
            {
                "area_id": [1, 2, 3, 4, 5, 6, 7],
                "lat": [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 50.0],
                "lon": [0.0, 1.0, -1.0, 0.0, 0.0, 2.0, 50.0],
            }
        )

    def test_one_degree_equator_distance(self):
        value = spatial.haversine_distance_km(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(value, 111.1950802335, rtol=1e-10)

    def test_haversine_broadcasts_to_pairwise_shape(self):
        values = spatial.haversine_distance_km(
            np.array([[0.0], [1.0]]),
            np.array([[0.0], [0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 2.0]]),
        )
        self.assertEqual(values.shape, (2, 3))
        self.assertEqual(values.dtype, np.dtype("float64"))
        self.assertEqual(values[0, 0], 0.0)
        np.testing.assert_allclose(values[0, 1], 111.1950802335, rtol=1e-10)

    def test_knn5_excludes_self_and_breaks_ties_by_area_id(self):
        matrix = spatial.build_distance_matrix(
            self.coordinates.sample(frac=1, random_state=2)
        )
        neighbors = spatial.build_knn5_neighbors(matrix)
        position = matrix.area_to_pos[1]
        ids = matrix.area_ids[neighbors.neighbor_positions[position]].tolist()
        self.assertEqual(ids, [2, 3, 4, 5, 6])
        self.assertNotIn(1, ids)
        self.assertTrue(all(len(item) == 5 for item in neighbors.neighbor_positions))

    def test_knn_rejects_invalid_cardinality(self):
        matrix = spatial.build_distance_matrix(self.coordinates)
        for invalid in (0, len(self.coordinates)):
            with self.assertRaisesRegex(ValueError, "k"):
                spatial.build_knn5_neighbors(matrix, k=invalid)

    def test_d200_boundary_is_inclusive_and_has_no_fallback(self):
        manual = spatial.DistanceMatrix(
            area_ids=np.array([1, 2, 3], dtype=np.int64),
            distances_km=np.array(
                [
                    [0.0, 200.0, 200.0000001],
                    [200.0, 0.0, 500.0],
                    [200.0000001, 500.0, 0.0],
                ],
                dtype=np.float64,
            ),
            area_to_pos={1: 0, 2: 1, 3: 2},
        )
        neighbors = spatial.build_d200_neighbors(manual)
        self.assertEqual(
            manual.area_ids[neighbors.neighbor_positions[0]].tolist(), [2]
        )
        self.assertEqual(len(neighbors.neighbor_positions[2]), 0)
        self.assertEqual(neighbors.neighbor_positions[2].dtype, np.dtype("int64"))

    def test_weight_diagnostics_preserve_neighbor_order_and_empty_hash(self):
        manual = spatial.DistanceMatrix(
            area_ids=np.array([1, 2, 3], dtype=np.int64),
            distances_km=np.array(
                [[0.0, 10.0, 300.0], [10.0, 0.0, 400.0], [300.0, 400.0, 0.0]],
                dtype=np.float64,
            ),
            area_to_pos={1: 0, 2: 1, 3: 2},
        )
        neighbors = spatial.build_d200_neighbors(manual, radius_km=20.0)
        lookup = pd.DataFrame(
            {"area_id": [3, 1, 2], "country_code_3": ["BBB", "AAA", "AAA"]}
        )
        diagnostics = spatial.build_weight_diagnostics(neighbors, lookup)
        self.assertEqual(
            diagnostics.columns.tolist(), list(spatial.WEIGHT_DIAGNOSTIC_COLUMNS)
        )
        self.assertEqual(diagnostics["area_id"].tolist(), [1, 2, 3])
        first = diagnostics.set_index("area_id").loc[1]
        self.assertEqual(first["neighbor_ids"], "2")
        self.assertEqual(
            first["neighbor_ids_sha256"],
            hashlib.sha256(b"2").hexdigest(),
        )
        empty = diagnostics.set_index("area_id").loc[3]
        self.assertTrue(empty["zero_neighbor"])
        self.assertEqual(empty["neighbor_ids"], "")
        self.assertEqual(
            empty["neighbor_ids_sha256"], hashlib.sha256(b"").hexdigest()
        )
        self.assertTrue(pd.isna(empty["min_distance_km"]))

    def test_weight_diagnostics_require_exact_lookup_area_set(self):
        matrix = spatial.build_distance_matrix(self.coordinates)
        neighbors = spatial.build_knn5_neighbors(matrix)
        incomplete = pd.DataFrame(
            {
                "area_id": self.coordinates["area_id"].iloc[:-1],
                "country_code_3": ["AAA"] * (len(self.coordinates) - 1),
            }
        )
        with self.assertRaisesRegex(ValueError, "area set"):
            spatial.build_weight_diagnostics(neighbors, incomplete)


def build_interpolation_fixture(
    rows,
    *,
    feature="dyn",
    feature_time_type="dynamic",
    layer="nowcasting_layer2",
    coordinates=None,
    lookup=None,
):
    if coordinates is None:
        coordinates = pd.DataFrame(
            {
                "area_id": [1, 2, 3, 4],
                "lat": [0.0, 0.0, 0.0, 0.0],
                "lon": [0.0, 1.0, 2.0, 3.0],
            }
        )
    if lookup is None:
        lookup = pd.DataFrame(
            {
                "area_id": coordinates["area_id"],
                "country_code_3": ["AAA", "AAA", "AAA", "BBB"][: len(coordinates)],
            }
        )
    distance_matrix = spatial.build_distance_matrix(coordinates)
    source_index = spatial.build_observed_feature_index(
        observations=pd.DataFrame(rows),
        feature=feature,
        feature_time_type=feature_time_type,
        layer=layer,
        distance_matrix=distance_matrix,
        area_country=lookup,
    )
    area_country = dict(
        lookup[["area_id", "country_code_3"]].itertuples(index=False, name=None)
    )
    return source_index, distance_matrix, area_country


def standard_interpolation_fixture(
    *,
    include_own=True,
    include_same_country=True,
    include_global=True,
    layer="nowcasting_layer2",
):
    rows = [
        {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
        {"area_id": 2, "date": "2022-01-01", "dyn": np.nan},
        {
            "area_id": 2,
            "date": "2021-01-01",
            "dyn": 10.0 if include_own else np.nan,
        },
        {
            "area_id": 3,
            "date": "2022-01-01",
            "dyn": 20.0 if include_same_country else np.nan,
        },
        {
            "area_id": 4,
            "date": "2022-01-01",
            "dyn": 30.0 if include_global else np.nan,
        },
        {"area_id": 4, "date": "2022-02-01", "dyn": 40.0},
    ]
    return build_interpolation_fixture(rows, layer=layer)


def resolve_standard_interpolation(
    *,
    include_own=True,
    include_same_country=True,
    include_global=True,
    layer="nowcasting_layer2",
    condition="knn5_spatial_means",
    split="test",
):
    source_index, distance_matrix, area_country = standard_interpolation_fixture(
        include_own=include_own,
        include_same_country=include_same_country,
        include_global=include_global,
        layer=layer,
    )
    return spatial.resolve_neighbor_slot(
        aggregation_target_area_id=1,
        imputed_neighbor_area_id=2,
        target_month=pd.Timestamp("2022-01-01"),
        feature="dyn",
        feature_time_type="dynamic",
        layer=layer,
        source_index=source_index,
        distance_matrix=distance_matrix,
        area_country=area_country,
        condition=condition,
        split=split,
    )


class InterpolationTests(unittest.TestCase):
    def test_layer1_feature_month_uses_calendar_months(self):
        resolved = spatial.resolve_feature_month(
            pd.Timestamp("2020-02-29"), "dynamic", "layer1_shared"
        )
        self.assertEqual(resolved, pd.Timestamp("2019-02-28"))
        self.assertTrue(
            pd.isna(
                spatial.resolve_feature_month(
                    pd.Timestamp("2020-02-29"), "static", "layer1_shared"
                )
            )
        )

    def test_layer1_same_row_month_peer_is_eligible_with_twelve_month_gap(self):
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
            {"area_id": 2, "date": "2022-01-01", "dyn": np.nan},
            {"area_id": 3, "date": "2022-01-01", "dyn": 20.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(
            rows, layer="layer1_shared"
        )
        result = spatial.resolve_neighbor_slot(
            1,
            2,
            pd.Timestamp("2022-01-01"),
            "dyn",
            "dynamic",
            "layer1_shared",
            source_index,
            distance_matrix,
            area_country,
            "knn5_spatial_means",
            "test",
        )
        self.assertEqual(result.value, 20.0)
        self.assertEqual(result.audit_record["source_row_month"], "2022-01-01")
        self.assertEqual(result.audit_record["resolved_feature_month"], "2021-01-01")
        self.assertEqual(result.audit_record["max_permitted_feature_month"], "2021-01-01")
        self.assertEqual(result.audit_record["month_gap"], 12)
        self.assertTrue(result.audit_record["temporal_contract_passed"])

    def test_layer1_future_row_is_rejected_without_double_lagging(self):
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
            {"area_id": 2, "date": "2022-01-01", "dyn": np.nan},
            {"area_id": 3, "date": "2022-02-01", "dyn": 20.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(
            rows, layer="layer1_shared"
        )
        result = spatial.resolve_neighbor_slot(
            1,
            2,
            pd.Timestamp("2022-01-01"),
            "dyn",
            "dynamic",
            "layer1_shared",
            source_index,
            distance_matrix,
            area_country,
            "knn5_spatial_means",
            "test",
        )
        self.assertTrue(np.isnan(result.value))
        self.assertIsNone(result.audit_record)

    def test_layer2_allows_same_month_and_rejects_future(self):
        result = resolve_standard_interpolation(
            include_own=False,
            include_same_country=True,
            include_global=True,
        )
        self.assertEqual(result.value, 20.0)
        self.assertEqual(result.audit_record["source_area_id"], 3)
        self.assertEqual(result.audit_record["resolved_feature_month"], "2022-01-01")
        self.assertTrue(result.audit_record["temporal_contract_passed"])

    def test_layer2_future_only_returns_no_source(self):
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
            {"area_id": 2, "date": "2022-01-01", "dyn": np.nan},
            {"area_id": 3, "date": "2022-02-01", "dyn": 20.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(rows)
        result = spatial.resolve_neighbor_slot(
            1,
            2,
            pd.Timestamp("2022-01-01"),
            "dyn",
            "dynamic",
            "nowcasting_layer2",
            source_index,
            distance_matrix,
            area_country,
            "d200_spatial_means",
            "test",
        )
        self.assertTrue(np.isnan(result.value))
        self.assertIsNone(result.audit_record)

    def test_tier_priority_prefers_older_own_history(self):
        result = resolve_standard_interpolation()
        self.assertEqual(result.value, 10.0)
        self.assertEqual(result.audit_record["source_tier"], "own_history")
        self.assertEqual(result.audit_record["source_area_id"], 2)

    def test_same_country_prefers_recency_before_distance(self):
        coordinates = pd.DataFrame(
            {
                "area_id": [1, 2, 3, 4],
                "lat": [0.0] * 4,
                "lon": [-2.0, 0.0, 5.0, 1.0],
            }
        )
        lookup = pd.DataFrame(
            {"area_id": [1, 2, 3, 4], "country_code_3": ["AAA"] * 4}
        )
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
            {"area_id": 2, "date": "2022-01-01", "dyn": np.nan},
            {"area_id": 3, "date": "2022-01-01", "dyn": 30.0},
            {"area_id": 4, "date": "2021-12-01", "dyn": 40.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(
            rows, coordinates=coordinates, lookup=lookup
        )
        result = spatial.resolve_neighbor_slot(
            1, 2, pd.Timestamp("2022-01-01"), "dyn", "dynamic",
            "nowcasting_layer2", source_index, distance_matrix, area_country,
            "knn5_spatial_means", "test",
        )
        self.assertEqual(result.audit_record["source_area_id"], 3)

    def test_same_country_prefers_distance_then_area_id_for_same_month(self):
        coordinates = pd.DataFrame(
            {
                "area_id": [1, 2, 3, 4, 5],
                "lat": [0.0] * 5,
                "lon": [-3.0, 0.0, 1.0, -1.0, 2.0],
            }
        )
        lookup = pd.DataFrame(
            {"area_id": [1, 2, 3, 4, 5], "country_code_3": ["AAA"] * 5}
        )
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
            {"area_id": 2, "date": "2022-01-01", "dyn": np.nan},
            {"area_id": 3, "date": "2022-01-01", "dyn": 30.0},
            {"area_id": 4, "date": "2022-01-01", "dyn": 40.0},
            {"area_id": 5, "date": "2022-01-01", "dyn": 50.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(
            rows, coordinates=coordinates, lookup=lookup
        )
        result = spatial.resolve_neighbor_slot(
            1, 2, pd.Timestamp("2022-01-01"), "dyn", "dynamic",
            "nowcasting_layer2", source_index, distance_matrix, area_country,
            "knn5_spatial_means", "test",
        )
        self.assertEqual(result.audit_record["source_area_id"], 3)

    def test_spatial_fallback_excludes_aggregation_target_and_neighbor(self):
        result = resolve_standard_interpolation(
            include_own=False,
            include_same_country=False,
            include_global=True,
        )
        self.assertEqual(result.value, 30.0)
        self.assertEqual(result.audit_record["source_tier"], "global")
        self.assertNotIn(result.audit_record["source_area_id"], {1, 2})

    def test_no_source_returns_nan_without_lineage(self):
        result = resolve_standard_interpolation(
            include_own=False,
            include_same_country=False,
            include_global=False,
        )
        self.assertTrue(np.isnan(result.value))
        self.assertIsNone(result.audit_record)

    def test_static_own_value_has_missing_feature_month_fields(self):
        rows = [
            {"area_id": 1, "date": "2022-01-01", "static_x": 999.0},
            {"area_id": 2, "date": "2022-01-01", "static_x": np.nan},
            {"area_id": 2, "date": "2018-01-01", "static_x": 10.0},
            {"area_id": 3, "date": "2022-01-01", "static_x": 20.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(
            rows,
            feature="static_x",
            feature_time_type="static",
            layer="layer1_shared",
        )
        result = spatial.resolve_neighbor_slot(
            1, 2, pd.Timestamp("2022-01-01"), "static_x", "static",
            "layer1_shared", source_index, distance_matrix, area_country,
            "knn5_spatial_means", "test",
        )
        self.assertEqual(result.value, 10.0)
        self.assertEqual(result.audit_record["source_tier"], "own_history")
        self.assertTrue(pd.isna(result.audit_record["resolved_feature_month"]))
        self.assertTrue(pd.isna(result.audit_record["max_permitted_feature_month"]))
        self.assertTrue(pd.isna(result.audit_record["month_gap"]))

    def test_static_peer_selection_ignores_recency(self):
        coordinates = pd.DataFrame(
            {
                "area_id": [1, 2, 3, 4],
                "lat": [0.0] * 4,
                "lon": [-2.0, 0.0, 5.0, 1.0],
            }
        )
        lookup = pd.DataFrame(
            {"area_id": [1, 2, 3, 4], "country_code_3": ["AAA"] * 4}
        )
        rows = [
            {"area_id": 1, "date": "2022-01-01", "static_x": 999.0},
            {"area_id": 2, "date": "2022-01-01", "static_x": np.nan},
            {"area_id": 3, "date": "2022-01-01", "static_x": 30.0},
            {"area_id": 4, "date": "2018-01-01", "static_x": 40.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(
            rows,
            feature="static_x",
            feature_time_type="static",
            layer="layer1_shared",
            coordinates=coordinates,
            lookup=lookup,
        )
        result = spatial.resolve_neighbor_slot(
            1, 2, pd.Timestamp("2022-01-01"), "static_x", "static",
            "layer1_shared", source_index, distance_matrix, area_country,
            "d200_spatial_means", "test",
        )
        self.assertEqual(result.audit_record["source_area_id"], 4)

    def test_static_conflicts_fail_when_building_observed_index(self):
        rows = [
            {"area_id": 2, "date": "2018-01-01", "static_x": 10.0},
            {"area_id": 2, "date": "2019-01-01", "static_x": 11.0},
        ]
        with self.assertRaisesRegex(ValueError, "static"):
            build_interpolation_fixture(
                rows,
                feature="static_x",
                feature_time_type="static",
                layer="layer1_shared",
            )

    def test_observed_target_slot_cannot_be_sent_to_resolver(self):
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 999.0},
            {"area_id": 2, "date": "2022-01-01", "dyn": 10.0},
        ]
        source_index, distance_matrix, area_country = build_interpolation_fixture(rows)
        with self.assertRaisesRegex(ValueError, "already has an observed"):
            spatial.resolve_neighbor_slot(
                1, 2, pd.Timestamp("2022-01-01"), "dyn", "dynamic",
                "nowcasting_layer2", source_index, distance_matrix, area_country,
                "knn5_spatial_means", "test",
            )

    def test_index_rejects_generated_features_and_normalized_duplicate_keys(self):
        generated_rows = pd.DataFrame(
            {
                "area_id": [1],
                "date": ["2022-01-01"],
                "dyn__knn5_mean": [1.0],
            }
        )
        coordinates = pd.DataFrame(
            {"area_id": [1], "lat": [0.0], "lon": [0.0]}
        )
        lookup = pd.DataFrame(
            {"area_id": [1], "country_code_3": ["AAA"]}
        )
        matrix = spatial.build_distance_matrix(coordinates)
        with self.assertRaisesRegex(ValueError, "generated"):
            spatial.build_observed_feature_index(
                generated_rows,
                "dyn__knn5_mean",
                "dynamic",
                "nowcasting_layer2",
                matrix,
                lookup,
            )

        duplicate_rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 1.0},
            {"area_id": 1, "date": "2022-01-01 00:00:00", "dyn": 2.0},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_interpolation_fixture(duplicate_rows)

    def test_index_requires_exact_country_area_set(self):
        rows = [{"area_id": 1, "date": "2022-01-01", "dyn": 1.0}]
        coordinates = pd.DataFrame(
            {"area_id": [1, 2], "lat": [0.0, 0.0], "lon": [0.0, 1.0]}
        )
        incomplete = pd.DataFrame(
            {"area_id": [1], "country_code_3": ["AAA"]}
        )
        with self.assertRaisesRegex(ValueError, "area set"):
            build_interpolation_fixture(
                rows, coordinates=coordinates, lookup=incomplete
            )

    def test_resolution_does_not_mutate_observed_source_index(self):
        source_index, distance_matrix, area_country = standard_interpolation_fixture(
            include_own=False,
            include_same_country=True,
            include_global=True,
        )
        target_month = pd.Timestamp("2022-01-01")
        neighbor_position = source_index.area_to_pos[2]
        date_position = source_index.date_to_pos[target_month]
        self.assertFalse(source_index.observed[neighbor_position, date_position])
        first = spatial.resolve_neighbor_slot(
            1, 2, target_month, "dyn", "dynamic", "nowcasting_layer2",
            source_index, distance_matrix, area_country,
            "knn5_spatial_means", "test",
        )
        self.assertEqual(first.value, 20.0)
        self.assertFalse(source_index.observed[neighbor_position, date_position])
        self.assertTrue(np.isnan(source_index.values[neighbor_position, date_position]))

    def test_cached_resolution_keeps_condition_and_split_labels_separate(self):
        source_index, distance_matrix, area_country = standard_interpolation_fixture()
        labels = []
        for condition, split in (
            ("knn5_spatial_means", "train"),
            ("d200_spatial_means", "test"),
        ):
            result = spatial.resolve_neighbor_slot(
                1, 2, pd.Timestamp("2022-01-01"), "dyn", "dynamic",
                "nowcasting_layer2", source_index, distance_matrix, area_country,
                condition, split,
            )
            labels.append(
                (result.audit_record["condition"], result.audit_record["split"])
            )
        self.assertEqual(
            labels,
            [("knn5_spatial_means", "train"), ("d200_spatial_means", "test")],
        )


def six_area_augmentation_inputs(*, all_missing=False, absent_neighbor=False):
    coordinates = pd.DataFrame(
        {
            "area_id": [1, 2, 3, 4, 5, 6],
            "lat": [0.0] * 6,
            "lon": [0.0, 1.0, -1.0, 2.0, -2.0, 3.0],
        }
    )
    lookup = pd.DataFrame(
        {
            "area_id": [1, 2, 3, 4, 5, 6],
            "country_code_3": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
        }
    )
    if all_missing:
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 0.0},
            *[
                {"area_id": area_id, "date": "2022-01-01", "dyn": np.nan}
                for area_id in range(2, 7)
            ],
        ]
    elif absent_neighbor:
        rows = [
            {"area_id": 1, "date": "2022-01-01", "dyn": 0.0},
            {"area_id": 2, "date": "2021-01-01", "dyn": 10.0},
            {"area_id": 3, "date": "2022-01-01", "dyn": 3.0},
            {"area_id": 4, "date": "2022-01-01", "dyn": 4.0},
            {"area_id": 5, "date": "2022-01-01", "dyn": 5.0},
            {"area_id": 6, "date": "2022-01-01", "dyn": 6.0},
        ]
    else:
        rows = [
            {"area_id": area_id, "date": "2022-01-01", "dyn": float(area_id)}
            for area_id in range(1, 7)
        ]
    observations = pd.DataFrame(rows)
    target_rows = observations.loc[
        observations["area_id"].eq(1) & observations["date"].eq("2022-01-01")
    ].copy()
    distance_matrix = spatial.build_distance_matrix(coordinates)
    neighbors = spatial.build_knn5_neighbors(distance_matrix)
    return target_rows, observations, lookup, distance_matrix, neighbors


class SpatialAugmentationTests(unittest.TestCase):
    def test_mean_available_slots_uses_only_finite_values(self):
        mean, effective_count = spatial.mean_available_slots(
            [1.0, 2.0, np.nan, 4.0, np.inf]
        )
        self.assertAlmostEqual(mean, 7.0 / 3.0)
        self.assertEqual(effective_count, 3)
        missing_mean, missing_count = spatial.mean_available_slots(
            [np.nan, np.inf, -np.inf]
        )
        self.assertTrue(np.isnan(missing_mean))
        self.assertEqual(missing_count, 0)

    def test_knn_all_missing_neighbors_produce_nan_and_no_events(self):
        target, observations, lookup, distances, neighbors = (
            six_area_augmentation_inputs(all_missing=True)
        )
        result = spatial.augment_spatial_features(
            target,
            observations,
            ("dyn",),
            {"dyn": "dynamic"},
            neighbors,
            distances,
            lookup,
            "knn5_spatial_means",
            "nowcasting_layer2",
            "test",
        )
        self.assertTrue(np.isnan(result.data.loc[target.index[0], "dyn__knn5_mean"]))
        summary = result.interpolation_summary.iloc[0]
        self.assertEqual(summary["total_neighbor_slots"], 5)
        self.assertEqual(summary["original_missing_slots"], 5)
        self.assertEqual(summary["imputed_slots"], 0)
        self.assertEqual(summary["remaining_missing_slots"], 5)
        self.assertEqual(summary["rows_all_missing"], 1)
        self.assertEqual(len(result.interpolation_audit), 0)

    def test_absent_neighbor_row_uses_own_history_and_slot_identities(self):
        target, observations, lookup, distances, neighbors = (
            six_area_augmentation_inputs(absent_neighbor=True)
        )
        original = target.copy(deep=True)
        result = spatial.augment_spatial_features(
            target,
            observations,
            ("dyn",),
            {"dyn": "dynamic"},
            neighbors,
            distances,
            lookup,
            "knn5_spatial_means",
            "nowcasting_layer2",
            "test",
        )
        self.assertAlmostEqual(
            result.data.loc[target.index[0], "dyn__knn5_mean"], 5.6
        )
        pd.testing.assert_frame_equal(result.data[original.columns], original)
        self.assertTrue(result.data.index.equals(original.index))
        self.assertEqual(
            result.interpolation_summary.columns.tolist(),
            list(spatial.INTERPOLATION_SUMMARY_COLUMNS),
        )
        summary = result.interpolation_summary.iloc[0]
        self.assertEqual(summary["observed_slots"], 4)
        self.assertEqual(summary["original_missing_slots"], 1)
        self.assertEqual(summary["imputed_slots"], 1)
        self.assertEqual(summary["remaining_missing_slots"], 0)
        self.assertEqual(summary["effective_nonmissing_slots"], 5)
        self.assertEqual(summary["own_history_source_count"], 1)
        self.assertEqual(len(result.interpolation_audit), 1)
        self.assertEqual(
            result.interpolation_audit.iloc[0]["source_tier"], "own_history"
        )
        self.assertEqual(
            summary["observed_slots"] + summary["original_missing_slots"],
            summary["total_neighbor_slots"],
        )
        self.assertEqual(
            summary["imputed_slots"] + summary["remaining_missing_slots"],
            summary["original_missing_slots"],
        )
        self.assertEqual(
            summary["observed_slots"] + summary["imputed_slots"],
            summary["effective_nonmissing_slots"],
        )

    def test_d200_variable_support_includes_zero_neighbor_rows(self):
        coordinates = pd.DataFrame(
            {
                "area_id": [1, 2, 3],
                "lat": [0.0, 0.0, 0.0],
                "lon": [0.0, 10.0, 11.0],
            }
        )
        lookup = pd.DataFrame(
            {"area_id": [1, 2, 3], "country_code_3": ["AAA"] * 3}
        )
        observations = pd.DataFrame(
            {
                "area_id": [1, 2, 3],
                "date": ["2022-01-01"] * 3,
                "dyn": [1.0, 2.0, 3.0],
            }
        )
        distances = spatial.build_distance_matrix(coordinates)
        neighbors = spatial.build_d200_neighbors(distances)
        result = spatial.augment_spatial_features(
            observations,
            observations,
            ("dyn",),
            {"dyn": "dynamic"},
            neighbors,
            distances,
            lookup,
            "d200_spatial_means",
            "nowcasting_layer2",
            "test",
        )
        by_area = result.data.set_index("area_id")["dyn__d200_mean"]
        self.assertTrue(np.isnan(by_area.loc[1]))
        self.assertEqual(by_area.loc[2], 3.0)
        self.assertEqual(by_area.loc[3], 2.0)
        summary = result.interpolation_summary.iloc[0]
        self.assertEqual(summary["target_rows"], 3)
        self.assertEqual(summary["total_neighbor_slots"], 2)
        self.assertEqual(summary["observed_slots"], 2)
        self.assertEqual(summary["rows_with_effective_mean"], 2)
        self.assertEqual(summary["rows_all_missing"], 1)

    def test_augmentation_is_invariant_to_target_and_source_row_order(self):
        target, observations, lookup, distances, neighbors = (
            six_area_augmentation_inputs()
        )
        second_target = observations.loc[observations["area_id"].eq(2)]
        targets = pd.concat([target, second_target])
        left = spatial.augment_spatial_features(
            targets,
            observations,
            ("dyn",),
            {"dyn": "dynamic"},
            neighbors,
            distances,
            lookup,
            "knn5_spatial_means",
            "nowcasting_layer2",
            "test",
        )
        right = spatial.augment_spatial_features(
            targets.sample(frac=1.0, random_state=5),
            observations.sample(frac=1.0, random_state=6),
            ("dyn",),
            {"dyn": "dynamic"},
            neighbors,
            distances,
            lookup.sample(frac=1.0, random_state=7),
            "knn5_spatial_means",
            "nowcasting_layer2",
            "test",
        )
        columns = ["area_id", "date", "dyn", "dyn__knn5_mean"]
        pd.testing.assert_frame_equal(
            left.data[columns].sort_values(["area_id", "date"]).reset_index(drop=True),
            right.data[columns].sort_values(["area_id", "date"]).reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            left.interpolation_summary.reset_index(drop=True),
            right.interpolation_summary.reset_index(drop=True),
        )
        self.assertEqual(left.matrix_sha256, right.matrix_sha256)

    def test_live_condition_feature_lists_have_exact_order_and_counts(self):
        forecasting, _, _ = spatial.load_prepared_inputs(
            spatial.DEFAULT_FORECASTING_INPUT,
            spatial.DEFAULT_NOWCASTING_INPUT,
            spatial.DEFAULT_COUNTRY_LOOKUP,
        )
        original_layer1 = tuple(spatial.loco.select_layer1_features(forecasting))
        layer1, layer2 = spatial.build_condition_feature_lists(original_layer1)
        ablated = tuple(
            feature for feature in original_layer1 if feature not in {"lat", "lon"}
        )
        self.assertEqual(layer1["baseline_with_lat_lon"], original_layer1)
        self.assertEqual(layer1["no_lat_lon"], ablated)
        self.assertEqual(len(layer1["baseline_with_lat_lon"]), 106)
        self.assertEqual(len(layer1["no_lat_lon"]), 104)
        self.assertEqual(len(layer1["knn5_spatial_means"]), 208)
        self.assertEqual(len(layer1["d200_spatial_means"]), 208)
        self.assertEqual(
            layer1["knn5_spatial_means"][104:],
            tuple(f"{feature}__knn5_mean" for feature in ablated),
        )
        self.assertEqual(len(layer2["baseline_with_lat_lon"]), 69)
        self.assertEqual(layer2["no_lat_lon"], tuple(spatial.loco.NOWCAST_FEATURES))
        self.assertEqual(len(layer2["knn5_spatial_means"]), 138)
        self.assertEqual(len(layer2["d200_spatial_means"]), 138)

    def test_nonspatial_condition_matrices_preserve_nonrange_indices_and_are_lazy(self):
        forecasting, nowcasting, lookup = spatial.load_prepared_inputs(
            spatial.DEFAULT_FORECASTING_INPUT,
            spatial.DEFAULT_NOWCASTING_INPUT,
            spatial.DEFAULT_COUNTRY_LOOKUP,
        )
        custom_index = pd.Index(np.arange(10000, 10000 + len(forecasting)) * 3)
        forecasting = forecasting.copy()
        nowcasting = nowcasting.copy()
        forecasting.index = custom_index
        nowcasting.index = custom_index
        cutoff = pd.Timestamp("2022-01-01")
        train_mask = pd.to_datetime(forecasting["date"]).lt(cutoff)
        test_mask = ~train_mask
        now_train_mask = pd.to_datetime(nowcasting["date"]).lt(cutoff)
        now_test_mask = ~now_train_mask
        manifest = spatial.build_feature_manifest(
            spatial.loco.select_layer1_features(forecasting)
        )
        bundle = spatial.build_condition_matrices(
            forecasting,
            nowcasting,
            train_mask,
            test_mask,
            now_train_mask,
            now_test_mask,
            manifest,
            None,
            None,
            None,
            lookup,
            ("baseline_with_lat_lon", "no_lat_lon"),
        )
        self.assertEqual(
            tuple(bundle.forecasting),
            ("baseline_with_lat_lon", "no_lat_lon"),
        )
        for condition in bundle.forecasting:
            self.assertTrue(bundle.forecasting[condition].index.equals(custom_index))
            self.assertTrue(bundle.nowcasting[condition].index.equals(custom_index))
            pd.testing.assert_frame_equal(
                bundle.forecasting[condition][forecasting.columns], forecasting
            )
            pd.testing.assert_frame_equal(
                bundle.nowcasting[condition][nowcasting.columns], nowcasting
            )
        self.assertEqual(len(bundle.interpolation_audit), 0)
        self.assertEqual(len(bundle.interpolation_summary), 0)
        self.assertEqual(
            bundle.matrix_hashes.columns.tolist(),
            list(spatial.MATRIX_HASH_COLUMNS),
        )
        self.assertEqual(len(bundle.matrix_hashes), 8)

    def test_many_generated_columns_do_not_fragment_the_dataframe(self):
        feature_names = tuple(f"x{index}" for index in range(101))
        observations = pd.DataFrame(
            {
                "area_id": np.arange(1, 7),
                "date": ["2022-01-01"] * 6,
                **{
                    feature: np.arange(1, 7, dtype=float)
                    for feature in feature_names
                },
            }
        )
        coordinates = pd.DataFrame(
            {
                "area_id": np.arange(1, 7),
                "lat": [0.0] * 6,
                "lon": [0.0, 1.0, -1.0, 2.0, -2.0, 3.0],
            }
        )
        lookup = pd.DataFrame(
            {
                "area_id": np.arange(1, 7),
                "country_code_3": ["AAA"] * 6,
            }
        )
        distances = spatial.build_distance_matrix(coordinates)
        neighbors = spatial.build_knn5_neighbors(distances)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.PerformanceWarning)
            result = spatial.augment_spatial_features(
                observations.iloc[[0]],
                observations,
                feature_names,
                {feature: "dynamic" for feature in feature_names},
                neighbors,
                distances,
                lookup,
                "knn5_spatial_means",
                "nowcasting_layer2",
                "test",
            )
        performance_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, pd.errors.PerformanceWarning)
        ]
        self.assertEqual(performance_warnings, [])
        self.assertEqual(
            result.data.shape[1], observations.shape[1] + len(feature_names)
        )

    def test_interpolation_audit_uses_compact_columnar_dtypes(self):
        month_count = 120
        dates = pd.date_range("2012-01-01", periods=month_count, freq="MS")
        observations = pd.DataFrame(
            [
                {
                    "area_id": area_id,
                    "date": month,
                    "dyn": (
                        float(area_id)
                        if area_id == 1 or month == dates[0]
                        else np.nan
                    ),
                }
                for month in dates
                for area_id in range(1, 7)
            ]
        )
        target_rows = observations.loc[observations["area_id"].eq(1)].copy()
        coordinates = pd.DataFrame(
            {
                "area_id": np.arange(1, 7),
                "lat": [0.0] * 6,
                "lon": [0.0, 1.0, -1.0, 2.0, -2.0, 3.0],
            }
        )
        lookup = pd.DataFrame(
            {
                "area_id": np.arange(1, 7),
                "country_code_3": ["AAA"] * 6,
            }
        )
        distances = spatial.build_distance_matrix(coordinates)
        neighbors = spatial.build_knn5_neighbors(distances)

        result = spatial.augment_spatial_features(
            target_rows,
            observations,
            ("dyn",),
            {"dyn": "dynamic"},
            neighbors,
            distances,
            lookup,
            "knn5_spatial_means",
            "nowcasting_layer2",
            "test",
        )
        audit = result.interpolation_audit

        self.assertEqual(len(audit), (month_count - 1) * 5)
        self.assertEqual(
            audit.columns.tolist(), list(spatial.INTERPOLATION_AUDIT_COLUMNS)
        )
        for column in (
            "condition",
            "layer",
            "split",
            "feature",
            "feature_time_type",
            "source_country_code_3",
            "source_tier",
        ):
            self.assertIsInstance(audit[column].dtype, pd.CategoricalDtype)
        for column in (
            "target_month",
            "max_permitted_feature_month",
            "source_row_month",
            "resolved_feature_month",
        ):
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(audit[column]))
        bytes_per_event = (
            audit.memory_usage(index=True, deep=True).sum() / len(audit)
        )
        self.assertLess(
            bytes_per_event,
            160.0,
            f"interpolation audit uses {bytes_per_event:.1f} deep bytes/event",
        )


def prediction_frame_with_raw_value(phase3_raw):
    return pd.DataFrame(
        {
            "phase2_test": [0.60],
            "phase3_test": [0.30],
            "phase4_test": [0.10],
            "phase5_test": [0.00],
            "phase2_pred_raw": [0.10],
            "phase3_pred_raw": [phase3_raw],
            "phase4_pred_raw": [0.10],
            "phase5_pred_raw": [0.00],
        }
    )


def prediction_frame_with_all_raw_predictions(value):
    frame = prediction_frame_with_raw_value(value)
    for phase in range(2, 6):
        frame[f"phase{phase}_pred_raw"] = value
    return frame


def metric_fixture_where_raw_and_rounded_r2_differ():
    frame = pd.DataFrame(
        {
            "area_id": [1, 2, 3],
            "date": pd.to_datetime(["2022-01-01"] * 3),
            "country_code_3": ["AAA", "AAA", "BBB"],
            "phase2_test": [0.60, 0.80, 0.30],
            "phase3_test": [0.10, 0.40, 0.30],
            "phase4_test": [0.05, 0.10, 0.10],
            "phase5_test": [0.00, 0.00, 0.00],
            "phase2_pred_raw": [0.55, 0.75, 0.25],
            "phase3_pred_raw": [0.104, 0.204, 0.894],
            "phase4_pred_raw": [0.04, 0.09, 0.08],
            "phase5_pred_raw": [0.00, 0.00, 0.00],
        }
    )
    return spatial.wide_predictions_to_phases_preserving_raw(frame)


def complete_synthetic_prediction_fixture():
    frames = []
    for condition_index, condition in enumerate(spatial.CONDITION_ORDER):
        for model_index, model in enumerate(spatial.MODEL_ORDER):
            frame = pd.DataFrame(
                {
                    "area_id": [1, 2, 3],
                    "date": pd.to_datetime(["2022-01-01"] * 3),
                    "country_code_3": ["AAA", "AAA", "BBB"],
                    "phase2_test": [0.60, 0.80, 0.30],
                    "phase3_test": [0.10, 0.40, 0.30],
                    "phase4_test": [0.05, 0.10, 0.10],
                    "phase5_test": [0.00, 0.00, 0.00],
                    "phase2_pred_raw": [0.55, 0.75, 0.25],
                    "phase3_pred_raw": [0.104, 0.204, 0.894],
                    "phase4_pred_raw": [0.04, 0.09, 0.08],
                    "phase5_pred_raw": [0.00, 0.00, 0.00],
                }
            )
            shift = condition_index * 0.005 + model_index * 0.002
            frame["phase3_pred_raw"] = frame["phase3_pred_raw"] + shift
            frame = spatial.wide_predictions_to_phases_preserving_raw(frame)
            frame.insert(0, "model", model)
            frame.insert(0, "condition_label", spatial.CONDITION_LABELS[condition])
            frame.insert(0, "condition", condition)
            frame["source_row_index"] = np.arange(len(frame))
            frame["split_id"] = "temporal_2022"
            frame["source_overall_phase"] = frame["overall_phase"]
            for phase in range(2, 6):
                if model == "Nowcasting":
                    frame[f"phase{phase}_layer1_pred"] = (
                        frame[f"phase{phase}_pred_raw"] * 0.75
                    )
                    frame[f"phase{phase}_residual_pred"] = (
                        frame[f"phase{phase}_pred_raw"] * 0.25
                    )
                else:
                    frame[f"phase{phase}_layer1_pred"] = np.nan
                    frame[f"phase{phase}_residual_pred"] = np.nan
            frames.append(frame.loc[:, spatial.PREDICTION_COLUMNS])
    return pd.concat(frames, ignore_index=True)


class RecordingMeanRegressor:
    def __init__(self, **params):
        self.params = params
        self.fit_columns = ()
        self.fit_x = pd.DataFrame()
        self.fit_y = np.array([], dtype=float)
        self.mean_ = 0.0

    def fit(self, x, y):
        self.fit_columns = tuple(x.columns)
        self.fit_x = x.reset_index(drop=True).copy()
        self.fit_y = np.asarray(y, dtype=float)
        self.mean_ = float(np.mean(self.fit_y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_, dtype=float)


class SequentialRecordingFactory:
    def __init__(self):
        self.models = []

    def __call__(self, **params):
        model = RecordingMeanRegressor(**params)
        self.models.append(model)
        return model

    @property
    def max_depth_sequence(self):
        return [model.params["max_depth"] for model in self.models]

    @property
    def fit_columns(self):
        return [model.fit_columns for model in self.models]


class RecordingInlineExecutor:
    last_max_workers = None
    last_start_method = None

    def __init__(self, max_workers, mp_context=None):
        type(self).last_max_workers = max_workers
        type(self).last_start_method = (
            None if mp_context is None else mp_context.get_start_method()
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


def synthetic_prepared_inputs():
    data = pd.DataFrame(
        {
            "area_id": [1, 1, 2, 2],
            "date": pd.to_datetime(
                ["2020-01-01", "2022-01-01", "2020-01-01", "2022-01-01"]
            ),
            "country_code_3": ["AAA", "AAA", "BBB", "BBB"],
            "overall_phase": [2, 3, 4, 2],
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [4.0, 3.0, 2.0, 1.0],
            "phase2_worse": [0.60, 0.70, 0.80, 0.40],
            "phase3_worse": [0.10, 0.30, 0.50, 0.10],
            "phase4_worse": [0.05, 0.10, 0.20, 0.05],
            "phase5_worse": [0.00, 0.00, 0.05, 0.00],
        }
    )
    return data.sort_values(["area_id", "date"], kind="mergesort").reset_index(
        drop=True
    )


def synthetic_train_mask(data=None):
    data = synthetic_prepared_inputs() if data is None else data
    return pd.to_datetime(data["date"]).lt(pd.Timestamp("2022-01-01"))


def synthetic_test_mask(data=None):
    data = synthetic_prepared_inputs() if data is None else data
    return pd.to_datetime(data["date"]).ge(pd.Timestamp("2022-01-01"))


def synthetic_prepared_pair_with_reversed_nowcast_rows():
    forecasting = synthetic_prepared_inputs()
    nowcasting = forecasting[["area_id", "date", "country_code_3"]].copy()
    nowcasting["CPI"] = [101.0, 102.0, 201.0, 202.0]
    nowcasting = nowcasting.iloc[::-1].reset_index(drop=True)
    return forecasting, nowcasting


class ModelRunnerTests(unittest.TestCase):
    def test_forecasting_uses_four_models_and_condition_columns(self):
        factory = SequentialRecordingFactory()
        result = spatial.fit_forecasting_condition(
            forecasting=synthetic_prepared_inputs(),
            train_mask=synthetic_train_mask(),
            test_mask=synthetic_test_mask(),
            split_id="temporal_2022",
            layer1_features=("x1", "x2"),
            general_params={"max_depth": 11, "random_state": 0, "n_jobs": 1},
            phase3_params={"max_depth": 9, "random_state": 0, "n_jobs": 1},
            condition="no_lat_lon",
            estimator_factory=factory,
        )
        self.assertEqual(len(factory.models), 4)
        self.assertEqual(factory.max_depth_sequence, [11, 9, 9, 9])
        self.assertEqual(factory.fit_columns, [("x1", "x2")] * 4)
        self.assertEqual(result["condition"].unique().tolist(), ["no_lat_lon"])
        self.assertEqual(result.columns.tolist(), list(spatial.PREDICTION_COLUMNS))
        component_columns = [
            f"phase{phase}_{component}_pred"
            for phase in range(2, 6)
            for component in ("layer1", "residual")
        ]
        self.assertTrue(result[component_columns].isna().all().all())

    def test_local_runners_retain_key_target_and_numeric_guards(self):
        duplicate = pd.concat(
            [synthetic_prepared_inputs(), synthetic_prepared_inputs().iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            spatial.fit_forecasting_condition(
                duplicate,
                duplicate["date"].lt(pd.Timestamp("2022-01-01")),
                duplicate["date"].ge(pd.Timestamp("2022-01-01")),
                "temporal_2022",
                ("x1", "x2"),
                {"max_depth": 11},
                {"max_depth": 9},
                condition="no_lat_lon",
                estimator_factory=SequentialRecordingFactory(),
            )
        missing_target = synthetic_prepared_inputs()
        missing_target.loc[0, "phase2_worse"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing phase2_worse"):
            spatial.fit_forecasting_condition(
                missing_target,
                synthetic_train_mask(missing_target),
                synthetic_test_mask(missing_target),
                "temporal_2022",
                ("x1", "x2"),
                {"max_depth": 11},
                {"max_depth": 9},
                condition="no_lat_lon",
                estimator_factory=SequentialRecordingFactory(),
            )
        forecasting, nowcasting = synthetic_prepared_pair_with_reversed_nowcast_rows()
        nowcasting["CPI"] = nowcasting["CPI"].astype(str)
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            spatial.fit_nowcasting_condition(
                forecasting,
                nowcasting,
                synthetic_train_mask(forecasting),
                synthetic_test_mask(forecasting),
                synthetic_train_mask(nowcasting),
                synthetic_test_mask(nowcasting),
                "temporal_2022",
                ("x1", "x2"),
                ("CPI",),
                {"max_depth": 11},
                {"max_depth": 9},
                condition="knn5_spatial_means",
                estimator_factory=SequentialRecordingFactory(),
            )

    def test_nowcasting_uses_eight_models_and_keyed_residual_merge(self):
        factory = SequentialRecordingFactory()
        forecasting, nowcasting = synthetic_prepared_pair_with_reversed_nowcast_rows()
        result = spatial.fit_nowcasting_condition(
            forecasting=forecasting,
            nowcasting=nowcasting,
            train_mask=synthetic_train_mask(forecasting),
            test_mask=synthetic_test_mask(forecasting),
            now_train_mask=synthetic_train_mask(nowcasting),
            now_test_mask=synthetic_test_mask(nowcasting),
            split_id="temporal_2022",
            layer1_features=("x1", "x2"),
            layer2_features=("CPI",),
            general_params={"max_depth": 11, "random_state": 0, "n_jobs": 1},
            phase3_params={"max_depth": 9, "random_state": 0, "n_jobs": 1},
            condition="knn5_spatial_means",
            estimator_factory=factory,
        )
        self.assertEqual(len(factory.models), 8)
        self.assertEqual(factory.max_depth_sequence, [11, 11, 9, 9, 9, 9, 9, 9])
        first_layer1 = factory.models[0]
        first_layer2 = factory.models[1]
        forecast_train = forecasting.loc[synthetic_train_mask(forecasting)].copy()
        expected_residual = (
            forecast_train["phase2_worse"].to_numpy() - first_layer1.mean_
        )
        expected = dict(
            zip(
                forecast_train["area_id"].map({1: 101.0, 2: 201.0}),
                expected_residual,
            )
        )
        observed = dict(zip(first_layer2.fit_x["CPI"], first_layer2.fit_y))
        self.assertEqual(observed, expected)
        for phase in range(2, 6):
            np.testing.assert_allclose(
                result[f"phase{phase}_pred_raw"],
                result[f"phase{phase}_layer1_pred"]
                + result[f"phase{phase}_residual_pred"],
            )

    def test_run_condition_models_real_serial_and_parallel_paths_agree(self):
        forecasting, nowcasting = synthetic_prepared_pair_with_reversed_nowcast_rows()
        general_params = {
            "n_estimators": 1,
            "max_depth": 1,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "random_state": 0,
            "n_jobs": 1,
        }
        results_by_workers = {}
        for workers in (1, 2):
            results_by_workers[workers] = spatial.run_condition_models(
                "no_lat_lon",
                forecasting,
                nowcasting,
                synthetic_train_mask(forecasting),
                synthetic_test_mask(forecasting),
                synthetic_train_mask(nowcasting),
                synthetic_test_mask(nowcasting),
                ("x1", "x2"),
                ("CPI",),
                general_params,
                general_params,
                workers=workers,
            )
            self.assertEqual(tuple(results_by_workers[workers]), spatial.MODEL_ORDER)
            for model in spatial.MODEL_ORDER:
                self.assertEqual(
                    results_by_workers[workers][model].columns.tolist(),
                    list(spatial.PREDICTION_COLUMNS),
                )
                self.assertEqual(len(results_by_workers[workers][model]), 2)
        for model in spatial.MODEL_ORDER:
            pd.testing.assert_frame_equal(
                results_by_workers[1][model],
                results_by_workers[2][model],
                check_exact=True,
            )

    def test_run_condition_models_caps_process_workers_at_two(self):
        original_executor = spatial.ProcessPoolExecutor
        original_fit = spatial._fit_condition_model
        calls = []

        def fake_fit(model_name, payload):
            calls.append(model_name)
            return model_name, pd.DataFrame({"model": [model_name]})

        try:
            spatial.ProcessPoolExecutor = RecordingInlineExecutor
            spatial._fit_condition_model = fake_fit
            forecasting, nowcasting = synthetic_prepared_pair_with_reversed_nowcast_rows()
            results = spatial.run_condition_models(
                "no_lat_lon",
                forecasting,
                nowcasting,
                synthetic_train_mask(forecasting),
                synthetic_test_mask(forecasting),
                synthetic_train_mask(nowcasting),
                synthetic_test_mask(nowcasting),
                ("x1", "x2"),
                ("CPI",),
                {"max_depth": 1},
                {"max_depth": 1},
                workers=7,
            )
        finally:
            spatial.ProcessPoolExecutor = original_executor
            spatial._fit_condition_model = original_fit
        self.assertEqual(RecordingInlineExecutor.last_max_workers, 2)
        self.assertEqual(RecordingInlineExecutor.last_start_method, "spawn")
        self.assertEqual(calls, list(spatial.MODEL_ORDER))
        self.assertEqual(tuple(results), spatial.MODEL_ORDER)

    def test_run_condition_models_wraps_worker_failures_with_context(self):
        original_executor = spatial.ProcessPoolExecutor
        original_fit = spatial._fit_condition_model

        def failing_fit(model_name, payload):
            raise ValueError(f"synthetic {model_name} failure")

        try:
            spatial.ProcessPoolExecutor = RecordingInlineExecutor
            spatial._fit_condition_model = failing_fit
            forecasting, nowcasting = synthetic_prepared_pair_with_reversed_nowcast_rows()
            with self.assertRaisesRegex(
                RuntimeError,
                r"no_lat_lon (Forecasting|Nowcasting) model failed",
            ):
                spatial.run_condition_models(
                    "no_lat_lon",
                    forecasting,
                    nowcasting,
                    synthetic_train_mask(forecasting),
                    synthetic_test_mask(forecasting),
                    synthetic_train_mask(nowcasting),
                    synthetic_test_mask(nowcasting),
                    ("x1", "x2"),
                    ("CPI",),
                    {"max_depth": 1},
                    {"max_depth": 1},
                    workers=2,
                )
        finally:
            spatial.ProcessPoolExecutor = original_executor
            spatial._fit_condition_model = original_fit

    def test_run_condition_models_rejects_worker_return_mismatch(self):
        original_executor = spatial.ProcessPoolExecutor
        original_fit = spatial._fit_condition_model

        def mismatched_fit(model_name, payload):
            returned = "Nowcasting" if model_name == "Forecasting" else "Forecasting"
            return returned, pd.DataFrame({"model": [returned]})

        try:
            spatial.ProcessPoolExecutor = RecordingInlineExecutor
            spatial._fit_condition_model = mismatched_fit
            forecasting, nowcasting = synthetic_prepared_pair_with_reversed_nowcast_rows()
            with self.assertRaisesRegex(RuntimeError, "Model worker mismatch"):
                spatial.run_condition_models(
                    "no_lat_lon",
                    forecasting,
                    nowcasting,
                    synthetic_train_mask(forecasting),
                    synthetic_test_mask(forecasting),
                    synthetic_train_mask(nowcasting),
                    synthetic_test_mask(nowcasting),
                    ("x1", "x2"),
                    ("CPI",),
                    {"max_depth": 1},
                    {"max_depth": 1},
                    workers=2,
                )
        finally:
            spatial.ProcessPoolExecutor = original_executor
            spatial._fit_condition_model = original_fit


class PredictionMetricTests(unittest.TestCase):
    def test_prediction_and_metric_schemas_are_exact(self):
        self.assertEqual(len(spatial.PREDICTION_COLUMNS), 32)
        self.assertEqual(
            spatial.PREDICTION_COLUMNS[:11],
            (
                "condition",
                "condition_label",
                "model",
                "area_id",
                "date",
                "country_code_3",
                "source_row_index",
                "split_id",
                "source_overall_phase",
                "overall_phase",
                "overall_phase_pred",
            ),
        )
        self.assertEqual(len(spatial.METRIC_COLUMNS), 33)

    def test_raw_predictions_are_preserved_and_rounded_only_for_phase(self):
        frame = prediction_frame_with_raw_value(phase3_raw=0.204)
        original_raw = frame.filter(like="_pred_raw").copy(deep=True)
        converted = spatial.wide_predictions_to_phases_preserving_raw(frame)
        pd.testing.assert_frame_equal(
            converted[original_raw.columns], original_raw, check_exact=True
        )
        self.assertEqual(converted.loc[0, "phase3_pred_rounded"], 0.20)
        self.assertEqual(converted.loc[0, "overall_phase_pred"], 3)
        self.assertEqual(
            [f"phase{phase}_pred_rounded" for phase in range(2, 6)],
            [
                column
                for column in converted.columns
                if column.endswith("_pred_rounded")
            ],
        )

    def test_nonpositive_row_is_retained_as_phase_one(self):
        converted = spatial.wide_predictions_to_phases_preserving_raw(
            prediction_frame_with_all_raw_predictions(-0.01)
        )
        self.assertEqual(len(converted), 1)
        self.assertEqual(converted.loc[0, "overall_phase_pred"], 1)
        self.assertTrue(converted.loc[0, "nonpositive_cumulative_prediction_sum"])

    def test_missing_or_infinite_raw_prediction_fails(self):
        for invalid in (np.nan, np.inf, -np.inf):
            frame = prediction_frame_with_raw_value(phase3_raw=invalid)
            with self.assertRaisesRegex(ValueError, "finite"):
                spatial.wide_predictions_to_phases_preserving_raw(frame)

    def test_r2_uses_raw_not_rounded_predictions(self):
        predictions = metric_fixture_where_raw_and_rounded_r2_differ()
        record = spatial.calculate_comparison_metrics(
            predictions,
            "baseline_with_lat_lon",
            spatial.CONDITION_LABELS["baseline_with_lat_lon"],
            "Forecasting",
        )
        expected = r2_score(
            predictions["phase3_test"], predictions["phase3_pred_raw"]
        )
        rounded = r2_score(
            predictions["phase3_test"], predictions["phase3_pred_rounded"]
        )
        self.assertNotEqual(expected, rounded)
        self.assertAlmostEqual(record["phase3plus_r2"], expected)

    def test_overall_accuracy_is_exact_five_phase_accuracy(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1, 2],
                "date": pd.to_datetime(["2022-01-01", "2022-01-01"]),
                "country_code_3": ["AAA", "BBB"],
                "overall_phase": [3, 4],
                "overall_phase_pred": [4, 4],
                "phase3_test": [0.20, 0.40],
                "phase3_pred_raw": [0.25, 0.35],
                "nonpositive_cumulative_prediction_sum": [False, False],
            }
        )
        record = spatial.calculate_comparison_metrics(
            predictions,
            "no_lat_lon",
            spatial.CONDITION_LABELS["no_lat_lon"],
            "Forecasting",
        )
        self.assertEqual(record["phase3plus_precision"], 1.0)
        self.assertEqual(record["phase3plus_recall"], 1.0)
        self.assertEqual(record["overall_accuracy"], 0.5)

    def test_undefined_metric_reasons_are_explicit(self):
        predictions = pd.DataFrame(
            {
                "area_id": [1],
                "date": pd.to_datetime(["2022-01-01"]),
                "country_code_3": ["AAA"],
                "overall_phase": [1],
                "overall_phase_pred": [1],
                "phase3_test": [0.0],
                "phase3_pred_raw": [0.1],
                "nonpositive_cumulative_prediction_sum": [False],
            }
        )
        record = spatial.calculate_comparison_metrics(
            predictions,
            "no_lat_lon",
            spatial.CONDITION_LABELS["no_lat_lon"],
            "Forecasting",
        )
        self.assertTrue(pd.isna(record["phase3plus_precision"]))
        self.assertEqual(
            record["phase3plus_precision_undefined_reason"],
            "no_predicted_phase3plus",
        )
        self.assertTrue(pd.isna(record["phase3plus_recall"]))
        self.assertEqual(
            record["phase3plus_recall_undefined_reason"],
            "no_actual_phase3plus",
        )
        self.assertTrue(pd.isna(record["phase3plus_r2"]))
        self.assertEqual(
            record["phase3plus_r2_undefined_reason"],
            "insufficient_observations",
        )

    def test_fractional_overall_phase_is_rejected(self):
        predictions = metric_fixture_where_raw_and_rounded_r2_differ()
        predictions["overall_phase"] = predictions["overall_phase"].astype(float)
        predictions.loc[0, "overall_phase"] = 2.5
        with self.assertRaisesRegex(ValueError, "integer"):
            spatial.calculate_comparison_metrics(
                predictions,
                "no_lat_lon",
                spatial.CONDITION_LABELS["no_lat_lon"],
                "Forecasting",
            )

    def test_integer_nonpositive_flags_are_rejected(self):
        predictions = metric_fixture_where_raw_and_rounded_r2_differ()
        predictions["nonpositive_cumulative_prediction_sum"] = [0, 1, 0]
        with self.assertRaisesRegex(ValueError, "Boolean"):
            spatial.calculate_comparison_metrics(
                predictions,
                "no_lat_lon",
                spatial.CONDITION_LABELS["no_lat_lon"],
                "Forecasting",
            )

    def test_metrics_use_original_notebook_references_without_overwriting_reruns(self):
        combined = complete_synthetic_prediction_fixture()
        metrics = spatial.build_metrics_table(combined, production_run=True)
        self.assertEqual(metrics.columns.tolist(), list(spatial.METRIC_COLUMNS))
        self.assertEqual(metrics.shape, (8, 33))
        self.assertEqual(
            list(metrics[["condition", "model"]].itertuples(index=False, name=None)),
            [
                (condition, model)
                for condition in spatial.CONDITION_ORDER
                for model in spatial.MODEL_ORDER
            ],
        )
        expected_references = {
            "Forecasting": {
                "phase3plus_precision": 0.7750702905342081,
                "phase3plus_recall": 0.9408418657565415,
                "overall_accuracy": 0.6495726495726496,
                "phase3plus_r2": 0.2489985704986828,
            },
            "Nowcasting": {
                "phase3plus_precision": 0.8035892323030908,
                "phase3plus_recall": 0.9169510807736063,
                "overall_accuracy": 0.6666666666666666,
                "phase3plus_r2": 0.27554513043222684,
            },
        }
        metric_columns = {
            "phase3plus_precision": "baseline_phase3plus_precision",
            "phase3plus_recall": "baseline_phase3plus_recall",
            "overall_accuracy": "baseline_overall_accuracy",
            "phase3plus_r2": "baseline_phase3plus_r2",
        }
        for model in spatial.MODEL_ORDER:
            model_rows = metrics.loc[metrics["model"].eq(model)].set_index("condition")
            rerun_predictions = combined.loc[
                combined["condition"].eq("baseline_with_lat_lon")
                & combined["model"].eq(model)
            ]
            rerun_record = spatial.calculate_comparison_metrics(
                rerun_predictions,
                "baseline_with_lat_lon",
                spatial.CONDITION_LABELS["baseline_with_lat_lon"],
                model,
            )
            for metric, baseline_column in metric_columns.items():
                reference = expected_references[model][metric]
                self.assertTrue(np.allclose(model_rows[baseline_column], reference))
                self.assertAlmostEqual(
                    model_rows.loc["baseline_with_lat_lon", metric],
                    rerun_record[metric],
                )
                for condition in spatial.CONDITION_ORDER:
                    current = model_rows.loc[condition, metric]
                    self.assertAlmostEqual(
                        model_rows.loc[condition, f"{metric}_signed_delta"],
                        current - reference,
                    )
                    self.assertAlmostEqual(
                        model_rows.loc[condition, f"{metric}_absolute_delta"],
                        abs(current - reference),
                    )

    def test_diagnostic_without_baseline_keeps_schema_and_nan_deltas(self):
        predictions = complete_synthetic_prediction_fixture()
        predictions = predictions.loc[
            predictions["condition"].eq("no_lat_lon")
        ].reset_index(drop=True)
        metrics = spatial.build_metrics_table(predictions, production_run=False)
        self.assertEqual(metrics.shape, (2, 33))
        baseline_and_delta_columns = [
            column
            for column in spatial.METRIC_COLUMNS
            if column.startswith("baseline_")
            or column.endswith("_signed_delta")
            or column.endswith("_absolute_delta")
        ]
        self.assertEqual(len(baseline_and_delta_columns), 12)
        self.assertTrue(metrics[baseline_and_delta_columns].isna().all().all())


def complete_synthetic_metrics():
    metrics = spatial.build_metrics_table(
        complete_synthetic_prediction_fixture(),
        production_run=True,
    )
    metrics["n_test"] = spatial.DEFAULT_EXPECTED_TEST_ROWS
    metrics["n_test_areas"] = 646
    return metrics


class SpatialFigureTests(unittest.TestCase):
    def test_figure_has_exact_twelve_panel_contract(self):
        metrics = complete_synthetic_metrics().sample(frac=1.0, random_state=9)
        figure = spatial.create_spatial_feature_comparison_figure(metrics)
        self.assertEqual(len(figure.axes), 12)
        letters = [
            text.get_text()
            for axis in figure.axes
            for text in axis.texts
            if text.get_text() in list("abcdefghijkl")
        ]
        self.assertEqual(letters, list("abcdefghijkl"))
        self.assertIn("2022", figure._suptitle.get_text())
        self.assertIn("1,170", figure._suptitle.get_text())
        self.assertEqual(
            [figure.axes[row * 4].get_ylabel() for row in range(3)],
            [spatial.CONDITION_LABELS[condition] for condition in spatial.EXPERIMENT_CONDITIONS],
        )
        expected_titles = [
            "Phase 3+ precision",
            "Phase 3+ recall",
            "Overall-phase accuracy",
            "Phase 3+ share R²",
        ]
        for row in range(3):
            self.assertEqual(
                [figure.axes[row * 4 + column].get_title(loc="left") for column in range(4)],
                expected_titles,
            )
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual(
            [text.get_text() for text in figure.legends[0].get_texts()],
            ["Experiment", "Original notebook reference"],
        )
        spatial.plt.close(figure)

    def test_each_panel_has_experiment_baseline_connectors_and_value_labels(self):
        figure = spatial.create_spatial_feature_comparison_figure(
            complete_synthetic_metrics()
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for index, axis in enumerate(figure.axes):
            self.assertEqual(len(axis.collections), 4)
            self.assertEqual(len(axis.lines), 2 + int(index % 4 == 3))
            for collection in axis.collections:
                self.assertEqual(len(collection.get_offsets()), 1)
                x_position, y_position = collection.get_offsets()[0]
                center_x, center_y = axis.transData.transform(
                    (x_position, y_position)
                )
                marker_bbox = collection.get_paths()[0].get_extents(
                    spatial.mpl.transforms.Affine2D(collection.get_transforms()[0])
                ).translated(center_x, center_y)
                stroke_padding = (
                    float(collection.get_linewidths()[0]) * figure.dpi / 72.0 / 2.0
                )
                marker_bbox = spatial.mpl.transforms.Bbox.from_extents(
                    marker_bbox.x0 - stroke_padding,
                    marker_bbox.y0 - stroke_padding,
                    marker_bbox.x1 + stroke_padding,
                    marker_bbox.y1 + stroke_padding,
                )
                self.assertGreaterEqual(marker_bbox.x0, axis.bbox.x0)
                self.assertLessEqual(marker_bbox.x1, axis.bbox.x1)
                self.assertGreaterEqual(marker_bbox.y0, axis.bbox.y0)
                self.assertLessEqual(marker_bbox.y1, axis.bbox.y1)
            panel_letter = next(
                text
                for text in axis.texts
                if text.get_text() == chr(ord("a") + index)
            )
            letter_bbox = panel_letter.get_window_extent(renderer)
            protected_text = [
                *axis.get_xticklabels(),
                *axis.get_yticklabels(),
                axis._left_title,
            ]
            for text in protected_text:
                if text.get_visible() and text.get_text().strip():
                    self.assertFalse(
                        letter_bbox.overlaps(text.get_window_extent(renderer)),
                        (panel_letter.get_text(), text.get_text()),
                    )
            numeric_annotations = [
                text
                for text in axis.texts
                if re.fullmatch(r"-?\d+\.\d{3}", text.get_text())
            ]
            self.assertEqual(len(numeric_annotations), 2)
            for annotation in numeric_annotations:
                bbox = annotation.get_window_extent(renderer)
                self.assertGreaterEqual(bbox.x0, axis.bbox.x0)
                self.assertLessEqual(bbox.x1, axis.bbox.x1)
                self.assertGreaterEqual(bbox.y0, axis.bbox.y0)
                self.assertLessEqual(bbox.y1, axis.bbox.y1)
        spatial.plt.close(figure)

    def test_metric_columns_share_limits_and_r2_has_zero_line(self):
        figure = spatial.create_spatial_feature_comparison_figure(
            complete_synthetic_metrics()
        )
        for column in range(4):
            limits = [figure.axes[row * 4 + column].get_ylim() for row in range(3)]
            self.assertEqual(limits[0], limits[1])
            self.assertEqual(limits[1], limits[2])
        for row in range(3):
            axis = figure.axes[row * 4 + 3]
            self.assertTrue(
                any(
                    np.asarray(line.get_ydata()).shape == (2,)
                    and np.allclose(line.get_ydata(), [0.0, 0.0])
                    for line in axis.lines
                )
            )
        spatial.plt.close(figure)

    def test_figure_rejects_duplicate_drifted_or_undefined_metrics(self):
        base = complete_synthetic_metrics()
        cases = []
        cases.append(("duplicate", pd.concat([base, base.iloc[[0]]], ignore_index=True)))
        bad_baseline = base.copy()
        selector = bad_baseline["condition"].eq("no_lat_lon") & bad_baseline[
            "model"
        ].eq("Forecasting")
        bad_baseline.loc[selector, "baseline_phase3plus_precision"] += 0.1
        cases.append(("baseline", bad_baseline))
        bad_delta = base.copy()
        bad_delta.loc[selector, "phase3plus_precision_signed_delta"] += 0.1
        cases.append(("delta", bad_delta))
        undefined = base.copy()
        undefined.loc[selector, "phase3plus_precision"] = np.nan
        cases.append(("finite", undefined))
        for message, metrics in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    spatial.create_spatial_feature_comparison_figure(metrics)

    def test_save_figure_writes_jpg_png_and_pdf(self):
        figure = spatial.create_spatial_feature_comparison_figure(
            complete_synthetic_metrics()
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = spatial.save_spatial_feature_comparison_figure(
                figure,
                Path(directory),
            )
            self.assertEqual(
                set(paths),
                {"figure_jpg", "figure_png", "figure_pdf"},
            )
            self.assertEqual(paths["figure_jpg"].read_bytes()[:2], b"\xff\xd8")
            self.assertEqual(
                paths["figure_png"].read_bytes()[:8],
                b"\x89PNG\r\n\x1a\n",
            )
            self.assertEqual(paths["figure_pdf"].read_bytes()[:4], b"%PDF")
        spatial.plt.close(figure)


def complete_prediction_mapping():
    combined = complete_synthetic_prediction_fixture()
    return {
        (condition, model): group.reset_index(drop=True)
        for (condition, model), group in combined.groupby(
            ["condition", "model"], sort=False
        )
    }


def synthetic_subset_predictions(condition="no_lat_lon"):
    combined = complete_synthetic_prediction_fixture()
    return combined.loc[combined["condition"].eq(condition)].reset_index(drop=True)


def synthetic_subset_metrics(condition="no_lat_lon"):
    return spatial.build_metrics_table(
        synthetic_subset_predictions(condition), production_run=False
    )


def complete_formal_prediction_fixture():
    frames = []
    combined = complete_synthetic_prediction_fixture()
    row_positions = np.arange(spatial.DEFAULT_EXPECTED_TEST_ROWS) % 3
    for condition in spatial.CONDITION_ORDER:
        for model in spatial.MODEL_ORDER:
            template = combined.loc[
                combined["condition"].eq(condition)
                & combined["model"].eq(model)
            ].reset_index(drop=True)
            frame = template.iloc[row_positions].reset_index(drop=True).copy()
            frame["area_id"] = np.arange(1, spatial.DEFAULT_EXPECTED_TEST_ROWS + 1)
            frame["source_row_index"] = np.arange(spatial.DEFAULT_EXPECTED_TEST_ROWS)
            frames.append(frame.loc[:, spatial.PREDICTION_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def empty_frame(columns):
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def synthetic_manifest():
    return empty_frame(spatial.FEATURE_MANIFEST_COLUMNS)


def synthetic_weights():
    return empty_frame(spatial.WEIGHT_DIAGNOSTIC_COLUMNS)


def synthetic_interpolation_audit():
    return empty_frame(spatial.INTERPOLATION_AUDIT_COLUMNS)


def nonempty_interpolation_audit():
    frame = pd.DataFrame(
        {
            "condition": ["knn5_spatial_means"],
            "layer": ["layer1_shared"],
            "split": ["test"],
            "feature": ["dynamic_feature"],
            "feature_time_type": ["dynamic"],
            "aggregation_target_area_id": [1],
            "imputed_neighbor_area_id": [2],
            "target_month": pd.to_datetime(["2022-01-01"]),
            "max_permitted_feature_month": pd.to_datetime(["2021-01-01"]),
            "source_area_id": [3],
            "source_country_code_3": ["AAA"],
            "source_row_month": pd.to_datetime(["2022-01-01"]),
            "resolved_feature_month": pd.to_datetime(["2021-01-01"]),
            "month_gap": pd.Series([12], dtype="Int32"),
            "source_tier": ["same_country"],
            "distance_imputed_neighbor_to_source_km": [12.5],
            "temporal_contract_passed": [True],
        }
    )
    frame["condition"] = frame["condition"].astype("category")
    frame["layer"] = frame["layer"].astype("category")
    frame["split"] = frame["split"].astype("category")
    frame["feature_time_type"] = frame["feature_time_type"].astype("category")
    frame["source_tier"] = frame["source_tier"].astype("category")
    return frame.loc[:, spatial.INTERPOLATION_AUDIT_COLUMNS]


def synthetic_interpolation_summary():
    return empty_frame(spatial.INTERPOLATION_SUMMARY_COLUMNS)


def synthetic_source_audit(
    production_run,
    condition="no_lat_lon",
    predictions=None,
):
    if predictions is None:
        predictions = (
            complete_synthetic_prediction_fixture()
            if production_run
            else synthetic_subset_predictions(condition)
        )
    metrics = spatial.build_metrics_table(
        predictions,
        production_run=production_run,
    )
    audit = metrics[
        [
            "condition",
            "condition_label",
            "model",
            "n_test",
            "test_key_sha256",
        ]
    ].copy()
    audit["production_run"] = bool(production_run)
    audit["run_label"] = "formal" if production_run else "diagnostic"
    audit["coordinates_complete_passed"] = True
    audit["coordinates_unique_within_area_passed"] = True
    audit["coordinates_cross_table_equal_passed"] = True
    audit["coordinate_area_count_passed"] = True
    audit["knn5_contract_passed"] = True
    audit["d200_contract_passed"] = True
    audit["temporal_violation_count"] = 0
    return audit


def synthetic_matrix_hashes(condition):
    return pd.DataFrame(
        [
            {
                "condition": condition,
                "layer": layer,
                "split": split,
                "feature_count": 208 if layer == "layer1_shared" else 138,
                "matrix_sha256": f"{condition}-{layer}-{split}",
            }
            for layer in ("layer1_shared", "nowcasting_layer2")
            for split in ("train", "test")
        ],
        columns=spatial.MATRIX_HASH_COLUMNS,
    )


def synthetic_coordinate_validation():
    return {
        "coordinates_complete_passed": True,
        "coordinates_unique_within_area_passed": True,
        "coordinates_cross_table_equal_passed": True,
        "coordinate_area_count": 1198,
        "coordinate_area_count_passed": True,
        "coordinate_table_sha256": "coordinate-hash",
    }


def source_audit_input_paths():
    return {
        "forecasting_input": spatial.DEFAULT_FORECASTING_INPUT,
        "nowcasting_input": spatial.DEFAULT_NOWCASTING_INPUT,
        "country_lookup": spatial.DEFAULT_COUNTRY_LOOKUP,
        "general_params": spatial.DEFAULT_GENERAL_PARAMS,
        "phase3_params": spatial.DEFAULT_PHASE3_PARAMS,
        "generator": Path(spatial.__file__),
    }


class ArtifactOrchestrationTests(unittest.TestCase):
    def test_source_audit_separates_original_notebook_reference_from_rerun(self):
        predictions = synthetic_subset_predictions("baseline_with_lat_lon")
        metrics = spatial.build_metrics_table(predictions, production_run=False)
        audit = spatial.build_source_audit(
            predictions,
            metrics,
            synthetic_matrix_hashes("baseline_with_lat_lon"),
            synthetic_weights(),
            synthetic_interpolation_audit(),
            synthetic_coordinate_validation(),
            source_audit_input_paths(),
            "manifest-hash",
            random_state=0,
            workers=1,
            production_run=False,
        ).set_index("model")

        expected = {
            "Forecasting": {
                "path": "Table1_Forecasting_main.ipynb",
                "cell_index": 1,
                "precision": 0.7750702905342081,
                "recall": 0.9408418657565415,
                "accuracy": 0.6495726495726496,
                "r2": 0.2489985704986828,
            },
            "Nowcasting": {
                "path": "Table1_Nowcasting_two_layer.ipynb",
                "cell_index": 2,
                "precision": 0.8035892323030908,
                "recall": 0.9169510807736063,
                "accuracy": 0.6666666666666666,
                "r2": 0.27554513043222684,
            },
        }
        metric_rows = metrics.set_index("model")
        for model, reference in expected.items():
            row = audit.loc[model]
            self.assertEqual(row["metric_reference_type"], "original_notebook_stored_output")
            self.assertEqual(row["legacy_notebook_path"], reference["path"])
            self.assertEqual(row["legacy_notebook_cell_index"], reference["cell_index"])
            self.assertEqual(row["legacy_notebook_environment"], "unpinned_windows_xgboost")
            self.assertFalse(row["legacy_prediction_rows_available"])
            self.assertAlmostEqual(
                row["legacy_notebook_phase3plus_precision"], reference["precision"]
            )
            self.assertAlmostEqual(
                row["legacy_notebook_phase3plus_recall"], reference["recall"]
            )
            self.assertAlmostEqual(
                row["legacy_notebook_overall_accuracy"], reference["accuracy"]
            )
            self.assertAlmostEqual(row["legacy_notebook_phase3plus_r2"], reference["r2"])
            self.assertAlmostEqual(
                row["controlled_rerun_baseline_phase3plus_precision"],
                metric_rows.loc[model, "phase3plus_precision"],
            )
            self.assertAlmostEqual(
                row["controlled_rerun_baseline_phase3plus_recall"],
                metric_rows.loc[model, "phase3plus_recall"],
            )
            self.assertAlmostEqual(
                row["controlled_rerun_baseline_overall_accuracy"],
                metric_rows.loc[model, "overall_accuracy"],
            )
            self.assertAlmostEqual(
                row["controlled_rerun_baseline_phase3plus_r2_raw"],
                metric_rows.loc[model, "phase3plus_r2"],
            )

    def test_combined_predictions_require_eight_identical_key_sets(self):
        groups = complete_prediction_mapping()
        combined = spatial.build_combined_predictions(
            groups, spatial.CONDITION_ORDER, expected_test_rows=3
        )
        self.assertEqual(len(combined), 24)
        hashes = {
            spatial.canonical_key_sha256(group)
            for _, group in combined.groupby(["condition", "model"], sort=False)
        }
        self.assertEqual(len(hashes), 1)

    def test_combined_predictions_reject_duplicate_test_keys(self):
        groups = complete_prediction_mapping()
        original = groups[("no_lat_lon", "Forecasting")]
        groups[("no_lat_lon", "Forecasting")] = pd.concat(
            [
                original.iloc[[0]],
                original.iloc[[0]],
                original.iloc[[1]],
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            spatial.build_combined_predictions(
                groups, spatial.CONDITION_ORDER, expected_test_rows=3
            )

    def test_combined_predictions_reject_unique_but_different_key_set(self):
        groups = complete_prediction_mapping()
        changed = groups[("no_lat_lon", "Forecasting")].copy()
        changed.loc[0, "area_id"] = 9999
        groups[("no_lat_lon", "Forecasting")] = changed
        with self.assertRaisesRegex(ValueError, "test-key hashes differ"):
            spatial.build_combined_predictions(
                groups, spatial.CONDITION_ORDER, expected_test_rows=3
            )

    def test_combined_predictions_reject_missing_or_extra_groups(self):
        groups = complete_prediction_mapping()
        groups.pop(("d200_spatial_means", "Nowcasting"))
        with self.assertRaisesRegex(ValueError, "mapping"):
            spatial.build_combined_predictions(
                groups, spatial.CONDITION_ORDER, expected_test_rows=3
            )
        groups = complete_prediction_mapping()
        groups[("bogus", "Forecasting")] = next(iter(groups.values())).copy()
        with self.assertRaisesRegex(ValueError, "mapping"):
            spatial.build_combined_predictions(
                groups, spatial.CONDITION_ORDER, expected_test_rows=3
            )

    def test_subset_run_never_writes_formal_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = spatial.write_artifacts(
                predictions=synthetic_subset_predictions(),
                metrics=synthetic_subset_metrics(),
                feature_manifest=synthetic_manifest(),
                weight_diagnostics=synthetic_weights(),
                interpolation_audit=synthetic_interpolation_audit(),
                interpolation_summary=synthetic_interpolation_summary(),
                source_audit=synthetic_source_audit(production_run=False),
                output_dir=Path(directory),
                production_run=False,
            )
            self.assertFalse(any(key.startswith("figure_") for key in artifacts))
            self.assertFalse(
                any(
                    Path(directory).glob(
                        "precision_recall_accuracy_p3r2_spatial_feature_comparison.*"
                    )
                )
            )

    def test_subset_run_rejects_stale_formal_figure_without_modifying_it(self):
        for filename in spatial.FORMAL_FIGURE_FILENAMES:
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory)
                    stale_figure = output_dir / filename
                    previous = b"previous-formal-figure"
                    stale_figure.write_bytes(previous)
                    with self.assertRaisesRegex(
                        ValueError, "clean diagnostic output directory"
                    ):
                        spatial.write_artifacts(
                            predictions=synthetic_subset_predictions(),
                            metrics=synthetic_subset_metrics(),
                            feature_manifest=synthetic_manifest(),
                            weight_diagnostics=synthetic_weights(),
                            interpolation_audit=synthetic_interpolation_audit(),
                            interpolation_summary=synthetic_interpolation_summary(),
                            source_audit=synthetic_source_audit(production_run=False),
                            output_dir=output_dir,
                            production_run=False,
                        )
                    self.assertEqual(stale_figure.read_bytes(), previous)
                    self.assertEqual(list(output_dir.iterdir()), [stale_figure])

    def test_source_audit_publication_rejects_bogus_group_label_and_hash(self):
        predictions = synthetic_subset_predictions()
        metrics = synthetic_subset_metrics()
        mutations = {
            "condition": "bogus_condition",
            "condition_label": "Bogus label",
            "model": "BogusModel",
            "test_key_sha256": "bogus-hash",
        }
        for column, value in mutations.items():
            with self.subTest(column=column):
                audit = synthetic_source_audit(production_run=False)
                audit.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "Source audit"):
                    spatial._validate_source_audit_for_publication(
                        audit,
                        predictions,
                        metrics,
                        production_run=False,
                    )

    def test_source_audit_rejects_fractional_or_string_n_test(self):
        predictions = synthetic_subset_predictions()
        metrics = synthetic_subset_metrics()
        for table_name, value in (
            ("audit", 3.9),
            ("audit", "3"),
            ("metrics", 3.9),
            ("metrics", "3"),
        ):
            with self.subTest(table_name=table_name, value=value):
                audit = synthetic_source_audit(production_run=False)
                changed_metrics = metrics.copy()
                target = audit if table_name == "audit" else changed_metrics
                target["n_test"] = target["n_test"].astype(object)
                target.loc[0, "n_test"] = value
                with self.assertRaisesRegex(ValueError, "n_test"):
                    spatial._validate_source_audit_for_publication(
                        audit,
                        predictions,
                        changed_metrics,
                        production_run=False,
                    )

    def test_source_audit_rejects_aligned_unknown_condition_or_model(self):
        for column, value in (
            ("condition", "bogus_condition"),
            ("model", "BogusModel"),
        ):
            with self.subTest(column=column):
                predictions = synthetic_subset_predictions()
                metrics = synthetic_subset_metrics()
                audit = synthetic_source_audit(production_run=False)
                selector = predictions["model"].eq("Forecasting")
                predictions.loc[selector, column] = value
                metrics.loc[metrics["model"].eq("Forecasting"), column] = value
                audit.loc[audit["model"].eq("Forecasting"), column] = value
                with self.assertRaisesRegex(ValueError, "unknown condition or model"):
                    spatial._validate_source_audit_for_publication(
                        audit,
                        predictions,
                        metrics,
                        production_run=False,
                    )

    def test_source_audit_rejects_duplicate_and_metric_group_drift(self):
        predictions = synthetic_subset_predictions()
        metrics = synthetic_subset_metrics()
        audit = synthetic_source_audit(production_run=False)
        cases = (
            (
                "duplicate_audit",
                pd.concat([audit, audit.iloc[[0]]], ignore_index=True),
                metrics,
            ),
            (
                "duplicate_metrics",
                audit,
                pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True),
            ),
            (
                "metric_group_drift",
                audit,
                metrics.assign(
                    condition=metrics["condition"].mask(
                        metrics["model"].eq("Forecasting"),
                        "knn5_spatial_means",
                    )
                ),
            ),
        )
        for name, changed_audit, changed_metrics in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "Source audit"):
                    spatial._validate_source_audit_for_publication(
                        changed_audit,
                        predictions,
                        changed_metrics,
                        production_run=False,
                    )

    def test_formal_source_audit_requires_contract_flags_and_zero_violations(self):
        predictions = complete_synthetic_prediction_fixture()
        metrics = spatial.build_metrics_table(predictions, production_run=True)
        failures = {
            "coordinates_complete_passed": False,
            "coordinates_unique_within_area_passed": False,
            "coordinates_cross_table_equal_passed": False,
            "coordinate_area_count_passed": False,
            "knn5_contract_passed": False,
            "d200_contract_passed": False,
            "temporal_violation_count": 1,
        }
        for column, value in failures.items():
            with self.subTest(column=column):
                audit = synthetic_source_audit(production_run=True)
                audit.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "formal source audit"):
                    spatial._validate_source_audit_for_publication(
                        audit,
                        predictions,
                        metrics,
                        production_run=True,
                    )

    def test_nonempty_interpolation_gzip_is_reproducible_and_roundtrips(self):
        hashes = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                artifacts = spatial.write_artifacts(
                    predictions=synthetic_subset_predictions(),
                    metrics=synthetic_subset_metrics(),
                    feature_manifest=synthetic_manifest(),
                    weight_diagnostics=synthetic_weights(),
                    interpolation_audit=nonempty_interpolation_audit(),
                    interpolation_summary=synthetic_interpolation_summary(),
                    source_audit=synthetic_source_audit(production_run=False),
                    output_dir=Path(directory),
                    production_run=False,
                )
                path = artifacts["interpolation_audit_csv_gz"]
                hashes.append(spatial.sha256_file(path))
                restored = pd.read_csv(path, compression="gzip")
                self.assertEqual(
                    restored.columns.tolist(), list(spatial.INTERPOLATION_AUDIT_COLUMNS)
                )
                self.assertEqual(restored.loc[0, "month_gap"], 12)
                self.assertEqual(restored.loc[0, "source_tier"], "same_country")
                self.assertTrue(restored.loc[0, "temporal_contract_passed"])
        self.assertEqual(hashes[0], hashes[1])

    def test_source_audit_scopes_events_by_condition_and_model_layer(self):
        predictions = synthetic_subset_predictions("knn5_spatial_means")
        metrics = spatial.build_metrics_table(predictions, production_run=False)
        matrix_hashes = synthetic_matrix_hashes("knn5_spatial_means")
        event_defaults = {
            column: None for column in spatial.INTERPOLATION_AUDIT_COLUMNS
        }
        events = pd.DataFrame(
            [
                {
                    **event_defaults,
                    "condition": "knn5_spatial_means",
                    "layer": "layer1_shared",
                    "source_tier": "own_history",
                    "temporal_contract_passed": True,
                },
                {
                    **event_defaults,
                    "condition": "knn5_spatial_means",
                    "layer": "nowcasting_layer2",
                    "source_tier": "global",
                    "temporal_contract_passed": False,
                },
                {
                    **event_defaults,
                    "condition": "knn5_spatial_means",
                    "layer": "nowcasting_layer2",
                    "source_tier": "global",
                    "temporal_contract_passed": True,
                },
                {
                    **event_defaults,
                    "condition": "knn5_spatial_means",
                    "layer": "nowcasting_layer2",
                    "source_tier": "global",
                    "temporal_contract_passed": False,
                },
                {
                    **event_defaults,
                    "condition": "d200_spatial_means",
                    "layer": "layer1_shared",
                    "source_tier": "same_country",
                    "temporal_contract_passed": True,
                },
            ],
            columns=spatial.INTERPOLATION_AUDIT_COLUMNS,
        )
        audit = spatial.build_source_audit(
            predictions,
            metrics,
            matrix_hashes,
            synthetic_weights(),
            events,
            synthetic_coordinate_validation(),
            source_audit_input_paths(),
            "manifest-hash",
            random_state=0,
            workers=1,
            production_run=False,
        ).set_index("model")
        self.assertEqual(audit.loc["Forecasting", "consumed_interpolation_event_count"], 1)
        self.assertEqual(audit.loc["Forecasting", "layer2_interpolation_event_count"], 0)
        self.assertEqual(audit.loc["Forecasting", "temporal_violation_count"], 0)
        self.assertEqual(audit.loc["Nowcasting", "consumed_interpolation_event_count"], 4)
        self.assertEqual(audit.loc["Nowcasting", "layer2_interpolation_event_count"], 3)
        self.assertEqual(audit.loc["Nowcasting", "layer2_global_event_count"], 3)
        self.assertEqual(audit.loc["Nowcasting", "temporal_violation_count"], 2)
        expected_hashes = metrics.set_index("model")["test_key_sha256"]
        self.assertEqual(audit["test_key_sha256"].to_dict(), expected_hashes.to_dict())

    def test_all_zero_d200_diagnostics_fail_the_contract(self):
        predictions = synthetic_subset_predictions("d200_spatial_means")
        metrics = spatial.build_metrics_table(predictions, production_run=False)
        diagnostics = pd.DataFrame(
            {
                "scheme": ["d200"] * spatial.DEFAULT_EXPECTED_AREAS,
                "area_id": np.arange(1, spatial.DEFAULT_EXPECTED_AREAS + 1),
                "country_code_3": ["AAA"] * spatial.DEFAULT_EXPECTED_AREAS,
                "neighbor_count": [0] * spatial.DEFAULT_EXPECTED_AREAS,
                "min_distance_km": [np.nan] * spatial.DEFAULT_EXPECTED_AREAS,
                "max_distance_km": [np.nan] * spatial.DEFAULT_EXPECTED_AREAS,
                "mean_distance_km": [np.nan] * spatial.DEFAULT_EXPECTED_AREAS,
                "zero_neighbor": [True] * spatial.DEFAULT_EXPECTED_AREAS,
                "neighbor_ids": [""] * spatial.DEFAULT_EXPECTED_AREAS,
                "neighbor_ids_sha256": [hashlib.sha256(b"").hexdigest()]
                * spatial.DEFAULT_EXPECTED_AREAS,
            },
            columns=spatial.WEIGHT_DIAGNOSTIC_COLUMNS,
        )
        audit = spatial.build_source_audit(
            predictions,
            metrics,
            synthetic_matrix_hashes("d200_spatial_means"),
            diagnostics,
            synthetic_interpolation_audit(),
            synthetic_coordinate_validation(),
            source_audit_input_paths(),
            "manifest-hash",
            random_state=0,
            workers=1,
            production_run=False,
        )
        self.assertFalse(audit["d200_contract_passed"].any())

    def test_publication_failure_restores_all_previous_artifacts(self):
        names = (
            "spatial_feature_comparison_predictions.csv",
            "spatial_feature_comparison_metrics.csv",
            "spatial_feature_comparison_feature_manifest.csv",
            "spatial_feature_comparison_weight_diagnostics.csv",
            "spatial_feature_interpolation_audit.csv.gz",
            "spatial_feature_interpolation_summary.csv",
            *spatial.FORMAL_FIGURE_FILENAMES,
            "spatial_feature_comparison_source_audit.csv",
        )
        for error_type in (OSError, KeyboardInterrupt):
            with self.subTest(error_type=error_type.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory)
                    expected = {}
                    for index, name in enumerate(names):
                        content = f"previous-{index}-{name}".encode("utf-8")
                        (output_dir / name).write_bytes(content)
                        expected[name] = content

                    original_replace = Path.replace
                    original_create = getattr(
                        spatial, "create_spatial_feature_comparison_figure", None
                    )
                    original_save = getattr(
                        spatial, "save_spatial_feature_comparison_figure", None
                    )
                    staged_publications = 0

                    def create_figure(_metrics):
                        return spatial.plt.figure()

                    def save_figure(_figure, staging):
                        paths = {
                            f"figure_{suffix}": Path(staging) / filename
                            for suffix, filename in zip(
                                ("jpg", "png", "pdf"),
                                spatial.FORMAL_FIGURE_FILENAMES,
                            )
                        }
                        for path in paths.values():
                            path.write_bytes(b"synthetic-formal-figure")
                        return paths

                    def flaky_replace(path, target):
                        nonlocal staged_publications
                        target = Path(target)
                        if (
                            path.parent.name.startswith(
                                ".spatial_feature_comparison_staging_"
                            )
                            and target.parent == output_dir
                        ):
                            staged_publications += 1
                            if staged_publications == len(names):
                                raise error_type("synthetic publication failure")
                        return original_replace(path, target)

                    try:
                        Path.replace = flaky_replace
                        spatial.create_spatial_feature_comparison_figure = create_figure
                        spatial.save_spatial_feature_comparison_figure = save_figure
                        predictions = complete_formal_prediction_fixture()
                        with self.assertRaisesRegex(
                            error_type, "synthetic publication failure"
                        ):
                            spatial.write_artifacts(
                                predictions=predictions,
                                metrics=spatial.build_metrics_table(
                                    predictions, production_run=True
                                ),
                                feature_manifest=synthetic_manifest(),
                                weight_diagnostics=synthetic_weights(),
                                interpolation_audit=synthetic_interpolation_audit(),
                                interpolation_summary=synthetic_interpolation_summary(),
                                source_audit=synthetic_source_audit(
                                    production_run=True,
                                    predictions=predictions,
                                ),
                                output_dir=output_dir,
                                production_run=True,
                            )
                    finally:
                        Path.replace = original_replace
                        if original_create is None:
                            del spatial.create_spatial_feature_comparison_figure
                        else:
                            spatial.create_spatial_feature_comparison_figure = original_create
                        if original_save is None:
                            del spatial.save_spatial_feature_comparison_figure
                        else:
                            spatial.save_spatial_feature_comparison_figure = original_save
                    self.assertEqual(staged_publications, len(names))
                    for name, content in expected.items():
                        self.assertEqual((output_dir / name).read_bytes(), content)
                    self.assertFalse(
                        any(
                            path.name.startswith(
                                ".spatial_feature_comparison_staging_"
                            )
                            for path in output_dir.iterdir()
                        )
                    )


if __name__ == "__main__":
    unittest.main()
