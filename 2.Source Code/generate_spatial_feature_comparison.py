"""Compare three evaluation models across spatial feature schemes.

Forecasting and cascading two-layer Nowcasting retain the fixed 2022 temporal
holdout. Contemporaneous uses the package's canonical seed-0 random five-fold
row-level cross-validation and saves one out-of-fold prediction for all 5,575
source rows. Metrics across the temporal and random-CV protocols are descriptive
and are not directly comparable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold

import generate_all_prediction_evaluation as evaluation
import generate_leave_one_country_out_robustness as loco
import generate_phase_cumulative_scatter_comparison as scatter
import main_result_figure1_v1 as frozen_main_result


DEFAULT_FORECASTING_INPUT = loco.DEFAULT_FORECASTING_INPUT
DEFAULT_NOWCASTING_INPUT = loco.DEFAULT_NOWCASTING_INPUT
DEFAULT_COUNTRY_LOOKUP = loco.DEFAULT_COUNTRY_LOOKUP
DEFAULT_GENERAL_PARAMS = loco.DEFAULT_GENERAL_PARAMS
DEFAULT_PHASE3_PARAMS = loco.DEFAULT_PHASE3_PARAMS
DEFAULT_OUTPUT_DIR = loco.DEFAULT_OUTPUT_DIR
DEFAULT_RANDOM_STATE: int | None = None
DEFAULT_WORKERS = 1
DEFAULT_ESTIMATOR_N_JOBS: int | None = None
DEFAULT_CUTOFF = "2022-01-01"
DEFAULT_EXPECTED_SOURCE_ROWS = 5575
DEFAULT_EXPECTED_AREAS = 1198
DEFAULT_EXPECTED_TEST_ROWS = 1170
DEFAULT_EXPECTED_TEST_AREAS = 646
DEFAULT_CONTEMPORANEOUS_PARAMS = evaluation.DEFAULT_CONTEMPORANEOUS_PARAMS_PATH
DEFAULT_CONTEMPORANEOUS_REFERENCE_PREDICTIONS = (
    evaluation.DEFAULT_CONTEMPORANEOUS_PREDICTIONS_PATH
)
DEFAULT_CONTEMPORANEOUS_REFERENCE_AUDIT = (
    DEFAULT_OUTPUT_DIR / f"{evaluation.CONTEMPORANEOUS_AUDIT_STEM}.csv"
)
CONTEMPORANEOUS_RANDOM_STATE = evaluation.DEFAULT_RANDOM_STATE
EXPECTED_CONTEMPORANEOUS_ROWS = evaluation.EXPECTED_SOURCE_ROWS
EXPECTED_CONTEMPORANEOUS_AREAS = evaluation.EXPECTED_CONTEMPORANEOUS_AREAS
EXPECTED_CONTEMPORANEOUS_FOLDS = evaluation.EXPECTED_CONTEMPORANEOUS_FOLDS
EXPECTED_CONTEMPORANEOUS_ROWS_PER_FOLD = evaluation.EXPECTED_ROWS_PER_FOLD

CONDITION_ORDER = (
    "baseline_with_lat_lon",
    "no_lat_lon",
    "knn5_spatial_means",
    "d200_spatial_means",
)
EXPERIMENT_CONDITIONS = CONDITION_ORDER[1:]
TEMPORAL_MODEL_ORDER = ("Forecasting", "Nowcasting")
CONTEMPORANEOUS_MODEL = "Contemporaneous"
MODEL_ORDER = (*TEMPORAL_MODEL_ORDER, CONTEMPORANEOUS_MODEL)
CONDITION_LABELS: Mapping[str, str] = {
    "baseline_with_lat_lon": "Baseline with latitude/longitude",
    "no_lat_lon": "No latitude/longitude",
    "knn5_spatial_means": "KNN-5 spatial means",
    "d200_spatial_means": "200 km spatial means",
}
MAIN_RESULT_REFERENCES: Mapping[str, Mapping[str, object]] = (
    frozen_main_result.spatial_references()
)
REFERENCE_METRIC_NAMES = (
    "phase3plus_precision",
    "phase3plus_recall",
    "overall_accuracy",
    "phase3plus_r2",
)
FORMAL_FIGURE_FILENAMES = tuple(
    f"precision_recall_accuracy_p3r2_spatial_feature_comparison.{suffix}"
    for suffix in ("jpg", "png", "pdf")
)

LAYER1_STATIC_FEATURES = (
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

FEATURE_MANIFEST_COLUMNS = (
    "layer",
    "feature_order",
    "original_feature",
    "feature_time_type",
    "reference_month_rule",
    "neighbor_eligible",
    "knn5_feature_name",
    "d200_feature_name",
)

EARTH_RADIUS_KM = 6371.0088

WEIGHT_DIAGNOSTIC_COLUMNS = (
    "scheme",
    "area_id",
    "country_code_3",
    "neighbor_count",
    "min_distance_km",
    "max_distance_km",
    "mean_distance_km",
    "zero_neighbor",
    "neighbor_ids",
    "neighbor_ids_sha256",
)

INTERPOLATION_AUDIT_COLUMNS = (
    "condition",
    "layer",
    "split",
    "feature",
    "feature_time_type",
    "aggregation_target_area_id",
    "imputed_neighbor_area_id",
    "target_month",
    "max_permitted_feature_month",
    "source_area_id",
    "source_country_code_3",
    "source_row_month",
    "resolved_feature_month",
    "month_gap",
    "source_tier",
    "distance_imputed_neighbor_to_source_km",
    "temporal_contract_passed",
)

_INTERPOLATION_AUDIT_FIXED_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "condition": CONDITION_ORDER,
    "layer": ("layer1_shared", "nowcasting_layer2"),
    "split": ("train", "test"),
    "feature_time_type": ("dynamic", "static"),
    "source_tier": ("global", "own_history", "same_country"),
}
_INTERPOLATION_AUDIT_DATETIME_COLUMNS = (
    "target_month",
    "max_permitted_feature_month",
    "source_row_month",
    "resolved_feature_month",
)
_INTERPOLATION_AUDIT_AREA_COLUMNS = (
    "aggregation_target_area_id",
    "imputed_neighbor_area_id",
    "source_area_id",
)

INTERPOLATION_SUMMARY_COLUMNS = (
    "condition",
    "layer",
    "split",
    "feature",
    "target_rows",
    "total_neighbor_slots",
    "observed_slots",
    "original_missing_slots",
    "imputed_slots",
    "remaining_missing_slots",
    "effective_nonmissing_slots",
    "rows_with_effective_mean",
    "rows_all_missing",
    "own_history_source_count",
    "same_country_source_count",
    "global_source_count",
    "imputation_month_gap_min",
    "imputation_month_gap_median",
    "imputation_month_gap_max",
    "imputation_distance_km_min",
    "imputation_distance_km_median",
    "imputation_distance_km_max",
)

MATRIX_HASH_COLUMNS = (
    "condition",
    "layer",
    "split",
    "feature_count",
    "matrix_sha256",
)

PREDICTION_COLUMNS = (
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
    "phase2_test",
    "phase3_test",
    "phase4_test",
    "phase5_test",
    "phase2_pred_raw",
    "phase3_pred_raw",
    "phase4_pred_raw",
    "phase5_pred_raw",
    "phase2_pred_rounded",
    "phase3_pred_rounded",
    "phase4_pred_rounded",
    "phase5_pred_rounded",
    "nonpositive_cumulative_prediction_sum",
    "phase2_layer1_pred",
    "phase2_residual_pred",
    "phase3_layer1_pred",
    "phase3_residual_pred",
    "phase4_layer1_pred",
    "phase4_residual_pred",
    "phase5_layer1_pred",
    "phase5_residual_pred",
)

METRIC_COLUMNS = (
    "condition",
    "condition_label",
    "model",
    "evaluation_protocol",
    "evaluation_population",
    "n_splits",
    "fold_assignment_sha256",
    "n_test",
    "n_test_areas",
    "n_test_countries",
    "test_key_sha256",
    "actual_phase3plus_count",
    "predicted_phase3plus_count",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "nonpositive_cumulative_prediction_count",
    "phase3plus_precision",
    "phase3plus_precision_undefined_reason",
    "baseline_phase3plus_precision",
    "phase3plus_precision_signed_delta",
    "phase3plus_precision_absolute_delta",
    "phase3plus_recall",
    "phase3plus_recall_undefined_reason",
    "baseline_phase3plus_recall",
    "phase3plus_recall_signed_delta",
    "phase3plus_recall_absolute_delta",
    "overall_accuracy",
    "baseline_overall_accuracy",
    "overall_accuracy_signed_delta",
    "overall_accuracy_absolute_delta",
    "phase3plus_r2",
    "phase3plus_r2_undefined_reason",
    "phase3plus_r2_raw",
    "phase3plus_r2_raw_undefined_reason",
    "baseline_phase3plus_r2",
    "phase3plus_r2_signed_delta",
    "phase3plus_r2_absolute_delta",
)


@dataclass(frozen=True)
class DistanceMatrix:
    area_ids: np.ndarray
    distances_km: np.ndarray
    area_to_pos: Mapping[int, int]

    def __post_init__(self) -> None:
        area_ids = np.asarray(self.area_ids)
        distances = np.asarray(self.distances_km)
        if area_ids.ndim != 1 or len(area_ids) == 0:
            raise ValueError("DistanceMatrix area_ids must be a non-empty 1D array.")
        if not np.issubdtype(area_ids.dtype, np.integer):
            raise ValueError("DistanceMatrix area_ids must be integers.")
        if len(np.unique(area_ids)) != len(area_ids):
            raise ValueError("DistanceMatrix area_ids must be unique.")
        if distances.shape != (len(area_ids), len(area_ids)):
            raise ValueError("DistanceMatrix distances must be square and align to area_ids.")
        if distances.dtype != np.dtype("float64"):
            raise ValueError("DistanceMatrix distances must use float64.")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("DistanceMatrix distances must be finite and non-negative.")
        if not np.allclose(distances, distances.T, rtol=0.0, atol=1e-10):
            raise ValueError("DistanceMatrix distances must be symmetric.")
        if not np.array_equal(np.diag(distances), np.zeros(len(area_ids))):
            raise ValueError("DistanceMatrix diagonal must be exactly zero.")
        expected_mapping = {
            int(area_id): int(position)
            for position, area_id in enumerate(area_ids.tolist())
        }
        if dict(self.area_to_pos) != expected_mapping:
            raise ValueError("DistanceMatrix area_to_pos does not match area_ids.")


@dataclass(frozen=True)
class NeighborIndex:
    scheme: str
    area_ids: np.ndarray
    neighbor_positions: tuple[np.ndarray, ...]
    neighbor_distances_km: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        area_ids = np.asarray(self.area_ids)
        count = len(area_ids)
        if self.scheme not in {"knn5", "d200"}:
            raise ValueError(f"Unknown neighbor scheme: {self.scheme}")
        if len(self.neighbor_positions) != count or len(
            self.neighbor_distances_km
        ) != count:
            raise ValueError("NeighborIndex rows must align with area_ids.")
        for target_position, (positions, distances) in enumerate(
            zip(self.neighbor_positions, self.neighbor_distances_km)
        ):
            positions = np.asarray(positions)
            distances = np.asarray(distances)
            if not np.issubdtype(positions.dtype, np.integer):
                raise ValueError("Neighbor positions must use an integer dtype.")
            if distances.dtype != np.dtype("float64"):
                raise ValueError("Neighbor distances must use float64.")
            if len(positions) != len(distances):
                raise ValueError("Neighbor positions and distances must have equal length.")
            if len(np.unique(positions)) != len(positions):
                raise ValueError("Neighbor positions must be unique within a row.")
            if ((positions < 0) | (positions >= count)).any():
                raise ValueError("Neighbor positions are out of range.")
            if target_position in set(positions.tolist()):
                raise ValueError("Neighbor rows must exclude the target area itself.")
            if not np.isfinite(distances).all() or (distances < 0).any():
                raise ValueError("Neighbor distances must be finite and non-negative.")
            if len(positions) > 1:
                ordered = np.lexsort((area_ids[positions], distances))
                if not np.array_equal(ordered, np.arange(len(positions))):
                    raise ValueError("Neighbors must be ordered by distance then area_id.")


@dataclass(frozen=True)
class ObservedFeatureIndex:
    feature: str
    feature_time_type: str
    layer: str
    area_ids: np.ndarray
    dates: np.ndarray
    resolved_feature_months: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    area_to_pos: Mapping[int, int]
    date_to_pos: Mapping[pd.Timestamp, int]
    country_by_pos: np.ndarray
    positions_by_country: Mapping[str, np.ndarray]
    observed_date_positions_by_area: tuple[np.ndarray, ...]
    observed_area_positions_by_date: tuple[np.ndarray, ...]
    resolution_cache: dict[tuple[object, ...], "InterpolationResult"] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    candidate_cache: dict[
        tuple[object, ...], tuple[tuple[int, int], ...]
    ] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class InterpolationResult:
    value: float
    audit_record: dict[str, object] | None


@dataclass(frozen=True)
class SpatialAugmentationResult:
    data: pd.DataFrame
    interpolation_audit: pd.DataFrame
    interpolation_summary: pd.DataFrame
    matrix_sha256: str


@dataclass(frozen=True)
class ConditionMatrices:
    forecasting: Mapping[str, pd.DataFrame]
    nowcasting: Mapping[str, pd.DataFrame]
    layer1_features: Mapping[str, tuple[str, ...]]
    layer2_features: Mapping[str, tuple[str, ...]]
    interpolation_audit: pd.DataFrame
    interpolation_summary: pd.DataFrame
    matrix_hashes: pd.DataFrame


@dataclass(frozen=True)
class ContemporaneousFoldContract:
    fold_table: pd.DataFrame
    reference_predictions: pd.DataFrame
    reference_audit: pd.DataFrame
    fold_assignment_sha256: str
    source_row_index_sha256: str
    population_key_sha256: str


def _normalize_month(value: object, label: str) -> pd.Timestamp:
    try:
        month = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a valid month.") from error
    if pd.isna(month):
        raise ValueError(f"{label} is missing.")
    if month.tz is not None:
        raise ValueError(f"{label} must be timezone-naive.")
    if month != month.normalize():
        raise ValueError(f"{label} must contain a midnight value.")
    return month


def _calendar_month_gap(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def resolve_feature_month(
    row_month: pd.Timestamp,
    feature_time_type: str,
    layer: str,
) -> pd.Timestamp | pd.NaT:
    """Resolve the information month represented by one stored feature value."""
    row_month = _normalize_month(row_month, "row_month")
    if feature_time_type not in {"dynamic", "static"}:
        raise ValueError(f"Unknown feature_time_type: {feature_time_type}")
    if layer not in {"layer1_shared", "nowcasting_layer2"}:
        raise ValueError(f"Unknown layer: {layer}")
    if feature_time_type == "static":
        return pd.NaT
    if layer == "layer1_shared":
        return row_month - pd.DateOffset(months=12)
    return row_month


def build_observed_feature_index(
    observations: pd.DataFrame,
    feature: str,
    feature_time_type: str,
    layer: str,
    distance_matrix: DistanceMatrix,
    area_country: pd.DataFrame,
) -> ObservedFeatureIndex:
    """Index one original feature without ever materializing imputed sources."""
    if feature.endswith("__knn5_mean") or feature.endswith("__d200_mean"):
        raise ValueError("generated spatial features cannot be observed sources.")
    if feature_time_type not in {"dynamic", "static"}:
        raise ValueError(f"Unknown feature_time_type: {feature_time_type}")
    if layer not in {"layer1_shared", "nowcasting_layer2"}:
        raise ValueError(f"Unknown layer: {layer}")
    if feature_time_type == "static" and layer != "layer1_shared":
        raise ValueError("Static features are only permitted in layer1_shared.")
    loco._require_columns(observations, ["area_id", "date", feature])
    data = observations.loc[:, ["area_id", "date", feature]].copy()
    if data[["area_id", "date"]].isna().any().any():
        raise ValueError("Observed feature keys contain missing values.")
    try:
        parsed_dates = pd.to_datetime(data["date"], errors="raise", format="mixed")
    except (TypeError, ValueError) as error:
        raise ValueError("Observed feature dates contain invalid values.") from error
    if isinstance(parsed_dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("Observed feature dates must be timezone-naive.")
    if not parsed_dates.eq(parsed_dates.dt.normalize()).all():
        raise ValueError("Observed feature dates must contain midnight values.")
    data["date"] = parsed_dates
    try:
        numeric_area_ids = pd.to_numeric(data["area_id"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Observed feature area_id values must be numeric.") from error
    if not np.equal(numeric_area_ids, np.floor(numeric_area_ids)).all():
        raise ValueError("Observed feature area_id values must be integer-valued.")
    data["area_id"] = numeric_area_ids.astype("int64")
    if data.duplicated(["area_id", "date"]).any():
        raise ValueError("Observed feature data contain duplicate area-date keys.")
    if not pd.api.types.is_numeric_dtype(data[feature]):
        raise ValueError(f"Observed feature {feature} must be numeric.")
    feature_values = data[feature].to_numpy(dtype=np.float64, na_value=np.nan)
    nonmissing = ~pd.isna(data[feature]).to_numpy()
    if not np.isfinite(feature_values[nonmissing]).all():
        raise ValueError(f"Observed feature {feature} contains non-finite values.")

    lookup = loco.normalize_country_lookup(area_country)
    expected_area_set = set(int(value) for value in distance_matrix.area_ids.tolist())
    if set(int(value) for value in lookup["area_id"].tolist()) != expected_area_set:
        raise ValueError("Country lookup area set does not match distance areas.")
    unknown_observation_areas = sorted(
        set(int(value) for value in data["area_id"].tolist()).difference(
            expected_area_set
        )
    )
    if unknown_observation_areas:
        raise ValueError(
            f"Observed feature contains unknown area_id values: {unknown_observation_areas}"
        )
    if feature_time_type == "static":
        inconsistent = [
            int(area_id)
            for area_id, values in data.groupby("area_id", sort=False)[feature]
            if values.dropna().nunique() > 1
        ]
        if inconsistent:
            raise ValueError(
                f"Observed static feature {feature} varies within areas: {inconsistent}"
            )

    area_ids = distance_matrix.area_ids.astype(np.int64, copy=True)
    area_to_pos = {
        int(area_id): int(position)
        for position, area_id in enumerate(area_ids.tolist())
    }
    dates_index = pd.DatetimeIndex(data["date"].drop_duplicates()).sort_values()
    dates = dates_index.to_numpy().astype("datetime64[us]", copy=False)
    date_to_pos = {
        pd.Timestamp(date): int(position)
        for position, date in enumerate(dates.tolist())
    }
    values = np.full((len(area_ids), len(dates)), np.nan, dtype=np.float64)
    row_area_positions = data["area_id"].map(area_to_pos).to_numpy(dtype=np.int64)
    row_date_positions = data["date"].map(date_to_pos).to_numpy(dtype=np.int64)
    values[row_area_positions, row_date_positions] = feature_values
    observed = ~np.isnan(values)

    if feature_time_type == "static":
        resolved_feature_months = np.full(
            len(dates), np.datetime64("NaT", "us"), dtype="datetime64[us]"
        )
    else:
        resolved_feature_months = np.asarray(
            [
                resolve_feature_month(pd.Timestamp(date), feature_time_type, layer)
                .to_datetime64()
                .astype("datetime64[us]")
                for date in dates
            ],
            dtype="datetime64[us]",
        )

    country_lookup = lookup.set_index("area_id")["country_code_3"]
    country_by_pos = np.asarray(
        [country_lookup.loc[int(area_id)] for area_id in area_ids], dtype=object
    )
    positions_by_country = {
        str(country): np.flatnonzero(country_by_pos == country).astype(np.int64)
        for country in sorted(set(country_by_pos.tolist()))
    }
    observed_date_positions_by_area = tuple(
        np.flatnonzero(observed[position])[::-1].astype(np.int64)
        for position in range(len(area_ids))
    )
    observed_area_positions_by_date = tuple(
        np.flatnonzero(observed[:, position]).astype(np.int64)
        for position in range(len(dates))
    )

    for array in (
        area_ids,
        dates,
        resolved_feature_months,
        values,
        observed,
        country_by_pos,
        *positions_by_country.values(),
        *observed_date_positions_by_area,
        *observed_area_positions_by_date,
    ):
        array.setflags(write=False)
    return ObservedFeatureIndex(
        feature=feature,
        feature_time_type=feature_time_type,
        layer=layer,
        area_ids=area_ids,
        dates=dates,
        resolved_feature_months=resolved_feature_months,
        values=values,
        observed=observed,
        area_to_pos=area_to_pos,
        date_to_pos=date_to_pos,
        country_by_pos=country_by_pos,
        positions_by_country=positions_by_country,
        observed_date_positions_by_area=observed_date_positions_by_area,
        observed_area_positions_by_date=observed_area_positions_by_date,
    )


def _rank_spatial_sources(
    source_index: ObservedFeatureIndex,
    distance_matrix: DistanceMatrix,
    imputed_neighbor_position: int,
    max_permitted_feature_month: pd.Timestamp | pd.NaT,
    required_country: str | None,
) -> tuple[tuple[int, int], ...]:
    cache_key = (
        "spatial",
        imputed_neighbor_position,
        None
        if pd.isna(max_permitted_feature_month)
        else pd.Timestamp(max_permitted_feature_month),
        required_country,
    )
    cached = source_index.candidate_cache.get(cache_key)
    if cached is not None:
        return cached
    ranked: list[tuple[int, int]] = []
    if source_index.feature_time_type == "static":
        candidates = np.flatnonzero(source_index.observed.any(axis=1)).astype(np.int64)
        candidates = np.asarray(
            [
                position
                for position in candidates.tolist()
                if position != imputed_neighbor_position
                and (
                    required_country is None
                    or source_index.country_by_pos[position] == required_country
                )
            ],
            dtype=np.int64,
        )
        if not len(candidates):
            result: tuple[tuple[int, int], ...] = ()
        else:
            distances = distance_matrix.distances_km[
                imputed_neighbor_position, candidates
            ]
            order = np.lexsort((source_index.area_ids[candidates], distances))
            for source_position in candidates[order[:2]].tolist():
                source_date_position = int(
                    source_index.observed_date_positions_by_area[
                        source_position
                    ][0]
                )
                ranked.append((int(source_position), source_date_position))
            result = tuple(ranked)
        source_index.candidate_cache[cache_key] = result
        return result

    max_month_value = np.datetime64(
        pd.Timestamp(max_permitted_feature_month).to_datetime64(), "us"
    )
    selected_areas: set[int] = set()
    for date_position in range(len(source_index.dates) - 1, -1, -1):
        if source_index.resolved_feature_months[date_position] > max_month_value:
            continue
        candidates = source_index.observed_area_positions_by_date[date_position]
        candidates = np.asarray(
            [
                position
                for position in candidates.tolist()
                if position != imputed_neighbor_position
                and position not in selected_areas
                and (
                    required_country is None
                    or source_index.country_by_pos[position] == required_country
                )
            ],
            dtype=np.int64,
        )
        if not len(candidates):
            continue
        distances = distance_matrix.distances_km[
            imputed_neighbor_position, candidates
        ]
        order = np.lexsort((source_index.area_ids[candidates], distances))
        for source_position in candidates[order].tolist():
            source_position = int(source_position)
            ranked.append((source_position, int(date_position)))
            selected_areas.add(source_position)
            if len(ranked) == 2:
                result = tuple(ranked)
                source_index.candidate_cache[cache_key] = result
                return result
    result = tuple(ranked)
    source_index.candidate_cache[cache_key] = result
    return result


def _select_spatial_source(
    source_index: ObservedFeatureIndex,
    distance_matrix: DistanceMatrix,
    aggregation_target_position: int,
    imputed_neighbor_position: int,
    max_permitted_feature_month: pd.Timestamp | pd.NaT,
    required_country: str | None,
) -> tuple[int, int] | None:
    for source_position, source_date_position in _rank_spatial_sources(
        source_index,
        distance_matrix,
        imputed_neighbor_position,
        max_permitted_feature_month,
        required_country,
    ):
        if source_position != aggregation_target_position:
            return source_position, source_date_position
    return None


def _select_own_source(
    source_index: ObservedFeatureIndex,
    imputed_neighbor_position: int,
    max_permitted_feature_month: pd.Timestamp | pd.NaT,
) -> tuple[int, int] | None:
    cache_key = (
        "own",
        imputed_neighbor_position,
        None
        if pd.isna(max_permitted_feature_month)
        else pd.Timestamp(max_permitted_feature_month),
    )
    cached = source_index.candidate_cache.get(cache_key)
    if cached is not None:
        return cached[0] if cached else None
    choice = None
    for source_date_position in source_index.observed_date_positions_by_area[
        imputed_neighbor_position
    ].tolist():
        if source_index.feature_time_type == "static" or (
            source_index.resolved_feature_months[source_date_position]
            <= np.datetime64(
                pd.Timestamp(max_permitted_feature_month).to_datetime64(), "us"
            )
        ):
            choice = (imputed_neighbor_position, int(source_date_position))
            break
    source_index.candidate_cache[cache_key] = () if choice is None else (choice,)
    return choice


def resolve_neighbor_slot(
    aggregation_target_area_id: int,
    imputed_neighbor_area_id: int,
    target_month: pd.Timestamp,
    feature: str,
    feature_time_type: str,
    layer: str,
    source_index: ObservedFeatureIndex,
    distance_matrix: DistanceMatrix,
    area_country: Mapping[int, str],
    condition: str,
    split: str,
) -> InterpolationResult:
    """Resolve one missing neighbor slot from immutable observed values only."""
    if (
        source_index.feature != feature
        or source_index.feature_time_type != feature_time_type
        or source_index.layer != layer
    ):
        raise ValueError("Requested feature metadata do not match the observed index.")
    target_month = _normalize_month(target_month, "target_month")
    aggregation_target_area_id = int(aggregation_target_area_id)
    imputed_neighbor_area_id = int(imputed_neighbor_area_id)
    if aggregation_target_area_id == imputed_neighbor_area_id:
        raise ValueError("Aggregation target and imputed neighbor must differ.")
    try:
        aggregation_target_position = source_index.area_to_pos[
            aggregation_target_area_id
        ]
        imputed_neighbor_position = source_index.area_to_pos[
            imputed_neighbor_area_id
        ]
    except KeyError as error:
        raise ValueError("Aggregation target or imputed neighbor is unknown.") from error
    if (
        distance_matrix.area_to_pos.get(aggregation_target_area_id)
        != aggregation_target_position
        or distance_matrix.area_to_pos.get(imputed_neighbor_area_id)
        != imputed_neighbor_position
    ):
        raise ValueError("Observed index and distance matrix area order differ.")
    if aggregation_target_area_id not in area_country or imputed_neighbor_area_id not in area_country:
        raise ValueError("area_country is missing the aggregation target or imputed neighbor.")
    if (
        str(area_country[imputed_neighbor_area_id])
        != str(source_index.country_by_pos[imputed_neighbor_position])
    ):
        raise ValueError("area_country disagrees with the observed index.")

    target_date_position = source_index.date_to_pos.get(target_month)
    if (
        target_date_position is not None
        and source_index.observed[
            imputed_neighbor_position, target_date_position
        ]
    ):
        raise ValueError("Imputed neighbor already has an observed target-month value.")

    cache_key = (
        condition,
        split,
        feature,
        layer,
        target_month,
        aggregation_target_area_id,
        imputed_neighbor_area_id,
    )
    cached = source_index.resolution_cache.get(cache_key)
    if cached is not None:
        return cached

    if feature_time_type == "static":
        max_permitted_feature_month = pd.NaT
    else:
        max_permitted_feature_month = resolve_feature_month(
            target_month, "dynamic", layer
        )

    source_choice = None
    own_source = _select_own_source(
        source_index,
        imputed_neighbor_position,
        max_permitted_feature_month,
    )
    if own_source is not None:
        source_choice = (*own_source, "own_history")

    if source_choice is None:
        imputed_country = str(source_index.country_by_pos[imputed_neighbor_position])
        same_country = _select_spatial_source(
            source_index,
            distance_matrix,
            aggregation_target_position,
            imputed_neighbor_position,
            max_permitted_feature_month,
            imputed_country,
        )
        if same_country is not None:
            source_choice = (*same_country, "same_country")
    if source_choice is None:
        global_source = _select_spatial_source(
            source_index,
            distance_matrix,
            aggregation_target_position,
            imputed_neighbor_position,
            max_permitted_feature_month,
            None,
        )
        if global_source is not None:
            source_choice = (*global_source, "global")
    if source_choice is None:
        result = InterpolationResult(value=np.nan, audit_record=None)
        source_index.resolution_cache[cache_key] = result
        return result

    source_position, source_date_position, source_tier = source_choice
    source_area_id = int(source_index.area_ids[source_position])
    source_row_month = pd.Timestamp(source_index.dates[source_date_position])
    value = float(source_index.values[source_position, source_date_position])
    if not np.isfinite(value):
        raise ValueError("Resolved source value is not finite and observed.")
    distance_km = float(
        distance_matrix.distances_km[imputed_neighbor_position, source_position]
    )
    if feature_time_type == "static":
        resolved_feature_month = pd.NaT
        month_gap = pd.NA
        temporal_contract_passed = True
        max_month_value: object = pd.NA
        resolved_month_value: object = pd.NA
    else:
        resolved_feature_month = resolve_feature_month(
            source_row_month, feature_time_type, layer
        )
        month_gap = _calendar_month_gap(target_month, resolved_feature_month)
        if layer == "layer1_shared":
            temporal_contract_passed = bool(
                resolved_feature_month <= max_permitted_feature_month
                and month_gap >= 12
            )
        else:
            temporal_contract_passed = bool(
                resolved_feature_month <= max_permitted_feature_month
            )
        if not temporal_contract_passed:
            raise ValueError("Resolved source violates the temporal contract.")
        max_month_value = max_permitted_feature_month.strftime("%Y-%m-%d")
        resolved_month_value = resolved_feature_month.strftime("%Y-%m-%d")

    audit_record = {
        "condition": condition,
        "layer": layer,
        "split": split,
        "feature": feature,
        "feature_time_type": feature_time_type,
        "aggregation_target_area_id": aggregation_target_area_id,
        "imputed_neighbor_area_id": imputed_neighbor_area_id,
        "target_month": target_month.strftime("%Y-%m-%d"),
        "max_permitted_feature_month": max_month_value,
        "source_area_id": source_area_id,
        "source_country_code_3": str(source_index.country_by_pos[source_position]),
        "source_row_month": source_row_month.strftime("%Y-%m-%d"),
        "resolved_feature_month": resolved_month_value,
        "month_gap": month_gap,
        "source_tier": source_tier,
        "distance_imputed_neighbor_to_source_km": distance_km,
        "temporal_contract_passed": temporal_contract_passed,
    }
    if tuple(audit_record) != INTERPOLATION_AUDIT_COLUMNS:
        raise ValueError("Interpolation audit record has an unexpected schema.")
    result = InterpolationResult(value=value, audit_record=audit_record)
    source_index.resolution_cache[cache_key] = result
    return result


def mean_available_slots(values: Sequence[float]) -> tuple[float, int]:
    """Return the arithmetic mean and count of finite neighbor-slot values."""
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    count = int(finite.sum())
    if count == 0:
        return np.nan, 0
    return float(array[finite].mean()), count


def build_condition_feature_lists(
    layer1_features: Sequence[str],
    layer2_features: Sequence[str] = loco.NOWCAST_FEATURES,
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]]]:
    """Build the exact ordered Layer 1 and Layer 2 columns for each condition."""
    layer1 = tuple(layer1_features)
    layer2 = tuple(layer2_features)
    build_feature_manifest(layer1, layer2)
    ablated = tuple(feature for feature in layer1 if feature not in {"lat", "lon"})
    layer1_by_condition = {
        "baseline_with_lat_lon": layer1,
        "no_lat_lon": ablated,
        "knn5_spatial_means": ablated
        + tuple(f"{feature}__knn5_mean" for feature in ablated),
        "d200_spatial_means": ablated
        + tuple(f"{feature}__d200_mean" for feature in ablated),
    }
    layer2_by_condition = {
        "baseline_with_lat_lon": layer2,
        "no_lat_lon": layer2,
        "knn5_spatial_means": layer2
        + tuple(f"{feature}__knn5_mean" for feature in layer2),
        "d200_spatial_means": layer2
        + tuple(f"{feature}__d200_mean" for feature in layer2),
    }
    expected_layer1_counts = {
        "baseline_with_lat_lon": 106,
        "no_lat_lon": 104,
        "knn5_spatial_means": 208,
        "d200_spatial_means": 208,
    }
    expected_layer2_counts = {
        "baseline_with_lat_lon": 69,
        "no_lat_lon": 69,
        "knn5_spatial_means": 138,
        "d200_spatial_means": 138,
    }
    if {
        condition: len(features)
        for condition, features in layer1_by_condition.items()
    } != expected_layer1_counts:
        raise ValueError("Condition Layer 1 feature counts are incorrect.")
    if {
        condition: len(features)
        for condition, features in layer2_by_condition.items()
    } != expected_layer2_counts:
        raise ValueError("Condition Layer 2 feature counts are incorrect.")
    return layer1_by_condition, layer2_by_condition


def build_contemporaneous_feature_lists(
    nowcasting_source_path: Path,
    layer1_features: Sequence[str],
    layer2_features: Sequence[str] = loco.NOWCAST_FEATURES,
) -> Mapping[str, tuple[str, ...]]:
    """Return the canonical random-CV predictors under every feature condition."""
    source_columns = pd.read_csv(nowcasting_source_path, nrows=0).columns.tolist()
    excluded = {
        "area_id",
        "date",
        "overall_phase",
        *loco.PHASE_SHARE_COLUMNS,
        *loco.CUMULATIVE_TARGETS.values(),
    }
    base_features = tuple(column for column in source_columns if column not in excluded)
    if len(base_features) != 173 or "kfolds" in base_features:
        raise ValueError(
            "Expected 173 source predictors before adding the canonical kfolds feature."
        )

    layer1 = tuple(layer1_features)
    layer2 = tuple(layer2_features)
    if set(layer1).intersection(layer2):
        raise ValueError("Contemporaneous Layer 1 and Layer 2 feature sets overlap.")
    available = set(layer1).union(layer2)
    missing = sorted(set(base_features).difference(available))
    if missing:
        raise ValueError(
            f"Canonical contemporaneous predictors are missing from model layers: {missing}"
        )
    unavailable_layer1 = sorted(set(layer1).difference(base_features))
    if unavailable_layer1 != ["infra_index_m12_l12", "infra_index_s12_l12"]:
        raise ValueError(
            "Unexpected Forecasting-only predictors in the contemporaneous contract: "
            f"{unavailable_layer1}"
        )

    active_original = {
        "baseline_with_lat_lon": base_features,
        "no_lat_lon": tuple(
            feature for feature in base_features if feature not in {"lat", "lon"}
        ),
    }
    active_original["knn5_spatial_means"] = active_original["no_lat_lon"]
    active_original["d200_spatial_means"] = active_original["no_lat_lon"]
    by_condition: dict[str, tuple[str, ...]] = {}
    for condition in CONDITION_ORDER:
        originals = active_original[condition]
        if condition == "knn5_spatial_means":
            generated = tuple(f"{feature}__knn5_mean" for feature in originals)
        elif condition == "d200_spatial_means":
            generated = tuple(f"{feature}__d200_mean" for feature in originals)
        else:
            generated = ()
        by_condition[condition] = (*originals, *generated, "kfolds")

    expected_counts = {
        "baseline_with_lat_lon": 174,
        "no_lat_lon": 172,
        "knn5_spatial_means": 343,
        "d200_spatial_means": 343,
    }
    observed_counts = {
        condition: len(features) for condition, features in by_condition.items()
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Contemporaneous feature counts are incorrect: {observed_counts}"
        )
    return by_condition


def load_contemporaneous_fold_contract(
    source_path: Path,
    reference_predictions_path: Path,
    reference_audit_path: Path,
) -> ContemporaneousFoldContract:
    """Rebuild and verify the canonical seed-0 random five-fold assignment."""
    source_path = Path(source_path)
    reference_predictions_path = Path(reference_predictions_path)
    reference_audit_path = Path(reference_audit_path)
    source = pd.read_csv(source_path)
    if len(source) != EXPECTED_CONTEMPORANEOUS_ROWS:
        raise ValueError("Contemporaneous source row count differs from 5,575.")
    if "source_row_index" in source:
        raise ValueError("Contemporaneous source unexpectedly contains source_row_index.")
    source.insert(0, "source_row_index", np.arange(len(source), dtype=int))
    loco._require_columns(source, ["area_id", "date", "phase1_percent"])
    source = source.loc[source["phase1_percent"].notna()].copy()
    if len(source) != EXPECTED_CONTEMPORANEOUS_ROWS:
        raise ValueError("phase1_percent filtering changed the random-CV population.")
    source["date"] = pd.to_datetime(source["date"], errors="raise", format="mixed")
    source = source.sort_values(["area_id", "date"], kind="mergesort").reset_index(
        drop=True
    )
    source["fold"] = -1
    shuffled = source.sample(
        frac=1,
        random_state=CONTEMPORANEOUS_RANDOM_STATE,
    ).reset_index(drop=True)
    splitter = KFold(n_splits=EXPECTED_CONTEMPORANEOUS_FOLDS, shuffle=False)
    for fold, (_, validation_index) in enumerate(splitter.split(shuffled)):
        shuffled.loc[validation_index, "fold"] = fold
    shuffled["shuffle_position"] = np.arange(len(shuffled), dtype=int)
    fold_table = shuffled.loc[
        :, ["source_row_index", "area_id", "date", "fold", "shuffle_position"]
    ].sort_values("source_row_index", kind="mergesort").reset_index(drop=True)
    fold_counts = fold_table["fold"].value_counts().sort_index().to_dict()
    expected_fold_counts = {
        fold: EXPECTED_CONTEMPORANEOUS_ROWS_PER_FOLD
        for fold in range(EXPECTED_CONTEMPORANEOUS_FOLDS)
    }
    if fold_counts != expected_fold_counts:
        raise ValueError(f"Unexpected contemporaneous fold sizes: {fold_counts}")

    reference = pd.read_csv(reference_predictions_path)
    loco._require_columns(
        reference,
        [
            "source_row_index",
            "area_id",
            "date",
            "fold",
            "evaluation_protocol",
            "evaluation_population",
            "shuffle_seed",
        ],
    )
    reference["date"] = pd.to_datetime(
        reference["date"], errors="raise", format="mixed"
    )
    if len(reference) != EXPECTED_CONTEMPORANEOUS_ROWS:
        raise ValueError("Canonical contemporaneous reference does not contain 5,575 rows.")
    if not reference["evaluation_protocol"].eq("random_5fold_row_cv").all():
        raise ValueError("Canonical contemporaneous reference protocol drifted.")
    if not reference["evaluation_population"].eq(
        "random_5fold_full_oof_5575"
    ).all():
        raise ValueError("Canonical contemporaneous reference population drifted.")
    if not pd.to_numeric(reference["shuffle_seed"], errors="raise").eq(
        CONTEMPORANEOUS_RANDOM_STATE
    ).all():
        raise ValueError("Canonical contemporaneous reference seed drifted.")
    reference_fold_table = reference.loc[
        :, ["source_row_index", "area_id", "date", "fold"]
    ].sort_values("source_row_index", kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fold_table.drop(columns="shuffle_position"),
        reference_fold_table,
        check_dtype=False,
        check_exact=True,
    )

    audit = pd.read_csv(reference_audit_path)
    if len(audit) != 1:
        raise ValueError("Canonical contemporaneous reference audit must contain one row.")
    audit_row = audit.iloc[0]
    required_audit_values = {
        "evaluation_protocol": "random_5fold_row_cv",
        "evaluation_population": "random_5fold_full_oof_5575",
        "source_rows": EXPECTED_CONTEMPORANEOUS_ROWS,
        "oof_rows": EXPECTED_CONTEMPORANEOUS_ROWS,
        "test_areas": EXPECTED_CONTEMPORANEOUS_AREAS,
        "n_splits": EXPECTED_CONTEMPORANEOUS_FOLDS,
        "shuffle_seed": CONTEMPORANEOUS_RANDOM_STATE,
    }
    for column, expected in required_audit_values.items():
        if column not in audit or audit_row[column] != expected:
            raise ValueError(
                f"Canonical contemporaneous reference audit drifted for {column}."
            )
    if audit_row["predictions_sha256"] != sha256_file(reference_predictions_path):
        raise ValueError("Canonical contemporaneous prediction hash does not match its audit.")
    if audit_row["source_sha256"] != sha256_file(source_path):
        raise ValueError("Canonical contemporaneous source hash does not match its audit.")

    fold_hash = evaluation._fold_assignment_sha256(reference)
    source_index_hash = evaluation._source_row_index_sha256(reference)
    population_hash = evaluation._population_key_sha256(reference)
    for column, observed in (
        ("fold_assignment_sha256", fold_hash),
        ("source_row_index_sha256", source_index_hash),
        ("population_key_sha256", population_hash),
    ):
        if audit_row[column] != observed:
            raise ValueError(f"Canonical contemporaneous {column} does not match its audit.")
    return ContemporaneousFoldContract(
        fold_table=fold_table,
        reference_predictions=reference,
        reference_audit=audit,
        fold_assignment_sha256=fold_hash,
        source_row_index_sha256=source_index_hash,
        population_key_sha256=population_hash,
    )


def _summary_stat(values: Sequence[float], statistic: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan
    if statistic == "min":
        return float(np.min(array))
    if statistic == "median":
        return float(np.median(array))
    if statistic == "max":
        return float(np.max(array))
    raise ValueError(f"Unknown summary statistic: {statistic}")


def _compact_interpolation_audit(
    data: pd.DataFrame,
    *,
    feature_categories: Sequence[str],
    country_categories: Sequence[str],
) -> pd.DataFrame:
    """Store the event-only audit in compact, concat-stable column dtypes."""
    if data.columns.tolist() != list(INTERPOLATION_AUDIT_COLUMNS):
        raise ValueError("Interpolation audit frame has an unexpected schema.")
    result = data.loc[:, INTERPOLATION_AUDIT_COLUMNS].copy(deep=False)
    category_values = {
        **_INTERPOLATION_AUDIT_FIXED_CATEGORIES,
        "feature": tuple(sorted({str(value) for value in feature_categories})),
        "source_country_code_3": tuple(
            sorted({str(value) for value in country_categories})
        ),
    }
    for column, categories in category_values.items():
        observed = {
            str(value) for value in result[column].dropna().unique().tolist()
        }
        unknown = observed.difference(categories)
        if unknown:
            raise ValueError(
                f"Interpolation audit {column} contains unknown values: "
                f"{sorted(unknown)}"
            )
        result[column] = pd.Categorical(
            result[column], categories=categories, ordered=True
        )

    for column in _INTERPOLATION_AUDIT_DATETIME_COLUMNS:
        result[column] = pd.to_datetime(result[column], errors="raise")

    int32_info = np.iinfo(np.int32)
    for column in _INTERPOLATION_AUDIT_AREA_COLUMNS:
        numeric = pd.to_numeric(result[column], errors="raise")
        if numeric.isna().any():
            raise ValueError(f"Interpolation audit {column} contains missing values.")
        values = numeric.to_numpy(dtype=np.float64)
        if len(values) and (
            not np.equal(values, np.floor(values)).all()
            or values.min() < int32_info.min
            or values.max() > int32_info.max
        ):
            raise ValueError(f"Interpolation audit {column} is not int32-safe.")
        result[column] = numeric.astype(np.int32)

    month_gap = pd.to_numeric(result["month_gap"], errors="raise")
    observed_gaps = month_gap.dropna().to_numpy(dtype=np.float64)
    if len(observed_gaps) and not np.equal(
        observed_gaps, np.floor(observed_gaps)
    ).all():
        raise ValueError("Interpolation audit month_gap must be integer-valued.")
    result["month_gap"] = month_gap.astype("Int32")

    distance = pd.to_numeric(
        result["distance_imputed_neighbor_to_source_km"], errors="raise"
    ).astype(np.float64)
    if len(distance) and (not np.isfinite(distance).all() or (distance < 0).any()):
        raise ValueError("Interpolation audit distances must be finite and non-negative.")
    result["distance_imputed_neighbor_to_source_km"] = distance

    temporal = result["temporal_contract_passed"]
    if temporal.isna().any() or not temporal.isin([True, False]).all():
        raise ValueError("Interpolation audit temporal flags must be Boolean.")
    result["temporal_contract_passed"] = temporal.astype(bool)
    return result.reset_index(drop=True)


def _audit_category_values(data: pd.DataFrame, column: str) -> tuple[str, ...]:
    if isinstance(data[column].dtype, pd.CategoricalDtype):
        return tuple(str(value) for value in data[column].cat.categories)
    return tuple(str(value) for value in data[column].dropna().unique().tolist())


def _concat_interpolation_audit_blocks(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate pre-sorted audit blocks without one giant object sort."""
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return _compact_interpolation_audit(
            pd.DataFrame(columns=INTERPOLATION_AUDIT_COLUMNS),
            feature_categories=(),
            country_categories=(),
        )
    feature_categories = tuple(
        sorted(
            {
                value
                for frame in nonempty
                for value in _audit_category_values(frame, "feature")
            }
        )
    )
    country_categories = tuple(
        sorted(
            {
                value
                for frame in nonempty
                for value in _audit_category_values(
                    frame, "source_country_code_3"
                )
            }
        )
    )
    compact = [
        _compact_interpolation_audit(
            frame,
            feature_categories=feature_categories,
            country_categories=country_categories,
        )
        for frame in nonempty
    ]
    condition_rank = {value: rank for rank, value in enumerate(CONDITION_ORDER)}
    layer_rank = {"layer1_shared": 0, "nowcasting_layer2": 1}
    split_rank = {"train": 0, "test": 1}

    def block_key(frame: pd.DataFrame) -> tuple[int, int, int, str]:
        labels = []
        for column in ("condition", "layer", "split"):
            values = tuple(str(value) for value in frame[column].dropna().unique())
            if len(values) != 1:
                raise ValueError(
                    "Each interpolation audit block must contain one "
                    f"{column} value."
                )
            labels.append(values[0])
        features = tuple(str(value) for value in frame["feature"].dropna().unique())
        if not features:
            raise ValueError("Interpolation audit block has no feature values.")
        return (
            condition_rank[labels[0]],
            layer_rank[labels[1]],
            split_rank[labels[2]],
            min(features),
        )

    compact.sort(key=block_key)
    result = pd.concat(compact, ignore_index=True)
    if result.columns.tolist() != list(INTERPOLATION_AUDIT_COLUMNS):
        raise ValueError("Combined interpolation audit has an unexpected schema.")
    return result


def augment_spatial_features(
    target_rows: pd.DataFrame,
    source_observations: pd.DataFrame,
    original_features: Sequence[str],
    feature_time_types: Mapping[str, str],
    neighbors: NeighborIndex,
    distance_matrix: DistanceMatrix,
    area_country: pd.DataFrame,
    scheme: str,
    layer: str,
    split: str,
) -> SpatialAugmentationResult:
    """Append leakage-safe neighbor means while preserving every original cell."""
    scheme_details = {
        "knn5_spatial_means": ("knn5", "knn5"),
        "d200_spatial_means": ("d200", "d200"),
    }
    if scheme not in scheme_details:
        raise ValueError(f"Unknown spatial scheme: {scheme}")
    neighbor_scheme, feature_suffix = scheme_details[scheme]
    if neighbors.scheme != neighbor_scheme:
        raise ValueError("Neighbor index scheme does not match the requested condition.")
    if layer not in {"layer1_shared", "nowcasting_layer2"}:
        raise ValueError(f"Unknown layer: {layer}")
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    features = tuple(original_features)
    if not features or len(set(features)) != len(features):
        raise ValueError("original_features must be non-empty and unique.")
    if set(feature_time_types) != set(features):
        raise ValueError("feature_time_types must match original_features exactly.")
    loco._require_columns(target_rows, [*loco.KEY_COLUMNS, *features])
    loco._require_columns(source_observations, [*loco.KEY_COLUMNS, *features])
    canonical_key_sha256(target_rows)
    canonical_key_sha256(source_observations)
    if not target_rows.index.is_unique:
        raise ValueError("target_rows index must be unique.")
    if not np.array_equal(neighbors.area_ids, distance_matrix.area_ids):
        raise ValueError("Neighbor and distance area orders differ.")

    result_data = target_rows.copy(deep=True)
    original_target = target_rows.copy(deep=True)
    target_area_ids = result_data["area_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(result_data["area_id"].to_numpy(), target_area_ids):
        raise ValueError("Target area_id values must be integer-valued.")
    unknown_targets = sorted(
        set(int(value) for value in target_area_ids.tolist()).difference(
            distance_matrix.area_to_pos
        )
    )
    if unknown_targets:
        raise ValueError(f"Target rows contain unknown areas: {unknown_targets}")
    target_area_positions = np.asarray(
        [distance_matrix.area_to_pos[int(area_id)] for area_id in target_area_ids],
        dtype=np.int64,
    )
    target_months = tuple(
        _normalize_month(value, "target row date") for value in result_data["date"]
    )
    area_country_mapping = dict(
        loco.normalize_country_lookup(area_country)[
            ["area_id", "country_code_3"]
        ].itertuples(index=False, name=None)
    )

    audit_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    generated_features = []
    generated_columns: dict[str, np.ndarray] = {}
    audit_feature_categories = tuple(sorted(features))
    audit_country_categories = tuple(
        sorted({str(value) for value in area_country_mapping.values()})
    )
    for feature in features:
        feature_time_type = feature_time_types[feature]
        generated_feature = f"{feature}__{feature_suffix}_mean"
        if generated_feature in result_data.columns:
            raise ValueError(f"Generated feature already exists: {generated_feature}")
        generated_features.append(generated_feature)
        source_index = build_observed_feature_index(
            source_observations,
            feature,
            feature_time_type,
            layer,
            distance_matrix,
            area_country,
        )
        generated_values = np.full(len(result_data), np.nan, dtype=np.float64)
        total_neighbor_slots = 0
        observed_slots = 0
        original_missing_slots = 0
        imputed_slots = 0
        remaining_missing_slots = 0
        effective_nonmissing_slots = 0
        rows_with_effective_mean = 0
        rows_all_missing = 0
        tier_counts = {"own_history": 0, "same_country": 0, "global": 0}
        feature_audit_records: list[dict[str, object]] = []

        for row_position, (
            target_area_id,
            target_area_position,
            target_month,
        ) in enumerate(
            zip(target_area_ids, target_area_positions, target_months)
        ):
            neighbor_positions = neighbors.neighbor_positions[target_area_position]
            slot_count = len(neighbor_positions)
            total_neighbor_slots += slot_count
            if slot_count == 0:
                rows_all_missing += 1
                continue
            date_position = source_index.date_to_pos.get(target_month)
            if date_position is None:
                direct_observed = np.zeros(slot_count, dtype=bool)
                slot_values = np.full(slot_count, np.nan, dtype=np.float64)
            else:
                direct_observed = source_index.observed[
                    neighbor_positions, date_position
                ].copy()
                slot_values = source_index.values[
                    neighbor_positions, date_position
                ].astype(np.float64, copy=True)
            observed_count = int(direct_observed.sum())
            missing_slot_indices = np.flatnonzero(~direct_observed)
            observed_slots += observed_count
            original_missing_slots += int(len(missing_slot_indices))

            for missing_slot_index in missing_slot_indices.tolist():
                neighbor_position = int(neighbor_positions[missing_slot_index])
                neighbor_area_id = int(source_index.area_ids[neighbor_position])
                resolution = resolve_neighbor_slot(
                    aggregation_target_area_id=int(target_area_id),
                    imputed_neighbor_area_id=neighbor_area_id,
                    target_month=target_month,
                    feature=feature,
                    feature_time_type=feature_time_type,
                    layer=layer,
                    source_index=source_index,
                    distance_matrix=distance_matrix,
                    area_country=area_country_mapping,
                    condition=scheme,
                    split=split,
                )
                if np.isfinite(resolution.value):
                    if resolution.audit_record is None:
                        raise ValueError("Successful interpolation is missing lineage.")
                    slot_values[missing_slot_index] = resolution.value
                    imputed_slots += 1
                    tier = str(resolution.audit_record["source_tier"])
                    tier_counts[tier] += 1
                    feature_audit_records.append(resolution.audit_record)
                else:
                    if resolution.audit_record is not None:
                        raise ValueError("Unresolved interpolation unexpectedly has lineage.")
                    remaining_missing_slots += 1

            mean_value, effective_count = mean_available_slots(slot_values)
            generated_values[row_position] = mean_value
            effective_nonmissing_slots += effective_count
            if effective_count:
                rows_with_effective_mean += 1
            else:
                rows_all_missing += 1

        if neighbors.scheme == "knn5":
            if not all(
                len(neighbors.neighbor_positions[position]) == 5
                for position in target_area_positions
            ):
                raise ValueError("KNN-5 target rows do not all have five neighbors.")
            if total_neighbor_slots != len(result_data) * 5:
                raise ValueError("KNN-5 total slot count is incorrect.")
        else:
            expected_slots = sum(
                len(neighbors.neighbor_positions[position])
                for position in target_area_positions
            )
            if total_neighbor_slots != expected_slots:
                raise ValueError("D200 total slot count is incorrect.")
        if observed_slots + original_missing_slots != total_neighbor_slots:
            raise ValueError("Observed/missing slot identity failed.")
        if imputed_slots + remaining_missing_slots != original_missing_slots:
            raise ValueError("Imputed/remaining slot identity failed.")
        if observed_slots + imputed_slots != effective_nonmissing_slots:
            raise ValueError("Effective slot identity failed.")
        if rows_with_effective_mean + rows_all_missing != len(result_data):
            raise ValueError("Effective/all-missing row identity failed.")
        if sum(tier_counts.values()) != imputed_slots or len(
            feature_audit_records
        ) != imputed_slots:
            raise ValueError("Interpolation tier counts do not match events.")

        generated_columns[generated_feature] = generated_values
        feature_audit = _compact_interpolation_audit(
            pd.DataFrame.from_records(
                feature_audit_records, columns=INTERPOLATION_AUDIT_COLUMNS
            ),
            feature_categories=audit_feature_categories,
            country_categories=audit_country_categories,
        )
        if not feature_audit.empty:
            feature_audit = feature_audit.sort_values(
                [
                    "target_month",
                    "aggregation_target_area_id",
                    "imputed_neighbor_area_id",
                    "source_tier",
                    "source_area_id",
                ],
                kind="mergesort",
            ).reset_index(drop=True)
            audit_frames.append(feature_audit)
        month_gaps = feature_audit["month_gap"].dropna().to_numpy(dtype=np.float64)
        distances = feature_audit[
            "distance_imputed_neighbor_to_source_km"
        ].to_numpy(dtype=np.float64)
        summary_records.append(
            {
                "condition": scheme,
                "layer": layer,
                "split": split,
                "feature": feature,
                "target_rows": int(len(result_data)),
                "total_neighbor_slots": int(total_neighbor_slots),
                "observed_slots": int(observed_slots),
                "original_missing_slots": int(original_missing_slots),
                "imputed_slots": int(imputed_slots),
                "remaining_missing_slots": int(remaining_missing_slots),
                "effective_nonmissing_slots": int(effective_nonmissing_slots),
                "rows_with_effective_mean": int(rows_with_effective_mean),
                "rows_all_missing": int(rows_all_missing),
                "own_history_source_count": int(tier_counts["own_history"]),
                "same_country_source_count": int(tier_counts["same_country"]),
                "global_source_count": int(tier_counts["global"]),
                "imputation_month_gap_min": _summary_stat(month_gaps, "min"),
                "imputation_month_gap_median": _summary_stat(
                    month_gaps, "median"
                ),
                "imputation_month_gap_max": _summary_stat(month_gaps, "max"),
                "imputation_distance_km_min": _summary_stat(distances, "min"),
                "imputation_distance_km_median": _summary_stat(
                    distances, "median"
                ),
                "imputation_distance_km_max": _summary_stat(distances, "max"),
            }
        )
        source_index.resolution_cache.clear()
        source_index.candidate_cache.clear()
        feature_audit_records.clear()

    result_data = pd.concat(
        [
            result_data,
            pd.DataFrame(generated_columns, index=result_data.index),
        ],
        axis=1,
    )
    pd.testing.assert_frame_equal(
        result_data.loc[:, original_target.columns],
        original_target,
        check_exact=True,
    )
    if not result_data.index.equals(original_target.index):
        raise ValueError("Spatial augmentation changed the target-row index.")
    interpolation_audit = _concat_interpolation_audit_blocks(audit_frames)
    interpolation_summary = pd.DataFrame.from_records(
        summary_records, columns=INTERPOLATION_SUMMARY_COLUMNS
    )
    matrix_sha256 = canonical_dataframe_sha256(
        result_data,
        loco.KEY_COLUMNS,
        [*loco.KEY_COLUMNS, *features, *generated_features],
    )
    return SpatialAugmentationResult(
        data=result_data,
        interpolation_audit=interpolation_audit,
        interpolation_summary=interpolation_summary,
        matrix_sha256=matrix_sha256,
    )


def _reassemble_partitioned_data(
    original: pd.DataFrame,
    parts: Sequence[pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    combined = pd.concat(list(parts), axis=0)
    if combined.index.duplicated().any():
        raise ValueError(f"{label} partition reassembly duplicated row indices.")
    if set(combined.index) != set(original.index):
        raise ValueError(f"{label} partition reassembly changed the row-index set.")
    combined = combined.reindex(original.index)
    pd.testing.assert_frame_equal(
        combined.loc[:, original.columns], original, check_exact=True
    )
    return combined


def build_condition_matrices(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    now_train_mask: pd.Series,
    now_test_mask: pd.Series,
    feature_manifest: pd.DataFrame,
    distance_matrix: DistanceMatrix,
    knn5_neighbors: NeighborIndex,
    d200_neighbors: NeighborIndex,
    area_country: pd.DataFrame,
    conditions: Sequence[str],
) -> ConditionMatrices:
    """Build only selected condition matrices and their deterministic audits."""
    selected = tuple(condition for condition in CONDITION_ORDER if condition in conditions)
    if not selected or len(selected) != len(tuple(conditions)):
        raise ValueError("conditions must be unique approved condition identifiers.")
    if not forecasting.index.is_unique or not nowcasting.index.is_unique:
        raise ValueError("Condition source table indices must be unique.")
    loco._validate_unique_keys(forecasting, "Condition forecasting source")
    loco._validate_unique_keys(nowcasting, "Condition nowcasting source")
    train_mask, test_mask = loco._validate_split_masks(
        forecasting, train_mask, test_mask, DEFAULT_CUTOFF, "Forecasting"
    )
    now_train_mask, now_test_mask = loco._validate_split_masks(
        nowcasting, now_train_mask, now_test_mask, DEFAULT_CUTOFF, "Nowcasting"
    )
    layer1_original = tuple(loco.select_layer1_features(forecasting))
    expected_manifest = build_feature_manifest(layer1_original)
    pd.testing.assert_frame_equal(
        feature_manifest.reset_index(drop=True),
        expected_manifest.reset_index(drop=True),
        check_dtype=False,
    )
    layer1_features, layer2_features = build_condition_feature_lists(
        layer1_original
    )
    layer1_ablated = layer1_features["no_lat_lon"]
    layer1_time_types = dict(
        feature_manifest.loc[
            feature_manifest["layer"].eq("layer1_shared")
            & feature_manifest["neighbor_eligible"].astype(bool),
            ["original_feature", "feature_time_type"],
        ].itertuples(index=False, name=None)
    )
    layer2_original = tuple(loco.NOWCAST_FEATURES)
    layer2_time_types = dict(
        feature_manifest.loc[
            feature_manifest["layer"].eq("nowcasting_layer2"),
            ["original_feature", "feature_time_type"],
        ].itertuples(index=False, name=None)
    )

    forecasting_by_condition: dict[str, pd.DataFrame] = {}
    nowcasting_by_condition: dict[str, pd.DataFrame] = {}
    audit_frames = []
    summary_frames = []
    matrix_hash_records = []
    split_specs = (
        ("train", train_mask, now_train_mask),
        ("test", test_mask, now_test_mask),
    )

    for condition in selected:
        if condition in {"baseline_with_lat_lon", "no_lat_lon"}:
            condition_forecasting = forecasting.copy(deep=True)
            condition_nowcasting = nowcasting.copy(deep=True)
        else:
            if distance_matrix is None:
                raise ValueError("Spatial conditions require a distance matrix.")
            neighbors = (
                knn5_neighbors
                if condition == "knn5_spatial_means"
                else d200_neighbors
            )
            if neighbors is None:
                raise ValueError(f"Spatial condition {condition} is missing neighbors.")
            layer1_parts = []
            layer2_parts = []
            for split_name, forecast_mask, nowcast_mask in split_specs:
                layer1_result = augment_spatial_features(
                    forecasting.loc[forecast_mask].copy(),
                    forecasting,
                    layer1_ablated,
                    layer1_time_types,
                    neighbors,
                    distance_matrix,
                    area_country,
                    condition,
                    "layer1_shared",
                    split_name,
                )
                layer2_result = augment_spatial_features(
                    nowcasting.loc[nowcast_mask].copy(),
                    nowcasting,
                    layer2_original,
                    layer2_time_types,
                    neighbors,
                    distance_matrix,
                    area_country,
                    condition,
                    "nowcasting_layer2",
                    split_name,
                )
                layer1_parts.append(layer1_result.data)
                layer2_parts.append(layer2_result.data)
                audit_frames.extend(
                    [
                        layer1_result.interpolation_audit,
                        layer2_result.interpolation_audit,
                    ]
                )
                summary_frames.extend(
                    [
                        layer1_result.interpolation_summary,
                        layer2_result.interpolation_summary,
                    ]
                )
            condition_forecasting = _reassemble_partitioned_data(
                forecasting, layer1_parts, f"{condition} Layer 1"
            )
            condition_nowcasting = _reassemble_partitioned_data(
                nowcasting, layer2_parts, f"{condition} Layer 2"
            )
        forecasting_by_condition[condition] = condition_forecasting
        nowcasting_by_condition[condition] = condition_nowcasting

        for split_name, forecast_mask, nowcast_mask in split_specs:
            for layer_name, data, mask, active_features in (
                (
                    "layer1_shared",
                    condition_forecasting,
                    forecast_mask,
                    layer1_features[condition],
                ),
                (
                    "nowcasting_layer2",
                    condition_nowcasting,
                    nowcast_mask,
                    layer2_features[condition],
                ),
            ):
                matrix_hash_records.append(
                    {
                        "condition": condition,
                        "layer": layer_name,
                        "split": split_name,
                        "feature_count": int(len(active_features)),
                        "matrix_sha256": canonical_dataframe_sha256(
                            data.loc[mask],
                            loco.KEY_COLUMNS,
                            [*loco.KEY_COLUMNS, *active_features],
                        ),
                    }
                )
        for layer_name, data, active_features in (
            (
                "layer1_shared",
                condition_forecasting,
                layer1_features[condition],
            ),
            (
                "nowcasting_layer2",
                condition_nowcasting,
                layer2_features[condition],
            ),
        ):
            matrix_hash_records.append(
                {
                    "condition": condition,
                    "layer": layer_name,
                    "split": "full_oof",
                    "feature_count": int(len(active_features)),
                    "matrix_sha256": canonical_dataframe_sha256(
                        data,
                        loco.KEY_COLUMNS,
                        [*loco.KEY_COLUMNS, *active_features],
                    ),
                }
            )

    interpolation_audit = _concat_interpolation_audit_blocks(audit_frames)
    if summary_frames:
        interpolation_summary = pd.concat(summary_frames, ignore_index=True)
        condition_rank = {value: rank for rank, value in enumerate(CONDITION_ORDER)}
        layer_rank = {"layer1_shared": 0, "nowcasting_layer2": 1}
        split_rank = {"train": 0, "test": 1}
        feature_rank = {
            (row.layer, row.original_feature): int(row.feature_order)
            for row in feature_manifest.itertuples(index=False)
        }
        interpolation_summary = (
            interpolation_summary.assign(
                _condition_rank=interpolation_summary["condition"].map(
                    condition_rank
                ),
                _layer_rank=interpolation_summary["layer"].map(layer_rank),
                _split_rank=interpolation_summary["split"].map(split_rank),
                _feature_rank=[
                    feature_rank[(layer, feature)]
                    for layer, feature in interpolation_summary[
                        ["layer", "feature"]
                    ].itertuples(index=False, name=None)
                ],
            )
            .sort_values(
                [
                    "_condition_rank",
                    "_layer_rank",
                    "_split_rank",
                    "_feature_rank",
                ],
                kind="mergesort",
            )
            .drop(
                columns=[
                    "_condition_rank",
                    "_layer_rank",
                    "_split_rank",
                    "_feature_rank",
                ]
            )
            .reset_index(drop=True)
        )
    else:
        interpolation_summary = pd.DataFrame(columns=INTERPOLATION_SUMMARY_COLUMNS)
    matrix_hashes = pd.DataFrame.from_records(
        matrix_hash_records, columns=MATRIX_HASH_COLUMNS
    )
    return ConditionMatrices(
        forecasting=forecasting_by_condition,
        nowcasting=nowcasting_by_condition,
        layer1_features={
            condition: layer1_features[condition] for condition in selected
        },
        layer2_features={
            condition: layer2_features[condition] for condition in selected
        },
        interpolation_audit=interpolation_audit,
        interpolation_summary=interpolation_summary,
        matrix_hashes=matrix_hashes,
    )


def haversine_distance_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> np.ndarray:
    """Return broadcast Haversine great-circle distances in kilometres."""
    if not np.isfinite(earth_radius_km) or earth_radius_km <= 0:
        raise ValueError("earth_radius_km must be finite and positive.")
    lat1_r, lon1_r, lat2_r, lon2_r = (
        np.radians(np.asarray(value, dtype=np.float64))
        for value in (lat1, lon1, lat2, lon2)
    )
    delta_lat = lat2_r - lat1_r
    delta_lon = lon2_r - lon1_r
    haversine_value = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_r)
        * np.cos(lat2_r)
        * np.sin(delta_lon / 2.0) ** 2
    )
    distances = earth_radius_km * 2.0 * np.arcsin(
        np.sqrt(np.clip(haversine_value, 0.0, 1.0))
    )
    return np.asarray(distances, dtype=np.float64)


def build_distance_matrix(
    coordinates: pd.DataFrame,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> DistanceMatrix:
    """Build one deterministic area-sorted pairwise distance matrix."""
    loco._require_columns(coordinates, ["area_id", "lat", "lon"])
    selected = coordinates.loc[:, ["area_id", "lat", "lon"]].copy()
    if selected.isna().any().any():
        raise ValueError("Coordinates contain missing values.")
    non_numeric = [
        column
        for column in selected.columns
        if not pd.api.types.is_numeric_dtype(selected[column])
    ]
    if non_numeric:
        raise ValueError(f"Coordinates contain non-numeric columns: {non_numeric}")
    if selected.duplicated(["area_id"]).any():
        raise ValueError("Coordinates contain duplicate area_id values.")
    ordered = selected.sort_values("area_id", kind="mergesort").reset_index(drop=True)
    area_ids = ordered["area_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ordered["area_id"].to_numpy(), area_ids):
        raise ValueError("Coordinate area_id values must be integer-valued.")
    latitudes = ordered["lat"].to_numpy(dtype=np.float64)
    longitudes = ordered["lon"].to_numpy(dtype=np.float64)
    distances = haversine_distance_km(
        latitudes[:, None],
        longitudes[:, None],
        latitudes[None, :],
        longitudes[None, :],
        earth_radius_km,
    ).astype(np.float64, copy=False)
    np.fill_diagonal(distances, 0.0)
    if not np.allclose(distances, distances.T, rtol=0.0, atol=1e-10):
        raise ValueError("Haversine distance matrix is not symmetric.")
    return DistanceMatrix(
        area_ids=area_ids,
        distances_km=distances,
        area_to_pos={
            int(area_id): int(position)
            for position, area_id in enumerate(area_ids.tolist())
        },
    )


def build_knn5_neighbors(
    distance_matrix: DistanceMatrix,
    k: int = 5,
) -> NeighborIndex:
    """Select each area's directed nearest-k non-self neighbors."""
    area_count = len(distance_matrix.area_ids)
    if (
        isinstance(k, bool)
        or not isinstance(k, (int, np.integer))
        or k < 1
        or k >= area_count
    ):
        raise ValueError("k must be an integer between 1 and area_count - 1.")
    all_positions = np.arange(area_count, dtype=np.int64)
    neighbor_positions = []
    neighbor_distances = []
    for target_position, row in enumerate(distance_matrix.distances_km):
        candidates = all_positions[all_positions != target_position]
        order = np.lexsort(
            (distance_matrix.area_ids[candidates], row[candidates])
        )
        selected = candidates[order[: int(k)]].astype(np.int64, copy=False)
        neighbor_positions.append(selected)
        neighbor_distances.append(row[selected].astype(np.float64, copy=False))
    return NeighborIndex(
        scheme="knn5",
        area_ids=distance_matrix.area_ids,
        neighbor_positions=tuple(neighbor_positions),
        neighbor_distances_km=tuple(neighbor_distances),
    )


def build_d200_neighbors(
    distance_matrix: DistanceMatrix,
    radius_km: float = 200.0,
) -> NeighborIndex:
    """Select every non-self area within an inclusive distance radius."""
    if not np.isfinite(radius_km) or radius_km < 0:
        raise ValueError("radius_km must be finite and non-negative.")
    area_count = len(distance_matrix.area_ids)
    all_positions = np.arange(area_count, dtype=np.int64)
    neighbor_positions = []
    neighbor_distances = []
    for target_position, row in enumerate(distance_matrix.distances_km):
        selected = all_positions[
            (all_positions != target_position) & (row <= radius_km)
        ]
        order = np.lexsort(
            (distance_matrix.area_ids[selected], row[selected])
        )
        selected = selected[order].astype(np.int64, copy=False)
        neighbor_positions.append(selected)
        neighbor_distances.append(row[selected].astype(np.float64, copy=False))
    return NeighborIndex(
        scheme="d200",
        area_ids=distance_matrix.area_ids,
        neighbor_positions=tuple(neighbor_positions),
        neighbor_distances_km=tuple(neighbor_distances),
    )


def build_weight_diagnostics(
    neighbors: NeighborIndex,
    area_country: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize deterministic neighbor membership and distance support."""
    loco._require_columns(area_country, ["area_id", "country_code_3"])
    lookup = area_country.loc[:, ["area_id", "country_code_3"]].copy()
    if lookup["area_id"].isna().any() or lookup["country_code_3"].isna().any():
        raise ValueError("Country lookup contains missing values.")
    if lookup.duplicated(["area_id"]).any():
        raise ValueError("Country lookup area set contains duplicate area_id values.")
    lookup_ids = lookup["area_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(lookup["area_id"].to_numpy(), lookup_ids):
        raise ValueError("Country lookup area_id values must be integer-valued.")
    expected_area_set = set(int(value) for value in neighbors.area_ids.tolist())
    if set(int(value) for value in lookup_ids.tolist()) != expected_area_set:
        raise ValueError("Country lookup area set does not match neighbor areas.")
    country_by_area = dict(
        lookup.astype({"area_id": "int64"})[
            ["area_id", "country_code_3"]
        ].itertuples(index=False, name=None)
    )

    records = []
    for target_position, area_id in enumerate(neighbors.area_ids.tolist()):
        positions = neighbors.neighbor_positions[target_position]
        distances = neighbors.neighbor_distances_km[target_position]
        neighbor_ids = neighbors.area_ids[positions]
        neighbor_string = ";".join(str(int(value)) for value in neighbor_ids.tolist())
        zero_neighbor = len(positions) == 0
        records.append(
            {
                "scheme": neighbors.scheme,
                "area_id": int(area_id),
                "country_code_3": country_by_area[int(area_id)],
                "neighbor_count": int(len(positions)),
                "min_distance_km": (
                    np.nan if zero_neighbor else float(np.min(distances))
                ),
                "max_distance_km": (
                    np.nan if zero_neighbor else float(np.max(distances))
                ),
                "mean_distance_km": (
                    np.nan if zero_neighbor else float(np.mean(distances))
                ),
                "zero_neighbor": bool(zero_neighbor),
                "neighbor_ids": neighbor_string,
                "neighbor_ids_sha256": hashlib.sha256(
                    neighbor_string.encode("utf-8")
                ).hexdigest(),
            }
        )
    return pd.DataFrame.from_records(records, columns=WEIGHT_DIAGNOSTIC_COLUMNS)


def validate_run_configuration(
    conditions: Sequence[str] | None,
    output_dir: Path,
    random_state: int | None,
    workers: int,
    estimator_n_jobs: int | None,
) -> tuple[tuple[str, ...], bool]:
    """Validate condition selection and protect formal output paths."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")

    selected = CONDITION_ORDER if conditions is None else tuple(conditions)
    if not selected:
        raise ValueError("at least one condition must be selected.")

    unknown = [item for item in selected if item not in CONDITION_ORDER]
    if unknown:
        raise ValueError(f"Unknown condition values: {unknown}")
    if len(set(selected)) != len(selected):
        raise ValueError("conditions contains duplicate identifiers.")

    selected = tuple(item for item in CONDITION_ORDER if item in selected)
    formal_output = Path(output_dir).resolve() == DEFAULT_OUTPUT_DIR.resolve()
    frozen_controls = (
        selected == CONDITION_ORDER
        and random_state is None
        and estimator_n_jobs is None
        and workers == 1
    )
    production_run = formal_output and frozen_controls
    if formal_output and not production_run:
        raise ValueError(
            "Formal output requires all four conditions, no explicit XGBoost seed or "
            "n_jobs override, and one outer model worker."
        )
    if production_run:
        frozen_main_result.assert_frozen_environment(("matplotlib",))
    return selected, production_run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the approved command-line option surface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecasting-input", type=Path, default=DEFAULT_FORECASTING_INPUT
    )
    parser.add_argument(
        "--nowcasting-input", type=Path, default=DEFAULT_NOWCASTING_INPUT
    )
    parser.add_argument("--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument(
        "--contemporaneous-params",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_PARAMS,
    )
    parser.add_argument(
        "--contemporaneous-reference-predictions",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_REFERENCE_PREDICTIONS,
    )
    parser.add_argument(
        "--contemporaneous-reference-audit",
        type=Path,
        default=DEFAULT_CONTEMPORANEOUS_REFERENCE_AUDIT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conditions", nargs="+", choices=CONDITION_ORDER)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Supplying a value creates a diagnostic, non-frozen run.",
    )
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=DEFAULT_ESTIMATOR_N_JOBS,
        help="Supplying a value overrides frozen default XGBoost threads.",
    )
    return parser.parse_args(argv)


def load_prepared_inputs(
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the two keyed model tables and country lookup."""
    forecasting_raw = pd.read_csv(forecasting_path)
    nowcasting_raw = pd.read_csv(nowcasting_path)
    lookup = loco.load_country_lookup(country_lookup_path)
    forecasting, nowcasting = loco.prepare_model_inputs(
        forecasting_raw, nowcasting_raw, lookup
    )
    forecasting = loco.add_cumulative_targets(forecasting)
    if len(forecasting) != DEFAULT_EXPECTED_SOURCE_ROWS:
        raise ValueError("Prepared forecasting row count differs from 5,575.")
    if len(nowcasting) != DEFAULT_EXPECTED_SOURCE_ROWS:
        raise ValueError("Prepared nowcasting row count differs from 5,575.")
    return forecasting, nowcasting, lookup


def build_coordinate_table(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    expected_area_count: int = DEFAULT_EXPECTED_AREAS,
) -> pd.DataFrame:
    """Return the complete, unique, cross-table-consistent area coordinates."""
    coordinate_frames = []
    for name, data in (("Forecasting", forecasting), ("Nowcasting", nowcasting)):
        loco._require_columns(data, ["area_id", "lat", "lon"])
        if data[["lat", "lon"]].isna().any().any():
            raise ValueError(f"{name} coordinates contain missing values.")
        non_numeric = [
            column
            for column in ("area_id", "lat", "lon")
            if not pd.api.types.is_numeric_dtype(data[column])
        ]
        if non_numeric:
            raise ValueError(f"{name} coordinates contain non-numeric columns: {non_numeric}")
        counts = data.groupby("area_id", sort=False)[["lat", "lon"]].nunique(
            dropna=False
        )
        if counts.ne(1).any().any():
            raise ValueError(f"{name} has non-unique coordinates within area_id.")
        coordinate_frames.append(
            data[["area_id", "lat", "lon"]]
            .drop_duplicates()
            .sort_values("area_id", kind="mergesort")
            .reset_index(drop=True)
            .astype({"area_id": "int64", "lat": "float64", "lon": "float64"})
        )
    if not coordinate_frames[0].equals(coordinate_frames[1]):
        raise ValueError("Forecasting and Nowcasting coordinates differ.")
    coordinates = coordinate_frames[0]
    if len(coordinates) != expected_area_count:
        raise ValueError(
            f"Expected {expected_area_count} coordinate rows, found {len(coordinates)}."
        )
    coordinates.attrs["coordinates_cross_table_equal_passed"] = True
    coordinates.attrs["expected_area_count"] = int(expected_area_count)
    return coordinates


def build_coordinate_validation_record(
    coordinates: pd.DataFrame,
) -> Mapping[str, object]:
    """Summarize the coordinate validations retained by the canonical table."""
    loco._require_columns(coordinates, ["area_id", "lat", "lon"])
    expected_area_count = int(
        coordinates.attrs.get("expected_area_count", DEFAULT_EXPECTED_AREAS)
    )
    return {
        "coordinates_complete_passed": bool(
            coordinates[["lat", "lon"]].notna().all().all()
        ),
        "coordinates_unique_within_area_passed": bool(
            ~coordinates.duplicated(["area_id"]).any()
        ),
        "coordinates_cross_table_equal_passed": bool(
            coordinates.attrs.get("coordinates_cross_table_equal_passed", False)
        ),
        "coordinate_area_count": int(len(coordinates)),
        "coordinate_area_count_passed": len(coordinates) == expected_area_count,
        "coordinate_table_sha256": canonical_dataframe_sha256(
            coordinates,
            ["area_id"],
            ["area_id", "lat", "lon"],
        ),
    }


def build_feature_manifest(
    layer1_features: Sequence[str],
    layer2_features: Sequence[str] = loco.NOWCAST_FEATURES,
) -> pd.DataFrame:
    """Build the authoritative ordered feature-time and spatial-name manifest."""
    layer1 = tuple(layer1_features)
    layer2 = tuple(layer2_features)
    if len(layer1) != 106:
        raise ValueError(f"Layer 1 must contain exactly 106 features; found {len(layer1)}.")
    if len(set(layer1)) != len(layer1):
        raise ValueError("Layer 1 feature names must be unique.")
    if layer1.count("lat") != 1 or layer1.count("lon") != 1:
        raise ValueError("Layer 1 must contain lat and lon exactly once.")
    ablated = tuple(feature for feature in layer1 if feature not in {"lat", "lon"})
    if len(ablated) != 104:
        raise ValueError("Removing only lat and lon must leave 104 Layer 1 features.")
    if len(LAYER1_STATIC_FEATURES) != 31 or len(set(LAYER1_STATIC_FEATURES)) != 31:
        raise ValueError("The Layer 1 static allowlist must contain 31 unique features.")
    missing_static = sorted(set(LAYER1_STATIC_FEATURES).difference(ablated))
    if missing_static:
        raise ValueError(f"Layer 1 is missing static features: {missing_static}")
    static_set = set(LAYER1_STATIC_FEATURES)
    dynamic = tuple(feature for feature in ablated if feature not in static_set)
    if len(dynamic) != 73 or set(dynamic).intersection(static_set):
        raise ValueError("Layer 1 must contain exactly 31 static and 73 dynamic features.")
    reconstructed = tuple(
        feature for feature in layer1 if feature not in {"lat", "lon"}
    )
    if reconstructed != ablated or set(ablated) != static_set.union(dynamic):
        raise ValueError("Layer 1 static/dynamic classification does not rebuild the live set.")
    if layer2 != tuple(loco.NOWCAST_FEATURES) or len(layer2) != 69:
        raise ValueError("Layer 2 must match the exact ordered 69 NOWCAST_FEATURES.")

    records = []
    for feature_order, feature in enumerate(layer1):
        if feature in {"lat", "lon"}:
            feature_time_type = "coordinate"
            reference_month_rule = "distance_only"
            neighbor_eligible = False
        elif feature in static_set:
            feature_time_type = "static"
            reference_month_rule = "package_snapshot_static"
            neighbor_eligible = True
        else:
            feature_time_type = "dynamic"
            reference_month_rule = "row_month_minus_12_calendar_months"
            neighbor_eligible = True
        records.append(
            {
                "layer": "layer1_shared",
                "feature_order": feature_order,
                "original_feature": feature,
                "feature_time_type": feature_time_type,
                "reference_month_rule": reference_month_rule,
                "neighbor_eligible": neighbor_eligible,
                "knn5_feature_name": (
                    f"{feature}__knn5_mean" if neighbor_eligible else pd.NA
                ),
                "d200_feature_name": (
                    f"{feature}__d200_mean" if neighbor_eligible else pd.NA
                ),
            }
        )
    for feature_order, feature in enumerate(layer2):
        records.append(
            {
                "layer": "nowcasting_layer2",
                "feature_order": feature_order,
                "original_feature": feature,
                "feature_time_type": "dynamic",
                "reference_month_rule": "row_month",
                "neighbor_eligible": True,
                "knn5_feature_name": f"{feature}__knn5_mean",
                "d200_feature_name": f"{feature}__d200_mean",
            }
        )
    return pd.DataFrame.from_records(records, columns=FEATURE_MANIFEST_COLUMNS)


def _normalize_canonical_date_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    date_columns = [
        column
        for column in result.columns
        if column == "date" or column.endswith("_date") or column.endswith("_month")
    ]
    for column in date_columns:
        present = result[column].notna()
        if not present.any():
            continue
        try:
            parsed = pd.to_datetime(
                result.loc[present, column], errors="raise", format="mixed"
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Canonical date column {column} contains invalid dates.") from error
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise ValueError(f"Canonical date column {column} must be timezone-naive.")
        if not parsed.eq(parsed.dt.normalize()).all():
            raise ValueError(f"Canonical date column {column} must contain midnight values.")
        normalized = pd.Series(pd.NA, index=result.index, dtype="string")
        normalized.loc[present] = parsed.dt.strftime("%Y-%m-%d").to_numpy()
        result[column] = normalized
    return result


def canonical_dataframe_sha256(
    data: pd.DataFrame,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
) -> str:
    """Hash a key-sorted, schema-ordered canonical CSV representation."""
    keys = tuple(key_columns)
    if not keys:
        raise ValueError("Canonical hashes require at least one key column.")
    selected_columns = list(dict.fromkeys([*keys, *value_columns]))
    loco._require_columns(data, selected_columns)
    ordered = data.loc[:, selected_columns].copy()
    if ordered[list(keys)].isna().any().any():
        raise ValueError("Canonical hash keys contain missing values.")
    ordered = _normalize_canonical_date_columns(ordered)
    if ordered.duplicated(list(keys)).any():
        raise ValueError("Canonical hash keys are not unique.")
    ordered = ordered.sort_values(list(keys), kind="mergesort")
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_key_sha256(
    data: pd.DataFrame,
    key_columns: Sequence[str] = ("area_id", "date"),
) -> str:
    """Hash only the canonical observation-key columns."""
    return canonical_dataframe_sha256(data, key_columns, key_columns)


def wide_predictions_to_phases_preserving_raw(data: pd.DataFrame) -> pd.DataFrame:
    """Convert cumulative predictions without overwriting raw model outputs."""
    required = [
        f"phase{phase}_{suffix}"
        for phase in range(2, 6)
        for suffix in ("test", "pred_raw")
    ]
    loco._require_columns(data, required)
    result = data.copy().reset_index(drop=True)
    raw_columns = [f"phase{phase}_pred_raw" for phase in range(2, 6)]
    actual_columns = [f"phase{phase}_test" for phase in range(2, 6)]
    try:
        raw_values = result[raw_columns].to_numpy(dtype=np.float64)
        actual_values = result[actual_columns].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Cumulative actual and raw prediction columns must be numeric.") from error
    if not np.isfinite(raw_values).all():
        raise ValueError("All raw cumulative predictions must be finite.")
    if not np.isfinite(actual_values).all():
        raise ValueError("All actual cumulative shares must be finite.")

    rounded_columns = []
    for phase in range(2, 6):
        column = f"phase{phase}_pred_rounded"
        result[column] = result[f"phase{phase}_pred_raw"].round(2)
        rounded_columns.append(column)
    result["nonpositive_cumulative_prediction_sum"] = (
        result[rounded_columns].sum(axis=1, min_count=4) <= 0
    )
    result["overall_phase"] = loco._phase_from_cumulative(result, "test")
    rounded_for_conversion = result.rename(
        columns={
            f"phase{phase}_pred_rounded": f"phase{phase}_rounded"
            for phase in range(2, 6)
        }
    )
    result["overall_phase_pred"] = loco._phase_from_cumulative(
        rounded_for_conversion, "rounded"
    )
    return result


def fit_forecasting_condition(
    forecasting: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    split_id: object,
    layer1_features: Sequence[str],
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
    *,
    condition: str,
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit four cumulative Forecasting regressors for one feature condition."""
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown condition: {condition}")
    layer1_features = tuple(layer1_features)
    loco._require_columns(
        forecasting,
        [
            *loco.KEY_COLUMNS,
            "country_code_3",
            "overall_phase",
            *loco.CUMULATIVE_TARGETS.values(),
            *layer1_features,
        ],
    )
    loco._validate_unique_keys(forecasting, "Condition forecasting input")
    non_numeric_layer1 = [
        column
        for column in layer1_features
        if not pd.api.types.is_numeric_dtype(forecasting[column])
    ]
    if non_numeric_layer1:
        raise ValueError(
            f"Layer 1 contains non-numeric features: {non_numeric_layer1}"
        )
    train_mask, test_mask = loco._validate_split_masks(
        forecasting, train_mask, test_mask, split_id, "Forecasting"
    )

    keys = [*loco.KEY_COLUMNS, "country_code_3"]
    test_rows = forecasting.loc[test_mask]
    wide = test_rows[keys].copy()
    wide["source_row_index"] = test_rows.index.to_numpy()
    wide["split_id"] = split_id
    wide["source_overall_phase"] = test_rows["overall_phase"].to_numpy()
    for phase, target_column in loco.CUMULATIVE_TARGETS.items():
        y_train = forecasting.loc[train_mask, target_column]
        y_test = forecasting.loc[test_mask, target_column]
        if y_train.isna().any() or y_test.isna().any():
            raise ValueError(
                f"Missing {target_column} values for condition {condition}."
            )
        params = general_params if phase == 2 else phase3_params
        model = estimator_factory(**dict(params))
        model.fit(
            forecasting.loc[train_mask, list(layer1_features)],
            y_train,
        )
        phase_frame = test_rows[keys].copy()
        phase_frame[f"phase{phase}_test"] = y_test.to_numpy()
        phase_frame[f"phase{phase}_pred_raw"] = np.asarray(
            model.predict(forecasting.loc[test_mask, list(layer1_features)])
        )
        wide = wide.merge(phase_frame, on=keys, how="inner", validate="one_to_one")

    if len(wide) != int(test_mask.sum()):
        raise ValueError(f"Forecasting condition {condition} lost test rows.")
    result = wide_predictions_to_phases_preserving_raw(wide)
    result.insert(0, "model", "Forecasting")
    result.insert(0, "condition_label", CONDITION_LABELS[condition])
    result.insert(0, "condition", condition)
    for phase in range(2, 6):
        result[f"phase{phase}_layer1_pred"] = np.nan
        result[f"phase{phase}_residual_pred"] = np.nan
    return result.loc[:, PREDICTION_COLUMNS]


def fit_nowcasting_condition(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    now_train_mask: pd.Series,
    now_test_mask: pd.Series,
    split_id: object,
    layer1_features: Sequence[str],
    layer2_features: Sequence[str],
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
    *,
    condition: str,
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit the cascading two-layer Nowcasting model for one feature condition."""
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown condition: {condition}")
    layer1_features = tuple(layer1_features)
    layer2_features = tuple(layer2_features)
    loco._require_columns(
        forecasting,
        [
            *loco.KEY_COLUMNS,
            "country_code_3",
            "overall_phase",
            *loco.CUMULATIVE_TARGETS.values(),
            *layer1_features,
        ],
    )
    loco._require_columns(
        nowcasting,
        [*loco.KEY_COLUMNS, "country_code_3", *layer2_features],
    )
    loco._validate_unique_keys(forecasting, "Condition forecasting input")
    loco._validate_unique_keys(nowcasting, "Condition nowcasting input")
    forecasting_keys = pd.MultiIndex.from_frame(forecasting[loco.KEY_COLUMNS])
    nowcasting_keys = pd.MultiIndex.from_frame(nowcasting[loco.KEY_COLUMNS])
    if set(forecasting_keys) != set(nowcasting_keys):
        raise ValueError("Condition forecasting and nowcasting key sets differ.")

    non_numeric_layer1 = [
        column
        for column in layer1_features
        if not pd.api.types.is_numeric_dtype(forecasting[column])
    ]
    non_numeric_layer2 = [
        column
        for column in layer2_features
        if not pd.api.types.is_numeric_dtype(nowcasting[column])
    ]
    if non_numeric_layer1:
        raise ValueError(
            f"Layer 1 contains non-numeric features: {non_numeric_layer1}"
        )
    if non_numeric_layer2:
        raise ValueError(
            f"Layer 2 contains non-numeric features: {non_numeric_layer2}"
        )

    train_mask, test_mask = loco._validate_split_masks(
        forecasting, train_mask, test_mask, split_id, "Forecasting"
    )
    now_train_mask, now_test_mask = loco._validate_split_masks(
        nowcasting, now_train_mask, now_test_mask, split_id, "Nowcasting"
    )
    for label, left_data, left_mask, right_data, right_mask in (
        ("train", forecasting, train_mask, nowcasting, now_train_mask),
        ("test", forecasting, test_mask, nowcasting, now_test_mask),
    ):
        left_keys = pd.MultiIndex.from_frame(
            left_data.loc[left_mask, loco.KEY_COLUMNS]
        )
        right_keys = pd.MultiIndex.from_frame(
            right_data.loc[right_mask, loco.KEY_COLUMNS]
        )
        if set(left_keys) != set(right_keys):
            raise ValueError(f"Condition {condition} has different {label} keys.")

    keys = [*loco.KEY_COLUMNS, "country_code_3"]
    test_rows = forecasting.loc[test_mask]
    wide = test_rows[keys].copy()
    wide["source_row_index"] = test_rows.index.to_numpy()
    wide["split_id"] = split_id
    wide["source_overall_phase"] = test_rows["overall_phase"].to_numpy()
    for phase, target_column in loco.CUMULATIVE_TARGETS.items():
        y_train = forecasting.loc[train_mask, target_column]
        y_test = forecasting.loc[test_mask, target_column]
        if y_train.isna().any() or y_test.isna().any():
            raise ValueError(
                f"Missing {target_column} values for condition {condition}."
            )
        params = general_params if phase == 2 else phase3_params
        layer1 = estimator_factory(**dict(params))
        layer1.fit(
            forecasting.loc[train_mask, list(layer1_features)],
            y_train,
        )
        layer1_train = np.asarray(
            layer1.predict(forecasting.loc[train_mask, list(layer1_features)])
        )
        layer1_test = np.asarray(
            layer1.predict(forecasting.loc[test_mask, list(layer1_features)])
        )

        residual_frame = forecasting.loc[train_mask, keys].copy()
        residual_frame["layer1_residual"] = y_train.to_numpy() - layer1_train
        keyed_train = nowcasting.loc[
            now_train_mask, [*keys, *layer2_features]
        ].merge(
            residual_frame,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if len(keyed_train) != int(train_mask.sum()):
            raise ValueError(f"Condition {condition} lost residual training rows.")

        layer2 = estimator_factory(**dict(params))
        layer2.fit(
            keyed_train[list(layer2_features)], keyed_train["layer1_residual"]
        )
        now_test = nowcasting.loc[now_test_mask, [*keys, *layer2_features]]
        residual_frame = now_test[keys].copy()
        residual_frame[f"phase{phase}_residual_pred"] = np.asarray(
            layer2.predict(now_test[list(layer2_features)])
        )
        phase_frame = test_rows[keys].copy()
        phase_frame[f"phase{phase}_test"] = y_test.to_numpy()
        phase_frame[f"phase{phase}_layer1_pred"] = layer1_test
        phase_frame = phase_frame.merge(
            residual_frame, on=keys, how="inner", validate="one_to_one"
        )
        phase_frame[f"phase{phase}_pred_raw"] = (
            phase_frame[f"phase{phase}_layer1_pred"]
            + phase_frame[f"phase{phase}_residual_pred"]
        )
        wide = wide.merge(phase_frame, on=keys, how="inner", validate="one_to_one")

    if len(wide) != int(test_mask.sum()):
        raise ValueError(f"Nowcasting condition {condition} lost test rows.")
    result = wide_predictions_to_phases_preserving_raw(wide)
    result.insert(0, "model", "Nowcasting")
    result.insert(0, "condition_label", CONDITION_LABELS[condition])
    result.insert(0, "condition", condition)
    return result.loc[:, PREDICTION_COLUMNS]


def load_contemporaneous_hyperparameters(
    params_path: Path,
    estimator_n_jobs: int | None,
) -> dict[str, object]:
    """Load the notebook-effective all-target Contemporaneous parameters."""
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    if not isinstance(params, dict) or not params:
        raise ValueError("Contemporaneous hyperparameters must be a non-empty object.")
    params["random_state"] = CONTEMPORANEOUS_RANDOM_STATE
    if estimator_n_jobs is None:
        params.pop("n_jobs", None)
    else:
        params["n_jobs"] = estimator_n_jobs
    return params


def build_contemporaneous_design_matrix(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    fold_table: pd.DataFrame,
    feature_columns: Sequence[str],
    layer1_features: Sequence[str],
    layer2_features: Sequence[str],
) -> pd.DataFrame:
    """Assemble one condition's canonical random-CV matrix in shuffle order."""
    features = tuple(feature_columns)
    layer1 = set(layer1_features)
    layer2 = set(layer2_features)
    if not features or features[-1] != "kfolds" or len(set(features)) != len(features):
        raise ValueError("Contemporaneous feature order must be unique and end in kfolds.")
    if layer1.intersection(layer2):
        raise ValueError("Condition Layer 1 and Layer 2 feature sets overlap.")
    loco._require_columns(
        forecasting,
        [
            *loco.KEY_COLUMNS,
            "country_code_3",
            "overall_phase",
            *loco.CUMULATIVE_TARGETS.values(),
            *[feature for feature in features if feature in layer1],
        ],
    )
    loco._require_columns(
        nowcasting,
        [
            *loco.KEY_COLUMNS,
            *[feature for feature in features if feature in layer2],
        ],
    )
    loco._require_columns(
        fold_table,
        [*loco.KEY_COLUMNS, "source_row_index", "fold", "shuffle_position"],
    )
    forecast_keys = pd.MultiIndex.from_frame(forecasting[list(loco.KEY_COLUMNS)])
    nowcast_keys = pd.MultiIndex.from_frame(nowcasting[list(loco.KEY_COLUMNS)])
    if not forecast_keys.equals(nowcast_keys):
        raise ValueError("Contemporaneous condition matrices are not identically key-ordered.")

    metadata = forecasting.loc[
        :,
        [
            *loco.KEY_COLUMNS,
            "country_code_3",
            "overall_phase",
            *loco.CUMULATIVE_TARGETS.values(),
        ],
    ].copy()
    metadata["date"] = pd.to_datetime(metadata["date"], errors="raise", format="mixed")
    normalized_folds = fold_table.copy()
    normalized_folds["date"] = pd.to_datetime(
        normalized_folds["date"], errors="raise", format="mixed"
    )
    metadata = metadata.merge(
        normalized_folds,
        on=list(loco.KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(metadata) != EXPECTED_CONTEMPORANEOUS_ROWS:
        raise ValueError("Contemporaneous fold merge changed the 5,575-row population.")

    feature_values: dict[str, np.ndarray] = {}
    for feature in features:
        if feature == "kfolds":
            feature_values[feature] = metadata["fold"].to_numpy(dtype=int)
        elif feature in layer1:
            feature_values[feature] = forecasting[feature].to_numpy()
        elif feature in layer2:
            feature_values[feature] = nowcasting[feature].to_numpy()
        else:
            raise ValueError(
                f"Contemporaneous feature {feature} is not assigned to a model layer."
            )
    design = pd.concat(
        [metadata.reset_index(drop=True), pd.DataFrame(feature_values)],
        axis=1,
    )
    design = design.sort_values("shuffle_position", kind="mergesort").reset_index(
        drop=True
    )
    if not np.array_equal(
        design["shuffle_position"].to_numpy(dtype=int),
        np.arange(EXPECTED_CONTEMPORANEOUS_ROWS, dtype=int),
    ):
        raise ValueError("Contemporaneous shuffle positions are incomplete.")
    non_numeric = [
        feature
        for feature in features
        if not pd.api.types.is_numeric_dtype(design[feature])
    ]
    if non_numeric:
        raise ValueError(
            f"Contemporaneous design contains non-numeric predictors: {non_numeric}"
        )
    return design


def fit_contemporaneous_condition(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    fold_table: pd.DataFrame,
    feature_columns: Sequence[str],
    layer1_features: Sequence[str],
    layer2_features: Sequence[str],
    params: Mapping[str, object],
    *,
    condition: str,
    estimator_factory: Callable[..., object] = xgb.XGBRegressor,
) -> pd.DataFrame:
    """Fit one full-OOF seed-0 random five-fold Contemporaneous condition."""
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown condition: {condition}")
    features = tuple(feature_columns)
    design = build_contemporaneous_design_matrix(
        forecasting,
        nowcasting,
        fold_table,
        features,
        layer1_features,
        layer2_features,
    )
    fold_counts = design["fold"].value_counts().sort_index().to_dict()
    expected_fold_counts = {
        fold: EXPECTED_CONTEMPORANEOUS_ROWS_PER_FOLD
        for fold in range(EXPECTED_CONTEMPORANEOUS_FOLDS)
    }
    if fold_counts != expected_fold_counts:
        raise ValueError(f"Unexpected contemporaneous fold sizes: {fold_counts}")

    fold_frames: list[pd.DataFrame] = []
    keys = [*loco.KEY_COLUMNS, "country_code_3"]
    for fold in range(EXPECTED_CONTEMPORANEOUS_FOLDS):
        train_mask = design["fold"].ne(fold)
        validation_mask = design["fold"].eq(fold)
        validation = design.loc[validation_mask]
        wide = validation[keys].copy()
        wide["source_row_index"] = validation["source_row_index"].to_numpy(dtype=int)
        wide["split_id"] = f"random5fold_fold_{fold}"
        wide["source_overall_phase"] = validation["overall_phase"].to_numpy()
        for phase, target_column in loco.CUMULATIVE_TARGETS.items():
            y_train = design.loc[train_mask, target_column]
            y_validation = design.loc[validation_mask, target_column]
            if y_train.isna().any() or y_validation.isna().any():
                raise ValueError(
                    f"Missing {target_column} values for Contemporaneous fold {fold}."
                )
            model = estimator_factory(**dict(params))
            model.fit(design.loc[train_mask, list(features)], y_train)
            predicted = np.asarray(
                model.predict(design.loc[validation_mask, list(features)])
            )
            if not np.isfinite(predicted).all():
                raise ValueError(
                    f"Contemporaneous {condition} Phase {phase}+ predictions are non-finite."
                )
            wide[f"phase{phase}_test"] = y_validation.to_numpy(dtype=float)
            wide[f"phase{phase}_pred_raw"] = predicted
        fold_frames.append(wide_predictions_to_phases_preserving_raw(wide))

    result = pd.concat(fold_frames, ignore_index=True)
    if len(result) != EXPECTED_CONTEMPORANEOUS_ROWS:
        raise ValueError("Contemporaneous OOF assembly changed the source population.")
    if result["source_row_index"].nunique() != EXPECTED_CONTEMPORANEOUS_ROWS:
        raise ValueError("Contemporaneous OOF predictions do not cover every source row once.")
    if result["area_id"].nunique() != EXPECTED_CONTEMPORANEOUS_AREAS:
        raise ValueError("Contemporaneous OOF predictions have an unexpected area count.")
    result.insert(0, "model", CONTEMPORANEOUS_MODEL)
    result.insert(0, "condition_label", CONDITION_LABELS[condition])
    result.insert(0, "condition", condition)
    for phase in range(2, 6):
        result[f"phase{phase}_layer1_pred"] = np.nan
        result[f"phase{phase}_residual_pred"] = np.nan
    return (
        result.sort_values("source_row_index", kind="mergesort")
        .reset_index(drop=True)
        .loc[:, PREDICTION_COLUMNS]
    )


def validate_contemporaneous_baseline_reference(
    predictions: pd.DataFrame,
    fold_contract: ContemporaneousFoldContract,
) -> None:
    """Require the latitude/longitude OOF rerun to reproduce the canonical sidecar."""
    baseline = predictions.loc[
        predictions["condition"].eq("baseline_with_lat_lon")
        & predictions["model"].eq(CONTEMPORANEOUS_MODEL)
    ].sort_values("source_row_index", kind="mergesort")
    reference = fold_contract.reference_predictions.sort_values(
        "source_row_index", kind="mergesort"
    )
    if len(baseline) != EXPECTED_CONTEMPORANEOUS_ROWS or len(reference) != len(
        baseline
    ):
        raise ValueError("Contemporaneous baseline/reference row counts differ.")
    for column in ("source_row_index", "area_id"):
        if not np.array_equal(
            baseline[column].to_numpy(), reference[column].to_numpy()
        ):
            raise ValueError(f"Contemporaneous baseline/reference {column} drifted.")
    baseline_dates = pd.to_datetime(baseline["date"], errors="raise")
    reference_dates = pd.to_datetime(reference["date"], errors="raise")
    if not np.array_equal(baseline_dates.to_numpy(), reference_dates.to_numpy()):
        raise ValueError("Contemporaneous baseline/reference dates drifted.")
    expected_split_ids = reference["fold"].map(
        lambda value: f"random5fold_fold_{int(value)}"
    )
    if not baseline["split_id"].reset_index(drop=True).equals(
        expected_split_ids.reset_index(drop=True)
    ):
        raise ValueError("Contemporaneous baseline/reference fold IDs drifted.")
    exact_mapping = {
        "source_overall_phase": "source_overall_phase",
        "overall_phase": "overall_phase",
        "overall_phase_pred": "contemporaneous_predict",
    }
    for observed_column, reference_column in exact_mapping.items():
        if not np.array_equal(
            baseline[observed_column].to_numpy(),
            reference[reference_column].to_numpy(),
        ):
            raise ValueError(
                f"Contemporaneous baseline/reference {observed_column} drifted."
            )
    for phase in range(2, 6):
        actual_column = f"phase{phase}_test"
        if not np.allclose(
            baseline[actual_column].to_numpy(dtype=float),
            reference[f"phase{phase}_actual"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Contemporaneous baseline/reference {actual_column} drifted."
            )

        rounded_column = f"phase{phase}_pred_rounded"
        observed_rounded = np.round(
            baseline[rounded_column].to_numpy(dtype=np.float64), 2
        )
        reference_rounded = np.round(
            reference[f"phase{phase}_contemporaneous"].to_numpy(dtype=np.float64),
            2,
        )
        if not np.array_equal(observed_rounded, reference_rounded):
            raise ValueError(
                f"Contemporaneous baseline/reference {rounded_column} drifted."
            )


@dataclass(frozen=True)
class ModelRunPayload:
    condition: str
    forecasting: pd.DataFrame
    nowcasting: pd.DataFrame
    train_mask: pd.Series
    test_mask: pd.Series
    now_train_mask: pd.Series
    now_test_mask: pd.Series
    layer1_features: tuple[str, ...]
    layer2_features: tuple[str, ...]
    general_params: Mapping[str, object]
    phase3_params: Mapping[str, object]
    contemporaneous_features: tuple[str, ...]
    contemporaneous_params: Mapping[str, object]
    contemporaneous_fold_table: pd.DataFrame


def _fit_condition_model(
    model_name: str,
    payload: ModelRunPayload,
) -> tuple[str, pd.DataFrame]:
    if model_name == "Forecasting":
        result = fit_forecasting_condition(
            payload.forecasting,
            payload.train_mask,
            payload.test_mask,
            "temporal_2022",
            payload.layer1_features,
            payload.general_params,
            payload.phase3_params,
            condition=payload.condition,
        )
    elif model_name == "Nowcasting":
        result = fit_nowcasting_condition(
            payload.forecasting,
            payload.nowcasting,
            payload.train_mask,
            payload.test_mask,
            payload.now_train_mask,
            payload.now_test_mask,
            "temporal_2022",
            payload.layer1_features,
            payload.layer2_features,
            payload.general_params,
            payload.phase3_params,
            condition=payload.condition,
        )
    elif model_name == CONTEMPORANEOUS_MODEL:
        result = fit_contemporaneous_condition(
            payload.forecasting,
            payload.nowcasting,
            payload.contemporaneous_fold_table,
            payload.contemporaneous_features,
            payload.layer1_features,
            payload.layer2_features,
            payload.contemporaneous_params,
            condition=payload.condition,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model_name, result


def run_condition_models(
    condition: str,
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    train_mask: pd.Series,
    test_mask: pd.Series,
    now_train_mask: pd.Series,
    now_test_mask: pd.Series,
    layer1_features: Sequence[str],
    layer2_features: Sequence[str],
    general_params: Mapping[str, object],
    phase3_params: Mapping[str, object],
    contemporaneous_features: Sequence[str],
    contemporaneous_params: Mapping[str, object],
    contemporaneous_fold_table: pd.DataFrame,
    workers: int = 2,
) -> Mapping[str, pd.DataFrame]:
    """Run all three model families for one condition with deterministic ordering."""
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown condition: {condition}")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    payload = ModelRunPayload(
        condition=condition,
        forecasting=forecasting,
        nowcasting=nowcasting,
        train_mask=train_mask,
        test_mask=test_mask,
        now_train_mask=now_train_mask,
        now_test_mask=now_test_mask,
        layer1_features=tuple(layer1_features),
        layer2_features=tuple(layer2_features),
        general_params=dict(general_params),
        phase3_params=dict(phase3_params),
        contemporaneous_features=tuple(contemporaneous_features),
        contemporaneous_params=dict(contemporaneous_params),
        contemporaneous_fold_table=contemporaneous_fold_table,
    )
    if workers == 1:
        pairs = [_fit_condition_model(model_name, payload) for model_name in MODEL_ORDER]
    else:
        pairs = []
        with ProcessPoolExecutor(
            max_workers=min(len(MODEL_ORDER), workers),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            futures = {
                executor.submit(_fit_condition_model, model_name, payload): model_name
                for model_name in MODEL_ORDER
            }
            for future in as_completed(futures):
                expected_model = futures[future]
                try:
                    returned_model, result = future.result()
                except Exception as error:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"{condition} {expected_model} model failed"
                    ) from error
                if returned_model != expected_model:
                    raise RuntimeError(
                        f"Model worker mismatch: expected {expected_model}, "
                        f"got {returned_model}."
                    )
                pairs.append((returned_model, result))
    results = dict(pairs)
    return {model_name: results[model_name] for model_name in MODEL_ORDER}


def calculate_comparison_metrics(
    predictions: pd.DataFrame,
    condition: str,
    condition_label: str,
    model_name: str,
    baseline_values: Mapping[str, float] | None = None,
    fold_assignment_sha256: str | None = None,
) -> dict[str, object]:
    """Calculate one condition-model record with explicit undefined metrics."""
    if condition not in CONDITION_ORDER:
        raise ValueError(f"Unknown condition: {condition}")
    if condition_label != CONDITION_LABELS[condition]:
        raise ValueError("condition_label does not match the approved condition.")
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name == CONTEMPORANEOUS_MODEL:
        evaluation_protocol = "random_5fold_row_cv"
        evaluation_population = "random_5fold_full_oof_5575"
        n_splits = EXPECTED_CONTEMPORANEOUS_FOLDS
        if not fold_assignment_sha256:
            raise ValueError("Contemporaneous metrics require a fold-assignment hash.")
    else:
        evaluation_protocol = "fixed_2022_temporal_test"
        evaluation_population = "temporal_test_2022_1170"
        n_splits = 1
        if fold_assignment_sha256 is not None:
            raise ValueError("Temporal metrics must not carry a random-CV fold hash.")
    loco._require_columns(
        predictions,
        [
            "area_id",
            "date",
            "country_code_3",
            "overall_phase",
            "overall_phase_pred",
            "phase3_test",
            "phase3_pred_raw",
            "phase3_pred_rounded",
            "nonpositive_cumulative_prediction_sum",
        ],
    )
    if predictions.empty:
        raise ValueError("Comparison metrics require at least one prediction row.")
    test_key_sha256 = canonical_key_sha256(predictions)
    overall_actual = pd.to_numeric(predictions["overall_phase"], errors="raise")
    overall_predicted = pd.to_numeric(
        predictions["overall_phase_pred"], errors="raise"
    )
    overall_actual_values = overall_actual.to_numpy(dtype=np.float64)
    overall_predicted_values = overall_predicted.to_numpy(dtype=np.float64)
    if (
        overall_actual.isna().any()
        or overall_predicted.isna().any()
        or not np.equal(overall_actual_values, np.floor(overall_actual_values)).all()
        or not np.equal(
            overall_predicted_values, np.floor(overall_predicted_values)
        ).all()
        or not overall_actual.between(1, 5).all()
        or not overall_predicted.between(1, 5).all()
    ):
        raise ValueError(
            "Overall phases must be complete integer values from 1 through 5."
        )
    actual_share = pd.to_numeric(predictions["phase3_test"], errors="raise")
    predicted_share = pd.to_numeric(
        predictions["phase3_pred_rounded"], errors="raise"
    )
    raw_predicted_share = pd.to_numeric(
        predictions["phase3_pred_raw"], errors="raise"
    )
    if not np.isfinite(actual_share.to_numpy(dtype=np.float64)).all():
        raise ValueError("Actual Phase 3+ shares must be finite.")
    if not np.isfinite(predicted_share.to_numpy(dtype=np.float64)).all():
        raise ValueError("Rounded predicted Phase 3+ shares must be finite.")
    if not np.isfinite(raw_predicted_share.to_numpy(dtype=np.float64)).all():
        raise ValueError("Raw predicted Phase 3+ shares must be finite.")
    nonpositive = predictions["nonpositive_cumulative_prediction_sum"]
    if nonpositive.isna().any() or not pd.api.types.is_bool_dtype(nonpositive.dtype):
        raise ValueError("Nonpositive cumulative prediction flags must be Boolean.")

    actual_positive = overall_actual.ge(3)
    predicted_positive = overall_predicted.ge(3)
    true_positive = int((actual_positive & predicted_positive).sum())
    false_positive = int((~actual_positive & predicted_positive).sum())
    false_negative = int((actual_positive & ~predicted_positive).sum())
    true_negative = int((~actual_positive & ~predicted_positive).sum())
    predicted_positive_count = true_positive + false_positive
    actual_positive_count = true_positive + false_negative
    precision = (
        true_positive / predicted_positive_count
        if predicted_positive_count
        else np.nan
    )
    recall = (
        true_positive / actual_positive_count if actual_positive_count else np.nan
    )
    precision_reason = (
        None if predicted_positive_count else "no_predicted_phase3plus"
    )
    recall_reason = None if actual_positive_count else "no_actual_phase3plus"

    if len(predictions) < 2:
        phase3plus_r2 = np.nan
        r2_reason = "insufficient_observations"
    elif actual_share.nunique() < 2:
        phase3plus_r2 = np.nan
        r2_reason = "constant_actual_phase3plus_share"
    else:
        phase3plus_r2 = float(r2_score(actual_share, predicted_share))
        r2_reason = None
    if len(predictions) < 2:
        phase3plus_r2_raw = np.nan
        r2_raw_reason = "insufficient_observations"
    elif actual_share.nunique() < 2:
        phase3plus_r2_raw = np.nan
        r2_raw_reason = "constant_actual_phase3plus_share"
    else:
        phase3plus_r2_raw = float(r2_score(actual_share, raw_predicted_share))
        r2_raw_reason = None

    current = {
        "phase3plus_precision": precision,
        "phase3plus_recall": recall,
        "overall_accuracy": float(
            accuracy_score(overall_actual, overall_predicted)
        ),
        "phase3plus_r2": phase3plus_r2,
    }
    metric_names = tuple(current)
    if baseline_values is not None:
        if set(baseline_values) != set(metric_names):
            raise ValueError("baseline_values has an unexpected metric set.")
        baseline = {name: baseline_values[name] for name in metric_names}
    elif condition == "baseline_with_lat_lon":
        baseline = dict(current)
    else:
        baseline = {name: np.nan for name in metric_names}

    def metric_delta(name: str) -> tuple[float, float]:
        value = current[name]
        reference = baseline[name]
        if pd.isna(value) or pd.isna(reference):
            return np.nan, np.nan
        signed = float(value - reference)
        return signed, abs(signed)

    precision_delta = metric_delta("phase3plus_precision")
    recall_delta = metric_delta("phase3plus_recall")
    accuracy_delta = metric_delta("overall_accuracy")
    r2_delta = metric_delta("phase3plus_r2")
    record = {
        "condition": condition,
        "condition_label": condition_label,
        "model": model_name,
        "evaluation_protocol": evaluation_protocol,
        "evaluation_population": evaluation_population,
        "n_splits": n_splits,
        "fold_assignment_sha256": (
            fold_assignment_sha256
            if model_name == CONTEMPORANEOUS_MODEL
            else pd.NA
        ),
        "n_test": int(len(predictions)),
        "n_test_areas": int(predictions["area_id"].nunique()),
        "n_test_countries": int(predictions["country_code_3"].nunique()),
        "test_key_sha256": test_key_sha256,
        "actual_phase3plus_count": actual_positive_count,
        "predicted_phase3plus_count": predicted_positive_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "nonpositive_cumulative_prediction_count": int(nonpositive.sum()),
        "phase3plus_precision": precision,
        "phase3plus_precision_undefined_reason": precision_reason,
        "baseline_phase3plus_precision": baseline["phase3plus_precision"],
        "phase3plus_precision_signed_delta": precision_delta[0],
        "phase3plus_precision_absolute_delta": precision_delta[1],
        "phase3plus_recall": recall,
        "phase3plus_recall_undefined_reason": recall_reason,
        "baseline_phase3plus_recall": baseline["phase3plus_recall"],
        "phase3plus_recall_signed_delta": recall_delta[0],
        "phase3plus_recall_absolute_delta": recall_delta[1],
        "overall_accuracy": current["overall_accuracy"],
        "baseline_overall_accuracy": baseline["overall_accuracy"],
        "overall_accuracy_signed_delta": accuracy_delta[0],
        "overall_accuracy_absolute_delta": accuracy_delta[1],
        "phase3plus_r2": phase3plus_r2,
        "phase3plus_r2_undefined_reason": r2_reason,
        "phase3plus_r2_raw": phase3plus_r2_raw,
        "phase3plus_r2_raw_undefined_reason": r2_raw_reason,
        "baseline_phase3plus_r2": baseline["phase3plus_r2"],
        "phase3plus_r2_signed_delta": r2_delta[0],
        "phase3plus_r2_absolute_delta": r2_delta[1],
    }
    if tuple(record) != METRIC_COLUMNS:
        raise ValueError("Comparison metric record has an unexpected schema.")
    return record


def build_metrics_table(
    predictions: pd.DataFrame,
    production_run: bool,
    contemporaneous_fold_assignment_sha256: str,
) -> pd.DataFrame:
    """Build deterministic condition-major, model-minor comparison metrics."""
    loco._require_columns(predictions, ["condition", "condition_label", "model"])
    if predictions.empty:
        raise ValueError("Metrics table requires prediction rows.")
    observed_conditions = set(predictions["condition"].dropna().astype(str))
    unknown_conditions = observed_conditions.difference(CONDITION_ORDER)
    if unknown_conditions:
        raise ValueError(f"Predictions contain unknown conditions: {sorted(unknown_conditions)}")
    included = tuple(
        condition for condition in CONDITION_ORDER if condition in observed_conditions
    )
    if production_run and included != CONDITION_ORDER:
        raise ValueError("Production metrics require all four conditions.")
    expected_groups = {
        (condition, model) for condition in included for model in MODEL_ORDER
    }
    observed_groups = set(
        predictions[["condition", "model"]].itertuples(index=False, name=None)
    )
    if observed_groups != expected_groups:
        raise ValueError("Prediction groups do not match included conditions and models.")

    baselines: dict[str, dict[str, float]] = {}
    if "baseline_with_lat_lon" in included:
        for model in TEMPORAL_MODEL_ORDER:
            baselines[model] = {
                name: float(MAIN_RESULT_REFERENCES[model][name])
                for name in REFERENCE_METRIC_NAMES
            }
        contemporaneous_baseline = predictions.loc[
            predictions["condition"].eq("baseline_with_lat_lon")
            & predictions["model"].eq(CONTEMPORANEOUS_MODEL)
        ]
        if contemporaneous_baseline.empty:
            raise ValueError("Contemporaneous controlled baseline predictions are missing.")
        baseline_record = calculate_comparison_metrics(
            contemporaneous_baseline,
            "baseline_with_lat_lon",
            CONDITION_LABELS["baseline_with_lat_lon"],
            CONTEMPORANEOUS_MODEL,
            baseline_values=None,
            fold_assignment_sha256=contemporaneous_fold_assignment_sha256,
        )
        baselines[CONTEMPORANEOUS_MODEL] = {
            name: float(baseline_record[name]) for name in REFERENCE_METRIC_NAMES
        }

    records = []
    key_hashes_by_model: dict[str, set[str]] = {
        model: set() for model in MODEL_ORDER
    }
    for condition in included:
        expected_label = CONDITION_LABELS[condition]
        for model in MODEL_ORDER:
            group = predictions.loc[
                predictions["condition"].eq(condition)
                & predictions["model"].eq(model)
            ]
            labels = group["condition_label"].drop_duplicates().tolist()
            if labels != [expected_label]:
                raise ValueError(
                    f"Condition label mismatch for {condition} {model}."
                )
            record = calculate_comparison_metrics(
                group,
                condition,
                expected_label,
                model,
                baselines.get(model),
                (
                    contemporaneous_fold_assignment_sha256
                    if model == CONTEMPORANEOUS_MODEL
                    else None
                ),
            )
            key_hashes_by_model[model].add(record["test_key_sha256"])
            records.append(record)
    if any(len(values) != 1 for values in key_hashes_by_model.values()):
        raise ValueError("Condition test-key hashes differ within a model protocol.")
    temporal_hashes = {
        next(iter(key_hashes_by_model[model])) for model in TEMPORAL_MODEL_ORDER
    }
    if len(temporal_hashes) != 1:
        raise ValueError("Forecasting and Nowcasting temporal test-key hashes differ.")
    if next(iter(key_hashes_by_model[CONTEMPORANEOUS_MODEL])) in temporal_hashes:
        raise ValueError("Temporal and random-CV populations unexpectedly share a key hash.")
    return pd.DataFrame.from_records(records, columns=METRIC_COLUMNS)


def assert_frozen_main_result_reproduced(metrics: pd.DataFrame) -> None:
    """Require the same-environment latitude/longitude baseline to match Figure 1."""
    for model in TEMPORAL_MODEL_ORDER:
        match = metrics.loc[
            metrics["condition"].eq("baseline_with_lat_lon")
            & metrics["model"].eq(model)
        ]
        if len(match) != 1:
            raise ValueError(f"Frozen baseline metric row is incomplete for {model}.")
        row = match.iloc[0]
        expected = frozen_main_result.RESULTS[model]
        for metric_name in REFERENCE_METRIC_NAMES:
            if not np.isclose(
                float(row[metric_name]),
                float(expected[metric_name]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Frozen baseline did not reproduce {model} {metric_name}."
                )
        count_expectations = {
            "true_positive": "true_positive",
            "false_positive": "false_positive",
            "false_negative": "false_negative",
            "true_negative": "true_negative",
        }
        for metric_column, result_key in count_expectations.items():
            if int(row[metric_column]) != int(expected[result_key]):
                raise ValueError(
                    f"Frozen baseline did not reproduce {model} {metric_column}."
                )
        correct_rows = int(round(float(row["overall_accuracy"]) * int(row["n_test"])))
        if correct_rows != int(expected["correct_rows"]):
            raise ValueError(f"Frozen baseline did not reproduce {model} correct rows.")


def build_combined_predictions(
    predictions_by_condition_model: Mapping[tuple[str, str], pd.DataFrame],
    conditions: Sequence[str],
    expected_test_rows: int = DEFAULT_EXPECTED_TEST_ROWS,
) -> pd.DataFrame:
    """Assemble deterministic predictions while preserving both populations."""
    selected_input = tuple(conditions)
    if not selected_input:
        raise ValueError("At least one condition is required for prediction assembly.")
    if len(set(selected_input)) != len(selected_input):
        raise ValueError("Prediction assembly conditions contain duplicates.")
    unknown = set(selected_input).difference(CONDITION_ORDER)
    if unknown:
        raise ValueError(f"Unknown prediction conditions: {sorted(unknown)}")
    selected = tuple(
        condition for condition in CONDITION_ORDER if condition in selected_input
    )
    expected_groups = {
        (condition, model) for condition in selected for model in MODEL_ORDER
    }
    if set(predictions_by_condition_model) != expected_groups:
        raise ValueError(
            "Prediction mapping does not match selected condition-model groups."
        )

    frames = []
    key_hashes_by_model: dict[str, set[str]] = {
        model: set() for model in MODEL_ORDER
    }
    for condition in selected:
        for model in MODEL_ORDER:
            frame = predictions_by_condition_model[(condition, model)].copy()
            if frame.columns.tolist() != list(PREDICTION_COLUMNS):
                raise ValueError(f"{condition} {model} has an unexpected schema.")
            expected_rows = (
                EXPECTED_CONTEMPORANEOUS_ROWS
                if model == CONTEMPORANEOUS_MODEL
                else expected_test_rows
            )
            expected_areas = (
                EXPECTED_CONTEMPORANEOUS_AREAS
                if model == CONTEMPORANEOUS_MODEL
                else DEFAULT_EXPECTED_TEST_AREAS
            )
            if len(frame) != expected_rows:
                raise ValueError(
                    f"{condition} {model} has {len(frame)} rows; "
                    f"expected {expected_rows}."
                )
            if frame["area_id"].nunique() != expected_areas:
                raise ValueError(
                    f"{condition} {model} has an unexpected evaluation-area population."
                )
            if frame.duplicated(["area_id", "date"]).any():
                raise ValueError(f"{condition} {model} contains duplicate test keys.")
            if not frame["condition"].eq(condition).all():
                raise ValueError(f"{condition} {model} contains another condition label.")
            if not frame["model"].eq(model).all():
                raise ValueError(f"{condition} {model} contains another model label.")
            if not frame["condition_label"].eq(CONDITION_LABELS[condition]).all():
                raise ValueError(f"{condition} {model} has an invalid condition label.")
            key_hashes_by_model[model].add(canonical_key_sha256(frame))
            frames.append(frame)
    if any(len(values) != 1 for values in key_hashes_by_model.values()):
        raise ValueError("Condition key hashes differ within a model protocol.")
    temporal_hashes = {
        next(iter(key_hashes_by_model[model])) for model in TEMPORAL_MODEL_ORDER
    }
    if len(temporal_hashes) != 1:
        raise ValueError("Forecasting and Nowcasting temporal test-key hashes differ.")
    if next(iter(key_hashes_by_model[CONTEMPORANEOUS_MODEL])) in temporal_hashes:
        raise ValueError("Temporal and random-CV prediction populations are not distinct.")

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.assign(
            _condition_rank=pd.Categorical(
                combined["condition"], CONDITION_ORDER, ordered=True
            ),
            _model_rank=pd.Categorical(
                combined["model"], MODEL_ORDER, ordered=True
            ),
        )
        .sort_values(
            ["_condition_rank", "_model_rank", "area_id", "date"],
            kind="mergesort",
        )
        .drop(columns=["_condition_rank", "_model_rank"])
        .reset_index(drop=True)
    )
    if (
        selected == CONDITION_ORDER
        and expected_test_rows == DEFAULT_EXPECTED_TEST_ROWS
        and len(combined) != 31660
    ):
        raise ValueError("Formal combined predictions must contain 31,660 rows.")
    return combined.loc[:, PREDICTION_COLUMNS]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_audit_for_publication(
    source_audit: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    production_run: bool,
) -> None:
    """Require source-audit group and run semantics to match publishable data."""
    required_columns = {
        "condition",
        "condition_label",
        "model",
        "production_run",
        "run_label",
        "evaluation_protocol",
        "evaluation_population",
        "n_splits",
        "fold_assignment_sha256",
        "n_test",
        "n_test_areas",
        "test_key_sha256",
    }
    missing = required_columns.difference(source_audit.columns)
    if missing:
        raise ValueError(f"Source audit is missing required columns: {sorted(missing)}")
    if source_audit.empty:
        raise ValueError("Source audit must contain one row per condition-model group.")
    if source_audit.duplicated(["condition", "model"]).any():
        raise ValueError("Source audit contains duplicate condition-model rows.")
    if metrics.duplicated(["condition", "model"]).any():
        raise ValueError("Source audit validation found duplicate metric groups.")

    prediction_groups = set(
        predictions[["condition", "model"]].itertuples(index=False, name=None)
    )
    metric_groups = set(
        metrics[["condition", "model"]].itertuples(index=False, name=None)
    )
    audit_groups = set(
        source_audit[["condition", "model"]].itertuples(index=False, name=None)
    )
    if metric_groups != prediction_groups:
        raise ValueError("Source audit validation found prediction/metric group drift.")
    if audit_groups != prediction_groups:
        raise ValueError("Source audit condition-model groups do not match predictions.")
    if any(
        condition not in CONDITION_LABELS or model not in MODEL_ORDER
        for condition, model in prediction_groups
    ):
        raise ValueError("Source audit contains an unknown condition or model.")
    formal_groups = {
        (condition, model) for condition in CONDITION_ORDER for model in MODEL_ORDER
    }
    if production_run and prediction_groups != formal_groups:
        raise ValueError("Invalid formal source audit: all twelve groups are required.")

    expected_run_label = "formal" if production_run else "diagnostic"

    def matches_run_flag(value: object) -> bool:
        return isinstance(value, (bool, np.bool_)) and bool(value) is bool(production_run)

    def matches_exact_count(value: object, expected: int) -> bool:
        return (
            not isinstance(value, (bool, np.bool_))
            and isinstance(value, (int, np.integer))
            and int(value) == expected
        )

    if not source_audit["production_run"].map(matches_run_flag).all():
        raise ValueError("Source audit production_run values do not match the run.")
    if not source_audit["run_label"].eq(expected_run_label).all():
        raise ValueError("Source audit run_label values do not match the run.")

    for condition, model in sorted(
        prediction_groups,
        key=lambda item: (CONDITION_ORDER.index(item[0]), MODEL_ORDER.index(item[1])),
    ):
        group = predictions.loc[
            predictions["condition"].eq(condition)
            & predictions["model"].eq(model)
        ]
        metric = metrics.loc[
            metrics["condition"].eq(condition) & metrics["model"].eq(model)
        ]
        audit = source_audit.loc[
            source_audit["condition"].eq(condition)
            & source_audit["model"].eq(model)
        ]
        if len(metric) != 1 or len(audit) != 1:
            raise ValueError("Source audit requires exactly one row for every group.")
        metric = metric.iloc[0]
        audit = audit.iloc[0]
        expected_label = CONDITION_LABELS[condition]
        expected_n_test = int(len(group))
        expected_key_hash = canonical_key_sha256(group)
        if metric["condition_label"] != expected_label:
            raise ValueError("Source audit validation found an invalid metric condition label.")
        if audit["condition_label"] != expected_label:
            raise ValueError("Source audit condition_label does not match the condition.")
        if not matches_exact_count(metric["n_test"], expected_n_test):
            raise ValueError("Source audit validation found an invalid metric n_test value.")
        if not matches_exact_count(audit["n_test"], expected_n_test):
            raise ValueError("Source audit n_test does not match saved predictions.")
        if metric["test_key_sha256"] != expected_key_hash:
            raise ValueError("Source audit validation found an invalid metric test-key hash.")
        if audit["test_key_sha256"] != expected_key_hash:
            raise ValueError("Source audit test_key_sha256 does not match predictions.")
        is_contemporaneous = model == CONTEMPORANEOUS_MODEL
        expected_protocol = (
            "random_5fold_row_cv"
            if is_contemporaneous
            else "fixed_2022_temporal_test"
        )
        expected_population = (
            "random_5fold_full_oof_5575"
            if is_contemporaneous
            else "temporal_test_2022_1170"
        )
        expected_areas = (
            EXPECTED_CONTEMPORANEOUS_AREAS
            if is_contemporaneous
            else DEFAULT_EXPECTED_TEST_AREAS
        )
        expected_splits = EXPECTED_CONTEMPORANEOUS_FOLDS if is_contemporaneous else 1
        if metric["evaluation_protocol"] != expected_protocol or audit[
            "evaluation_protocol"
        ] != expected_protocol:
            raise ValueError("Source audit evaluation protocol does not match the model.")
        if metric["evaluation_population"] != expected_population or audit[
            "evaluation_population"
        ] != expected_population:
            raise ValueError("Source audit evaluation population does not match the model.")
        if not matches_exact_count(metric["n_test_areas"], expected_areas) or not matches_exact_count(
            audit["n_test_areas"], expected_areas
        ):
            raise ValueError("Source audit evaluation-area count does not match the model.")
        if not matches_exact_count(metric["n_splits"], expected_splits) or not matches_exact_count(
            audit["n_splits"], expected_splits
        ):
            raise ValueError("Source audit split count does not match the model.")
        if is_contemporaneous:
            if not isinstance(metric["fold_assignment_sha256"], str) or metric[
                "fold_assignment_sha256"
            ] != audit["fold_assignment_sha256"]:
                raise ValueError("Contemporaneous fold hash differs across artifacts.")
        elif pd.notna(metric["fold_assignment_sha256"]) or pd.notna(
            audit["fold_assignment_sha256"]
        ):
            raise ValueError("Temporal rows must not carry a random-CV fold hash.")

    if production_run:
        formal_flags = (
            "coordinates_complete_passed",
            "coordinates_unique_within_area_passed",
            "coordinates_cross_table_equal_passed",
            "coordinate_area_count_passed",
            "knn5_contract_passed",
            "d200_contract_passed",
        )
        missing_flags = set(formal_flags).difference(source_audit.columns)
        if missing_flags or "temporal_violation_count" not in source_audit.columns:
            raise ValueError(
                "Invalid formal source audit: required contract checks are missing."
            )
        for column in formal_flags:
            passed = source_audit[column].map(
                lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
            )
            if not passed.all():
                raise ValueError(
                    f"Invalid formal source audit: {column} must be true for every row."
                )
        violations = pd.to_numeric(
            source_audit["temporal_violation_count"], errors="coerce"
        )
        if violations.isna().any() or not violations.eq(0).all():
            raise ValueError(
                "Invalid formal source audit: temporal violations must be zero."
            )
        if not source_audit["freeze_id"].eq(frozen_main_result.FREEZE_ID).all():
            raise ValueError("Invalid formal source audit: freeze ID drifted.")
        if not source_audit["reference_environment_id"].eq(
            frozen_main_result.ENVIRONMENT["environment_id"]
        ).all():
            raise ValueError("Invalid formal source audit: environment ID drifted.")
        if source_audit["xgboost_random_state_override"].notna().any():
            raise ValueError("Invalid formal source audit: XGBoost seed was overridden.")
        if source_audit["xgboost_n_jobs_override"].notna().any():
            raise ValueError("Invalid formal source audit: XGBoost threads were overridden.")
        if not source_audit["xgboost_uses_default_threads"].map(bool).all():
            raise ValueError("Invalid formal source audit: default XGBoost threads required.")
        if not pd.to_numeric(source_audit["workers_requested"], errors="coerce").eq(1).all():
            raise ValueError("Invalid formal source audit: outer execution must be serial.")
        contemporaneous_rows = source_audit["model"].eq(CONTEMPORANEOUS_MODEL)
        if int(contemporaneous_rows.sum()) != len(CONDITION_ORDER):
            raise ValueError("Invalid formal source audit: four Contemporaneous rows required.")
        if not pd.to_numeric(
            source_audit.loc[
                contemporaneous_rows, "contemporaneous_estimator_random_state"
            ],
            errors="coerce",
        ).eq(CONTEMPORANEOUS_RANDOM_STATE).all():
            raise ValueError("Invalid formal source audit: Contemporaneous seed drifted.")
        if not source_audit.loc[
            contemporaneous_rows, "contemporaneous_reference_reproduced"
        ].map(bool).all():
            raise ValueError(
                "Invalid formal source audit: canonical Contemporaneous baseline not reproduced."
            )


def build_source_audit(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    matrix_hashes: pd.DataFrame,
    weight_diagnostics: pd.DataFrame,
    interpolation_audit: pd.DataFrame,
    contemporaneous_features: Mapping[str, tuple[str, ...]],
    contemporaneous_fold_contract: ContemporaneousFoldContract,
    coordinate_validation: Mapping[str, object],
    input_paths: Mapping[str, Path],
    feature_manifest_sha256: str,
    random_state: int | None,
    workers: int,
    estimator_n_jobs: int | None,
    production_run: bool,
) -> pd.DataFrame:
    """Build one self-contained provenance row per condition and model."""
    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("Predictions have an unexpected schema for source audit.")
    if metrics.columns.tolist() != list(METRIC_COLUMNS):
        raise ValueError("Metrics have an unexpected schema for source audit.")
    if matrix_hashes.columns.tolist() != list(MATRIX_HASH_COLUMNS):
        raise ValueError("Matrix hash table has an unexpected schema.")
    if weight_diagnostics.columns.tolist() != list(WEIGHT_DIAGNOSTIC_COLUMNS):
        raise ValueError("Weight diagnostics have an unexpected schema.")
    if interpolation_audit.columns.tolist() != list(INTERPOLATION_AUDIT_COLUMNS):
        raise ValueError("Interpolation audit has an unexpected schema.")
    if matrix_hashes.duplicated(["condition", "layer", "split"]).any():
        raise ValueError("Matrix hash table contains duplicate condition-layer-split rows.")
    if metrics.duplicated(["condition", "model"]).any():
        raise ValueError("Metrics contain duplicate condition-model rows.")
    if set(contemporaneous_features) != set(CONDITION_ORDER):
        raise ValueError("Contemporaneous feature contract must cover all four conditions.")

    required_input_paths = {
        "forecasting_input",
        "nowcasting_input",
        "country_lookup",
        "general_params",
        "phase3_params",
        "contemporaneous_params",
        "contemporaneous_reference_predictions",
        "contemporaneous_reference_audit",
        "generator",
    }
    if set(input_paths) != required_input_paths:
        raise ValueError("Source-audit input paths do not match the required set.")
    input_paths = {name: Path(path) for name, path in input_paths.items()}
    input_hashes = {
        f"{name}_sha256": sha256_file(path) for name, path in input_paths.items()
    }

    raw_source = pd.read_csv(
        input_paths["forecasting_input"], usecols=["area_id", "date"]
    )
    raw_dates = pd.to_datetime(raw_source["date"], errors="coerce")
    if raw_dates.isna().any():
        raise ValueError("Forecasting source contains unparseable dates.")
    lookup_source = pd.read_csv(
        input_paths["country_lookup"], usecols=["area_id", "country_code_3"]
    )
    cutoff = pd.Timestamp(DEFAULT_CUTOFF)

    def matrix_value(condition: str, layer: str, split: str, column: str):
        match = matrix_hashes.loc[
            matrix_hashes["condition"].eq(condition)
            & matrix_hashes["layer"].eq(layer)
            & matrix_hashes["split"].eq(split),
            column,
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one matrix audit row for {condition} {layer} {split}."
            )
        return match.iloc[0]

    knn_rows = weight_diagnostics.loc[weight_diagnostics["scheme"].eq("knn5")]
    d200_rows = weight_diagnostics.loc[weight_diagnostics["scheme"].eq("d200")]
    knn_ok = bool(
        len(knn_rows) == DEFAULT_EXPECTED_AREAS
        and knn_rows["area_id"].nunique() == DEFAULT_EXPECTED_AREAS
        and knn_rows["neighbor_count"].eq(5).all()
        and ~knn_rows["zero_neighbor"].astype(bool).any()
    )
    d200_ok = bool(
        len(d200_rows) == DEFAULT_EXPECTED_AREAS
        and d200_rows["area_id"].nunique() == DEFAULT_EXPECTED_AREAS
        and d200_rows["neighbor_count"].gt(0).any()
        and d200_rows["max_distance_km"].dropna().le(200.0).all()
        and d200_rows["zero_neighbor"].astype(bool).eq(
            d200_rows["neighbor_count"].eq(0)
        ).all()
    )

    if interpolation_audit.empty:
        layer_counts: dict[tuple[object, object], int] = {}
        tier_counts: dict[tuple[object, object, object], int] = {}
        violation_counts: dict[tuple[object, object], int] = {}
    else:
        if interpolation_audit[
            ["condition", "layer", "source_tier", "temporal_contract_passed"]
        ].isna().any().any():
            raise ValueError("Interpolation audit contains missing required audit fields.")
        lineage_summary = (
            interpolation_audit.groupby(
                ["condition", "layer", "source_tier"],
                observed=True,
                sort=False,
            )["temporal_contract_passed"]
            .agg(event_count="size", passed_count="sum")
            .reset_index()
        )
        layer_counts = {}
        tier_counts = {}
        passed_counts: dict[tuple[object, object], int] = {}
        for row in lineage_summary.itertuples(index=False):
            layer_key = (row.condition, row.layer)
            event_count = int(row.event_count)
            passed_count = int(row.passed_count)
            layer_counts[layer_key] = layer_counts.get(layer_key, 0) + event_count
            passed_counts[layer_key] = passed_counts.get(layer_key, 0) + passed_count
            tier_counts[(row.condition, row.layer, row.source_tier)] = event_count
        violation_counts = {
            key: layer_counts[key] - passed_counts.get(key, 0)
            for key in layer_counts
        }

    def layer_event_count(condition: str, layer: str) -> int:
        return int(layer_counts.get((condition, layer), 0))

    def tier_event_count(condition: str, layer: str, tier: str) -> int:
        return int(tier_counts.get((condition, layer, tier), 0))

    def violation_event_count(condition: str, layer: str) -> int:
        return int(violation_counts.get((condition, layer), 0))

    reference_applied = "baseline_with_lat_lon" in set(predictions["condition"])
    controlled_baselines: dict[str, dict[str, float]] = {}
    if reference_applied:
        for model in MODEL_ORDER:
            baseline_metric = metrics.loc[
                metrics["condition"].eq("baseline_with_lat_lon")
                & metrics["model"].eq(model)
            ]
            baseline_predictions = predictions.loc[
                predictions["condition"].eq("baseline_with_lat_lon")
                & predictions["model"].eq(model)
            ]
            if len(baseline_metric) != 1 or baseline_predictions.empty:
                raise ValueError(
                    f"Controlled rerun baseline is incomplete for {model}."
                )
            baseline_metric = baseline_metric.iloc[0]
            controlled_baselines[model] = {
                name: float(baseline_metric[name]) for name in REFERENCE_METRIC_NAMES
            }
            controlled_baselines[model]["phase3plus_r2_raw"] = float(
                baseline_metric["phase3plus_r2_raw"]
            )

    records = []
    for condition in CONDITION_ORDER:
        if condition not in set(predictions["condition"]):
            continue
        for model in MODEL_ORDER:
            group = predictions.loc[
                predictions["condition"].eq(condition)
                & predictions["model"].eq(model)
            ]
            metric_match = metrics.loc[
                metrics["condition"].eq(condition) & metrics["model"].eq(model)
            ]
            if len(group) == 0 or len(metric_match) != 1:
                raise ValueError(
                    f"Source audit requires one prediction and metric group for "
                    f"{condition} {model}."
                )
            metric = metric_match.iloc[0]
            layer1_events = layer_event_count(condition, "layer1_shared")
            layer2_events = layer_event_count(condition, "nowcasting_layer2")
            uses_layer2 = model in {"Nowcasting", CONTEMPORANEOUS_MODEL}
            consumed_events = (
                layer1_events
                if model == "Forecasting"
                else layer1_events + layer2_events
            )
            temporal_violations = violation_event_count(
                condition, "layer1_shared"
            )
            if uses_layer2:
                temporal_violations += violation_event_count(
                    condition, "nowcasting_layer2"
                )
            main_result_reference = MAIN_RESULT_REFERENCES.get(model)
            controlled_baseline = controlled_baselines.get(model, {})
            is_contemporaneous = model == CONTEMPORANEOUS_MODEL
            evaluation_protocol = (
                "random_5fold_row_cv"
                if is_contemporaneous
                else "fixed_2022_temporal_test"
            )
            evaluation_population = (
                "random_5fold_full_oof_5575"
                if is_contemporaneous
                else "temporal_test_2022_1170"
            )
            records.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABELS[condition],
                    "model": model,
                    "production_run": bool(production_run),
                    "run_label": "formal" if production_run else "diagnostic",
                    "evaluation_protocol": evaluation_protocol,
                    "evaluation_population": evaluation_population,
                    "validation_design": (
                        "random_row_cv" if is_contemporaneous else "fixed_temporal_holdout"
                    ),
                    "cutoff": pd.NA if is_contemporaneous else DEFAULT_CUTOFF,
                    "source_rows": int(len(raw_source)),
                    "train_rows": (
                        EXPECTED_CONTEMPORANEOUS_ROWS
                        - EXPECTED_CONTEMPORANEOUS_ROWS_PER_FOLD
                        if is_contemporaneous
                        else int(raw_dates.lt(cutoff).sum())
                    ),
                    "test_rows": (
                        EXPECTED_CONTEMPORANEOUS_ROWS
                        if is_contemporaneous
                        else int(raw_dates.ge(cutoff).sum())
                    ),
                    "n_splits": (
                        EXPECTED_CONTEMPORANEOUS_FOLDS if is_contemporaneous else 1
                    ),
                    "fold_rows": (
                        "|".join(
                            [str(EXPECTED_CONTEMPORANEOUS_ROWS_PER_FOLD)]
                            * EXPECTED_CONTEMPORANEOUS_FOLDS
                        )
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "fold_assignment_sha256": (
                        contemporaneous_fold_contract.fold_assignment_sha256
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "source_row_index_sha256": (
                        contemporaneous_fold_contract.source_row_index_sha256
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "population_key_sha256": (
                        contemporaneous_fold_contract.population_key_sha256
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "shuffle_seed": (
                        CONTEMPORANEOUS_RANDOM_STATE if is_contemporaneous else pd.NA
                    ),
                    "kfold_shuffle": False if is_contemporaneous else pd.NA,
                    "kfolds_predictor_included": (
                        True if is_contemporaneous else False
                    ),
                    "source_areas": int(raw_source["area_id"].nunique()),
                    "source_countries": int(
                        lookup_source["country_code_3"].nunique()
                    ),
                    **coordinate_validation,
                    "knn5_diagnostic_rows": int(len(knn_rows)),
                    "d200_diagnostic_rows": int(len(d200_rows)),
                    "d200_zero_neighbor_count": int(
                        d200_rows["neighbor_count"].eq(0).sum()
                    ),
                    "knn5_contract_passed": knn_ok,
                    "d200_contract_passed": d200_ok,
                    "freeze_id": frozen_main_result.FREEZE_ID,
                    "reference_environment_id": frozen_main_result.ENVIRONMENT[
                        "environment_id"
                    ],
                    "xgboost_random_state_override": random_state,
                    "contemporaneous_estimator_random_state": (
                        CONTEMPORANEOUS_RANDOM_STATE if is_contemporaneous else pd.NA
                    ),
                    "workers_requested": int(workers),
                    "max_parallel_models": min(len(MODEL_ORDER), int(workers)),
                    "process_start_method": "spawn" if workers > 1 else "serial",
                    "xgboost_n_jobs_override": estimator_n_jobs,
                    "xgboost_uses_default_threads": estimator_n_jobs is None,
                    "python_version": platform.python_version(),
                    "pandas_version": pd.__version__,
                    "numpy_version": np.__version__,
                    "sklearn_version": sklearn.__version__,
                    "xgboost_version": xgb.__version__,
                    "matplotlib_version": mpl.__version__,
                    "test_key_sha256": metric["test_key_sha256"],
                    "n_test": int(metric["n_test"]),
                    "n_test_areas": int(metric["n_test_areas"]),
                    "n_test_countries": int(metric["n_test_countries"]),
                    "layer1_feature_count": int(
                        matrix_value(
                            condition,
                            "layer1_shared",
                            "full_oof" if is_contemporaneous else "test",
                            "feature_count",
                        )
                    ),
                    "layer2_feature_count": (
                        int(
                            matrix_value(
                                condition,
                                "nowcasting_layer2",
                                "full_oof" if is_contemporaneous else "test",
                                "feature_count",
                            )
                        )
                        if uses_layer2
                        else np.nan
                    ),
                    "contemporaneous_feature_count": (
                        len(contemporaneous_features[condition])
                        if is_contemporaneous
                        else np.nan
                    ),
                    "contemporaneous_feature_order_sha256": (
                        hashlib.sha256(
                            "\n".join(contemporaneous_features[condition]).encode("utf-8")
                        ).hexdigest()
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "layer1_train_matrix_sha256": (
                        matrix_value(
                            condition, "layer1_shared", "train", "matrix_sha256"
                        )
                        if not is_contemporaneous
                        else pd.NA
                    ),
                    "layer1_test_matrix_sha256": (
                        matrix_value(
                            condition, "layer1_shared", "test", "matrix_sha256"
                        )
                        if not is_contemporaneous
                        else pd.NA
                    ),
                    "layer1_full_oof_matrix_sha256": (
                        matrix_value(
                            condition, "layer1_shared", "full_oof", "matrix_sha256"
                        )
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "layer2_train_matrix_sha256": (
                        matrix_value(
                            condition,
                            "nowcasting_layer2",
                            "train",
                            "matrix_sha256",
                        )
                        if model == "Nowcasting"
                        else np.nan
                    ),
                    "layer2_test_matrix_sha256": (
                        matrix_value(
                            condition,
                            "nowcasting_layer2",
                            "test",
                            "matrix_sha256",
                        )
                        if model == "Nowcasting"
                        else np.nan
                    ),
                    "layer2_full_oof_matrix_sha256": (
                        matrix_value(
                            condition,
                            "nowcasting_layer2",
                            "full_oof",
                            "matrix_sha256",
                        )
                        if is_contemporaneous
                        else pd.NA
                    ),
                    "consumed_interpolation_event_count": consumed_events,
                    "layer1_interpolation_event_count": layer1_events,
                    "layer1_own_history_event_count": tier_event_count(
                        condition, "layer1_shared", "own_history"
                    ),
                    "layer1_same_country_event_count": tier_event_count(
                        condition, "layer1_shared", "same_country"
                    ),
                    "layer1_global_event_count": tier_event_count(
                        condition, "layer1_shared", "global"
                    ),
                    "layer2_interpolation_event_count": (
                        layer2_events if uses_layer2 else 0
                    ),
                    "layer2_own_history_event_count": (
                        tier_event_count(
                            condition, "nowcasting_layer2", "own_history"
                        )
                        if uses_layer2
                        else 0
                    ),
                    "layer2_same_country_event_count": (
                        tier_event_count(
                            condition, "nowcasting_layer2", "same_country"
                        )
                        if uses_layer2
                        else 0
                    ),
                    "layer2_global_event_count": (
                        tier_event_count(
                            condition, "nowcasting_layer2", "global"
                        )
                        if uses_layer2
                        else 0
                    ),
                    "temporal_violation_count": int(temporal_violations),
                    "metric_reference_type": (
                        "canonical_contemporaneous_random_cv_baseline"
                        if is_contemporaneous and reference_applied
                        else "frozen_main_result_same_environment"
                        if reference_applied
                        else "not_applied_no_baseline_condition"
                    ),
                    "metric_reference_applied": bool(reference_applied),
                    "cross_protocol_directly_comparable": False,
                    "contemporaneous_reference_reproduced": (
                        True if is_contemporaneous else pd.NA
                    ),
                    "main_result_notebook_path": (
                        main_result_reference["notebook_path"]
                        if main_result_reference is not None
                        else pd.NA
                    ),
                    "main_result_notebook_cell_index": (
                        int(main_result_reference["cell_index"])
                        if main_result_reference is not None
                        else pd.NA
                    ),
                    "main_result_phase3plus_precision": (
                        float(main_result_reference["phase3plus_precision"])
                        if main_result_reference is not None
                        else np.nan
                    ),
                    "main_result_phase3plus_recall": (
                        float(main_result_reference["phase3plus_recall"])
                        if main_result_reference is not None
                        else np.nan
                    ),
                    "main_result_overall_accuracy": (
                        float(main_result_reference["overall_accuracy"])
                        if main_result_reference is not None
                        else np.nan
                    ),
                    "main_result_phase3plus_r2": (
                        float(main_result_reference["phase3plus_r2"])
                        if main_result_reference is not None
                        else np.nan
                    ),
                    "same_environment_baseline_phase3plus_precision": (
                        controlled_baseline.get("phase3plus_precision", np.nan)
                    ),
                    "same_environment_baseline_phase3plus_recall": (
                        controlled_baseline.get("phase3plus_recall", np.nan)
                    ),
                    "same_environment_baseline_overall_accuracy": (
                        controlled_baseline.get("overall_accuracy", np.nan)
                    ),
                    "same_environment_baseline_phase3plus_r2": (
                        controlled_baseline.get("phase3plus_r2", np.nan)
                    ),
                    "same_environment_baseline_phase3plus_r2_raw": (
                        controlled_baseline.get("phase3plus_r2_raw", np.nan)
                    ),
                    "freeze_source_path": str(Path(frozen_main_result.__file__)),
                    "freeze_source_sha256": sha256_file(
                        Path(frozen_main_result.__file__)
                    ),
                    "feature_manifest_sha256": feature_manifest_sha256,
                    **input_hashes,
                }
            )
    return pd.DataFrame.from_records(records)


def _validate_spatial_feature_figure_metrics(
    metrics: pd.DataFrame,
    condition_order: Sequence[str],
    model_order: Sequence[str],
) -> pd.DataFrame:
    """Validate every plotted value, baseline, and delta before rendering."""
    condition_order = tuple(condition_order)
    model_order = tuple(model_order)
    if condition_order != EXPERIMENT_CONDITIONS:
        raise ValueError("Figure condition order must match the three approved experiments.")
    if model_order != MODEL_ORDER:
        raise ValueError(
            "Figure model order must be Forecasting, Nowcasting, then Contemporaneous."
        )

    metric_specs = (
        (
            "phase3plus_precision",
            "baseline_phase3plus_precision",
            "phase3plus_precision_signed_delta",
            "phase3plus_precision_absolute_delta",
        ),
        (
            "phase3plus_recall",
            "baseline_phase3plus_recall",
            "phase3plus_recall_signed_delta",
            "phase3plus_recall_absolute_delta",
        ),
        (
            "overall_accuracy",
            "baseline_overall_accuracy",
            "overall_accuracy_signed_delta",
            "overall_accuracy_absolute_delta",
        ),
        (
            "phase3plus_r2",
            "baseline_phase3plus_r2",
            "phase3plus_r2_signed_delta",
            "phase3plus_r2_absolute_delta",
        ),
    )
    value_columns = [column for spec in metric_specs for column in spec]
    required = {
        "condition",
        "condition_label",
        "model",
        "n_test",
        "n_test_areas",
        "test_key_sha256",
        *value_columns,
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Figure metrics are missing columns: {sorted(missing)}")
    if metrics.duplicated(["condition", "model"]).any():
        raise ValueError("Figure metrics contain duplicate condition-model rows.")
    expected_groups = {
        (condition, model) for condition in CONDITION_ORDER for model in MODEL_ORDER
    }
    observed_groups = set(
        metrics[["condition", "model"]].itertuples(index=False, name=None)
    )
    if observed_groups != expected_groups:
        raise ValueError("Figure metrics do not contain the complete twelve groups.")

    plotting = metrics.copy()
    expected_labels = plotting["condition"].map(CONDITION_LABELS)
    if expected_labels.isna().any() or not plotting["condition_label"].eq(
        expected_labels
    ).all():
        raise ValueError("Figure metrics contain an invalid condition label.")
    n_test = pd.to_numeric(plotting["n_test"], errors="coerce")
    n_test_areas = pd.to_numeric(plotting["n_test_areas"], errors="coerce")
    if n_test.isna().any() or n_test_areas.isna().any():
        raise ValueError("Figure evaluation-population counts must be numeric.")
    temporal_rows = plotting["model"].isin(TEMPORAL_MODEL_ORDER)
    contemporaneous_rows = plotting["model"].eq(CONTEMPORANEOUS_MODEL)
    if not n_test.loc[temporal_rows].eq(DEFAULT_EXPECTED_TEST_ROWS).all() or not n_test_areas.loc[
        temporal_rows
    ].eq(DEFAULT_EXPECTED_TEST_AREAS).all():
        raise ValueError("Figure temporal metrics must use 1,170 rows and 646 areas.")
    if not n_test.loc[contemporaneous_rows].eq(
        EXPECTED_CONTEMPORANEOUS_ROWS
    ).all() or not n_test_areas.loc[contemporaneous_rows].eq(
        EXPECTED_CONTEMPORANEOUS_AREAS
    ).all():
        raise ValueError(
            "Figure Contemporaneous metrics must use 5,575 OOF rows and 1,198 areas."
        )
    if plotting["test_key_sha256"].isna().any():
        raise ValueError("Figure metrics contain missing population-key hashes.")
    temporal_hashes = plotting.loc[temporal_rows, "test_key_sha256"].unique()
    contemporaneous_hashes = plotting.loc[
        contemporaneous_rows, "test_key_sha256"
    ].unique()
    if len(temporal_hashes) != 1 or len(contemporaneous_hashes) != 1:
        raise ValueError("Figure key hashes drift within an evaluation protocol.")
    if temporal_hashes[0] == contemporaneous_hashes[0]:
        raise ValueError("Figure temporal and random-CV populations are not distinct.")

    for column in value_columns:
        plotting[column] = pd.to_numeric(plotting[column], errors="coerce")
        if not np.isfinite(plotting[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Figure metric {column} must contain finite values.")
    for column in (
        "phase3plus_precision",
        "baseline_phase3plus_precision",
        "phase3plus_recall",
        "baseline_phase3plus_recall",
        "overall_accuracy",
        "baseline_overall_accuracy",
    ):
        if not plotting[column].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"Figure classification metric {column} is outside [0, 1].")
    reason_columns = (
        "phase3plus_precision_undefined_reason",
        "phase3plus_recall_undefined_reason",
        "phase3plus_r2_undefined_reason",
        "phase3plus_r2_raw_undefined_reason",
    )
    present_reason_columns = [
        column for column in reason_columns if column in plotting.columns
    ]
    if present_reason_columns and plotting[present_reason_columns].notna().any().any():
        raise ValueError("Figure cannot plot metrics marked as undefined.")

    indexed = plotting.set_index(["condition", "model"], drop=False)
    for model in MODEL_ORDER:
        if model == CONTEMPORANEOUS_MODEL:
            expected_reference = {
                metric: float(
                    indexed.loc[("baseline_with_lat_lon", model), metric]
                )
                for metric, _, _, _ in metric_specs
            }
        else:
            expected_reference = {
                metric: float(MAIN_RESULT_REFERENCES[model][metric])
                for metric, _, _, _ in metric_specs
            }
        for metric, baseline_column, signed_column, absolute_column in metric_specs:
            expected_baseline = expected_reference[metric]
            for condition in CONDITION_ORDER:
                row = indexed.loc[(condition, model)]
                current = float(row[metric])
                baseline = float(row[baseline_column])
                signed = float(row[signed_column])
                absolute = float(row[absolute_column])
                if not np.isclose(
                    baseline,
                    expected_baseline,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Figure baseline drift for {condition} {model} {metric}."
                    )
                expected_signed = current - expected_baseline
                if not np.isclose(signed, expected_signed, rtol=0.0, atol=1e-12):
                    raise ValueError(
                        f"Figure signed delta drift for {condition} {model} {metric}."
                    )
                if not np.isclose(
                    absolute,
                    abs(expected_signed),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Figure absolute delta drift for {condition} {model} {metric}."
                    )
    return plotting


def create_spatial_feature_comparison_figure(
    metrics: pd.DataFrame,
    condition_order: Sequence[str] = EXPERIMENT_CONDITIONS,
    model_order: Sequence[str] = MODEL_ORDER,
) -> plt.Figure:
    """Create the formal 3 x 4 spatial-feature metric comparison figure."""
    plotting = _validate_spatial_feature_figure_metrics(
        metrics,
        condition_order,
        model_order,
    ).set_index(["condition", "model"])
    condition_order = tuple(condition_order)
    model_order = tuple(model_order)
    metric_specs = (
        (
            "phase3plus_precision",
            "baseline_phase3plus_precision",
            "Phase 3+ precision",
        ),
        (
            "phase3plus_recall",
            "baseline_phase3plus_recall",
            "Phase 3+ recall",
        ),
        (
            "overall_accuracy",
            "baseline_overall_accuracy",
            "Overall-phase accuracy",
        ),
        (
            "phase3plus_r2",
            "baseline_phase3plus_r2",
            "Phase 3+ share R²",
        ),
    )

    column_limits: dict[str, tuple[float, float]] = {}
    for metric, baseline_column, _ in metric_specs:
        values = []
        for condition in condition_order:
            for model in model_order:
                row = plotting.loc[(condition, model)]
                values.extend((float(row[metric]), float(row[baseline_column])))
        values_array = np.asarray(values, dtype=float)
        data_min = float(values_array.min())
        data_max = float(values_array.max())
        if metric != "phase3plus_r2":
            span = max(data_max - data_min, 0.02)
            lower = data_min - span * 0.55
            upper = data_max + span * 0.85
            if upper - lower < 0.04:
                midpoint = (upper + lower) / 2.0
                lower = midpoint - 0.02
                upper = midpoint + 0.02
        else:
            data_min = min(data_min, 0.0)
            data_max = max(data_max, 0.0)
            span = max(data_max - data_min, 0.10)
            lower = data_min - span * 0.15
            upper = data_max + span * 0.22
        column_limits[metric] = (lower, upper)

    loco.apply_figure_style()
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    figure, axes = plt.subplots(3, 4, figsize=(12.2, 7.3), squeeze=False)
    colors = {
        "Forecasting": "#0072B2",
        "Nowcasting": "#E69F00",
        CONTEMPORANEOUS_MODEL: "#009E73",
    }
    markers = {
        "Forecasting": "o",
        "Nowcasting": "s",
        CONTEMPORANEOUS_MODEL: "D",
    }
    x_positions = {model: index for index, model in enumerate(model_order)}

    for row_index, condition in enumerate(condition_order):
        for column_index, (metric, baseline_column, title) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            lower, upper = column_limits[metric]
            axis.set_ylim(lower, upper)
            if metric != "phase3plus_r2":
                bounded_ticks = [
                    tick for tick in axis.get_yticks() if 0.0 <= tick <= 1.0
                ]
                axis.set_yticks(bounded_ticks)
            axis.set_xlim(-0.25, len(model_order) - 0.75)
            axis.set_title(title, loc="left", fontweight="normal", pad=5)
            for model in model_order:
                x_position = x_positions[model]
                row = plotting.loc[(condition, model)]
                experiment_value = float(row[metric])
                baseline_value = float(row[baseline_column])
                axis.plot(
                    [x_position, x_position],
                    [baseline_value, experiment_value],
                    color="#9A9A9A",
                    linewidth=0.7,
                    zorder=1,
                )
                axis.scatter(
                    [x_position],
                    [experiment_value],
                    s=48,
                    color=colors[model],
                    marker=markers[model],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )
                axis.scatter(
                    [x_position],
                    [baseline_value],
                    s=48,
                    facecolors="none",
                    edgecolors="#7A7A7A",
                    marker=markers[model],
                    linewidth=1.0,
                    zorder=4,
                )
                y_span = upper - lower
                near_top = experiment_value >= upper - y_span * 0.13
                near_bottom = experiment_value <= lower + y_span * 0.13
                if baseline_value > experiment_value:
                    vertical_offset = -7
                    vertical_alignment = "top"
                else:
                    vertical_offset = 7
                    vertical_alignment = "bottom"
                if near_top and vertical_offset > 0:
                    vertical_offset = -7
                    vertical_alignment = "top"
                elif near_bottom and vertical_offset < 0:
                    vertical_offset = 7
                    vertical_alignment = "bottom"
                if model == "Forecasting":
                    horizontal_offset = 6
                    horizontal_alignment = "left"
                elif model == "Nowcasting":
                    horizontal_offset = 0
                    horizontal_alignment = "center"
                else:
                    horizontal_offset = -6
                    horizontal_alignment = "right"
                axis.annotate(
                    f"{experiment_value:.3f}",
                    (x_position, experiment_value),
                    xytext=(horizontal_offset, vertical_offset),
                    textcoords="offset points",
                    ha=horizontal_alignment,
                    va=vertical_alignment,
                    fontsize=7,
                    color="#222222",
                    zorder=5,
                )

            axis.set_xticks(
                list(range(len(model_order))),
                ["Forecast", "Nowcast", "Contemp."],
            )
            axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.7)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if column_index == 0:
                axis.set_ylabel(CONDITION_LABELS[condition], labelpad=24)
            axis.text(
                -0.17,
                1.16,
                chr(ord("a") + row_index * 4 + column_index),
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
                clip_on=False,
            )
            if metric == "phase3plus_r2":
                axis.axhline(0.0, color="#777777", linewidth=0.6, zorder=0)

    experiment_handle = mpl.lines.Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=5.5,
        markerfacecolor="#0072B2",
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="Condition result",
    )
    baseline_handle = mpl.lines.Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=5.5,
        markerfacecolor="none",
        markeredgecolor="#7A7A7A",
        markeredgewidth=1.0,
        label="Model reference",
    )
    figure.legend(
        handles=[experiment_handle, baseline_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.6,
    )
    figure.suptitle(
        "Spatial feature comparison",
        x=0.105,
        ha="left",
        fontsize=9,
        fontweight="normal",
    )
    figure.text(
        0.105,
        0.925,
        "Forecast/Nowcast: 2022 temporal holdout (n=1,170); "
        "Contemp.: random 5-fold full OOF (n=5,575; not directly comparable)",
        ha="left",
        va="top",
        fontsize=7.5,
        color="#333333",
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.99,
        bottom=0.105,
        top=0.875,
        wspace=0.30,
        hspace=0.48,
    )
    return figure


def save_spatial_feature_comparison_figure(
    figure: plt.Figure,
    output_dir: Path,
) -> Mapping[str, Path]:
    """Save the formal comparison figure as 300-DPI raster and vector files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        f"figure_{suffix}": output_dir / filename
        for suffix, filename in zip(
            ("jpg", "png", "pdf"),
            FORMAL_FIGURE_FILENAMES,
        )
    }
    figure.savefig(
        paths["figure_jpg"],
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        paths["figure_png"],
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        paths["figure_pdf"],
        bbox_inches="tight",
        facecolor="white",
    )
    return paths


def validate_staged_artifacts(
    staged_paths: Mapping[str, Path],
    production_run: bool,
) -> None:
    """Reread staged artifacts without duplicating the full lineage in memory."""
    required = {
        "predictions_csv",
        "metrics_csv",
        "feature_manifest_csv",
        "weight_diagnostics_csv",
        "interpolation_audit_csv_gz",
        "interpolation_summary_csv",
    }
    if not required.issubset(staged_paths):
        raise ValueError("Staged artifact mapping is incomplete.")
    predictions = pd.read_csv(
        staged_paths["predictions_csv"], parse_dates=["date"]
    )
    metrics = pd.read_csv(staged_paths["metrics_csv"])
    manifest = pd.read_csv(staged_paths["feature_manifest_csv"])
    weights = pd.read_csv(staged_paths["weight_diagnostics_csv"])
    audit_header = pd.read_csv(
        staged_paths["interpolation_audit_csv_gz"],
        compression="gzip",
        nrows=0,
    )
    with gzip.open(staged_paths["interpolation_audit_csv_gz"], "rb") as file:
        while file.read(1024 * 1024):
            pass
    summary = pd.read_csv(staged_paths["interpolation_summary_csv"])

    if predictions.columns.tolist() != list(PREDICTION_COLUMNS):
        raise ValueError("Staged predictions have an unexpected schema.")
    if metrics.columns.tolist() != list(METRIC_COLUMNS):
        raise ValueError("Staged metrics have an unexpected schema.")
    if manifest.columns.tolist() != list(FEATURE_MANIFEST_COLUMNS):
        raise ValueError("Staged feature manifest has an unexpected schema.")
    if weights.columns.tolist() != list(WEIGHT_DIAGNOSTIC_COLUMNS):
        raise ValueError("Staged weight diagnostics have an unexpected schema.")
    if audit_header.columns.tolist() != list(INTERPOLATION_AUDIT_COLUMNS):
        raise ValueError("Staged interpolation audit has an unexpected schema.")
    if summary.columns.tolist() != list(INTERPOLATION_SUMMARY_COLUMNS):
        raise ValueError("Staged interpolation summary has an unexpected schema.")
    if predictions.duplicated(
        ["condition", "model", "area_id", "date"]
    ).any():
        raise ValueError("Staged predictions contain duplicate keys.")
    if predictions["nonpositive_cumulative_prediction_sum"].astype(bool).any():
        raise ValueError("Staged predictions contain nonpositive cumulative sums.")
    group_area_counts = predictions.groupby(["condition", "model"])["area_id"].nunique()
    for (_, model), area_count in group_area_counts.items():
        expected_areas = (
            EXPECTED_CONTEMPORANEOUS_AREAS
            if model == CONTEMPORANEOUS_MODEL
            else DEFAULT_EXPECTED_TEST_AREAS
        )
        if int(area_count) != expected_areas:
            raise ValueError(
                f"Staged {model} prediction group has an unexpected area count."
            )

    contemporaneous_fold_hashes = metrics.loc[
        metrics["model"].eq(CONTEMPORANEOUS_MODEL), "fold_assignment_sha256"
    ].dropna().unique()
    if len(contemporaneous_fold_hashes) != 1:
        raise ValueError("Staged metrics do not contain one Contemporaneous fold hash.")
    recomputed = build_metrics_table(
        predictions,
        production_run=production_run,
        contemporaneous_fold_assignment_sha256=str(contemporaneous_fold_hashes[0]),
    )
    reason_columns = [
        column
        for column in METRIC_COLUMNS
        if column.endswith("_undefined_reason")
    ]
    for column in reason_columns:
        metrics[column] = metrics[column].fillna("<NA>").astype(str)
        recomputed[column] = recomputed[column].fillna("<NA>").astype(str)
    pd.testing.assert_frame_equal(
        metrics.reset_index(drop=True),
        recomputed.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=0.0,
        atol=1e-12,
    )
    for row in summary.itertuples(index=False):
        if row.observed_slots + row.original_missing_slots != row.total_neighbor_slots:
            raise ValueError("Observed/missing slot identity failed.")
        if row.imputed_slots + row.remaining_missing_slots != row.original_missing_slots:
            raise ValueError("Imputed/remaining slot identity failed.")
        if row.observed_slots + row.imputed_slots != row.effective_nonmissing_slots:
            raise ValueError("Effective slot identity failed.")

    figure_keys = {key for key in staged_paths if key.startswith("figure_")}
    if production_run:
        if figure_keys != {"figure_jpg", "figure_png", "figure_pdf"}:
            raise ValueError("Formal run is missing a figure format.")
        if len(predictions) != 31660 or len(metrics) != 12:
            raise ValueError("Formal output row counts are incorrect.")
        metric_areas = pd.to_numeric(metrics["n_test_areas"], errors="coerce")
        temporal_rows = metrics["model"].isin(TEMPORAL_MODEL_ORDER)
        contemporaneous_rows = metrics["model"].eq(CONTEMPORANEOUS_MODEL)
        if not metric_areas.loc[temporal_rows].eq(
            DEFAULT_EXPECTED_TEST_AREAS
        ).all() or not metric_areas.loc[contemporaneous_rows].eq(
            EXPECTED_CONTEMPORANEOUS_AREAS
        ).all():
            raise ValueError("Formal metrics contain incorrect protocol-specific areas.")
        if not pd.to_numeric(
            metrics["nonpositive_cumulative_prediction_count"], errors="coerce"
        ).eq(0).all():
            raise ValueError("Formal metrics contain nonpositive cumulative predictions.")
    elif figure_keys:
        raise ValueError("Non-production run must not stage a formal figure.")


def _replace_with_retry(source: Path, destination: Path) -> None:
    """Retry Windows atomic replacement while sync/indexing releases file locks."""
    last_error: OSError | None = None
    for attempt in range(20):
        try:
            Path(source).replace(destination)
            return
        except OSError as error:
            last_error = error
            if attempt == 19:
                break
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


def write_artifacts(
    *,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    weight_diagnostics: pd.DataFrame,
    interpolation_audit: pd.DataFrame,
    interpolation_summary: pd.DataFrame,
    source_audit: pd.DataFrame,
    output_dir: Path,
    production_run: bool,
) -> Mapping[str, Path]:
    """Stage, validate, and transactionally publish the comparison artifact set."""
    _validate_source_audit_for_publication(
        source_audit,
        predictions,
        metrics,
        production_run,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not production_run:
        stale_figures = [
            output_dir / filename
            for filename in FORMAL_FIGURE_FILENAMES
            if (output_dir / filename).exists()
        ]
        if stale_figures:
            names = ", ".join(path.name for path in stale_figures)
            raise ValueError(
                "Subset runs require a clean diagnostic output directory without "
                f"formal figure artifacts; found: {names}"
            )
    with tempfile.TemporaryDirectory(
        prefix=".spatial_feature_comparison_staging_",
    ) as staging_name:
        staging = Path(staging_name)
        predictions_path = staging / "spatial_feature_comparison_predictions.csv"
        predictions.to_csv(
            predictions_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        interpolation_audit_path = (
            staging / "spatial_feature_interpolation_audit.csv.gz"
        )
        interpolation_audit.to_csv(
            interpolation_audit_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
            compression={"method": "gzip", "mtime": 0},
        )
        frame_paths = {
            "metrics_csv": (
                metrics,
                staging / "spatial_feature_comparison_metrics.csv",
            ),
            "feature_manifest_csv": (
                feature_manifest,
                staging / "spatial_feature_comparison_feature_manifest.csv",
            ),
            "weight_diagnostics_csv": (
                weight_diagnostics,
                staging / "spatial_feature_comparison_weight_diagnostics.csv",
            ),
            "interpolation_summary_csv": (
                interpolation_summary,
                staging / "spatial_feature_interpolation_summary.csv",
            ),
        }
        for frame, path in frame_paths.values():
            frame.to_csv(
                path,
                index=False,
                float_format="%.17g",
                lineterminator="\n",
            )
        staged_paths: dict[str, Path] = {
            "predictions_csv": predictions_path,
            "metrics_csv": frame_paths["metrics_csv"][1],
            "feature_manifest_csv": frame_paths["feature_manifest_csv"][1],
            "weight_diagnostics_csv": frame_paths["weight_diagnostics_csv"][1],
            "interpolation_audit_csv_gz": interpolation_audit_path,
            "interpolation_summary_csv": frame_paths[
                "interpolation_summary_csv"
            ][1],
        }
        if production_run:
            figure = create_spatial_feature_comparison_figure(metrics)
            try:
                staged_paths.update(
                    save_spatial_feature_comparison_figure(figure, staging)
                )
            finally:
                plt.close(figure)
        validate_staged_artifacts(staged_paths, production_run)

        audited = source_audit.copy()
        for name, path in staged_paths.items():
            audited[f"{name}_path"] = Path(path).name
            audited[f"{name}_sha256"] = sha256_file(path)
        audit_path = staging / "spatial_feature_comparison_source_audit.csv"
        audited.to_csv(
            audit_path,
            index=False,
            float_format="%.17g",
            lineterminator="\n",
        )
        audit_check = pd.read_csv(audit_path)
        _validate_source_audit_for_publication(
            audit_check,
            predictions,
            metrics,
            production_run,
        )
        staged_paths["source_audit_csv"] = audit_path

        publication_order = [
            name for name in staged_paths if name != "source_audit_csv"
        ] + ["source_audit_csv"]
        backup_dir = staging / "previous_artifacts"
        backup_dir.mkdir()
        final_paths = {
            name: output_dir / staged_paths[name].name for name in publication_order
        }
        replaced = []
        try:
            for name in publication_order:
                final_path = final_paths[name]
                if final_path.exists():
                    _replace_with_retry(final_path, backup_dir / final_path.name)
                _replace_with_retry(staged_paths[name], final_path)
                replaced.append(name)
        except BaseException:
            for name in reversed(replaced):
                final_paths[name].unlink(missing_ok=True)
            for backup in backup_dir.iterdir():
                _replace_with_retry(backup, output_dir / backup.name)
            raise
        return final_paths


def run_analysis(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    contemporaneous_params_path: Path = DEFAULT_CONTEMPORANEOUS_PARAMS,
    contemporaneous_reference_predictions_path: Path = (
        DEFAULT_CONTEMPORANEOUS_REFERENCE_PREDICTIONS
    ),
    contemporaneous_reference_audit_path: Path = (
        DEFAULT_CONTEMPORANEOUS_REFERENCE_AUDIT
    ),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    conditions: Sequence[str] | None = None,
    workers: int = DEFAULT_WORKERS,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    estimator_n_jobs: int | None = DEFAULT_ESTIMATOR_N_JOBS,
) -> Mapping[str, Path]:
    """Run selected conditions and publish a validated comparison artifact set."""
    selected, production_run = validate_run_configuration(
        conditions, output_dir, random_state, workers, estimator_n_jobs
    )
    forecasting, nowcasting, lookup = load_prepared_inputs(
        forecasting_path, nowcasting_path, country_lookup_path
    )
    train_mask, test_mask, now_train_mask, now_test_mask = (
        scatter.temporal_split_masks(forecasting, nowcasting, DEFAULT_CUTOFF)
    )
    layer1_features = tuple(loco.select_layer1_features(forecasting))
    feature_manifest = build_feature_manifest(layer1_features)
    contemporaneous_features = build_contemporaneous_feature_lists(
        nowcasting_path,
        layer1_features,
    )
    contemporaneous_fold_contract = load_contemporaneous_fold_contract(
        nowcasting_path,
        contemporaneous_reference_predictions_path,
        contemporaneous_reference_audit_path,
    )
    coordinates = build_coordinate_table(forecasting, nowcasting)
    coordinate_validation = build_coordinate_validation_record(coordinates)
    distance_matrix = build_distance_matrix(coordinates)
    knn5_neighbors = build_knn5_neighbors(distance_matrix)
    d200_neighbors = build_d200_neighbors(distance_matrix)
    condition_matrices = build_condition_matrices(
        forecasting,
        nowcasting,
        train_mask,
        test_mask,
        now_train_mask,
        now_test_mask,
        feature_manifest,
        distance_matrix,
        knn5_neighbors,
        d200_neighbors,
        lookup,
        selected,
    )
    general_params, phase3_params = loco.load_hyperparameters(
        general_params_path,
        phase3_params_path,
        random_state,
        estimator_n_jobs=estimator_n_jobs,
    )
    contemporaneous_params = load_contemporaneous_hyperparameters(
        contemporaneous_params_path,
        estimator_n_jobs,
    )
    predictions_by_group = {}
    for condition in selected:
        model_results = run_condition_models(
            condition,
            condition_matrices.forecasting[condition],
            condition_matrices.nowcasting[condition],
            train_mask,
            test_mask,
            now_train_mask,
            now_test_mask,
            condition_matrices.layer1_features[condition],
            condition_matrices.layer2_features[condition],
            general_params,
            phase3_params,
            contemporaneous_features[condition],
            contemporaneous_params,
            contemporaneous_fold_contract.fold_table,
            workers,
        )
        for model_name in MODEL_ORDER:
            predictions_by_group[(condition, model_name)] = model_results[model_name]
    combined = build_combined_predictions(
        predictions_by_group,
        selected,
        expected_test_rows=DEFAULT_EXPECTED_TEST_ROWS,
    )
    if combined["nonpositive_cumulative_prediction_sum"].any():
        affected = int(combined["nonpositive_cumulative_prediction_sum"].sum())
        raise ValueError(
            "Frozen spatial run produced nonpositive cumulative predictions; "
            f"publication stopped with {affected} affected rows."
        )
    if "baseline_with_lat_lon" in selected:
        validate_contemporaneous_baseline_reference(
            combined,
            contemporaneous_fold_contract,
        )
    metrics = build_metrics_table(
        combined,
        production_run=production_run,
        contemporaneous_fold_assignment_sha256=(
            contemporaneous_fold_contract.fold_assignment_sha256
        ),
    )
    if production_run:
        assert_frozen_main_result_reproduced(metrics)
    weight_diagnostics = pd.concat(
        [
            build_weight_diagnostics(knn5_neighbors, lookup),
            build_weight_diagnostics(d200_neighbors, lookup),
        ],
        ignore_index=True,
    )
    source_audit = build_source_audit(
        combined,
        metrics,
        condition_matrices.matrix_hashes,
        weight_diagnostics,
        condition_matrices.interpolation_audit,
        contemporaneous_features,
        contemporaneous_fold_contract,
        coordinate_validation,
        {
            "forecasting_input": Path(forecasting_path),
            "nowcasting_input": Path(nowcasting_path),
            "country_lookup": Path(country_lookup_path),
            "general_params": Path(general_params_path),
            "phase3_params": Path(phase3_params_path),
            "contemporaneous_params": Path(contemporaneous_params_path),
            "contemporaneous_reference_predictions": Path(
                contemporaneous_reference_predictions_path
            ),
            "contemporaneous_reference_audit": Path(
                contemporaneous_reference_audit_path
            ),
            "generator": Path(__file__),
        },
        canonical_dataframe_sha256(
            feature_manifest,
            ["layer", "feature_order"],
            list(FEATURE_MANIFEST_COLUMNS),
        ),
        random_state,
        workers,
        estimator_n_jobs,
        production_run,
    )
    return write_artifacts(
        predictions=combined,
        metrics=metrics,
        feature_manifest=feature_manifest,
        weight_diagnostics=weight_diagnostics,
        interpolation_audit=condition_matrices.interpolation_audit,
        interpolation_summary=condition_matrices.interpolation_summary,
        source_audit=source_audit,
        output_dir=output_dir,
        production_run=production_run,
    )


def main() -> None:
    arguments = parse_args()
    artifacts = run_analysis(
        forecasting_path=arguments.forecasting_input,
        nowcasting_path=arguments.nowcasting_input,
        country_lookup_path=arguments.country_lookup,
        general_params_path=arguments.general_params,
        phase3_params_path=arguments.phase3_params,
        contemporaneous_params_path=arguments.contemporaneous_params,
        contemporaneous_reference_predictions_path=(
            arguments.contemporaneous_reference_predictions
        ),
        contemporaneous_reference_audit_path=arguments.contemporaneous_reference_audit,
        output_dir=arguments.output_dir,
        conditions=arguments.conditions,
        workers=arguments.workers,
        random_state=arguments.random_state,
        estimator_n_jobs=arguments.estimator_n_jobs,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
