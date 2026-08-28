"""Generate the audited 1,170-row Forecasting and Nowcasting test artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import r2_score

import generate_leave_one_country_out_robustness as loco
import generate_phase_cumulative_scatter_comparison as temporal


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIR = REPO_ROOT / "1.Source Data"
PRODUCED_GRAPH_DIR = REPO_ROOT / "2.Source Code" / "produced_graph"

EXPECTED_SOURCE_ROWS = 5575
EXPECTED_TRAIN_ROWS = 4405
EXPECTED_TEST_ROWS = 1170
EXPECTED_TEST_AREAS = 646
CUTOFF = "2022-01-01"
POPULATION_ID = "model_temporal_2022_1170"
CANONICAL_KEY = ["area_id", "date"]
RESTORED_TEST_INDICES = frozenset({3374, 3517, 3534, 3553, 3567})
CANONICAL_OUTPUT_COLUMNS = [
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
]
SOURCE_OUTCOME_COLUMNS = [
    "overall_phase",
    "phase1_percent",
    "phase2_percent",
    "phase3_percent",
    "phase4_percent",
    "phase5_percent",
]
SOURCE_REQUIRED_COLUMNS = [*CANONICAL_KEY, *SOURCE_OUTCOME_COLUMNS, "lat", "lon"]
PREDICTION_REQUIRED_COLUMNS = [
    "source_row_index",
    *CANONICAL_KEY,
    "overall_phase",
    "overall_phase_pred",
    "phase3_pred",
]

DEFAULT_FORECASTING_INPUT = temporal.DEFAULT_FORECASTING_INPUT
DEFAULT_NOWCASTING_INPUT = temporal.DEFAULT_NOWCASTING_INPUT
DEFAULT_COUNTRY_LOOKUP = temporal.DEFAULT_COUNTRY_LOOKUP
DEFAULT_GENERAL_PARAMS = temporal.DEFAULT_GENERAL_PARAMS
DEFAULT_PHASE3_PARAMS = temporal.DEFAULT_PHASE3_PARAMS
DEFAULT_OUTPUT_ROOT = REPO_ROOT


def _require_columns(data: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_date_series(values: pd.Series, column: str = "date") -> pd.Series:
    present = values.notna()
    if not bool(present.all()):
        raise ValueError(f"Canonical key {column} contains missing values.")
    try:
        parsed = pd.to_datetime(values, errors="raise", format="mixed")
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(values, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"Canonical date column {column} contains invalid dates.") from error
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"Canonical date column {column} must be timezone-naive.")
    if not bool(parsed.eq(parsed.dt.normalize()).all()):
        raise ValueError(f"Canonical date column {column} must contain midnight values.")
    return parsed.dt.strftime("%Y-%m-%d")


def _normalized_key_frame(data: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(data, CANONICAL_KEY, name)
    keys = data.loc[:, CANONICAL_KEY].copy()
    if keys["area_id"].isna().any():
        raise ValueError(f"{name} canonical keys contain missing area_id values.")
    keys["date"] = _normalize_date_series(keys["date"])
    if keys.duplicated(CANONICAL_KEY).any():
        raise ValueError(f"{name} canonical keys are not unique.")
    return keys


def canonical_key_sha256(data: pd.DataFrame) -> str:
    """Hash a key-sorted, date-normalized canonical CSV representation."""
    keys = _normalized_key_frame(data, "Canonical hash input")
    ordered = keys.sort_values(CANONICAL_KEY, kind="mergesort")
    csv_options = {
        "index": False,
        "float_format": "%.17g",
        "na_rep": "<NA>",
    }
    try:
        serialized = ordered.to_csv(lineterminator="\n", **csv_options)
    except TypeError as error:
        if "lineterminator" not in str(error):
            raise
        serialized = ordered.to_csv(line_terminator="\n", **csv_options)
    payload = serialized.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_values(data: pd.DataFrame, name: str) -> None:
    _require_columns(data, SOURCE_REQUIRED_COLUMNS, name)
    if data[["lat", "lon"]].isna().any().any():
        raise ValueError(f"{name} coordinates contain missing values.")
    coordinates = data[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} coordinates must be finite numeric values.")
    if data[SOURCE_OUTCOME_COLUMNS].isna().any().any():
        raise ValueError(f"{name} outcome columns contain missing values.")
    outcomes = data[SOURCE_OUTCOME_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(outcomes.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} outcome columns must be finite numeric values.")
    phases = outcomes["overall_phase"]
    if not np.allclose(phases, phases.round()) or not phases.between(1, 5).all():
        raise ValueError(f"{name} overall_phase must contain integer phase labels 1--5.")


def validate_temporal_population(
    forecasting: pd.DataFrame,
    nowcasting: pd.DataFrame,
    cutoff: str = CUTOFF,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Validate the fixed model population without filtering or positional joins."""
    for name, data in (("Forecasting", forecasting), ("Nowcasting", nowcasting)):
        if len(data) != EXPECTED_SOURCE_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_SOURCE_ROWS} source rows for {name}, found {len(data)}."
            )
        _validate_source_values(data, f"{name} input")
        _normalized_key_frame(data, f"{name} input")

    train, test, now_train, now_test = temporal.temporal_split_masks(
        forecasting, nowcasting, cutoff
    )
    for name, train_mask, test_mask in (
        ("Forecasting", train, test),
        ("Nowcasting", now_train, now_test),
    ):
        if int(test_mask.sum()) != EXPECTED_TEST_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_TEST_ROWS} test rows for {name}, "
                f"found {int(test_mask.sum())}."
            )
        if int(train_mask.sum()) != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_TRAIN_ROWS} train rows for {name}, "
                f"found {int(train_mask.sum())}."
            )

    forecasting_keys = _normalized_key_frame(forecasting, "Forecasting input")
    nowcasting_keys = _normalized_key_frame(nowcasting, "Nowcasting input")
    forecasting_index = pd.MultiIndex.from_frame(forecasting_keys)
    nowcasting_index = pd.MultiIndex.from_frame(nowcasting_keys)
    if set(forecasting_index) != set(nowcasting_index):
        raise ValueError("Forecasting and Nowcasting canonical key sets differ.")

    left = forecasting.copy()
    right = nowcasting.copy()
    left["date"] = forecasting_keys["date"].to_numpy()
    right["date"] = nowcasting_keys["date"].to_numpy()
    left = left.set_index(CANONICAL_KEY).sort_index()
    right = right.set_index(CANONICAL_KEY).sort_index()
    for column in SOURCE_OUTCOME_COLUMNS:
        left_values = pd.to_numeric(left[column], errors="coerce")
        right_values = pd.to_numeric(right[column], errors="coerce")
        if not np.array_equal(left_values.to_numpy(), right_values.to_numpy()):
            raise ValueError(
                f"Forecasting and Nowcasting outcome values differ for {column}."
            )

    forecast_test_keys = forecasting_keys.loc[test]
    nowcast_test_keys = nowcasting_keys.loc[now_test]
    if set(pd.MultiIndex.from_frame(forecast_test_keys)) != set(
        pd.MultiIndex.from_frame(nowcast_test_keys)
    ):
        raise ValueError("Forecasting and Nowcasting test key sets differ.")
    if forecast_test_keys["area_id"].nunique() != EXPECTED_TEST_AREAS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_AREAS} test areas, found "
            f"{forecast_test_keys['area_id'].nunique()}."
        )
    return train, test, now_train, now_test


def _normalized_predictions(data: pd.DataFrame, name: str) -> pd.DataFrame:
    _require_columns(data, PREDICTION_REQUIRED_COLUMNS, name)
    result = data.copy()
    result["date"] = _normalize_date_series(result["date"])
    if result[CANONICAL_KEY].isna().any().any():
        raise ValueError(f"{name} canonical keys contain missing values.")
    if result.duplicated(CANONICAL_KEY).any():
        raise ValueError(f"{name} canonical keys are not unique.")
    if result["source_row_index"].isna().any() or not result[
        "source_row_index"
    ].is_unique:
        raise ValueError(f"{name} source_row_index must be complete and unique.")
    return result.sort_values(CANONICAL_KEY, kind="mergesort").reset_index(drop=True)


def assemble_all_prediction(
    forecasting_predictions: pd.DataFrame,
    nowcasting_predictions: pd.DataFrame,
    forecasting_source: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble Forecasting, Nowcasting, and source metadata by canonical keys."""
    forecasting = _normalized_predictions(
        forecasting_predictions, "Forecasting predictions"
    )
    nowcasting = _normalized_predictions(nowcasting_predictions, "Nowcasting predictions")
    if canonical_key_sha256(forecasting) != canonical_key_sha256(nowcasting):
        raise ValueError("Forecasting and Nowcasting prediction key sets differ.")

    forecasting_columns = [
        "source_row_index",
        *CANONICAL_KEY,
        "overall_phase",
        "overall_phase_pred",
        "phase3_pred",
    ]
    if "country_code_3" in forecasting.columns:
        forecasting_columns.append("country_code_3")
    forecast = forecasting.loc[:, forecasting_columns].rename(
        columns={"source_row_index": "test_index"}
    )
    nowcast = nowcasting.loc[
        :,
        [
            "source_row_index",
            *CANONICAL_KEY,
            "overall_phase",
            "overall_phase_pred",
            "phase3_pred",
        ],
    ].rename(
        columns={
            "source_row_index": "nowcast_source_row_index",
            "overall_phase": "nowcast_overall_phase",
            "overall_phase_pred": "nowcast_predict",
            "phase3_pred": "phase3_nowcast",
        }
    )
    combined = forecast.merge(nowcast, on=CANONICAL_KEY, how="inner", validate="one_to_one")
    if len(combined) != len(forecasting):
        raise ValueError("Prediction assembly changed the evaluation population.")
    if not combined["test_index"].eq(combined["nowcast_source_row_index"]).all():
        raise ValueError("Forecasting and Nowcasting source_row_index values differ by key.")
    if not combined["overall_phase"].eq(combined["nowcast_overall_phase"]).all():
        raise ValueError("Forecasting and Nowcasting truth values differ by key.")

    source = forecasting_source.copy()
    _require_columns(source, SOURCE_REQUIRED_COLUMNS, "Forecasting source")
    source["source_row_index"] = source.index.to_numpy()
    source["date"] = _normalize_date_series(source["date"])
    if source.duplicated(CANONICAL_KEY).any():
        raise ValueError("Forecasting source canonical keys are not unique.")
    source_columns = [
        "source_row_index",
        *CANONICAL_KEY,
        *SOURCE_OUTCOME_COLUMNS,
        "lat",
        "lon",
    ]
    if "country_code_3" in source.columns and "country_code_3" not in combined.columns:
        source_columns.append("country_code_3")
    source = source.loc[:, source_columns].rename(
        columns={"overall_phase": "source_overall_phase"}
    )
    combined = combined.merge(source, on=CANONICAL_KEY, how="inner", validate="one_to_one")
    if len(combined) != len(forecasting):
        raise ValueError("Source metadata assembly changed the evaluation population.")
    if not combined["test_index"].eq(combined["source_row_index"]).all():
        raise ValueError("test_index does not match source_row_index lineage by key.")

    combined["phase2plus_share"] = combined[
        [f"phase{phase}_percent" for phase in range(2, 6)]
    ].sum(axis=1)
    combined["phase3plus_share"] = combined[
        [f"phase{phase}_percent" for phase in range(3, 6)]
    ].sum(axis=1)
    combined["phase4plus_share"] = combined[
        ["phase4_percent", "phase5_percent"]
    ].sum(axis=1)
    combined["phase5_share"] = combined["phase5_percent"]
    combined["truth_disagreement"] = combined["source_overall_phase"].ne(
        combined["overall_phase"]
    )
    ordered_audit_columns = [
        "source_row_index",
        "source_overall_phase",
        "truth_disagreement",
        "country_code_3",
        *[f"phase{phase}_percent" for phase in range(1, 6)],
        "phase2plus_share",
        "phase3plus_share",
        "phase4plus_share",
        "phase5_share",
    ]
    ordered_audit_columns = [
        column for column in ordered_audit_columns if column in combined.columns
    ]
    result = combined.loc[
        :, [*CANONICAL_OUTPUT_COLUMNS, *ordered_audit_columns]
    ].sort_values(CANONICAL_KEY, kind="mergesort")
    return result.reset_index(drop=True)


def build_truth_disagreements(combined: pd.DataFrame) -> pd.DataFrame:
    """Return every source-versus-evaluation phase disagreement with lineage."""
    required = [
        *CANONICAL_OUTPUT_COLUMNS,
        "source_row_index",
        "source_overall_phase",
        "truth_disagreement",
        *[f"phase{phase}_percent" for phase in range(1, 6)],
        "phase2plus_share",
        "phase3plus_share",
        "phase4plus_share",
        "phase5_share",
    ]
    _require_columns(combined, required, "Combined prediction table")
    disagreements = combined.loc[combined["truth_disagreement"], required].copy()
    optional = [column for column in ("country_code_3",) if column in combined.columns]
    if optional:
        disagreements = combined.loc[
            combined["truth_disagreement"], [*required, *optional]
        ].copy()
    return disagreements.sort_values("test_index", kind="mergesort").reset_index(drop=True)


def _validate_phase_labels(data: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Column {column} contains missing phase labels.")
        if not np.allclose(values.to_numpy(), values.round().to_numpy()):
            raise ValueError(f"Column {column} must contain integer phase labels.")
        if not values.between(1, 5).all():
            raise ValueError(f"Column {column} contains phase labels outside 1--5.")


def _validate_prediction_artifact_common(
    data: pd.DataFrame,
    expected_rows: int,
    require_restored_indices: bool,
) -> pd.DataFrame:
    _require_columns(data, CANONICAL_OUTPUT_COLUMNS, "Canonical prediction artifact")
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(data)}.")
    result = data.loc[:, CANONICAL_OUTPUT_COLUMNS].copy()
    result["date"] = _normalize_date_series(result["date"])
    if result[CANONICAL_KEY].isna().any().any():
        raise ValueError("Canonical prediction keys contain missing values.")
    if result.duplicated(CANONICAL_KEY).any():
        raise ValueError("Canonical prediction keys are not unique.")

    test_index = pd.to_numeric(result["test_index"], errors="coerce")
    if test_index.isna().any() or not np.allclose(test_index, test_index.round()):
        raise ValueError("test_index must contain complete integer lineage values.")
    result["test_index"] = test_index.astype(int)
    if not result["test_index"].is_unique:
        raise ValueError("test_index values are not unique.")
    if require_restored_indices:
        missing_restored = sorted(RESTORED_TEST_INDICES.difference(result["test_index"]))
        if missing_restored:
            raise ValueError(
                f"Canonical prediction artifact is missing restored indices: {missing_restored}"
            )

    _validate_phase_labels(
        result, ["overall_phase", "overall_phase_pred", "nowcast_predict"]
    )
    numeric_columns = ["phase3_pred", "phase3_nowcast", "lat", "lon"]
    numeric = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise ValueError("Canonical predictions and coordinates must be finite numeric values.")
    result[numeric_columns] = numeric
    return result.sort_values(CANONICAL_KEY, kind="mergesort").reset_index(drop=True)


def validate_canonical_prediction_artifact(
    data: pd.DataFrame, expected_rows: int = EXPECTED_TEST_ROWS
) -> pd.DataFrame:
    """Validate an already written current 1,170 prediction table."""
    return _validate_prediction_artifact_common(
        data, expected_rows=expected_rows, require_restored_indices=True
    )


def _classification_metrics(
    actual_phase: pd.Series, predicted_phase: pd.Series, predicted_share: pd.Series
) -> dict[str, float]:
    actual = pd.to_numeric(actual_phase, errors="raise").astype(int)
    predicted = pd.to_numeric(predicted_phase, errors="raise").astype(int)
    share = pd.to_numeric(predicted_share, errors="raise").astype(float)
    actual_positive = actual.ge(3)
    predicted_positive = predicted.ge(3)
    true_positive = int((actual_positive & predicted_positive).sum())
    false_positive = int((~actual_positive & predicted_positive).sum())
    false_negative = int((actual_positive & ~predicted_positive).sum())
    actual_positive_count = true_positive + false_negative
    predicted_positive_count = true_positive + false_positive
    if actual_positive_count:
        sensitivity = true_positive / actual_positive_count
    else:
        sensitivity = np.nan
    if predicted_positive_count:
        precision = true_positive / predicted_positive_count
    else:
        precision = np.nan
    actual_share = actual_positive.astype(float)
    if len(actual_share) < 2 or actual_share.nunique() < 2:
        phase3_r2 = np.nan
    else:
        phase3_r2 = float(r2_score(actual_share, share))
    return {
        "accuracy": float(actual.eq(predicted).mean()),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "phase3_r2": phase3_r2,
    }


def build_overlap_calibration(
    staged: pd.DataFrame, legacy: pd.DataFrame
) -> pd.DataFrame:
    """Compare one staged controlled rerun with all legacy overlapping keys."""
    staged_valid = _validate_prediction_artifact_common(
        staged, expected_rows=EXPECTED_TEST_ROWS, require_restored_indices=True
    )
    expected_overlap = EXPECTED_TEST_ROWS - len(RESTORED_TEST_INDICES)
    legacy_valid = _validate_prediction_artifact_common(
        legacy, expected_rows=expected_overlap, require_restored_indices=False
    )
    overlap = legacy_valid.merge(
        staged_valid,
        on=CANONICAL_KEY,
        how="inner",
        suffixes=("_legacy", "_staged"),
        validate="one_to_one",
    )
    if len(overlap) != expected_overlap:
        raise ValueError(
            f"Expected {expected_overlap} legacy overlap rows, found {len(overlap)}."
        )
    staged_only = staged_valid.merge(
        legacy_valid[CANONICAL_KEY],
        on=CANONICAL_KEY,
        how="left",
        indicator=True,
        validate="one_to_one",
    ).loc[lambda frame: frame["_merge"].eq("left_only")]
    if set(staged_only["test_index"]) != set(RESTORED_TEST_INDICES):
        raise ValueError("Staged-only rows do not equal the restored test indices.")
    if not overlap["overall_phase_legacy"].eq(overlap["overall_phase_staged"]).all():
        raise ValueError("Legacy and staged evaluation truth differs on shared keys.")

    records: list[dict[str, object]] = []
    model_specs = {
        "Forecasting": ("overall_phase_pred", "phase3_pred"),
        "Nowcasting": ("nowcast_predict", "phase3_nowcast"),
    }
    differing_by_key: dict[tuple[object, object], list[str]] = {}
    drift_detected = False
    for model_name, (phase_column, share_column) in model_specs.items():
        phase_legacy = overlap[f"{phase_column}_legacy"]
        phase_staged = overlap[f"{phase_column}_staged"]
        share_legacy = pd.to_numeric(
            overlap[f"{share_column}_legacy"], errors="raise"
        )
        share_staged = pd.to_numeric(
            overlap[f"{share_column}_staged"], errors="raise"
        )
        phase_equal = phase_legacy.eq(phase_staged)
        share_equal = share_legacy.eq(share_staged)
        model_difference = ~(phase_equal & share_equal)
        if bool(model_difference.any()):
            drift_detected = True
            for row in overlap.loc[model_difference, CANONICAL_KEY].itertuples(
                index=False, name=None
            ):
                differing_by_key.setdefault(row, []).append(model_name)
        absolute_share_difference = (share_staged - share_legacy).abs()
        legacy_metrics = _classification_metrics(
            overlap["overall_phase_legacy"], phase_legacy, share_legacy
        )
        staged_metrics = _classification_metrics(
            overlap["overall_phase_staged"], phase_staged, share_staged
        )
        record: dict[str, object] = {
            "record_type": "model_summary",
            "model": model_name,
            "overlap_rows": len(overlap),
            "restored_rows": len(staged_only),
            "phase_prediction_match_rows": int(phase_equal.sum()),
            "phase_prediction_difference_rows": int((~phase_equal).sum()),
            "phase3_exact_match_rows": int(share_equal.sum()),
            "phase3_max_abs_difference": float(absolute_share_difference.max()),
            "phase3_mean_abs_difference": float(absolute_share_difference.mean()),
            "prediction_drift_detected": bool(model_difference.any()),
            "formal_replacement_allowed": not bool(model_difference.any()),
        }
        for metric in ("accuracy", "sensitivity", "precision", "phase3_r2"):
            legacy_value = legacy_metrics[metric]
            staged_value = staged_metrics[metric]
            record[f"legacy_{metric}"] = legacy_value
            record[f"staged_{metric}"] = staged_value
            record[f"{metric}_delta"] = staged_value - legacy_value
        records.append(record)

    records.insert(
        0,
        {
            "record_type": "summary",
            "model": "All",
            "overlap_rows": len(overlap),
            "restored_rows": len(staged_only),
            "prediction_drift_detected": drift_detected,
            "formal_replacement_allowed": not drift_detected,
        },
    )
    records.append(
        {
            "record_type": "restored_coverage",
            "model": "All",
            "overlap_rows": len(overlap),
            "restored_rows": len(staged_only),
            "prediction_drift_detected": drift_detected,
            "formal_replacement_allowed": not drift_detected,
            "differing_models_json": "[]",
        }
    )
    for (area_id, date), model_names in sorted(differing_by_key.items()):
        row = staged_valid.loc[
            staged_valid["area_id"].eq(area_id) & staged_valid["date"].eq(date)
        ].iloc[0]
        records.append(
            {
                "record_type": "prediction_difference",
                "model": "Multiple" if len(model_names) > 1 else model_names[0],
                "area_id": area_id,
                "date": date,
                "test_index": int(row["test_index"]),
                "overlap_rows": len(overlap),
                "restored_rows": len(staged_only),
                "prediction_drift_detected": True,
                "formal_replacement_allowed": False,
                "differing_models_json": json.dumps(model_names),
            }
        )
    return pd.DataFrame(records)


def assert_promotion_authorized(calibration: pd.DataFrame) -> None:
    """Fail closed unless one calibration summary reports exact overlap."""
    summary = calibration.loc[calibration["record_type"].eq("summary")]
    if len(summary) != 1:
        raise ValueError("Overlap calibration must contain exactly one summary row.")
    if not bool(summary.iloc[0]["formal_replacement_allowed"]):
        raise RuntimeError(
            "Controlled 1,170 predictions differ from the legacy overlap; "
            "formal replacement is blocked pending legacy Windows/XGBoost reconstruction."
        )


def _nowcast_visualization_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    frame = canonical[
        ["test_index", "overall_phase", "area_id", "date"]
    ].copy()
    frame["overall_phase_pred"] = canonical["nowcast_predict"].to_numpy()
    frame["phase3_pred"] = canonical["phase3_nowcast"].to_numpy()
    frame["_merge"] = "both"
    return frame[
        [
            "test_index",
            "overall_phase",
            "overall_phase_pred",
            "phase3_pred",
            "area_id",
            "date",
            "_merge",
        ]
    ]


def write_staged_artifacts(
    combined: pd.DataFrame,
    truth_disagreements: pd.DataFrame,
    overlap_calibration: pd.DataFrame | None,
    staging_dir: Path,
) -> dict[str, Path]:
    """Write, reread, and validate the canonical prediction artifacts."""
    staging_dir = Path(staging_dir)
    source_dir = staging_dir / "1.Source Data"
    graph_dir = staging_dir / "2.Source Code" / "produced_graph"
    source_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "all_prediction": source_dir / "All_prediction.csv",
        "df_vis_nowcast": source_dir / "df_vis_nowacast.csv",
        "truth_disagreements": source_dir / "All_prediction_truth_disagreements.csv",
    }
    if overlap_calibration is not None:
        paths["overlap_calibration"] = (
            graph_dir / "all_prediction_temporal_test_overlap_calibration.csv"
        )
    canonical = validate_canonical_prediction_artifact(
        combined.loc[:, CANONICAL_OUTPUT_COLUMNS], expected_rows=EXPECTED_TEST_ROWS
    )
    disagreement_indices = set(
        pd.to_numeric(truth_disagreements["test_index"], errors="raise").astype(int)
    )
    if disagreement_indices != set(RESTORED_TEST_INDICES):
        raise ValueError(
            "Truth-disagreement rows do not equal the five restored test indices."
        )
    canonical.to_csv(paths["all_prediction"], index=False)
    _nowcast_visualization_frame(canonical).to_csv(paths["df_vis_nowcast"], index=False)
    truth_disagreements.to_csv(paths["truth_disagreements"], index=False)
    if overlap_calibration is not None:
        overlap_calibration.to_csv(paths["overlap_calibration"], index=False)

    reread = pd.read_csv(paths["all_prediction"])
    validate_canonical_prediction_artifact(reread, expected_rows=EXPECTED_TEST_ROWS)
    visualization = pd.read_csv(paths["df_vis_nowcast"])
    expected_visualization_columns = [
        "test_index",
        "overall_phase",
        "overall_phase_pred",
        "phase3_pred",
        "area_id",
        "date",
        "_merge",
    ]
    if visualization.columns.tolist() != expected_visualization_columns:
        raise ValueError("Staged nowcasting visualization schema changed.")
    if len(visualization) != EXPECTED_TEST_ROWS or not visualization["_merge"].eq(
        "both"
    ).all():
        raise ValueError("Staged nowcasting visualization population is incomplete.")
    return paths


def build_source_audit(
    paths: Mapping[str, Path],
    calibration: pd.DataFrame | None,
    truth_disagreements: pd.DataFrame,
    forecasting_path: Path,
    nowcasting_path: Path,
    country_lookup_path: Path,
    general_params_path: Path,
    phase3_params_path: Path,
    random_state: int,
    workers: int,
    estimator_n_jobs: int | None,
) -> pd.DataFrame:
    if calibration is None:
        prediction_drift_detected: object = pd.NA
        formal_replacement_allowed: object = pd.NA
        legacy_overlap_checked = False
    else:
        summary = calibration.loc[calibration["record_type"].eq("summary")].iloc[0]
        prediction_drift_detected = bool(summary["prediction_drift_detected"])
        formal_replacement_allowed = bool(summary["formal_replacement_allowed"])
        legacy_overlap_checked = True
    canonical = pd.read_csv(paths["all_prediction"])
    record: dict[str, object] = {
        "evaluation_population_id": POPULATION_ID,
        "source_rows": EXPECTED_SOURCE_ROWS,
        "train_rows": EXPECTED_TRAIN_ROWS,
        "test_rows": EXPECTED_TEST_ROWS,
        "test_areas": EXPECTED_TEST_AREAS,
        "cutoff": CUTOFF,
        "canonical_key": "+".join(CANONICAL_KEY),
        "canonical_key_sha256": canonical_key_sha256(canonical),
        "truth_disagreement_rows": len(truth_disagreements),
        "restored_test_indices_json": json.dumps(sorted(RESTORED_TEST_INDICES)),
        "legacy_overlap_checked": legacy_overlap_checked,
        "prediction_drift_detected": prediction_drift_detected,
        "formal_replacement_allowed": formal_replacement_allowed,
        "formal_complete": True,
        "random_state": random_state,
        "estimator_n_jobs": estimator_n_jobs if estimator_n_jobs is not None else pd.NA,
        "estimator_uses_default_n_jobs": estimator_n_jobs is None,
        "model_workers": workers,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "platform": platform.platform(),
        "generator_sha256": file_sha256(Path(__file__)),
    }
    input_paths = {
        "forecasting_input": forecasting_path,
        "nowcasting_input": nowcasting_path,
        "country_lookup": country_lookup_path,
        "general_params": general_params_path,
        "phase3_params": phase3_params_path,
    }
    for name, path in input_paths.items():
        record[f"{name}_path"] = str(Path(path).resolve())
        record[f"{name}_sha256"] = file_sha256(path)
    for name, path in paths.items():
        record[f"{name}_path"] = str(Path(path).resolve())
        record[f"{name}_sha256"] = file_sha256(path)
    return pd.DataFrame([record])


def run_generation(
    forecasting_path: Path = DEFAULT_FORECASTING_INPUT,
    nowcasting_path: Path = DEFAULT_NOWCASTING_INPUT,
    country_lookup_path: Path = DEFAULT_COUNTRY_LOOKUP,
    general_params_path: Path = DEFAULT_GENERAL_PARAMS,
    phase3_params_path: Path = DEFAULT_PHASE3_PARAMS,
    legacy_input_path: Path | None = None,
    staging_dir: Path = DEFAULT_OUTPUT_ROOT,
    cutoff: str = CUTOFF,
    random_state: int = 0,
    workers: int = 1,
    estimator_n_jobs: int | None = None,
) -> dict[str, Path]:
    """Run both models once and write the canonical 1,170-row artifact tree."""
    if workers < 1 or workers > 2:
        raise ValueError("workers must be 1 or 2 for the formal temporal run.")
    forecasting_path = Path(forecasting_path)
    nowcasting_path = Path(nowcasting_path)
    forecasting_source = pd.read_csv(forecasting_path)
    nowcasting_source = pd.read_csv(nowcasting_path)
    validate_temporal_population(forecasting_source, nowcasting_source, cutoff)
    predictions = temporal.run_temporal_predictions(
        forecasting_path=forecasting_path,
        nowcasting_path=nowcasting_path,
        country_lookup_path=Path(country_lookup_path),
        general_params_path=Path(general_params_path),
        phase3_params_path=Path(phase3_params_path),
        cutoff=cutoff,
        random_state=random_state,
        workers=workers,
        estimator_n_jobs=estimator_n_jobs,
    )
    if set(predictions) != {"Forecasting", "Nowcasting"}:
        raise ValueError("Temporal runner did not return both required models.")
    for name, frame in predictions.items():
        if len(frame) != EXPECTED_TEST_ROWS:
            raise ValueError(
                f"Expected {EXPECTED_TEST_ROWS} predictions for {name}, found {len(frame)}."
            )
    combined = assemble_all_prediction(
        predictions["Forecasting"], predictions["Nowcasting"], forecasting_source
    )
    canonical = validate_canonical_prediction_artifact(
        combined.loc[:, CANONICAL_OUTPUT_COLUMNS], expected_rows=EXPECTED_TEST_ROWS
    )
    if canonical["area_id"].nunique() != EXPECTED_TEST_AREAS:
        raise ValueError("Assembled predictions changed the test-area population.")
    disagreements = build_truth_disagreements(combined)
    if set(disagreements["test_index"]) != set(RESTORED_TEST_INDICES):
        raise ValueError("The live truth-disagreement set differs from the accepted five rows.")
    calibration = None
    if legacy_input_path is not None:
        legacy = pd.read_csv(legacy_input_path)
        calibration = build_overlap_calibration(canonical, legacy)
    paths = write_staged_artifacts(
        combined, disagreements, calibration, staging_dir=Path(staging_dir)
    )
    audit = build_source_audit(
        paths,
        calibration,
        disagreements,
        forecasting_path,
        nowcasting_path,
        Path(country_lookup_path),
        Path(general_params_path),
        Path(phase3_params_path),
        random_state,
        workers,
        estimator_n_jobs,
    )
    audit_path = (
        Path(staging_dir)
        / "2.Source Code"
        / "produced_graph"
        / "all_prediction_temporal_test_source_audit.csv"
    )
    audit.to_csv(audit_path, index=False)
    paths["source_audit"] = audit_path
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasting-input", type=Path, default=DEFAULT_FORECASTING_INPUT)
    parser.add_argument("--nowcasting-input", type=Path, default=DEFAULT_NOWCASTING_INPUT)
    parser.add_argument("--country-lookup", type=Path, default=DEFAULT_COUNTRY_LOOKUP)
    parser.add_argument("--general-params", type=Path, default=DEFAULT_GENERAL_PARAMS)
    parser.add_argument("--phase3-params", type=Path, default=DEFAULT_PHASE3_PARAMS)
    parser.add_argument(
        "--legacy-input",
        type=Path,
        default=None,
        help="Optional legacy 1,165 input for diagnostic overlap comparison only.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output root containing 1.Source Data and 2.Source Code directories.",
    )
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=None,
        help="XGBoost thread count; omit to preserve the notebook estimator default.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = run_generation(
        forecasting_path=args.forecasting_input,
        nowcasting_path=args.nowcasting_input,
        country_lookup_path=args.country_lookup,
        general_params_path=args.general_params,
        phase3_params_path=args.phase3_params,
        legacy_input_path=args.legacy_input,
        staging_dir=args.staging_dir,
        cutoff=args.cutoff,
        random_state=args.random_state,
        workers=args.workers,
        estimator_n_jobs=args.estimator_n_jobs,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
